import argparse
import os

import numpy as np
import tqdm
import yaml

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from .modules import StrokesPrediction, StrokesSynthesis
from .data import StrokesDataset
from .utils import plot_stroke, save_model, copy_to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strokes_path", type=str, default="data/strokes.npy")
    parser.add_argument("--texts_path", type=str, default="data/sentences.txt")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--grad_norm", type=float, default=10.0)
    parser.add_argument("--model_type", choices=["prediction", "synthesis"], required=True)

    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')

    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    print("Device: {}".format(args.device))
    strokes = np.load(args.strokes_path, encoding="latin1", allow_pickle=True)
    with open(args.texts_path) as fin:
        texts = list(map(lambda x: x.strip(), fin))

    attention_scale = 1. / np.mean([len(stroke) / len(sentence) for stroke, sentence in zip(strokes, texts)])
    print("Attention scale: {}".format(attention_scale))

    alphabet = {}
    for t in texts:
        for c in t:
            if c not in alphabet:
                alphabet[c] = len(alphabet)
    data_params = {
        "attention_scale": attention_scale,
        "alphabet": alphabet
    }
    with open("data/data_params.yaml", "w") as fout:
        yaml.dump(data_params, fout)
    inv_alphabet = {y: x for x, y in alphabet.items()}
    test_text = "How r you doin?"

    train_strokes, valid_strokes, train_texts, valid_texts = train_test_split(strokes, texts, test_size=0.1)
    no_text = args.model_type == "prediction"
    train_dataset = StrokesDataset(train_strokes, train_texts, alphabet, no_text)
    valid_dataset = StrokesDataset(valid_strokes, valid_texts, alphabet, no_text)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    if args.model_type == "prediction":
        model = StrokesPrediction(hidden_size=args.hidden_size)
        model.to(args.device)
        model.sample(device=args.device)
    elif args.model_type == "synthesis":
        model = StrokesSynthesis(
            alphabet=alphabet,
            attention_scale=attention_scale,
            hidden_size=args.hidden_size
        )
        model.to(args.device)
        model.sample(torch.LongTensor([alphabet[x] for x in test_text])[None, :].to(args.device), device=args.device)
    else:
        raise ValueError("unknown model type")

    optmizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    writer = SummaryWriter(args.model_dir)
    print("Starting to train...")
    global_step = 0
    for i in range(args.num_epochs):
        print("Epoch {} / {}".format(i + 1, args.num_epochs))
        for batch_X in tqdm.tqdm(train_dataloader):
            copy_to_device(batch_X, args.device)
            batch_X["strokes_inputs"] = batch_X["strokes_inputs"].permute((1, 0, 2))
            batch_X["strokes_targets"] = batch_X["strokes_targets"].permute((1, 0, 2))
            mixture_loss, end_loss = model.loss(**batch_X, device=args.device)
            loss = mixture_loss + end_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optmizer.step()
            optmizer.zero_grad()
            writer.add_scalar("Mixture loss/train", mixture_loss.cpu().detach().numpy(), global_step=global_step)
            writer.add_scalar("End loss/train", end_loss.cpu().detach().numpy(), global_step=global_step)
            loss = loss.cpu().detach().numpy()
            print("Loss: {}".format(loss))
            writer.add_scalar("Loss/train", loss, global_step=global_step)
            global_step += 1

        model_path = os.path.join(args.model_dir, "epoch_{}.pt".format(i))
        save_model(model, model_path)

        valid_losses = []
        valid_end_losses = []
        valid_mixture_losses = []
        for batch_X in tqdm.tqdm(valid_dataloader):
            copy_to_device(batch_X, args.device)
            batch_X["strokes_inputs"] = batch_X["strokes_inputs"].permute((1, 0, 2))
            batch_X["strokes_targets"] = batch_X["strokes_targets"].permute((1, 0, 2))
            mixture_loss, end_loss = model.loss(**batch_X, device=args.device)
            loss = mixture_loss + end_loss
            valid_losses.append(loss.cpu().detach().numpy())
            valid_end_losses.append(end_loss.cpu().detach().numpy())
            valid_mixture_losses.append(mixture_loss.cpu().detach().numpy())

        writer.add_scalar("Mixture loss/valid", np.mean(valid_mixture_losses), global_step=i)
        writer.add_scalar("End loss/valid", np.mean(valid_end_losses), global_step=i)
        writer.add_scalar("Loss/valid", np.mean(valid_losses), global_step=i)

        if args.model_type == "synthesis":
            h_0, h_1, h_2, prev_w_t, prev_k = model.init_states(1, device=args.device)
            strokes_inputs = torch.FloatTensor(valid_dataset[0]["strokes_inputs"])[:, None, :].to(args.device)
            text_inputs = torch.LongTensor(valid_dataset[0]["text"])[None, :].to(args.device)
            text_lengths = torch.LongTensor([valid_dataset[0]["text_lengths"]]).to(args.device)

            mixture_params, end_of_stroke_logits, h_0, h_1, h_2, prev_w_t, prev_k, alignments = model(
                strokes_inputs, text_inputs, text_lengths, h_0, h_1, h_2, prev_w_t, prev_k
            )
            first_alignment = alignments[: valid_dataset[0]["strokes_lengths"], 0, : text_lengths[0]]
            first_alignment = first_alignment.detach().cpu().numpy().T
            writer.add_image("alignment", first_alignment[None, :])

            text = "".join([inv_alphabet[int(x)] for x in valid_dataset[0]["text"]]).strip()
            sample = model.sample(text_inputs, device=args.device)
            img_dir = os.path.join(args.model_dir, "imgs")
            os.makedirs(img_dir, exist_ok=True)
            print(text)
            plot_stroke(sample, save_name=os.path.join(img_dir, "epoch_{}".format(i)))
        else:
            sample = model.sample(device=args.device)
            img_dir = os.path.join(args.model_dir, "imgs")
            os.makedirs(img_dir, exist_ok=True)
            plot_stroke(sample, save_name=os.path.join(img_dir, "epoch_{}".format(i)))


if __name__ == "__main__":
    main()
