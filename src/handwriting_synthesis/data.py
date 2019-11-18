import numpy as np
import torch

from torch.utils.data import Dataset


class StrokesDataset(Dataset):
    def __init__(self, strokes, texts, alphabet, no_text=False):
        self.strokes_lengths = np.array(list(map(lambda x: x.shape[0], strokes)))
        max_length = len(max(strokes, key=lambda x: len(x)))
        self.strokes = np.array(
            list(
                map(
                    lambda x: np.concatenate([
                        np.zeros((1, 3)),
                        x,
                        np.zeros((max_length - len(x), 3))
                    ], axis=0),
                    strokes
                )
            ),
            dtype=np.float32,
        )
        self.text_lengths = np.array(list(map(lambda x: len(x), texts)))
        self.max_text_length = np.max(self.text_lengths)

        self.alphabet = alphabet
        self.texts = texts
        self.no_text = no_text

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        char_indices = [self.alphabet[c] for c in self.texts[idx]]
        # padding
        for _ in range(self.max_text_length - self.text_lengths[idx]):
            char_indices.append(self.alphabet[" "])

        sample = {
            'strokes_inputs': self.strokes[idx][:-1],
            'strokes_targets': self.strokes[idx][1:],
            'strokes_lengths': self.strokes_lengths[idx],
        }
        if not self.no_text:
            sample['text'] = np.array(char_indices, dtype=np.int64)
            sample['text_lengths'] = self.text_lengths[idx]
        return sample
