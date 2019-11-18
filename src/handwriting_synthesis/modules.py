import numpy as np

import torch
import torch.nn as nn


def sequence_mask(sequence_lengths, maxlen=None):
    batch_size = sequence_lengths.shape[0]
    if maxlen is None:
        maxlen = int(torch.max(sequence_lengths))
    indices = (torch.arange(maxlen).unsqueeze(0).expand((batch_size, maxlen))).to(sequence_lengths.device)
    return (indices < sequence_lengths.unsqueeze(1).expand((batch_size, maxlen))).float()


class GaussianMixture(nn.Module):
    def __init__(self, input_size, num_components):
        super(GaussianMixture, self).__init__()
        self.num_components = num_components
        self.means_layer = nn.Linear(input_size, 2 * num_components)
        self.log_sigmas_layer = nn.Linear(input_size, 2 * num_components)
        self.correlations_layer = nn.Linear(input_size, num_components)
        self.mixture_weights_layer = nn.Linear(input_size, num_components)

    def forward(self, inputs):
        """
            inputs: [T, B, H]
        """
        batch_size = inputs.shape[1]
        means = (self.means_layer(inputs)).view(-1, batch_size, self.num_components, 2)
        log_sigmas = self.log_sigmas_layer(inputs)
        log_sigmas = log_sigmas.view(-1, batch_size, self.num_components, 2)
        correlations = torch.tanh(self.correlations_layer(inputs))
        mixture_weights_logits = self.mixture_weights_layer(inputs)
        return (means, log_sigmas, correlations), mixture_weights_logits

    def log_prob(self, data, mixture_params):
        """
        :param data: [T, B, 2]
        :param mixture_params: (means: [T, B, N, 2], log_sigmas: [T, B, N, 2], correlations: [T, B, N]),
                                mixture_weights_logits [T, B, N]
        :return:
        """
        (means, log_sigmas, correlations), mixture_weights_logits = mixture_params
        sigmas = torch.exp(log_sigmas)
        mu_x, mu_y = means[..., 0], means[..., 1]
        sigma_x, sigma_y = sigmas[..., 0], sigmas[..., 1]
        x, y = data[..., 0][..., None], data[..., 1][..., None]
        Z = ((x - mu_x) / sigma_x) ** 2 + ((y - mu_y) / sigma_y) ** 2 - \
            2 * correlations * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y)
        components_log_probs = -(log_sigmas[..., 0] + log_sigmas[..., 1] + np.log(2 * np.pi) +
                                 0.5 * torch.log(1 - correlations ** 2) + 0.5 * Z / (1 - correlations ** 2))
        mixture_weights_log_probs = torch.log_softmax(mixture_weights_logits, dim=-1)
        log_probs = torch.logsumexp(mixture_weights_log_probs + components_log_probs, dim=-1)
        return log_probs

    def sample(self, mixture_params, bias=None):
        (means, log_sigmas, correlations), mixture_weights_logits = mixture_params
        if bias is not None:
            mixture_weights_logits *= (1 + bias)
            log_sigmas -= bias

        cat_d = torch.distributions.Categorical(logits=mixture_weights_logits)
        index = int(cat_d.sample())
        mean = means[..., index, :]
        sigma = torch.exp(log_sigmas[..., index, :])
        correlation = correlations[..., index]
        d = torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(mean),
            covariance_matrix=torch.eye(2, device=mean.device)[None, None, :],
        )
        sample = d.sample()
        x, y = sample[..., 0], sample[..., 1]
        scaled_sample = torch.cat(
            [x * sigma[..., 0], sigma[..., 1] * (x * correlation + y * (1 - correlation ** 2) ** 0.5)],
            dim=-1
        )
        return scaled_sample + mean


class StrokesPrediction(nn.Module):
    def __init__(self, input_size=3, hidden_size=200, num_components=20):
        super(StrokesPrediction, self).__init__()
        self.rnn_0 = nn.LSTM(input_size, hidden_size, 1)
        self.rnn_1 = nn.LSTM(input_size + hidden_size, hidden_size, 1)
        self.rnn_2 = nn.LSTM(input_size + hidden_size, hidden_size, 1)
        self.mixture = GaussianMixture(hidden_size * 3, num_components)
        self.end_of_stroke_layer = nn.Linear(hidden_size * 3, 1)

        self.hidden_size = hidden_size
        self.num_components = num_components
        self.input_size = input_size

    def forward(self, inputs, h_0, h_1, h_2):
        y_0, h_0 = self.rnn_0(inputs, h_0)
        y_1, h_1 = self.rnn_1(torch.cat([inputs, y_0], dim=-1), h_1)
        y_2, h_2 = self.rnn_2(torch.cat([inputs, y_1], dim=-1), h_2)

        rnn_outputs = torch.cat([y_0, y_1, y_2], dim=-1)
        mixture_params = self.mixture(rnn_outputs)
        end_of_stroke_logits = self.end_of_stroke_layer(rnn_outputs)
        return mixture_params, end_of_stroke_logits, h_0, h_1, h_2

    def log_prob(self, x, mixture_params):
        return self.mixture.log_prob(x, mixture_params)

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        return [(weight.new_zeros(1, batch_size, self.hidden_size).to(device),
                 weight.new_zeros(1, batch_size, self.hidden_size).to(device)) for _ in range(3)]

    def loss(self, strokes_inputs, strokes_lengths, strokes_targets, device):
        hidden_states = self.init_hidden(int(strokes_inputs.shape[1]), device)
        mixture_params, end_of_stroke_logits, h_0, h_1, h_2 = self.forward(strokes_inputs, *hidden_states)
        model_log_prob = self.log_prob(strokes_targets[..., 1:], mixture_params)
        end_loss = nn.functional.binary_cross_entropy_with_logits(
            end_of_stroke_logits[..., 0],
            strokes_targets[..., 0],
            reduction="none",
        )
        mask = (sequence_mask(strokes_lengths, maxlen=strokes_inputs.shape[0])).permute(1, 0)
        strokes_lengths = strokes_lengths.float()
        mixture_loss = torch.mean(torch.sum(-model_log_prob * mask, dim=0) / strokes_lengths)
        end_loss = torch.mean(torch.sum(end_loss * mask, dim=0) / strokes_lengths)
        return mixture_loss, end_loss

    def sample(self, length=700, random_seed=1, device=None):
        torch.manual_seed(random_seed)
        h_0, h_1, h_2 = self.init_hidden(1, device=device)
        inputs = torch.zeros(1, 1, self.input_size).to(device)
        result = [inputs.cpu().numpy()]
        for i in range(length):
            mixture_params, end_of_stroke_logits, h_0, h_1, h_2 = self.forward(inputs, h_0, h_1, h_2)
            stroke_sample = self.mixture.sample(mixture_params)
            end_stroke = torch.distributions.Bernoulli(logits=end_of_stroke_logits).sample()
            inputs = torch.cat([end_stroke, stroke_sample], dim=-1)
            result.append(inputs.detach().cpu().numpy())
        return np.concatenate(result)[:, 0]


class WindowNetwork(nn.Module):
    def __init__(self, num_gaussians, num_letters=26, hidden_size=200, attention_scale=0.05):
        super(WindowNetwork, self).__init__()
        self.window_layer = nn.Linear(hidden_size, 3 * num_gaussians)

        self.num_gaussians = num_gaussians
        self.hidden_size = hidden_size
        self.num_letters = num_letters
        self.attention_scale = attention_scale  # defines the angle of the attention line

    def forward(self, text_inputs, text_lengths, y_0, prev_k):
        """
            text_inputs: [B, U] -> str
            text_lengths: [B,]
            y_0: [B, H]
            prev_k: [B, N]
        """
        text_outputs = nn.functional.one_hot(text_inputs, num_classes=self.num_letters).float()  # [B, U, L]
        outputs = self.window_layer(y_0)  # [B, N]
        a = torch.exp(outputs[..., : self.num_gaussians])[..., None]  # [B, N, 1]
        b = torch.exp(outputs[..., self.num_gaussians: 2 * self.num_gaussians])[..., None]  # [B, N, 1]
        k = (torch.exp(outputs[..., -self.num_gaussians:]) * self.attention_scale + prev_k)[..., None]  # [B, N, 1]

        maxlen = int(text_inputs.shape[1])
        text_mask = sequence_mask(text_lengths, maxlen=maxlen)
        indices = (torch.arange(end=maxlen)[None, None, :]).float().to(text_lengths.device)  # [1, 1, U]
        phi_t_u = torch.sum(a * torch.exp(-b * (k - indices) ** 2), dim=1) * text_mask  # [B, U]
        w_t = torch.sum(text_outputs * phi_t_u[..., None], dim=1)  # [B, L]
        return w_t, k[..., 0], phi_t_u


class StrokesSynthesis(nn.Module):
    def __init__(self, alphabet, attention_scale, window_num_gaussians=10,
                 input_size=3, hidden_size=400, num_components=20):
        super(StrokesSynthesis, self).__init__()
        num_letters = len(alphabet)
        self.rnn_0 = nn.LSTM(input_size + num_letters, hidden_size, 1)
        self.rnn_1 = nn.LSTM(hidden_size + input_size + num_letters, hidden_size, 1)
        self.rnn_2 = nn.LSTM(hidden_size + input_size + num_letters, hidden_size, 1)

        self.mixture = GaussianMixture(3 * hidden_size, num_components)
        self.end_of_stroke_layer = nn.Linear(3 * hidden_size, 1)
        self.window_network = WindowNetwork(
            num_gaussians=window_num_gaussians,
            num_letters=num_letters,
            hidden_size=hidden_size,
            attention_scale=attention_scale,
        )
        self.attention_scale = attention_scale
        self.window_num_gaussians = window_num_gaussians
        self.hidden_size = hidden_size
        self.alphabet = alphabet
        self.num_letters = num_letters
        self.num_components = num_components
        self.input_size = input_size

    def forward(self, strokes_inputs, text_inputs, text_lengths, h_0, h_1, h_2,
                prev_w_t, prev_k, bias=None):
        """
            strokes_inputs: [T, B, 3]
            text_inputs: [B, U]
            text_lengths: [B,]
            h_0: ([1, B, H], [1, B, H])
            h_upper: ([2, B, H], [2, B, H])
            prev_w_t: [B, L]
            prev_k: [B, N]
        """
        rnn_0_outputs = []
        alignments = []
        w_ts = []
        for x in strokes_inputs.unbind(0):
            rnn_input = torch.cat([x, prev_w_t], dim=-1)[None, ...]  # [1, B, num_letters + 3]
            y_0, h_0 = self.rnn_0(rnn_input, h_0)
            prev_w_t, prev_k, phi_t_u = self.window_network(
                text_inputs, text_lengths, y_0[0], prev_k
            )
            rnn_0_outputs.append(y_0)
            alignments.append(phi_t_u[None, :])
            w_ts.append(prev_w_t[None, :])

        w_ts = torch.cat(w_ts, dim=0)
        rnn_0_outputs = torch.cat(rnn_0_outputs, dim=0)
        alignments = torch.cat(alignments, dim=0)  # [T, B, U]
        rnn_1_inputs = torch.cat([rnn_0_outputs, w_ts, strokes_inputs], dim=-1)

        rnn_1_outputs, h_1 = self.rnn_1(rnn_1_inputs, h_1)
        rnn_2_inputs = torch.cat([rnn_1_outputs, w_ts, strokes_inputs], dim=-1)

        rnn_2_outputs, h_2 = self.rnn_2(rnn_2_inputs, h_2)

        rnn_outputs = torch.cat([rnn_0_outputs, rnn_1_outputs, rnn_2_outputs], dim=-1)

        mixture_params = self.mixture(rnn_outputs)
        end_of_stroke_logits = self.end_of_stroke_layer(rnn_outputs)

        return mixture_params, end_of_stroke_logits, h_0, h_1, h_2, prev_w_t, prev_k, alignments

    def log_prob(self, x, mixture_params):
        return self.mixture.log_prob(x, mixture_params)

    def loss(self, strokes_inputs, strokes_lengths, strokes_targets, text, text_lengths, device):
        batch_size = int(strokes_inputs.shape[1])
        h_0, h_1, h_2, prev_w_t, prev_k = self.init_states(batch_size, device)
        mixture_params, end_of_stroke_logits, h_0, h_1, h_2, prev_w_t, prev_k, alignments = self.forward(
            strokes_inputs, text, text_lengths,
            h_0, h_1, h_2, prev_w_t, prev_k,
        )
        model_log_prob = self.log_prob(strokes_targets[..., 1:], mixture_params)
        end_loss = nn.functional.binary_cross_entropy_with_logits(
            end_of_stroke_logits[..., 0],
            strokes_targets[..., 0],
            reduction="none",
        )
        mask = (sequence_mask(strokes_lengths, maxlen=strokes_inputs.shape[0])).permute((1, 0)).float()
        strokes_lengths = strokes_lengths.float()
        mixture_loss = torch.mean(torch.sum(-model_log_prob * mask, dim=0) / strokes_lengths)
        end_loss = torch.mean(torch.sum(end_loss * mask, dim=0) / strokes_lengths)
        return mixture_loss, end_loss

    def init_states(self, batch_size, device):
        weight = next(self.parameters())
        h_init_states = [
            (
                weight.new_zeros(1, batch_size, self.hidden_size).to(device),
                weight.new_zeros(1, batch_size, self.hidden_size).to(device)
            ) for _ in range(3)
        ]
        k = torch.zeros(batch_size, self.window_num_gaussians).to(device)
        w_t = torch.zeros(batch_size, self.num_letters).to(device)
        return (*h_init_states, w_t, k)

    def sample(self, text, max_length=1200, bias=None, device=None):
        h_0, h_1, h_2, prev_w_t, prev_k = self.init_states(1, device=device)
        strokes_inputs = torch.zeros(1, 1, self.input_size).to(text.device)
        result = [strokes_inputs.cpu().numpy()]
        text_lengths = torch.LongTensor([text.shape[1]]).to(text.device)

        for i in range(max_length):
            mixture_params, end_of_stroke_logits, h_0, h_1, h_2, prev_w_t, prev_k, alignments = self.forward(
                strokes_inputs, text, text_lengths, h_0, h_1, h_2, prev_w_t, prev_k
            )
            stroke_sample = self.mixture.sample(mixture_params, bias)
            end_stroke = torch.distributions.Bernoulli(logits=end_of_stroke_logits).sample()
            strokes_inputs = torch.cat([end_stroke, stroke_sample], dim=-1)
            result.append(strokes_inputs.detach().cpu().numpy())
            if alignments[0, 0, -1] > torch.max(alignments[..., :-1]):
                break
        return np.concatenate(result)[:, 0]

    def generate_conditionally(self, text, bias=10, random_seed=1, device=None):
        """
        Generate strokes for given text
        :param text: text string
        :param bias: float value controlling level of predictability of generated sample, the bigger is more
        predictable
        :param random_seed: int seed
        :param device:
        :return: stoke with shape [T, 3]
        """
        torch.manual_seed(random_seed)
        text += "  "  # need some padding to know where to stop
        text_indices = torch.LongTensor([self.alphabet.get(x, self.alphabet[" "]) for x in text])[None, :].to(device)
        sample = self.sample(text_indices, device=device, bias=bias)
        return sample
