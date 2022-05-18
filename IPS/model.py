import torch.nn as nn
import torch
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self):
        # def __init__(self, emb_weights):
        super(RNN, self).__init__()
        dropout_rate = 0.5
        input_size = 70
        hidden_size = 70
        n_labels = 3
        n_diagnosis_codes = 1104
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, n_labels)
        self.attention = SelfAttention(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.embed = nn.Embedding(n_diagnosis_codes, input_size)
        # self.embed.weight = nn.Parameter(torch.FloatTensor(emb_weights))

    # overload forward() method
    def forward(self, x, weight_of_x):
        x = x.long()
        x = self.embed(x)  # (n_visits, n_samples, visit_size)
        # multiply by weight here:
        weight_of_x = torch.unsqueeze(weight_of_x, dim=3)
        x = x * weight_of_x
        x = self.relu(x)

        x = torch.mean(x, dim=2)

        output, h_n = self.lstm(x)

        x, attn_weights = self.attention(output.transpose(0, 1))

        x = self.dropout(x)

        logit = self.fc(x)

        logit = self.softmax(logit)
        return logit


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights