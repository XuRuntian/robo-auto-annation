import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """用于捕捉动作转换期的轻量级 LSTM"""
    def __init__(self, input_size, hidden_size=64, output_size=1, num_layers=2):
        super(LSTMModel, self).__init__()
        # input_size 动态化，不再写死 16
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        max_len = x.shape[1]
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_len)
        output = self.fc(output)
        return self.sigmoid(output).squeeze(-1)

class CustomLoss(nn.Module):
    """变点专用的加权损失函数"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, penalty_factor):
        adjusted_loss = - (target * torch.log(pred + 1e-8) + penalty_factor * (1 - target) * torch.log(1 - pred + 1e-8))
        return adjusted_loss.mean()