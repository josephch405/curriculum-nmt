import torch

class RNNEncoder(torch.nn.Module):
    def __init__(input_d, output_d, layers):
        super(RNNEncoder)
        self.lstm = torch.nn.LSTM(300, 400, 2, bidirectional=True)

    def forward(x):
        return self.lstm(x)

class RNNDecoder(torch.nn.Module):
    def __init__(input_d, output_d, layers):
        super(RNNDencoder)
        self.lstm = torch.nn.LSTM(300, 400, 2, bidirectional=False)

    def forward(x):
        return self.lstm(x)