import torch
from torch import nn
from typing import List


class JasperBlock(nn.Module):
    """
    Jasper block
    This class implements the Jasper block from the paper:
    https://arxiv.org/pdf/1904.03288.pdf
    Args:
        n_subblocks (int): Number of subblocks to create
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        dropout (float): Probability of an element to be zeroed
    """
    def __init__(self, n_subblocks, in_channels: int, out_channels: int,
                 kernel_size: int, dropout: float) -> None:
        super(JasperBlock, self).__init__()

        self.n_subblocks = n_subblocks
        self.block = nn.ModuleList([
                        nn.ModuleList([
                            nn.Conv1d(in_channels=in_channels \
                                          if i == 0 else out_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      padding=kernel_size // 2),
                            nn.BatchNorm1d(num_features=out_channels),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=dropout)
                        ])
                        for i in range(n_subblocks)
        ])

    def forward(self, x: torch.Tensor, residuals: List[torch.Tensor]) -> torch.Tensor:
        for i, subblock in enumerate(self.block):
            for j, module in enumerate(subblock):
                if j == 2 and i == self.n_subblocks - 1:
                    for res in residuals:
                        x += res
                x = module(x)
        return x


class Jasper(nn.Module):
    """
    Jasper
    This class implements the Jasper from the paper:
    https://arxiv.org/pdf/1904.03288.pdf
    Args:
        B_repeats (int): Number of repetitions of each block (it is NOT B from paper).
            It means that to create Jasper 10x5 one has to set B_repeats=2, R=5
        R (int): Number of repetition of each subblock in block
        num_features (int): Number of input features
        vocab_size (int): Size of vocabulary
    """
    def __init__(self, B_repeats: int, R: int, num_features: int, vocab_size: int) -> None:
        super(Jasper, self).__init__()
 
        self.B = B_repeats * 5
        self.R = R

        default_parameters = [
                              {
                               'kernel_size': 11,
                               'out_channels': 256,
                               'dropout': 0.2
                              },
                              {
                               'kernel_size': 13,
                               'out_channels': 384,
                               'dropout': 0.2
                              },
                              {
                               'kernel_size': 17,
                               'out_channels': 512,
                               'dropout': 0.2
                              },
                              {
                               'kernel_size': 21,
                               'out_channels': 640,
                               'dropout': 0.2
                              },
                              {
                               'kernel_size': 25,
                               'out_channels': 768,
                               'dropout': 0.2
                              },
        ]

        parameters = []
        for i in range(5):
            for j in range(B_repeats):
                parameters.append(default_parameters[i])

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=num_features,
                      out_channels=256,
                      kernel_size=11,
                      stride=2,
                      padding=11//2),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.blocks = nn.ModuleList([
            JasperBlock(R,
                        in_channels=parameters[i - 1]['out_channels'] \
                            if i > 0 else 256,
                        out_channels=parameters[i]['out_channels'],
                        kernel_size=parameters[i]['kernel_size'],
                        dropout=parameters[i]['dropout'])
            for i in range(self.B)
        ])

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=parameters[-1]['out_channels'],
                      out_channels=896,
                      kernel_size=29,
                      dilation=2,
                      padding=28),
            nn.BatchNorm1d(num_features=896),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=896,
                      out_channels=1024,
                      kernel_size=1,),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )

        self.conv4 = nn.Conv1d(in_channels=1024,
                               out_channels=vocab_size,
                               kernel_size=1)

        self.res_connections = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=parameters[i-1]['out_channels'] \
                              if i > 0 else 256,
                          out_channels=parameters[i]['out_channels'],
                          kernel_size=1),
                nn.BatchNorm1d(num_features=parameters[i]['out_channels'])
            )
            for i in range(self.B)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for i, block in enumerate(self.blocks):
            residual = self.res_connections[i](x)
            block_residuals = [residual]
            x = block(x, block_residuals)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.log_softmax(dim=1)
