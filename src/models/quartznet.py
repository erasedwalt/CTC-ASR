import torch
from torch import nn


class QuartzBlock(nn.Module):
    def __init__(
        self,
        R: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int
    ):
        super(QuartzBlock, self).__init__()

        self.R = R
        self.block = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    groups=in_channels if i == 0 else out_channels,
                    padding=kernel_size // 2
                ),
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1
                ),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(inplace=True)
            ])
        for i in range(R)
        ])
        self.res_connection = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            ),
            nn.BatchNorm1d(num_features=out_channels)
        )

    def forward(self, x):
        residual = self.res_connection(x)
        for i, subblock in enumerate(self.block):
            for j, module in enumerate(subblock):
                if j == 3 and i == self.R - 1:
                    x += residual
                x = module(x)
        del residual
        return x


class QuartzNet(nn.Module):
    def __init__(
        self,
        B_repeats: int,
        R: int,
        num_features: int,
        vocab_size: int
    ) -> None:
        super(QuartzNet, self).__init__()

        default_parameters = [
                              {
                               'kernel_size': 33,
                               'out_channels': 256
                              },
                              {
                               'kernel_size': 39,
                               'out_channels': 256
                              },
                              {
                               'kernel_size': 51,
                               'out_channels': 512
                              },
                              {
                               'kernel_size': 63,
                               'out_channels': 512
                              },
                              {
                               'kernel_size': 75,
                               'out_channels': 512
                              },
        ]

        parameters = []
        for i in range(5):
            for j in range(B_repeats):
                parameters.append(default_parameters[i])

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=256,
                kernel_size=33,
                padding=33 // 2,
                stride=2
            ),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.Sequential(*[
            QuartzBlock(
                R=R,
                in_channels=parameters[i - 1]['out_channels'] if i > 0 else 256,
                out_channels=parameters[i]['out_channels'],
                kernel_size=parameters[i]['kernel_size']
            )
            for i in range(len(parameters))
        ])

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=parameters[-1]['out_channels'],
                out_channels=512,
                kernel_size=87,
                groups=512,
                dilation=2,
                padding=86
            ),
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=1
            ),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=512,
                out_channels=1024,
                kernel_size=1
            ),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=1024,
                out_channels=vocab_size,
                kernel_size=1
            )
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.log_softmax(dim=1)
