import torch
from torch import nn


KERNEL_SIZES = {
    'K1': [5, 3, 3, 3, 5, 5, 5, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7, 7, 9, 9, 9, 9, 41],
    'K2': [5, 5, 7, 7, 9, 9, 11, 7, 7, 9, 9, 11, 11, 13, 13, 13, 15, 15, 17, 17, 19, 19, 41],
    'K3': [5, 9, 9, 11, 13, 15, 15, 9, 11, 13, 15, 15, 17, 19, 19, 21, 21, 23, 25, 27, 27, 29, 41],
    'K4': [5, 11, 13, 15, 17, 19, 21, 13, 15, 17, 19, 21, 23, 25, 25, 27, 29, 31, 33, 35, 37, 39, 41]
}


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation sub-module.
    Args:
        channels (int): Input number of channels.
        reduction_ratio (int): Reduction ratio for "squeeze" layer.
        context_window (int): Integer number of timesteps that the context
            should be computed over, using stride 1 average pooling.
            If value < 1, then global context is computed.
        interpolation_mode (str): Interpolation mode of timestep dimension.
            Used only if context window is > 1.
            The modes available for resizing are: `nearest`, `linear` (3D-only),
            `bilinear`, `area`
    """
    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        context_window: int = -1,
        interpolation_mode: str = 'nearest',
    ) -> None:

        super(SqueezeExcite, self).__init__()
        self.interpolation_mode = interpolation_mode
        self.context_window = context_window
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, timesteps = x.size()[:3]

        if timesteps < self.context_window:
            y = self.gap(x)
        else:
            y = self.pool(x) 
        y = y.transpose(1, -1) 
        y = self.fc(y) 
        y = y.transpose(1, -1)

        if self.context_window > 0:
            y = torch.nn.functional.interpolate(y, size=timesteps, mode=self.interpolation_mode)
        y = torch.sigmoid(y)
        return x * y


class CitrinetBlock(nn.Module):
    """
    Citrinet block
    This class implements Citrinet block from the paper:
    https://arxiv.org/pdf/2104.01721.pdf
    Args:
        R (int): Number of repetition of each subblock in block
        C (int): Number of channels
        kernel_size (int): Kernel size
        stride (int, optional): Stride
        dropout (float, optional): Dropout probability
    """
    def __init__(
        self,
        R: int,
        C: int,
        kernel_size: int,
        stride: int = None,
        dropout: float = 0.
    ) -> None:

        super(CitrinetBlock, self).__init__()

        self.R = R

        self.block = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=kernel_size,
                    groups=C,
                    stride=stride if stride and i == 0 else 1,
                    padding=kernel_size // 2
                ),
                nn.Conv1d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=1
                ),
                nn.BatchNorm1d(num_features=C),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            for i in range(R)
        ])

        self.block.append(
            nn.ModuleList([
                nn.Conv1d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=kernel_size,
                    groups=C,
                    padding=kernel_size // 2
                ),
                nn.Conv1d(
                    in_channels=C,
                    out_channels=C,
                    kernel_size=1
                ),
                nn.BatchNorm1d(num_features=C),
                SqueezeExcite(
                    channels=C,
                    reduction_ratio=8
                ),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
        )

        self.res_connection = nn.Sequential(
            nn.Conv1d(
                in_channels=C,
                out_channels=C,
                kernel_size=1,
                stride=stride if stride else 1
            ),
            nn.BatchNorm1d(num_features=C)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_connection(x)
        for i, subblock in enumerate(self.block):
            for j, module in enumerate(subblock):
                if i == self.R and j == 4:
                    x += residual
                    del residual
                x = module(x)
        return x


class Citrinet(nn.Module):
    """
    Citrinet
    This class implements Citrinet from the paper:
    https://arxiv.org/pdf/2104.01721.pdf

    Args:
        C (int): Number of channels
        K (int): Type of kernel sizes
        R (int): Number of subblocks in each block
        num_features (int): Number of input channels
        vocab_size (int): Vocab size
        dropout (float, optional): Dropout probability
    """
    def __init__(
        self,
        C: int,
        K: int,
        R: int,
        num_features: int,
        vocab_size: int,
        dropout: float = 0.
    ) -> None:

        super(Citrinet, self).__init__()

        self.K = KERNEL_SIZES['K' + str(K)]
        self.megablocks = [
            [1, 7],
            [7, 14],
            [14, 22]
        ]

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=C,
                groups=num_features,
                kernel_size=self.K[0],
                padding=self.K[0] // 2,
            ),
            nn.Conv1d(
                in_channels=C,
                out_channels=C,
                kernel_size=1
            ),
            nn.BatchNorm1d(num_features=C),
            nn.ReLU(inplace=True)
        )

        self.megablock1 = nn.Sequential(*[
            CitrinetBlock(
                R=R,
                C=C,
                kernel_size=self.K[i + 1],
                stride=2 if i == 0 else None,
                dropout=dropout
            )
            for i in range(6) # 6 block in first megablock from paper
        ])

        self.megablock2 = nn.Sequential(*[
            CitrinetBlock(
                R=R,
                C=C,
                kernel_size=self.K[i + 7],
                stride=2 if i == 0 else None,
                dropout=dropout
            )
            for i in range(7) # 7 block in second
        ])

        self.megablock3 = nn.Sequential(*[
            CitrinetBlock(
                R=R,
                C=C,
                kernel_size=self.K[i + 14],
                stride=1, # 2 if i == 0 else None, #doesn't work with stride=2
                dropout=dropout
            )
            for i in range(8) # 8 block in third
        ])

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=C,
                out_channels=C,
                groups=C,
                kernel_size=self.K[-1],
                padding=self.K[-1] // 2
            ),
            nn.Conv1d(
                in_channels=C,
                out_channels=C,
                kernel_size=1
            ),
            nn.BatchNorm1d(num_features=C),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=C,
                out_channels=vocab_size,
                kernel_size=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.megablock1(x)
        x = self.megablock2(x)
        x = self.megablock3(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.log_softmax(dim=1)
