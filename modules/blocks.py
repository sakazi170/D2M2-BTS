import torch
import torch.nn as nn

class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MSGS(nn.Module):
    """
    Modality-Specific Gated Stem (MSGS)
    """

    def __init__(self, in_channels, out_channels, neg_slope=0.01):
        super().__init__()

        # 1) Feature extraction (DWS conv)
        self.feature_extract = nn.Sequential(
            DepthwiseSeparableConv3D(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=True),
        )

        # 2) Per-voxel adaptive gating
        self.gate = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # 3) Downsampling (DW stride-2 + PW for channel mixing)
        self.downsample = nn.Sequential(
            nn.Conv3d(
                out_channels, out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=out_channels,  # depthwise
                bias=False,
            ),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False),  # pointwise (IMPORTANT)
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=neg_slope, inplace=True),
        )

        # 4) Residual shortcut
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = self.shortcut(x)  # (B, C, H/2)

        feat = self.feature_extract(x)  # (B, C, H)
        gated = feat * self.gate(feat)  # (B, C, H)

        down = self.downsample(gated)  # (B, C, H/2)

        out = down + residual  # (B, C, H/2)

        return gated, out



class DRB(nn.Module):
    """Depthwise Residual Block."""

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv3D(in_channels, out_channels),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv3D(out_channels, out_channels),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True))
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups, out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.skip(x)