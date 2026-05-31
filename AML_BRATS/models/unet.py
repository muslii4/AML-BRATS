"""
Code adapted from Sovit Ranjan Rath
https://debuggercafe.com/unet-from-scratch-using-pytorch/

The architecture was changed slightly to match the BRATS scan dimensions.
"""

import torch
import torch.nn as nn


def double_convolution(in_channels, out_channels, batch_norm):
    """
    In the original paper implementation, the convolution operations were
    not padded but we are padding them here. This is because, we need the
    output result size to be same as input size.
    """
    if not batch_norm:
        conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    else:
        conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    return conv_op


class UNet(nn.Module):
    def __init__(self, num_classes, batch_norm):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = double_convolution(4, 64, batch_norm)
        self.down_convolution_2 = double_convolution(64, 128, batch_norm)
        self.down_convolution_3 = double_convolution(128, 256, batch_norm)
        self.down_convolution_4 = double_convolution(256, 512, batch_norm)

        # Expanding path.
        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_convolution_1 = double_convolution(512, 256, batch_norm)
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_convolution_2 = double_convolution(256, 128, batch_norm)
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.up_convolution_3 = double_convolution(128, 64, batch_norm)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(
            in_channels=64, out_channels=num_classes, kernel_size=1
        )

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        # *** DO NOT APPLY MAX POOL TO down_7 ***

        up_1 = self.up_transpose_1(down_7)
        x = self.up_convolution_1(torch.cat([down_5, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_3, up_2], 1))
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_1, up_3], 1))
        out = self.out(x)
        return out


if __name__ == "__main__":
    input_image = torch.rand((1, 4, 240, 240))
    model = UNet(num_classes=3, batch_norm=False)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")
    outputs = model(input_image)
    print(outputs.shape)
