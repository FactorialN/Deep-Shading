import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal

__all__ = [
    'runet'
]


def down(input_channels, output_channels, groups, pooling = True):
    if pooling:
        return nn.Sequential(
                nn.AvgPool2d(kernel_size = 2, stride = 2, ceil_mode= True),
                nn.Conv2d(input_channels, output_channels, groups = groups, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(0.01,inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, groups = groups, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(0.01,inplace=True)
                )

def up(input_channels, output_channels, groups, upsample = True):
    if upsample:
        return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, groups = groups, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(0.01,inplace=True),
                nn.Upsample(mode='bilinear', scale_factor=2)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, groups = groups, kernel_size = 3, stride = 1, padding = 1),
                nn.LeakyReLU(0.01,inplace=True)
                )


class RUNet(nn.Module):
    def __init__(self):
        super(RUNet, self).__init__()

        self.down1 = down(7, 16, 1, False)
        self.down2 = down(16, 32, 2)
        self.down3 = down(32, 64, 4)
        self.down4 = down(64, 128, 8)
        self.down5 = down(128, 256, 16)

        self.up3 = up(384, 128, 8, True, True)
        self.up2 = up(192, 64, 4)
        self.up1 = up(96, 32, 2)
        self.up0 = up(48, 1, 1, False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        out_down1 = self.down1(x)
        out_down2 = self.down2(out_down1)
        out_down3 = self.down3(out_down2)
        out_down4 = self.down4(out_down3)
        out_down5 = self.down5(out_down4)

        out_down5 = out_down5[:, :, :out_down4.shape[2], :]

        cat_input3 = torch.cat((out_down5, out_down4),1)
        out_up3 = self.up3(cat_input3)

        out_up3 = out_up3[:,:,:out_down3.shape[2], :]
        
        cat_input2 = torch.cat((out_up3, out_down3), 1)
        out_up2 = self.up2(cat_input2)
        cat_input1 = torch.cat((out_up2, out_down2), 1)
        out_up1 = self.up1(cat_input1)
        cat_input0 = torch.cat((out_up1, out_down1), 1)
        out_up0 = self.up0(cat_input0)

        return out_up0

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def unet(data=None):
    model = UNet()
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
