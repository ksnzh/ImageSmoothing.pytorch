import torch
import torch.nn as nn


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation, norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, dilation, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, dilation, norm_layer, use_dropout, use_bias):
        conv_block = []

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class BasicBlock(nn.Module):
    def __init__(self, type, inplane, outplane, stride):
        super(BasicBlock, self).__init__()
        conv_block = []
        if type == "Conv":
            conv_block += [nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1)]
        elif type == "Deconv":
            conv_block += [nn.ConvTranspose2d(inplane, outplane, kernel_size=4, stride=stride, padding=1)]

        conv_block +=[nn.BatchNorm2d(outplane),
                      nn.ReLU()]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out


class FCNSmooth(nn.Module):
    def __init__(self):
        super(FCNSmooth, self).__init__()

        model = []

        model += [BasicBlock("Conv", 3, 64, 1)]
        model += [BasicBlock("Conv", 64, 64, 1)]
        model += [BasicBlock("Conv", 64, 64, 2)]

        model += [ResnetBlock(64, 2)]
        model += [ResnetBlock(64, 2)]
        model += [ResnetBlock(64, 4)]
        model += [ResnetBlock(64, 4)]
        model += [ResnetBlock(64, 8)]
        model += [ResnetBlock(64, 8)]
        model += [ResnetBlock(64, 16)]
        model += [ResnetBlock(64, 16)]
        model += [ResnetBlock(64, 1)]
        model += [ResnetBlock(64, 1)]

        model += [BasicBlock("Deconv", 64, 64, 2)]
        model += [BasicBlock("Conv", 64, 64, 1)]
        model += [nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x - 128)
        return out


if __name__ == "__main__":
    from torchsummary import summary
    model = FCNSmooth().cuda()
    summary(model, (3, 640, 492))