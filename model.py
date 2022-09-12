import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_1(in_c, out_c, bt):
    if bt:
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
    return conv


def de_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )
    return conv


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngpu = 1
        self.nef = 32
        self.ngf = 16
        self.nBottleneck = 32
        self.nc = 1
        self.num_same_conv = 5

        # 1x80x80
        self.conv1 = conv_1(self.nc, self.nef, False)
        # 32x40x40
        self.conv2 = conv_1(self.nef, self.nef, True)
        # 32x20x20
        self.conv3 = conv_1(self.nef, self.nef*2, True)
        # 64x10x10
        self.conv4 = conv_1(self.nef*2+1, self.nef*4, True)
        # 128x5x5
        self.conv6 = nn.Conv2d(self.nef*4, self.nBottleneck, 2, bias=False)
        # 4000x4x4
        self.batchNorm1 = nn.BatchNorm2d(self.nBottleneck)
        self.leak_relu = nn.LeakyReLU(0.2, inplace=True)
        # 4000x4x4

        self.num_same_conv = self.num_same_conv
        self.sameconvs = nn.ModuleList([nn.ConvTranspose2d(
            32, 32, 3, 1, 1, bias=False) for _ in range(self.num_same_conv)])
        self.samepools = nn.ModuleList([nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1) for _ in range(self.num_same_conv)])
        self.samebns = nn.ModuleList(
            [nn.BatchNorm2d(32) for _ in range(self.num_same_conv)])

        self.convt1 = nn.ConvTranspose2d(
            self.nBottleneck, self.ngf * 8, 2, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(self.ngf * 8)
        self.relu = nn.ReLU(True)
        # 128x5x5
        self.convt2 = de_conv(256, 64)
        # 64x10x10
        self.convt3 = de_conv(128+1, 32)
        # 32x20x20
        self.convt4 = de_conv(64, 32)
        # 32x40x40
        self.convt6 = nn.ConvTranspose2d(64, self.nc, 4, 2, 1, bias=False)
        # 1x80x80
        self.tan = nn.Tanh()

    def forward(self, noise, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        mod_input = torch.cat([noise, x3], dim=1)
        x4 = self.conv4(mod_input)
        x6 = self.conv6(x4)
        x7 = self.batchNorm1(x6)
        x8 = self.leak_relu(x7)
        x9 = self.convt1(x8)

        x10 = self.batchNorm2(x9)
        x11 = self.relu(x10)
        x12 = self.convt2(torch.cat([x4, x11], 1))
        out = self.convt3(torch.cat([mod_input, x12], 1))

        for i in range(self.num_same_conv):
            conv = self.sameconvs[i]
            pool = self.samepools[i]
            bn = self.samebns[i]

            out = conv(out)
            out = pool(out)
            out = bn(out)
            out = F.leaky_relu(out, negative_slope=0.2)

        x14 = self.convt4(torch.cat([x2, out], 1))
        x15 = self.convt6(torch.cat([x1, x14], 1))

        return self.tan(x15)


class Discriminator(nn.Module):
    def __init__(self, d_project=128, d_hidden=128, dropout_p=0.5):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(64, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(64, 32, 4, 2, 1, bias=False)
        )

        self.decision = nn.Sequential(
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 5, 1, 0, bias=False)
        )

        self.projection1 = nn.Sequential(
            nn.Linear(800, d_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(d_hidden, d_project)
        )
        self.projection2 = nn.Sequential(
            nn.Linear(800, d_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(d_hidden, d_project)
        )

    def forward(self, input, sg_linear=False, projection1=False, projection2=False, penul=False):

        _aux = {}
        _return_aux = False

        penultimate = self.main(input)

        if sg_linear:
            out_d = penultimate.detach()
        else:
            out_d = penultimate

        discout = self.decision(out_d)

        if projection1:
            _return_aux = True
            _aux['projection1'] = self.projection1(
                penultimate.view(penultimate.shape[0], -1))

        if projection2:
            _return_aux = True
            _aux['projection2'] = self.projection2(
                penultimate.view(penultimate.shape[0], -1))

        if _return_aux:
            return discout, _aux

        if penul:
            return discout, penultimate

        return discout


class Discriminator_nodropout(nn.Module):
    def __init__(self, d_project=128, d_hidden=128):
        super(Discriminator_nodropout, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 4, 2, 1, bias=False)
        )

        self.decision = nn.Sequential(
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 5, 1, 0, bias=False)
        )

        self.projection1 = nn.Sequential(
            nn.Linear(800, d_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(d_hidden, d_project)
        )
        self.projection2 = nn.Sequential(
            nn.Linear(800, d_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(d_hidden, d_project)
        )

    def forward(self, input, sg_linear=False, projection1=False, projection2=False, penul=False):

        _aux = {}
        _return_aux = False

        penultimate = self.main(input)

        if sg_linear:
            out_d = penultimate.detach()
        else:
            out_d = penultimate

        discout = self.decision(out_d)

        if projection1:
            _return_aux = True
            _aux['projection1'] = self.projection1(
                penultimate.view(penultimate.shape[0], -1))

        if projection2:
            _return_aux = True
            _aux['projection2'] = self.projection2(
                penultimate.view(penultimate.shape[0], -1))

        if _return_aux:
            return discout, _aux

        if penul:
            return discout, penultimate

        return discout
