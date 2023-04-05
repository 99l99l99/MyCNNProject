import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class ConvTrans(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
        super(ConvTrans, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RegressionBasedNet(nn.Module):
    def __init__(self, cfg):
        super(RegressionBasedNet, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children()))
        self.device = cfg['device']
        self.midLayer = [5,6] #resnet中需要输出的层数
        self.in_out_channels = [(512,256), (256,128)] #与resnet输出层拼接，上采样的输入输出通道，倒着来的
        self.num_class = cfg['num_class']
        self.out_shape = cfg['out_shape']
        self.ratio = cfg['ratio']
        self.convTrans = []
        self.convs = []
        for in_out_channel in self.in_out_channels:
            self.convTrans.append(ConvTrans(in_out_channel[0], in_out_channel[1]))
            self.convs.append(Conv(in_out_channel[0], in_out_channel[1]))
        self.convTrans = nn.Sequential(*self.convTrans)
        self.convs = nn.Sequential(*self.convs)
        self.convKeypointsCoord = nn.Conv2d(
            in_channels=self.in_out_channels[-1][-1],
            out_channels=3+self.num_class,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.sig = nn.Sigmoid()
        w = np.array([i for i in range(self.out_shape[0])], dtype='float32')
        h = np.array([i for i in range(self.out_shape[1])], dtype='float32')
        [meshgrid_x, meshgrid_y] = np.meshgrid(w, h)
        meshgrid_x = meshgrid_x[np.newaxis,np.newaxis, ...]
        meshgrid_y = meshgrid_y[np.newaxis,np.newaxis, ...]
        meshgrid_x = meshgrid_x * self.ratio
        meshgrid_y = meshgrid_y * self.ratio
        meshgrid = np.concatenate([meshgrid_x, meshgrid_y], axis=1)
        self.meshgrid = nn.Parameter(torch.Tensor(meshgrid), requires_grad=False)
        self.to(self.device)

    def forward(self, x):
        # forward and get mid layers
        ResMidOutputs = []
        for i, feature in enumerate(self.features[:-2]):
            x = feature(x)
            if i in self.midLayer:
                ResMidOutputs.append(x)
        #unsample and cat with mid layers
        for i, ResMidOutput in enumerate(reversed(ResMidOutputs)):
            x = self.convTrans[i](x)
            x = torch.cat([x, ResMidOutput],dim = 1)
            x = self.convs[i](x)
        final = self.convKeypointsCoord(x)
        
        prob_cls, xy = torch.split(final, [1+self.num_class, 2], dim = 1)
        prob_cls = self.sig(prob_cls)
        xy = self.sig(xy) * self.ratio + self.meshgrid
        result = torch.cat([prob_cls, xy], dim=1)
        return  result