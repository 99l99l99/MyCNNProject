import torch
import torch.nn as nn
import torchvision.models as models

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

class HeatMapBasedNet(nn.Module):
    def __init__(self, cfg):
        super(HeatMapBasedNet, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children()))
        self.device = cfg['device']
        self.midLayer = [4,5,6]
        self.in_out_channels = [(512,256), (256,128), (128,64)]
        self.num_class = cfg['num_class']
        self.out_shape = cfg['out_shape']
        self.convTrans = []
        self.convs = []
        for in_out_channel in self.in_out_channels:
            self.convTrans.append(ConvTrans(in_out_channel[0], in_out_channel[1]))
            self.convs.append(Conv(in_out_channel[0], in_out_channel[1]))
        self.convTrans = nn.Sequential(*self.convTrans)
        self.convs = nn.Sequential(*self.convs)
        
        final_feature_channel = self.in_out_channels[-1][-1]
        self.Heatmaps = nn.Sequential(nn.Conv2d(final_feature_channel, final_feature_channel, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=final_feature_channel,out_channels=self.num_class,kernel_size=1,stride=1,padding=0),
                                       nn.Sigmoid())
       
        self.Offsets = nn.Sequential(nn.Conv2d(final_feature_channel, final_feature_channel, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=final_feature_channel,out_channels=2,kernel_size=1,stride=1,padding=0))
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
        # get keypoints map
        map = self.Heatmaps(x)
        offsets = self.Offsets(x)
        return map, offsets


