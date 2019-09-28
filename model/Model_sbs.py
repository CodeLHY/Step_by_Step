import torch
from torch import nn
import numpy as np
from torchvision.models import resnet50


class model_sys(nn.Module):

    def __init__(self):
        super(model_sys, self).__init__()
        self.extracter = resnet50(pretrained=True)
        #  对不同模态单独处理的部分网络，非共享
        net_IR = []
        net_IR += [nn.Conv2d(1000, 768, 1)]
        net_IR += [nn.Conv2d(768, 512, 1)]
        self.net_IR = nn.Sequential(*net_IR)
        net_RGB = []
        net_RGB += [nn.Conv2d(1000, 768, 1)]
        net_RGB += [nn.Conv2d(768, 512, 1)]
        self.net_RGB = nn.Sequential(*net_RGB)
        #  对从非共享网络中提取出的特征进行分类的网络
        classification = []
        classification += [nn.Linear(512, 513, bias=True)]
        classification += [nn.LeakyReLU()]
        self.classification = nn.Sequential(*classification)

    def forward(self, images, modals):
        feature_bases = self.extracter(images).unsqueeze(-1).unsqueeze(-1)
        features = torch.zeros((1, 512))
        predictions = torch.zeros((1, 513))
        for i, (feature_base, modal) in enumerate(zip(feature_bases, modals)):
            if modal == 0:
                feature_IR = self.net_IR(feature_base.unsqueeze(0)).squeeze().unsqueeze(0)
                prediction = self.classification(feature_IR.squeeze()).unsqueeze(0)
                if i == 0:
                    predictions = prediction
                    features = feature_IR
                else:
                    predictions = torch.cat((predictions, prediction), dim=0)
                    features = torch.cat((features, feature_IR), dim=0)
            if modal == 1:
                feature_RGB = self.net_RGB(feature_base.unsqueeze(0)).squeeze().unsqueeze(0)
                prediction = self.classification(feature_RGB.squeeze()).unsqueeze(0)
                if i == 0:
                    predictions = prediction
                    features = feature_RGB
                else:
                    predictions = torch.cat((predictions, prediction), dim=0)
                    features = torch.cat((features, feature_RGB), dim=0)
        return features, predictions
