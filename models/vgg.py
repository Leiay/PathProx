import torch
import torch.nn as nn
from typing import cast


class VGG19(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(VGG19, self).__init__()
        self.wo_linear = False
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
        layers = []
        self.grouped_layers = []
        self.num_neurons = []
        in_channels = 3
        group_flag = False
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                group_flag = False
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                if group_flag:
                    group_flag = False
                    self.grouped_layers.append([save_conv2d, conv2d])
                    this_conv2d = conv2d
                    self.num_neurons.append(in_channels)
                else:
                    try:
                        self.subgrouped_layers.append([this_conv2d, conv2d])
                    except:
                        pass
                    save_conv2d = conv2d
                    group_flag = True
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)
        self.other_layers = [self.fc]
        self.num_classes = num_classes

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


