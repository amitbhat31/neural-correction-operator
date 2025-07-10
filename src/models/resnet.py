import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.bn2(self.conv2(out))
    #     out += self.shortcut(x)
    #     out = F.relu(out)
    #     return out
    
    def forward(self, x): 
        print("start block")
        out = self.conv1(x)
        print(f"conv1: {out.shape}")
        out = self.bn1(out)
        print(f"batchnorm1: {out.shape}")
        out = F.relu(out)
        print(f"relu1: {out.shape}")
        out = self.conv2(out)
        print(f"conv2: {out.shape}")
        out = self.bn2(out)
        print(f"batchnorm2: {out.shape}")
        out += self.shortcut(x)
        print(f"shortcut shape: {self.shortcut(x).shape}")
        print(f"add skip: {out.shape}")
        out = F.relu(out)
        print(f"relu2: {out.shape}")
        return out


class ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks, in_channels=1, out_channels=1, output_res=64):
        super(ResNet18, self).__init__()

        conf = config.resnet
        self.in_planes = 64



        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, output_res * output_res)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        print(x.shape)        
        x = x.permute(0, 3, 1, 2) 
        print(x.shape)

        # x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.conv1(x)
        print(x.shape)
        x = self.bn1(x)
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        x = self.maxpool(x)
        print(f"maxpool: {x.shape}")
        print()
        
        x = self.layer1(x)
        print(f"fin block1: {x.shape}")
        print()
        x = self.layer2(x)
        print(f"fin block2: {x.shape}")
        print()
        x = self.layer3(x)
        print(f"fin block3: {x.shape}")
        print()
        x = self.layer4(x)
        print(f"fin block4: {x.shape}")
        print()
        
        x = self.avgpool(x)
        print(f"avgpool: {x.shape}")
        x = torch.flatten(x, start_dim=1) 
        print(f"flatten: {x.shape}")
        x = self.linear(x)
        print(f"linear: {x.shape}")


        print(x.shape, flush=True)
        
        # x = x.unsqueeze(1) 
        x = x.view(x.size(0), 1, 64, 64) 
        print(x.shape)

        x = x.permute(0, 2, 3, 1) 
        print(x.shape, flush=True)
        return x



