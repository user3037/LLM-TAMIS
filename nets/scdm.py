import torch
import torch.nn as nn
import torch.nn.functional as F

class SCDMModule(nn.Module):
    def __init__(self, in_channels):
        super(SCDMModule, self).__init__()
        
        # Initial concatenation layer for Down CNN and Up ViT
        self.conv1 = nn.Conv2d(in_channels * 2  , in_channels, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Parallel convolutional branches
        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Merging the branches
        self.conv_merge = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Further convolutional processing
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Attention-like multiplication
        self.conv_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, down_cnn, up_vit, up_cnn):
        # Concatenation of Down CNN and Up ViT
        # print(down_cnn.size(), up_vit.size(), up_cnn.size())
        if (up_vit.size() != up_cnn.size()):
            up_cnn = F.interpolate(up_cnn, scale_factor=2, mode="bilinear", align_corners=False)

        
        
        # print(down_cnn.size(), up_vit.size(), up_cnn.size())
        x = torch.cat((down_cnn, up_vit), dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        
        # Parallel branches
        x1 = self.conv_branch1(x)
        x2 = self.conv_branch2(x)
        
        # Merge branches
        x_merge = torch.cat((x1, x2), dim=1)
        x_merge = self.conv_merge(x_merge)
        x_gate = self.sigmoid(x_merge)
        
        # Apply gating
        x_out = x_gate * up_cnn
        
        # Further processing
        x_out = self.conv2(x_out)
        
        # Attention-based modulation
        attn = self.conv_attention(x_out)
        x_out = attn * x_out
        x_out = F.interpolate(x_out, scale_factor=2, mode="bilinear", align_corners=False)

        return x_out
