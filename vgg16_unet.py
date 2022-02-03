import torch
import torch.nn as nn
import torchvision
from torchsummary import summary

class convblockx2(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(convblockx2, self).__init__()
        
        self.doubleconvblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
            )
        
    def forward(self, x):
        return self.doubleconvblock(x)


class vgg_unet(nn.Module):
    
    def __init__(self, num_classes):
        
        super(vgg_unet, self).__init__()
        
        self.num_classes = num_classes
        
        #ENCODER LAYERS    
        self.encoder1 = torchvision.models.vgg16(pretrained = False).features[:4]
        self.encoder2 = torchvision.models.vgg16(pretrained = False).features[4:9]
        self.encoder3 = torchvision.models.vgg16(pretrained = False).features[9:16]
        self.encoder4 = torchvision.models.vgg16(pretrained = False).features[16:23]
        self.encoder5 = torchvision.models.vgg16(pretrained = False).features[23:30]
        
        self.upsample = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True)
        self.relu = nn.ReLU(True)
        
        self.decoder4 = convblockx2(512 + 512, 512)
        self.decoder3 = convblockx2(512 + 256, 256)
        self.decoder2 = convblockx2(256 + 128, 128)
        self.decoder1 = convblockx2(128 + 64, 64)
        self.out = convblockx2(64, 64)
        
        self.final = nn.Conv2d(64, self.num_classes, 1)
    
    def forward(self, x):
        
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        
        dec4 = torch.cat([self.upsample(x5), x4], dim = 1)
        dec4 = self.decoder4(dec4)
 
        dec3 = torch.cat([self.upsample(dec4), x3], dim = 1)
        dec3 = self.decoder3(dec3)
              
        dec2 = torch.cat([self.upsample(dec3), x2], dim = 1)
        dec2 = self.decoder2(dec2)
               
        dec1 = torch.cat([self.upsample(dec2), x1], dim = 1)
        dec1 = self.decoder1(dec1)
        
        output = self.final(dec1)
        return output
        

x = torch.randn(1,3,224,224)
model = vgg_unet(4)

summary(model, (3,224,224),device = "cpu")
