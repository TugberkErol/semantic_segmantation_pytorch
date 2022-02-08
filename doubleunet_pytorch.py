"""
DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation
Official code in https://github.com/DebeshJha/2020-CBMS-DoubleU-Net
Paper link: https://arxiv.org/abs/2006.04868
This is unofficial implementation in pytorch
"""
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision
from torchsummaryX import summary


class squeeze_excitation_block(nn.Module):
    
    def __init__(self, in_channels):
        
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//8),
            nn.ReLU(inplace = True),
            nn.Linear(in_channels//8, in_channels),
            nn.Sigmoid())
            
    
    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1, 1)
        return x*y.expand_as(x)
    

class convblockx2(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.se = squeeze_excitation_block(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x

class Encoder2(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.encoder1 = convblockx2(3, 32)
        self.encoder2 = convblockx2(32, 64)
        self.encoder3 = convblockx2(64, 128)
        self.encoder4 = convblockx2(128, 256)
        
        self.maxpool = nn.MaxPool2d((2,2))
        
    def forward(self,x):
        
        skip_connections = []
        enc1 = self.encoder1(x)
        skip_connections.append(enc1)
        enc1 = self.maxpool(enc1)
        
        enc2 = self.encoder2(enc1)
        skip_connections.append(enc2)
        enc2 = self.maxpool(enc2)
        
        enc3 = self.encoder3(enc2)
        skip_connections.append(enc3)
        enc3 = self.maxpool(enc3)
        
        enc4 = self.encoder4(enc3)
        skip_connections.append(enc4)
        bottleneck = self.maxpool(enc4)
        
        
        return bottleneck, skip_connections

       
class ASPP(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                     nn.Conv2d(in_channels, out_channels, 1, padding = 0, dilation = 1),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, dilation = 1, padding = 0, bias = False)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, dilation = 6, padding = 6, bias = False)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, dilation = 12, padding = 12, bias = False)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, dilation = 18, padding = 18, bias = False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(True)
        
        self.final_conv = nn.Conv2d(out_channels*5, out_channels, 1, dilation = 1, bias = False)
    
    def forward(self, x):
        
        aspp1 = self.avgpool(x)
        aspp1 = F.interpolate(aspp1, size = x.size()[2:], mode = "bilinear", align_corners = True)
        
        aspp2 = self.conv1(x)
        aspp2 = self.bn(aspp2)
        aspp2 = self.relu(aspp2)
                                     
        aspp3 = self.conv2(x)
        aspp3 = self.bn(aspp3)
        aspp3 = self.relu(aspp3)

        aspp4 = self.conv3(x)
        aspp4 = self.bn(aspp4)
        aspp4 = self.relu(aspp4)

        aspp5 = self.conv4(x)
        aspp5 = self.bn(aspp5)
        aspp5 = self.relu(aspp5)
        
        aspp_cat = torch.cat([aspp1, aspp2, aspp3, aspp4, aspp5], dim = 1)
        aspp_cat = self.final_conv(aspp_cat)
        aspp_cat = self.bn(aspp_cat)
        aspp_cat = self.relu(aspp_cat)
        
        return aspp_cat


class doubleunet(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.encoder1 = torchvision.models.vgg19(pretrained = False).features[:4]
        self.encoder2 = torchvision.models.vgg19(pretrained = False).features[4:9]
        self.encoder3 = torchvision.models.vgg19(pretrained = False).features[9:18]
        self.encoder4 = torchvision.models.vgg19(pretrained = False).features[18:27]
        self.encoder5 = torchvision.models.vgg19(pretrained = False).features[27:36]
        
        self.aspp = ASPP(512, 64)
        self.aspp1 = ASPP(256, 64)
        
        self.decoder1 = convblockx2(64 + 512, 256)
        self.decoder2 = convblockx2(256 + 256, 128)
        self.decoder3 = convblockx2(128 + 128, 64)
        self.decoder4 = convblockx2(64 + 64, 32)
        
        self.final_conv = nn.Conv2d(32, 1, 1, padding = 0)
        
        #second unet's encoder
        self.encoder_class = Encoder2()
        
        self.decoder1_ = convblockx2(512 + 256 + 64, 256)
        self.decoder2_ = convblockx2(256 + 128 + 256, 128)
        self.decoder3_ = convblockx2(128 + 64 + 128, 64)
        self.decoder4_ = convblockx2(64 + 32 + 64, 32)

        
    def forward(self, x):   
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        
        #Concatenate(encoder5, encoder4) => convblock => squeeze block
        enc5_aspp = self.aspp(enc5)
        enc5_enc4 = torch.cat([F.interpolate(enc5_aspp,enc4.size()[2:], mode = "bilinear", align_corners = True),enc4], dim = 1)
        dec1 = self.decoder1(enc5_enc4)
        
        #Concatenate(encoder3, decoder1) => convblock => squeeze block
        enc3_dec1 = torch.cat([F.interpolate(dec1, enc3.size()[2:], mode = "bilinear", align_corners = True), enc3], dim = 1)
        dec2 = self.decoder2(enc3_dec1)
        
        #Concatenate(encoder2, decoder2) => convblock => squeeze block
        enc2_dec2 = torch.cat([F.interpolate(dec2, enc2.size()[2:], mode = "bilinear", align_corners = True), enc2], dim = 1)
        dec3 = self.decoder3(enc2_dec2)
        
        #Concatenate(encoder1, decoder3) => convblock => squeeze block
        enc1_dec3 = torch.cat([F.interpolate(dec3, enc1.size()[2:], mode = "bilinear", align_corners = True), enc1], dim = 1)
        dec4 = self.decoder4(enc1_dec3)
        
        output1 = self.final_conv(dec4)
        output1 = F.sigmoid(output1)
        
        bottleneck, skip_connections = self.encoder_class(output1*x)
        bottleneck = self.aspp1(bottleneck)
        dec1_ = torch.cat([F.interpolate(bottleneck, enc4.size()[2:], mode = "bilinear", align_corners = True), enc4, skip_connections[-1]],dim = 1)
        dec1_ = self.decoder1_(dec1_)
        
        dec2_ = torch.cat([F.interpolate(dec1_, enc3.size()[2:], mode = "bilinear", align_corners = True), enc3, skip_connections[-2]], dim = 1)
        dec2_ = self.decoder2_(dec2_)

        dec3_ = torch.cat([F.interpolate(dec2_, enc2.size()[2:], mode = "bilinear", align_corners = True), enc2, skip_connections[-3]], dim = 1)
        dec3_ = self.decoder3_(dec3_)

        dec4_ = torch.cat([F.interpolate(dec3_, enc1.size()[2:], mode = "bilinear", align_corners = True), enc1, skip_connections[-4]], dim = 1)
        dec4_ = self.decoder4_(dec4_)
        
        output2 = self.final_conv(dec4_)
        output2 = F.sigmoid(output2)
        final_output = torch.cat([output1, output2], dim = 1)
        
        return final_output
        
        
                                     
x = torch.randn(2,3,224,224)
model = doubleunet()
out = model(x)
summary(model, x)                                    
                                     
                                     
                                     
                                     
                                     