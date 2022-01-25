import torch
import torch.nn as nn
import torch.nn.functional as F
    

class DiceLoss(nn.Module):
    
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.loss_list = []
        
    def forward(self, y_pred, y, smooth = 1e-15):
            
        if (y.shape[1] == 1):
                
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.view(-1)
        
            y = y.view(-1)   
            intersection = (y_pred * y).sum()
            dice = (2*intersection + smooth) / (y_pred.sum() + y.sum() + smooth)
            
            return 1 - dice
            
        else:
            
            for i in range(y.shape[1]):
                 
                y_prediction = F.softmax(y_pred, dim = 1)[:,i]
                y_prediction = y_prediction.view(-1)
                    
                y_real = y[:,i]
                y_real = y_real.view(-1)
                    
                intersection = (y_prediction * y_real).sum()
                dice = (2*intersection + smooth) / (y_prediction.sum() + y_real.sum() + smooth)
                self.loss_list.append(1 - dice)
                
            return sum(self.loss_list) / (y.shape[1])
                
                    

y = torch.randn(1,1,224,224).to("cuda")
y_pred = torch.randn(1,1,224,224).to("cuda")
                

loss = DiceLoss()
loss_ = loss(y_pred, y)
print(loss_)                       