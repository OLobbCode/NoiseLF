import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.prediction = nn.Conv2d(128, 2, 1, padding=0)

    def forward(self, x1, x2):
        
        assert(x1.shape == x2.shape)
        x = torch.cat((x1, x2),dim=1)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        out_sal = self.prediction(out)
        out_sal = F.upsample(out_sal, scale_factor=4, mode='bilinear')

        return out_sal



class Intergration(nn.Module): # 2+2 -> 2
    def __init__(self,n_class=2):
        super(Intergration, self).__init__()
        self.prediction = nn.Conv2d(4, 2, 1, padding=0)

    def forward(self, x1, x2):
        
        assert(x1.shape == x2.shape)
        out = torch.cat((x1, x2),dim=1)
        out_sal = self.prediction(out)

        return out_sal

if __name__ == '__main__':

    model_intergration = Intergration()

    x1 = torch.randn(1,2,256,256)
    x2 = torch.randn(1,2,256,256)
    #target = target.long()
    out = model_intergration(x1,x2)
    print(out.shape)
  
    
    