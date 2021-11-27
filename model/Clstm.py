import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from configg import cfg
import Model

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class Clstm(nn.Module):
    def __init__(self,n_class=2):
        super(Clstm, self).__init__()

        # ---------------------------- ConvLSTM1 ------------------------------ #

        # ------------------ ConvLSTM cell parameter ---------------------- #
        self.conv_cell2 = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)
        self.conv_cell3 = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)
        self.conv_cell4 = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)
        self.conv_cell5 = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)

        # attentive convlstm 2
        self.conv_cell = nn.Conv2d(64 + 64, 4 * 64, 5, padding=2)


        self.conv_w2 = nn.Conv2d(64 * 12, 12, 1, padding=0)  
        self.pool_avg_w2 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_w3 = nn.Conv2d(64 * 12, 12, 1, padding=0)  
        self.pool_avg_w3 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_w4 = nn.Conv2d(64 * 12, 12, 1, padding=0)  
        self.pool_avg_w4 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_w5 = nn.Conv2d(64 * 12, 12, 1, padding=0)  
        self.pool_avg_w5 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        
        # -----------------------------  Multi-scale2  ----------------------------- #
        self.Atrous_c1_2 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1_2 = nn.ReLU(inplace=True)
        self.Atrous_c3_2 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3_2 = nn.ReLU(inplace=True)
        self.Atrous_c5_2 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5_2 = nn.ReLU(inplace=True)
        self.Atrous_c7_2 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7_2 = nn.ReLU(inplace=True)
        self.Aconv_2 = nn.Conv2d(64 * 5, 64, 1, padding=0)

        # -----------------------------  Multi-scale3  ----------------------------- #
        self.Atrous_c1_3 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1_3 = nn.ReLU(inplace=True)
        self.Atrous_c3_3 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3_3 = nn.ReLU(inplace=True)
        self.Atrous_c5_3 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5_3 = nn.ReLU(inplace=True) 
        self.Atrous_c7_3 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7_3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7_3 = nn.ReLU(inplace=True)
        self.Aconv_3 = nn.Conv2d(64 * 5, 64, 1, padding=0)

        # -----------------------------  Multi-scale4  ----------------------------- #
        self.Atrous_c1_4 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1_4 = nn.ReLU(inplace=True)
        self.Atrous_c3_4 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3_4 = nn.ReLU(inplace=True)
        self.Atrous_c5_4 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5_4 = nn.ReLU(inplace=True)
        self.Atrous_c7_4 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7_4 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7_4 = nn.ReLU(inplace=True)
        self.Aconv_4 = nn.Conv2d(64 * 5, 64, 1, padding=0)

        # -----------------------------  Multi-scale5  ----------------------------- #
        self.Atrous_c1_5 = nn.Conv2d(64, 64, 3, padding=1, dilation=1)  # size:  64*64*64
        self.Atrous_b1_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r1_5 = nn.ReLU(inplace=True)
        self.Atrous_c3_5 = nn.Conv2d(64, 64, 3, padding=3, dilation=3)  # size:  64*64*64
        self.Atrous_b3_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r3_5 = nn.ReLU(inplace=True)
        self.Atrous_c5_5 = nn.Conv2d(64, 64, 3, padding=5, dilation=5)  # size:  64*64*64
        self.Atrous_b5_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r5_5 = nn.ReLU(inplace=True)
        self.Atrous_c7_5 = nn.Conv2d(64, 64, 3, padding=7, dilation=7)  # size:  64*64*64
        self.Atrous_b7_5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.Atrous_r7_5 = nn.ReLU(inplace=True)
        self.Aconv_5 = nn.Conv2d(64 * 5, 64, 1, padding=0)
 
        # ----------------------------- Attentive ConvLSTM 2 -------------------------- #
        # ConvLSTM-2
        self.conv_fcn2_1 = nn.Conv2d(64, 64, 1, padding=0)  
        self.conv_h_1 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_1 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_1 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv_fcn2_2 = nn.Conv2d(64, 64, 1, padding=0)  
        self.conv_h_2 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_2 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_2 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv_fcn2_3 = nn.Conv2d(64, 64, 1, padding=0)  
        self.conv_h_3 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_3 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_3 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv_fcn2_4 = nn.Conv2d(64, 64, 1, padding=0) 
        self.conv_h_4 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_4 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_4 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv_fcn2_5 = nn.Conv2d(64, 64, 1, padding=0) 
        self.conv_h_5 = nn.Conv2d(64, 64, 1, padding=0)
        self.pool_avg_5 = nn.AvgPool2d(64, stride=2, ceil_mode=True)
        self.conv_c_5 = nn.Conv2d(64, 64, 1, padding=0)

        # ----------------------------- Prediction  -------------------------- #
        self.prediction_focal = nn.Conv2d(64, 2, 1, padding=0)
        self.dfeature_map = nn.Conv2d(64, 1, 1, padding=0)
        self.dfeature_high_map = nn.Conv2d(64, 1, 1, padding=0)
        self.prediction_rgb = nn.Conv2d(64, 2, 1, padding=0)
        self.tfeature_map = nn.Conv2d(64, 1, 1, padding=0)
        self.tfeature_high_map = nn.Conv2d(64, 1, 1, padding=0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               # m.weight.data.zero_()
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def convlstm_cell(self,A, new_c):
        (ai, af, ao, ag) = torch.split(A, A.size()[1] // 4, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ag)
        g = torch.tanh(ag)
        new_c = f * new_c + i * g
        new_h = o * torch.tanh(new_c)
        return new_c , new_h



    def forward(self, r1, r2, r3, r4, r5, f1, f2, f3, f4, f5):
        
        # weighted focal slices
        f2_ori = f2
        f2 = torch.cat(torch.chunk(f2, 12, dim=0), dim=1)
        weight2 = self.conv_w2(f2)
        weight2 = self.pool_avg_w2(weight2)
        weight2 = torch.mul(F.softmax(weight2, dim=1), 12)
        weight2 = weight2.transpose(0, 1)
        f2 = torch.mul(f2_ori, weight2)
        f3_ori = f3
        f3 = torch.cat(torch.chunk(f3, 12, dim=0), dim=1)
        weight3 = self.conv_w3(f3)
        weight3 = self.pool_avg_w3(weight3)
        weight3 = torch.mul(F.softmax(weight3, dim=1), 12)
        weight3 = weight3.transpose(0, 1)
        f3 = torch.mul(f3_ori, weight3)
        f4_ori = f4
        f4 = torch.cat(torch.chunk(f4, 12, dim=0), dim=1)
        weight4 = self.conv_w4(f4)
        weight4 = self.pool_avg_w4(weight4)
        weight4 = torch.mul(F.softmax(weight4, dim=1), 12)
        weight4 = weight4.transpose(0, 1)
        f4 = torch.mul(f4_ori, weight4)
        f5_ori = f5
        f5 = torch.cat(torch.chunk(f5, 12, dim=0), dim=1)
        weight5 = self.conv_w5(f5)
        weight5 = self.pool_avg_w5(weight5)
        weight5 = torch.mul(F.softmax(weight5, dim=1), 12)
        weight5 = weight5.transpose(0, 1)
        f5 = torch.mul(f5_ori, weight5)
        
        #flow 1 rgb -> focal
        f1_r = torch.mul(f1, r1)
        f1 = f1 + f1_r
        f2_r = torch.mul(f2, r2)
        f2 = f2 + f2_r
        f3_r = torch.mul(f3, r3)
        f3 = f3 + f3_r
        f4_r = torch.mul(f4, r4)
        f4 = f4 + f4_r
        f5_r = torch.mul(f5, r5)
        f5 = f5 + f5_r
        
        # spatial-temporal ConvLSTM
        b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12 = torch.chunk(f2, 12, dim=0)	
        c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12 = torch.chunk(f3, 12, dim=0)	
        d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12 = torch.chunk(f4, 12, dim=0)	
        e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12 = torch.chunk(f5, 12, dim=0)	

        cell0 = b1
        h_state0 = b1
        combined = torch.cat((b1, h_state0), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, cell0)
        combined = torch.cat((b2, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((b3, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((b4, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((b5, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((b6, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((b7, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((b8, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((b9, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((b10, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((b11, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((b12, new_h), dim=1)
        A = self.conv_cell2(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        f2 = new_h
        # -----------------------------  level 3 spatial-temporal ConvLSTM  --------------------------------- #
        cell0 = c1
        h_state0 = c1
        combined = torch.cat((c1, h_state0), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, cell0)
        combined = torch.cat((c2, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((c3, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((c4, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((c5, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((c6, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((c7, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((c8, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((c9, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((c10, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((c11, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((c12, new_h), dim=1)
        A = self.conv_cell3(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        f3 = new_h
        # -----------------------------  level 4 spatial-temporal ConvLSTM  --------------------------------- #
        cell0 = d1
        h_state0 = d1
        combined = torch.cat((d1, h_state0), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, cell0)
        combined = torch.cat((d2, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((d3, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((d4, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((d5, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((d6, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((d7, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((d8, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((d9, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((d10, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((d11, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((d12, new_h), dim=1)
        A = self.conv_cell4(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        f4 = new_h
        # -----------------------------  level 5 spatial-temporal ConvLSTM  --------------------------------- #
        cell0 = e1
        h_state0 = e1
        combined = torch.cat((e1, h_state0), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, cell0)
        combined = torch.cat((e2, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((e3, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((e4, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((e5, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((e6, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((e7, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((e8, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((e9, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((e10, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((e11, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        combined = torch.cat((e12, new_h), dim=1)
        A = self.conv_cell5(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        f5 = new_h

        High_depth = f5
        
        #flow 2 focal -> rgb
        r1_f = torch.mul(r1, f1)
        r1 = r1 + r1_f
        r2_f = torch.mul(r2, f2)
        r2 = r2 + r2_f
        r3_f = torch.mul(r3, f3)
        r3 = r3 + r3_f
        r4_f = torch.mul(r4, f4)
        r4 = r4 + r4_f
        r5_f = torch.mul(r5, f5)
        r5 = r5 + r5_f
        
        # r2
        A1 = self.Atrous_r1_2(self.Atrous_b1_2(self.Atrous_c1_2(r2)))
        A3 = self.Atrous_r3_2(self.Atrous_b3_2(self.Atrous_c3_2(r2)))
        A5 = self.Atrous_r5_2(self.Atrous_b5_2(self.Atrous_c5_2(r2)))
        A7 = self.Atrous_r7_2(self.Atrous_b7_2(self.Atrous_c7_2(r2)))
        r2 = torch.cat([r2, A1, A3, A5, A7], dim=1)
        r2 = self.Aconv_2(r2)
        # r3
        A1 = self.Atrous_r1_3(self.Atrous_b1_3(self.Atrous_c1_3(r3)))
        A3 = self.Atrous_r3_3(self.Atrous_b3_3(self.Atrous_c3_3(r3)))
        A5 = self.Atrous_r5_3(self.Atrous_b5_3(self.Atrous_c5_3(r3)))
        A7 = self.Atrous_r7_3(self.Atrous_b7_3(self.Atrous_c7_3(r3)))
        r3 = torch.cat([r3, A1, A3, A5, A7], dim=1)
        r3 = self.Aconv_3(r3)
        # r4
        A1 = self.Atrous_r1_4(self.Atrous_b1_4(self.Atrous_c1_4(r4)))
        A3 = self.Atrous_r3_4(self.Atrous_b3_4(self.Atrous_c3_4(r4)))
        A5 = self.Atrous_r5_4(self.Atrous_b5_4(self.Atrous_c5_4(r4)))
        A7 = self.Atrous_r7_4(self.Atrous_b7_4(self.Atrous_c7_4(r4)))
        r4 = torch.cat([r4, A1, A3, A5, A7], dim=1)
        r4 = self.Aconv_4(r4)
        # r5
        A1 = self.Atrous_r1_5(self.Atrous_b1_5(self.Atrous_c1_5(r5)))
        A3 = self.Atrous_r3_5(self.Atrous_b3_5(self.Atrous_c3_5(r5)))
        A5 = self.Atrous_r5_5(self.Atrous_b5_5(self.Atrous_c5_5(r5)))
        A7 = self.Atrous_r7_5(self.Atrous_b7_5(self.Atrous_c7_5(r5)))
        r5 = torch.cat([r5, A1, A3, A5, A7], dim=1)
        r5 = self.Aconv_5(r5)

        High_texture = r5
        
        #flow 3 rgb -> focal
        f1_r = torch.mul(f1, r1)
        f1 = f1 + f1_r
        f2_r = torch.mul(f2, r2)
        f2 = f2 + f2_r
        f3_r = torch.mul(f3, r3)
        f3 = f3 + f3_r
        f4_r = torch.mul(f4, r4)
        f4 = f4 + f4_r
        f5_r = torch.mul(f5, r5)
        f5 = f5 + f5_r
        
        # Attention ConvLSTM2 Focal
        new_h = f5
        out5_ori = f5
        f5_c = self.conv_fcn2_5(f5)
        h_c = self.conv_h_5(new_h)
        fh5 = f5_c + h_c
        fh5 = self.pool_avg_5(fh5)
        fh5 = self.conv_c_5(fh5)
        w5 = torch.mul(F.softmax(fh5, dim=1), 64)
        fw5 = torch.mul(w5, out5_ori)
        combined = torch.cat((fw5, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)

        out4_ori = f4 + f5
        f4_c = self.conv_fcn2_4(out4_ori)
        h_c = self.conv_h_4(new_h)
        fh4 = f4_c + h_c
        fh4 = self.pool_avg_4(fh4)
        fh4 = self.conv_c_4(fh4)
        w4 = torch.mul(F.softmax(fh4, dim=1), 64)
        fw4 = torch.mul(w4, out4_ori)
        combined = torch.cat((fw4, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)

        out3_ori = f4 + f5 + f3
        f3_c = self.conv_fcn2_3(out3_ori)
        h_c = self.conv_h_3(new_h)
        fh3 = f3_c + h_c
        fh3 = self.pool_avg_3(fh3)
        fh3 = self.conv_c_3(fh3)
        w3 = torch.mul(F.softmax(fh3, dim=1), 64)
        fw3 = torch.mul(w3, out3_ori)
        combined = torch.cat((fw3, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        
        out2_ori = f2 + f4 + f5 + f3
        f2_c = self.conv_fcn2_2(out2_ori)
        h_c = self.conv_h_2(new_h)
        fh2 = f2_c + h_c
        fh2 = self.pool_avg_2(fh2)
        fh2 = self.conv_c_2(fh2)
        w2 = torch.mul(F.softmax(fh2, dim=1), 64)
        fw2 = torch.mul(w2, out2_ori)
        combined = torch.cat((fw2, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)

        out_f = new_h

        # Attention ConvLSTM2 RGB
        new_h = r5
        out5_ori = r5
        r5_c = self.conv_fcn2_5(r5)
        h_c = self.conv_h_5(new_h)
        fh5 = r5_c + h_c
        fh5 = self.pool_avg_5(fh5)
        fh5 = self.conv_c_5(fh5)
        w5 = torch.mul(F.softmax(fh5, dim=1), 64)
        fw5 = torch.mul(w5, out5_ori)
        combined = torch.cat((fw5, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        
        out4_ori = r4 + r5
        r4_c = self.conv_fcn2_4(out4_ori)
        h_c = self.conv_h_4(new_h)
        fh4 = r4_c + h_c
        fh4 = self.pool_avg_4(fh4)
        fh4 = self.conv_c_4(fh4)
        w4 = torch.mul(F.softmax(fh4, dim=1), 64)
        fw4 = torch.mul(w4, out4_ori)
        combined = torch.cat((fw4, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
        
        out3_ori = r4 + r5 + r3
        r3_c = self.conv_fcn2_3(out3_ori)
        h_c = self.conv_h_3(new_h)
        fh3 = r3_c + h_c
        fh3 = self.pool_avg_3(fh3)
        fh3 = self.conv_c_3(fh3)
        w3 = torch.mul(F.softmax(fh3, dim=1), 64)
        fw3 = torch.mul(w3, out3_ori)
        combined = torch.cat((fw3, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)
       
        out2_ori = r2 + r4 + r5 + r3
        r2_c = self.conv_fcn2_2(out2_ori)
        h_c = self.conv_h_2(new_h)
        fh2 = r2_c + h_c
        fh2 = self.pool_avg_2(fh2)
        fh2 = self.conv_c_2(fh2)
        w2 = torch.mul(F.softmax(fh2, dim=1), 64)
        fw2 = torch.mul(w2, out2_ori)
        combined = torch.cat((fw2, new_h), dim=1)
        A = self.conv_cell(combined)
        new_c, new_h = self.convlstm_cell(A, new_c)

        out_r = new_h


        output_f = self.prediction_focal(out_f)
        pred_focal = F.upsample(output_f, scale_factor=4, mode='bilinear')
        dfeature_map = self.dfeature_map(out_f)
        pred_depth =  F.interpolate(dfeature_map, scale_factor=4, mode='bilinear', align_corners=False)
        dfeature_high_map = self.dfeature_high_map(High_depth)
        pred_depth_high =  F.interpolate(dfeature_high_map, scale_factor=4, mode='bilinear', align_corners=False)

        output_r = self.prediction_rgb(out_r)
        pred_rgb = F.upsample(output_r, scale_factor=4, mode='bilinear')
        tfeature_map = self.tfeature_map(out_r)
        pred_texture =  F.interpolate(tfeature_map, scale_factor=4, mode='bilinear', align_corners=False)
        tfeature_high_map = self.tfeature_high_map(High_texture)
        pred_texture_high =  F.interpolate(tfeature_high_map, scale_factor=4, mode='bilinear', align_corners=False)

        return pred_focal, pred_depth_high, pred_depth, pred_rgb, pred_texture_high, pred_texture


from utils.math_utils import max2d, min2d
import Model
if __name__ == '__main__':

    model_rgb = Model.RGBNet(cfg.SOLVER.NCLASS)
    model_focal = Model.FocalNet(cfg.SOLVER.NCLASS)
    clstm = New(cfg.SOLVER.NCLASS)
   
    focal = torch.randn(12,3,256,256)
    rgb = torch.randn(1,3,256,256)
  
    r1,r2,r3,r4,r5 = model_rgb(rgb)
    f1,f2,f3,f4,f5 = model_focal(focal)
    pred_focal, pred_depth_high, pred_depth, pred_rgb, pred_texture_high, pred_texture = clstm(r1, r2, r3, r4, r5, f1, f2, f3, f4, f5) 
    
  
