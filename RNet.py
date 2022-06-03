import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *
import math
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding,stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, X):
        # print("xxx : " , X.shape)
        Y = F.relu(self.bn1(self.conv1(X)))
        # print("Y :" , Y.shape)
        Y = self.bn2(self.conv2(Y))
        # print("Y: " , Y.shape)
        if self.conv3:
            X = self.conv3(X)
        # print("Y+X:" , (Y+X).shape)
        return F.relu(Y + X)

class RNet(torch.nn.Module):
    def __init__(self ):
        super().__init__()
        self.conv3d_1 = torch.nn.Sequential(
            torch.nn.Conv3d(1 , 24 , (  3 , 3 ,7) , stride =1 , padding = 0) ,
            # torch.nn.BatchNorm3d(8) ,
            # torch.nn.ReLU(inplace = True)
            # torch.nn.PReLU()
        )

        self.upconv3d = torch.nn.ConvTranspose3d(24 , 1 , (3 , 3 ,7)  , stride=1 , padding=0)


        self.res_net1 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net2 = Residual(24,24, (3, 3, 1), (1, 1, 0))


        kernel_3d = math.ceil((30 - 6) / 2)
        # print("ker : " , kernel_3d)
        self.conv2 = nn.Conv3d(in_channels=24, out_channels=64, padding=(0, 0, 0),
                           kernel_size=(1, 1, 24), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
        nn.BatchNorm3d(64, eps=0.001, momentum=0.1, affine=True),  
        nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=24, padding=(0, 0, 0),
                           kernel_size=(3, 3, 64), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
        nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  
        nn.ReLU(inplace=True)
    )

        self.clip_order = nn.Sequential(
            # nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),

            # nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=24, out_channels=1, kernel_size=(9,9,1)) ,# 256
            nn.ReLU(inplace=True),
        )
        self.clip_order_drop = nn.Dropout(0.5)
        self.clip_order_linear = nn.Linear(24, 2)

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(
        # nn.Dropout(p=0.5),
        nn.Linear(24, 16)  # ,
        # nn.Softmax()
    )

    def forward(self , x ,clip=False):
        x = x.permute(0, 1, 4, 3, 2)
        # print(x.shape)
        x1 = self.conv3d_1(x)
        # print("x1 : " , x1.shape)
        x22 = self.upconv3d(x1)
        x22 = x22.permute(0,1,4,3,2)
        # print("x1 : " ,x1.shape)
        # print("x22 : " , x22.shape)
        if clip == True:
            b , _,_,_,_ = x1.shape
            # print("xxx : " , x1.shape)
            xx = self.clip_order(x1)
            # print("xxxxxx : " , xx.shape)
            xx = xx.view(b,-1)
            xx = self.clip_order_drop(xx)
            # print("xxxxx : " , xx.shape)
            xx = self.clip_order_linear(xx)
            return xx

        x2 = self.res_net1(x1)
        x2 = self.batch_norm2(self.conv2(x2))
        # print('x2', x2.shape)
        x2 = x2.permute(0, 4, 2, 3, 1)#permute 
        # print('x2', x2.shape)
        x2 = self.batch_norm3(self.conv3(x2))
        # print("X2 : " , x2.shape)
        x3 = self.res_net2(x2)
        x4 = self.avg_pooling(x3)#
        # print(x4.shape)
        x4 = x4.view(x4.size(0), -1)#resize
        # print(x4.shape)
        out = self.full_connection(x4)
        return out , x22 , x3




#
