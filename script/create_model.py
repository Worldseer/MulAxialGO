import math
import torch
import torch.nn as nn 
from . import axialnet

class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = nn.ReLU(inplace=True)
        self.layer_norm = nn.BatchNorm1d(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class axialbackbone(nn.Module):
    def __init__(self,emb_dim,winding_size):
        super().__init__()
        self.emb_1 = nn.Embedding(21,emb_dim,padding_idx=0)
        self.emb_2 = nn.Embedding(42,emb_dim,padding_idx=0)
        self.emb_3 = nn.Embedding(84,emb_dim,padding_idx=0)
        self.r = int(math.sqrt(emb_dim))
        self.pixel_shuffle = nn.PixelShuffle(self.r)
        self.backbone = axialnet.AxialAttentionNet(axialnet.AxialBlock,[1, 2, 4, 1],in_dim=3, s=0.5, con2d_groups=1, groups=8, image_size=winding_size*self.r)

    def forward(self,x):
        x_1 = self.emb_1(x)
        x_1 = x_1.permute(0,3,1,2)
        x_1 = self.pixel_shuffle(x_1)
        x_2 = self.emb_2(x).permute(0,3,1,2)
        x_2 = self.pixel_shuffle(x_2)
        x_3 = self.emb_3(x).permute(0,3,1,2)
        x_3 = self.pixel_shuffle(x_3)
        x = torch.cat([x_1,x_2,x_3],dim=1)
        x = self.backbone(x)
        # print('you can do it!')
        return x
       

class MulAxialGO(nn.Module):
    def __init__(self,emb_dim,winding_size,num_classes):
        super().__init__()
        self.axialnet = axialbackbone(emb_dim,winding_size)
        self.mlp_1 = MLPBlock(1280,1024)
        self.mlp_2 = MLPBlock(2048,2048)
        self.fc =  nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Sigmoid()
        )
    def forward(self,x_seq,x_esm1b):
        x_seq = self.axialnet(x_seq) #shape:(N,1024)
        x_esm1b = self.mlp_1(x_esm1b) #shape:(N,1024)
        x = self.mlp_2(torch.cat([x_seq,x_esm1b],dim=1)) #shape:(N,2048)
        x = self.fc(x + torch.cat([x_seq,x_esm1b],dim=1))
        return x 