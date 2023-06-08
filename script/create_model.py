import math
import torch
import torch.nn as nn 
from . import axialnet

class FCBLock(nn.Module): #no dropout

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm_layer = nn.BatchNorm1d(out_features) 
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm_layer(x)
        x = self.activation(x)
        return x

class AxialGO_old(nn.Module):
    def __init__(self,emb_dim,winding_size,num_classes,mlp_expand=2):
        super().__init__()
        self.emb_1 = nn.Embedding(21,emb_dim,padding_idx=0)
        self.emb_2 = nn.Embedding(42,emb_dim,padding_idx=0)
        self.emb_3 = nn.Embedding(84,emb_dim,padding_idx=0)
        self.r = int(math.sqrt(emb_dim))
        self.pixel_shuffle = nn.PixelShuffle(self.r)
        #axial26s
        self.backbone = axialnet.AxialAttentionNet(axialnet.AxialBlock,[1, 2, 4, 1],in_dim=3, s=0.5, con2d_groups=1, groups=8, image_size=winding_size*self.r)
        #axial50s
        #self.backbone = axialnet.AxialAttentionNet(axialnet.AxialBlock,[3, 4, 6, 3],in_dim=3, s=0.5, con2d_groups=1, groups=8, image_size=winding_size*self.r)
        self.out = nn.Sequential(
            nn.Linear(1024*1, 1024*mlp_expand),
            nn.BatchNorm1d(1024*mlp_expand),
            nn.ReLU(inplace=True),
            nn.Linear(1024*mlp_expand, num_classes),
            nn.Sigmoid()
        )
        
        
    def forward(self,x):
        x_1 = self.emb_1(x).permute(0,3,1,2)
        x_1 = self.pixel_shuffle(x_1)
        x_2 = self.emb_2(x).permute(0,3,1,2)
        x_2 = self.pixel_shuffle(x_2)
        x_3 = self.emb_3(x).permute(0,3,1,2)
        x_3 = self.pixel_shuffle(x_3)
        x = torch.cat([x_1,x_2,x_3],dim=1)
        x = self.backbone(x)
        x = self.out(x)
        return x

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

# class MLPBlock(nn.Module):

    # def __init__(self, in_features, out_features, dropout=0.1):
        # super().__init__()
        # self.linear = nn.Linear(in_features, out_features, False)
        # self.norm = nn.BatchNorm1d(out_features) 
        # self.activation = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(dropout)

    # def forward(self, x):
        # x = self.norm(self.linear(x))
        # x = self.activation(x)
        # x = self.dropout(x)
        # return x

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
        x_1 = self.emb_1(x).permute(0,3,1,2)
        x_1 = self.pixel_shuffle(x_1)
        x_2 = self.emb_2(x).permute(0,3,1,2)
        x_2 = self.pixel_shuffle(x_2)
        x_3 = self.emb_3(x).permute(0,3,1,2)
        x_3 = self.pixel_shuffle(x_3)
        x = torch.cat([x_1,x_2,x_3],dim=1)
        x = self.backbone(x)
        # print('you can do it!')
        return x
       
# class AxialGO(nn.Module):
    # def __init__(self,emb_dim,winding_size,num_classes):
        # super().__init__()
        # self.axialnet = axialbackbone(emb_dim,winding_size)
        # self.fc = FCBLock(1024,2048)
        # self.out = nn.Sequential(
            # nn.Linear(2048, num_classes), 
            # nn.Sigmoid()
        # )
    # def forward(self,x_seq):
        # x_seq = self.axialnet(x_seq) #shape:(N,1024)
        # x_seq = self.fc(x_seq)
        # x_seq = self.out(x_seq)
        # return x_seq
        
class AxialGO(nn.Module):
    def __init__(self,emb_dim,winding_size,num_classes):
        super().__init__()
        self.axialnet = axialbackbone(emb_dim,winding_size)
        self.fc_out = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
            nn.Sigmoid()
        )
    def forward(self,x_seq):
        x_seq = self.axialnet(x_seq) #shape:(N,1024)
        x_seq = self.fc_out(x_seq)
        return x_seq 

class AxialProGO(nn.Module):
    def __init__(self,emb_dim,winding_size,num_classes):
        super().__init__()
        self.axialnet = axialbackbone(emb_dim,winding_size)
        self.mlp_1 = MLPBlock(26406,1024)
        self.mlp_2 = MLPBlock(2048,2048)
        self.fc =  nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Sigmoid()
        )
    def forward(self,x_seq,x_interpro):
        x_seq = self.axialnet(x_seq) #shape:(N,1024)
        x_interpro = self.mlp_1(x_interpro) #shape:(N,1024)
        x = self.mlp_2(torch.cat([x_seq,x_interpro],dim=1)) #shape:(N,2048)
        x = self.fc(x + torch.cat([x_seq,x_interpro],dim=1))
        return x 

class AxialESM(nn.Module):
    def __init__(self,emb_dim,winding_size,num_classes): #输入esm1b
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

class ESM1b(nn.Module):
    def __init__(self,num_classes): #输入esm1b
        super().__init__()
        self.mlp_1 = MLPBlock(1280,1024)
        self.mlp_2 = MLPBlock(1024,2048)
        self.fc =  nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Sigmoid()
        )
    def forward(self,x_esm1b):
        x_esm1b = self.mlp_1(x_esm1b) #shape:(N,1024)
        x_esm1b = self.mlp_2(x_esm1b) #shape:(N,2048)
        x_esm1b = self.fc(x_esm1b)
        return x_esm1b
        
class InterPro(nn.Module):
    def __init__(self,num_classes): #输入esm1b
        super().__init__()
        self.mlp_1 = MLPBlock(26406,1024)
        self.mlp_2 = MLPBlock(1024,2048)
        self.fc =  nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Sigmoid()
        )
    def forward(self,x_interpro):
        x_interpro = self.mlp_1(x_interpro) #shape:(N,1024)
        x_interpro = self.mlp_2(x_interpro) #shape:(N,2048)
        x_interpro = self.fc(x_interpro)
        return x_interpro

class structure_2_nores(nn.Module):
    def __init__(self,emb_dim,winding_size,num_classes):
        super().__init__()
        self.axialnet = axialbackbone(emb_dim,winding_size)
        self.mlp_1 = MLPBlock(26406,1024,bias=False)
        self.mlp_2 = MLPBlock(2048,2048,bias=False)
        self.fc =  nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Sigmoid()
        )
    def forward(self,x_seq,x_interpro):
        x_seq = self.axialnet(x_seq) #shape:(N,1024)
        x_interpro = self.mlp_1(x_interpro) #shape:(N,1024)
        x = self.mlp_2(torch.cat([x_seq,x_interpro],dim=1)) #shape:(N,2048)
        x = self.fc(x)
        return x 

class AxialESM_nores(nn.Module):
    def __init__(self,emb_dim,winding_size,num_classes):
        super().__init__()
        self.axialnet = axialbackbone(emb_dim,winding_size)
        self.mlp_1 = MLPBlock(1280,1024,bias=False)
        self.mlp_2 = MLPBlock(2048,2048,bias=False)
        self.fc =  nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Sigmoid()
        )
    def forward(self,x_seq,x_interpro):
        x_seq = self.axialnet(x_seq) #shape:(N,1024)
        x_interpro = self.mlp_1(x_interpro) #shape:(N,1024)
        x = self.mlp_2(torch.cat([x_seq,x_interpro],dim=1)) #shape:(N,2048)
        x = self.fc(x)
        return x 

# class axialprogo(nn.Module):
    # def __init__(self,emb_dim,winding_size,num_classes):
        # super().__init__()
        # self.axialnet = axialbackbone(emb_dim,winding_size)
        # self.mlp_seq = MLPBlock(1024,1024)
        # self.mlp_interpro = MLPBlock(26406,1024)
        # self.mlp_concat = MLPBlock(2048,2048)
        # self.fc =  nn.Sequential(
            # nn.Linear(2048, num_classes),
            # nn.Sigmoid()
        # )
    # def forward(self,x_seq,x_interpro):
        # x_seq = self.axialnet(x_seq) #shape:(N,1024)
        # x_seq = self.mlp_seq(x_seq)
        # x_interpro = self.mlp_interpro(x_interpro)
        # x = self.mlp_concat(torch.cat([x_seq,x_interpro],dim=1))+torch.cat([x_seq,x_interpro],dim=1)
        # x = self.fc(x)
        # return x 

# class axialprogo2(nn.Module):
    # def __init__(self,emb_dim,winding_size,num_classes):
        # super().__init__()
        # self.axialnet = axialbackbone(emb_dim,winding_size)
        # self.mlp_seq = MLPBlock(1024,1024)
        # self.mlp_interpro = MLPBlock(26406,1024)
        # self.mlp_concat = MLPBlock(2048,2048)
        # self.fc =  nn.Sequential(
            # nn.Linear(2048, num_classes),
            # nn.Sigmoid()
        # )
    # def forward(self,x_seq,x_interpro):
        # x_seq = self.axialnet(x_seq) #shape:(N,1024)
        # x_seq = self.mlp_seq(x_seq)+x_seq
        # x_interpro = self.mlp_interpro(x_interpro)
        # x = self.mlp_concat(torch.cat([x_seq,x_interpro],dim=1))+torch.cat([x_seq,x_interpro],dim=1)
        # x = self.fc(x)
        # return x 



