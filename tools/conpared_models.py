import torch
from torch import nn
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
import math
from skimage import transform
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


NUM_CLASS = 9

class SSFTTnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(SSFTTnet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):
        x = x.reshape(x.shape[0],1, x.shape[1], x.shape[2], x.shape[3])
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x)
        x = rearrange(x,'b c h w -> b (h w) c')

        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)  # main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x

class BaseNet1(nn.Module):
    def __init__(self,num_features=103, dropout=0, num_classes=0):
        super(BaseNet1, self).__init__()

        self.conv0 = nn.Conv2d(5, 64, kernel_size=1, stride=1,
                               bias=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
      
        self.num_features = num_features
        
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)
        self.num_classes = num_classes
        # Append new layers
        n_fc1 = 1024
        n_fc2 = 256
        self.l2norm = Normalize(2)
        self.feat_spe = nn.Linear(self.num_features, n_fc1)
        self.feat_ss = nn.Linear(n_fc1+n_fc1, n_fc2)
        
        self.classifier = nn.Linear(n_fc2, self.num_classes)


    def forward(self, x,y):        
        x = self.conv0(x)
        x_res = x
        x = self.conv1(x)
        x = self.relu(x+x_res)
        x = self.avgpool(x)
        x_res = x
        x = self.conv2(x)
        x = self.relu(x+x_res)
        x = self.avgpool(x)        
        
        x = x.view(x.size(0), -1)

        y = self.feat_spe(y)   
        y = self.relu(y)      

        x = torch.cat([x,y],1)
        x_re = self.feat_ss(x)
        x = self.relu(x_re)
        
        if self.dropout > 0:
            x = self.drop(x)

        x = self.classifier(x)     

        return x, x_re





# class BaseNet2(nn.Module):
#     def __init__(self, num_features=103, dropout=0, num_classes=0):
#         super(BaseNet2, self).__init__()
#         # self.conv00 = nn.Conv2d(103, 30, kernel_size=1, stride=1,
#         #                        bias=True)
#         self.conv0 = nn.Conv2d(60,64, kernel_size=1, stride=1,
#                                bias=True)
#         self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
#                                bias=True)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
#                                bias=True)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#
#         self.num_features = num_features
#
#         self.dropout = dropout
#         self.drop = nn.Dropout(self.dropout)
#         self.num_classes = num_classes
#         # Append new layers
#         n_fc1 = 1024
#         n_fc2 = 256
#
#         self.feat_spe = nn.Linear(self.num_features, n_fc1)
#         self.feat_ss = nn.Linear(n_fc1 + n_fc1, n_fc2)
#         self.feat_ss2 = nn.Linear(n_fc2, 64)
#         self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
#         self.feat_ss3 = nn.Linear(n_fc2, 64)
#         self.classifier = nn.Linear(n_fc2, self.num_classes)
#         self.l2norm = Normalize(2)
#         self.attention_spatial = PAM_Module(64)
#     def forward(self, x, y):
#         # x = self.conv00(x)
#         x = self.conv0(x)
#         x_res = x
#         x = self.conv1(x)
#         x = self.relu(x + x_res)
#         x = self.avgpool(x)
#         x_res = x
#         x = self.conv2(x)
#         x0 = self.relu(x + x_res)
#         x = self.attention_spatial(x0)
#         # x = torch.mul(x, x)
#         x2 = torch.mul(x, x0)
#         x = self.avgpool(x2)
#
#         x = x.view(x.size(0), -1)
#
#         y = self.feat_spe(y)
#         y = self.relu(y)
#
#         x = torch.cat([x, y], 1)
#         x_re = self.feat_ss(x)
#         x = self.relu(x_re)
#         x_re1 = self.feat_ss2(x)
#         # x_re1 = self.relu_mlp(x_re1)
#         # x_re1 = self.feat_ss3(x_re1)
#         x_re1 = self.l2norm(x_re1)
#         if self.dropout > 0:
#             x = self.drop(x)
#
#         x = self.classifier(x)
#
#         return x, x_re1

# class BaseNet22(nn.Module):
#     def __init__(self, num_features=60, dropout=0, num_classes=0):
#         super(BaseNet22, self).__init__()
#         # self.conv00 = nn.Conv2d(103, 30, kernel_size=1, stride=1,
#         #                        bias=True)
#         self.conv0 = nn.Conv2d(30, 64, kernel_size=1, stride=1,
#                                bias=True)
#         self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
#                                bias=True)
#         self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1,
#                                bias=True)
#         self.relu = nn.ReLU(inplace=True)
#
#         # self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#
#         self.num_features = num_features
#
#         self.dropout = dropout
#         self.drop = nn.Dropout(self.dropout)
#         self.num_classes = num_classes
#         # Append new layers
#         n_fc1 = 1024
#         n_fc2 = 256
#         self.attention_spatial = PAM_Module(256)
#         self.feat_spe = nn.Linear(self.num_features, n_fc1)
#         self.feat_ss = nn.Linear(1280, n_fc2)
#         self.feat_ss2 = nn.Linear(n_fc2, 64)
#         self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
#         self.feat_ss3 = nn.Linear(n_fc2, 64)
#         self.classifier = nn.Linear(n_fc2, self.num_classes)
#         self.l2norm = Normalize(2)
#
#     def forward(self, x, y):
#         # x = self.conv00(x)
#         x = self.conv0(x)
#         x_res1 = x
#         x = self.conv1(x)
#         # x = self.relu(x + x_res)
#         x = self.relu(torch.concat((x, x_res1), dim=1))
#         x_res2 = x
#         x = self.conv2(x)
#         x0 = self.relu(torch.concat((x, x_res1, x_res2), dim=1))
#         x = self.attention_spatial(x0)
#         # x = torch.mul(x, x)
#         x2 = torch.mul(x, x0)
#
#         x = self.avgpool(x2)
#
#
#
#         x = x.view(x.size(0), -1)
#
#         y = self.feat_spe(y)
#         y = self.relu(y)
#
#         x = torch.cat([x, y], 1)
#         x_re = self.feat_ss(x)
#         x = self.relu(x_re)
#         x_re1 = self.feat_ss2(x)
#         # x_re1 = self.relu_mlp(x_re1)
#         # x_re1 = self.feat_ss3(x_re1)
#         x_re1 = self.l2norm(x_re1)
#         if self.dropout > 0:
#             x = self.drop(x)
#
#         x = self.classifier(x)
#
#         return x, x_re1

class BaseNet2(nn.Module):
    def __init__(self, num_features=103, dropout=0, num_classes=0):
        super(BaseNet2, self).__init__()
        # self.conv00 = nn.Conv2d(103, 30, kernel_size=1, stride=1,
        #                        bias=True)
        self.conv0 = nn.Conv2d(60, 64, kernel_size=1, stride=1,
                               bias=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.num_features = num_features

        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)
        self.num_classes = num_classes
        # Append new layers
        n_fc1 = 1024
        n_fc2 = 256
        self.attention_spatial = PAM_Module(64)
        self.feat_spe = nn.Linear(self.num_features, n_fc1)
        self.feat_ss = nn.Linear(n_fc1, n_fc2)
        self.feat_ss2 = nn.Linear(n_fc2, 64)
        self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.feat_ss3 = nn.Linear(n_fc2, 64)
        self.classifier = nn.Linear(2624, self.num_classes)
        self.l2norm = Normalize(2)

    def forward(self, x, y):
        # x = self.conv00(x)
        x = self.conv0(x)
        x_res = x
        x = self.conv1(x)
        x = self.relu(x + x_res)
        x = self.avgpool(x)
        x_res = x
        x = self.conv2(x)
        x = self.relu(x + x_res)
        # x = self.attention_spatial(x)
        # x = torch.mul(x, x)

        x = self.avgpool(x)



        x = x.view(x.size(0), -1)

        y = self.feat_spe(y)
        y = self.relu(y)

        x = torch.cat([x, y], 1)
        x_re = self.feat_ss(y)
        y = self.relu(x_re)
        x_re1 = self.feat_ss2(y)
        # x_re1 = self.relu_mlp(x_re1)
        # x_re1 = self.feat_ss3(x_re1)
        x_re1 = self.l2norm(x_re1)
        if self.dropout > 0:
            x = self.drop(x)

        x = self.classifier(x)

        return x, x_re1












# class BaseNet2(nn.Module):
#     def __init__(self, num_features=103, dropout=0, num_classes=0):
#         super(BaseNet2, self).__init__()
#
#         self.conv0 = nn.Conv2d(5, 64, kernel_size=1, stride=1,
#                                bias=True)
#         self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
#                                bias=True)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
#                                bias=True)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#
#         self.num_features = num_features
#
#         self.dropout = dropout
#         self.drop = nn.Dropout(self.dropout)
#         self.num_classes = num_classes
#         # Append new layers
#         n_fc1 = 1024
#         n_fc2 = 256
#
#         self.feat_spe = nn.Linear(self.num_features, n_fc1)
#         self.feat_ss = nn.Linear(n_fc1 + n_fc1, n_fc2)
#         self.feat_ss2 = nn.Linear(n_fc2, n_fc2)
#
#         self.classifier = nn.Linear(n_fc2, self.num_classes)
#
#     def forward(self, x, y):
#         x = self.conv0(x)
#         x_res = x
#         x = self.conv1(x)
#         x = self.relu(x + x_res)
#         x = self.avgpool(x)
#         x_res = x
#         x = self.conv2(x)
#         x = self.relu(x + x_res)
#         x = self.avgpool(x)
#
#         x = x.view(x.size(0), -1)
#
#         y = self.feat_spe(y)
#         y = self.relu(y)
#
#         x = torch.cat([x, y], 1)
#         x_re = self.feat_ss(x)
#         x = self.relu(x_re)
#         x_re1 = self.feat_ss2(x)
#         if self.dropout > 0:
#             x = self.drop(x)
#
#         x = self.classifier(x)
#
#         return x, x_re1


def WeightEMA_BN(Base,Ensemble,alpha):
    one_minus_alpha = 1.0 - alpha
    Base_params = Base.state_dict().copy()
    Ensemble_params = Ensemble.state_dict().copy()
    
    for b, e in zip(Base_params, Ensemble_params):
        Ensemble_params[e] = Base_params[b] * one_minus_alpha + Ensemble_params[e] * alpha
    
    Ensemble.load_state_dict(Ensemble_params)
    return Ensemble




class SpaRandomization(nn.Module):
    def __init__(self, num_features, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.alpha = nn.Parameter(torch.tensor(0.01), requires_grad=True).to(device)

    def forward(self, x, ):
        N, C, H, W = x.size()
        # x = self.norm(x)
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)
            mean = (1) * mean[idx_swap]
            var =  (1) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x, idx_swap




class SpeRandomization(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, idx_swap, y=None):
        N, C = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()
            if y != None:
                for i in range(len(y.unique())):
                    index = y == y.unique()[i]
                    tmp, mean_tmp, var_tmp = x[index], mean[index], var[index]
                    tmp = tmp[torch.randperm(tmp.size(0))].detach()
                    tmp = tmp * (var_tmp + self.eps).sqrt() + mean_tmp
                    x[index] = tmp
            else:
                # idx_swap = torch.randperm(N)
                x = x[idx_swap].detach()

                x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C)
        return x











from torch.nn import Module, Sequential, Conv2d, Conv3d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding





class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # m_batchsize, channle, height, width, C = x.size()
        # x = x.squeeze(-1)
        # m_batchsize, C, height, width, channle = x.size()

        # proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*channle).permute(0, 2, 1)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*channle)
        # energy = torch.bmm(proj_query, proj_key)
        # attention = self.softmax(energy)
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*channle)
        #
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)

        m_batchsize, C, height, width, _ = x.size()
        x = x.reshape(m_batchsize, C, height, width)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = (self.gamma*out + x)
        out = out.reshape(m_batchsize, C, height, width, 1)
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, channle = x.size()
        #print(x.size())
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) #形状转换并交换维度
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)

        out = self.gamma*out + x  #C*H*W
        return out











class DBDA_network(nn.Module):
    def __init__(self, band, classes):
        super(DBDA_network, self).__init__()
        self.name = 'DBDA_NET'

        # spectral branch

        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Conv3d(in_channels=48, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
                                    nn.BatchNorm3d(72, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Conv3d(in_channels=72, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm3d(96, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=96, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1)) # kernel size随数据变化

        #注意力机制模块

        #self.max_pooling1 = nn.MaxPool3d(kernel_size=(7, 7, 1))
        #self.avg_pooling1 = nn.AvgPool3d(kernel_size=(7, 7, 1))
        self.max_pooling1 = nn.AdaptiveAvgPool3d(1)
        self.avg_pooling1 = nn.AdaptiveAvgPool3d(1)

        self.shared_mlp = nn.Sequential(
                                    nn.Conv3d(in_channels=60, out_channels=30,
                                            kernel_size=(1, 1, 1), stride=(1, 1, 1)),
                                    nn.Conv3d(in_channels=30, out_channels=60,
                                            kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )

        self.activation1 = nn.Sigmoid()


        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        # 注意力机制模块

        # self.max_pooling2 = nn.MaxPool3d(kernel_size=(1, 1, 60))
        # self.avg_pooling2 = nn.AvgPool3d(kernel_size=(1, 1, 60))
        # self.max_pooling2 = nn.AdaptiveAvgPool3d(1)
        # self.avg_pooling2 = nn.AdaptiveAvgPool3d(1)

        self.conv25 = nn.Sequential(
                                nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                                kernel_size=(3, 3, 2), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.feature_out = nn.Linear(120, 64)
        self.full_connection = nn.Sequential(
                                # nn.Dropout(p=0.5),
                                nn.Linear(64, 64),
                                nn.Linear(64, 32),
                                nn.Linear(32, classes) # ,
                                # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)
        self.l2norm = Normalize(2)
        #fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        X = X.permute(0, 2, 3, 1)
        x00 = X.reshape(X.shape[0], 1, X.shape[2], X.shape[2], -1)
        # spectral
        x11 = self.conv11(x00)
        #print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        #print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        #print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        #print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        #print('x16', x16.shape)  # 7*7*97, 60

        #print('x16', x16.shape)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)
        x1 = torch.mul(x1, x16)


        # spatial
        #print('x', X.shape)
        x21 = self.conv21(x00)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        #print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        #print('x25', x25.shape)

        # 空间注意力机制
        x2 = self.attention_spatial(x25)
        x2 = torch.mul(x2, x25)

        # model1
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2= self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)
        feature = self.feature_out(x_pre)
        #print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        feature = self.l2norm(feature)
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(feature)
        # output = self.fc(x_pre)
        return output, feature


class DBDA_network(nn.Module):
    def __init__(self, band, classes):
        super(DBDA_network, self).__init__()

        # spectral branch

        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Conv3d(in_channels=48, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm3d(72, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Conv3d(in_channels=72, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm3d(96, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=96, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))  # kernel size随数据变化

        # 注意力机制模块

        # self.max_pooling1 = nn.MaxPool3d(kernel_size=(7, 7, 1))
        # self.avg_pooling1 = nn.AvgPool3d(kernel_size=(7, 7, 1))
        self.max_pooling1 = nn.AdaptiveAvgPool3d(1)
        self.avg_pooling1 = nn.AdaptiveAvgPool3d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv3d(in_channels=60, out_channels=30,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(in_channels=30, out_channels=60,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )

        self.activation1 = nn.Sigmoid()

        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        # 注意力机制模块

        # self.max_pooling2 = nn.MaxPool3d(kernel_size=(1, 1, 60))
        # self.avg_pooling2 = nn.AvgPool3d(kernel_size=(1, 1, 60))
        # self.max_pooling2 = nn.AdaptiveAvgPool3d(1)
        # self.avg_pooling2 = nn.AdaptiveAvgPool3d(1)

        self.conv25 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                      kernel_size=(3, 3, 2), stride=(1, 1, 1)),
            nn.Sigmoid()
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(120, classes)  # ,
            # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

        # fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):
        B, C,H, W= X.shape
        X = X.permute(0, 2, 3, 1)
        X = X.reshape(B, 1, H, W, C)
        # spectral
        x11 = self.conv11(X)
        # print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        # print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        # print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        # print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        # print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)
        x1 = torch.mul(x1, x16)

        # spatial
        # print('x', X.shape)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        # print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        # print('x25', x25.shape)

        # 空间注意力机制
        x2 = self.attention_spatial(x25)
        x2 = torch.mul(x2, x25)

        # model1
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)
        # print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        return output
class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()
    # Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
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
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

class SSRN_network(nn.Module):
    def __init__(self, band, classes):
        super(SSRN_network, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.res_net1 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net2 = Residual(24, 24, (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(24, 24, (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(24, 24, (3, 3, 1), (1, 1, 0))

        kernel_3d = math.ceil((band - 6) / 2)

        self.conv2 = nn.Conv3d(in_channels=24, out_channels=128, padding=(0, 0, 0),
                               kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=24, padding=(0, 0, 0),
                               kernel_size=(3, 3, 128), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(96, classes)  # ,
            # nn.Softmax()
        )

    def forward(self, X):
        if len(X.shape) == 4:
            X = X.unsqueeze(dim=1)
        X = X.permute(0, 1, 3, 4, 2)
        x1 = self.batch_norm1(self.conv1(X))
        x2 = self.res_net1(x1)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        # print(x10.shape)
        return self.full_connection(x4)

class FDSSC_network(nn.Module):
    def __init__(self, band, classes):
        super(FDSSC_network, self).__init__()

        # spectral branch
        self.name = 'FDSSC'
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm1 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    nn.PReLU()
        )
        self.conv2 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    nn.PReLU()
        )
        self.conv3 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    nn.PReLU()
        )
        self.conv4 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm4 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        kernel_3d = math.ceil((band - 6) / 2)
        # print(kernel_3d)
        self.conv5 = nn.Conv3d(in_channels=60, out_channels=200, padding=(0, 0, 0),
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))

        self.batch_norm5 = nn.Sequential(
                                    nn.BatchNorm3d(1, eps=0.001, momentum=0.1, affine=True),
                                    nn.PReLU()
        )
        self.conv6 = nn.Conv3d(in_channels=1, out_channels=24, padding=(0, 0, 0), kernel_size=(1, 1, 200), stride=(1, 1, 1))

        self.batch_norm6 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.PReLU()
        )
        self.conv7 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm7 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv8 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm8 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )
        self.conv9 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                               kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm9 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            nn.PReLU()
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Linear(60, classes)
                                # nn.Softmax()
        )


    def forward(self, X):
        # spectral
        if len(X.shape) == 4:
            X = X.unsqueeze(dim=1)
        X = X.permute((0, 1, 3, 4, 2))
        x1 = self.conv1(X)
        x2 = self.batch_norm1(x1)
        x2 = self.conv2(x2)

        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.batch_norm2(x3)
        x3 = self.conv3(x3)
        #print('x13', x13.shape)

        x4 = torch.cat((x1, x2, x3), dim=1)
        x4 = self.batch_norm3(x4)
        x4 = self.conv4(x4)

        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        # print('x15', x15.shape)

        # print(x5.shape)
        x6 = self.batch_norm4(x5)
        x6 = self.conv5(x6)
        #print('x16', x16.shape)  # 7*7*97, 60

        #print('x16', x16.shape)
        # 光谱注意力通道
        x6 = x6.permute((0, 4, 2, 3, 1))
        # print(x6.shape)

        x7 = self.batch_norm5(x6)
        x7 = self.conv6(x7)

        x8 = self.batch_norm6(x7)
        x8 = self.conv7(x8)

        x9 = torch.cat((x7, x8), dim=1)
        x9 = self.batch_norm7(x9)
        x9 = self.conv8(x9)

        x10 = torch.cat((x7, x8, x9), dim=1)
        x10 = self.batch_norm8(x10)
        x10 = self.conv9(x10)

        x10 = torch.cat((x7, x8, x9, x10), dim=1)
        x10 = self.batch_norm9(x10)
        x10 = self.global_pooling(x10)
        x10 = x10.view(x10.size(0), -1)

        output = self.full_connection(x10)
        # output = self.fc(x_pre)
        return output



# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class OurFE(nn.Module):
    def __init__(self, channel, dim):
        super(OurFE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(3 * channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out = self.out_conv(torch.cat((out1, out2, out3), dim=1))
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            DEPTHWISECONV(dim, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=dim, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x):
        b, d, c = x.shape
        w = int(math.sqrt(d))
        x1 = rearrange(x, 'b (w h) c -> b c w h', w=w, h=w)
        x1 = self.net(x1)
        x1 = rearrange(x1, 'b c w h -> b (w h) c')
        x = x + x1
        return x


class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, padding=0, stride=1, is_fe=False):
        super(DEPTHWISECONV, self).__init__()
        self.is_fe = is_fe
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        if self.is_fe:
            return out
        out = self.point_conv(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0., num_patches=10):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.spatial_norm = nn.BatchNorm2d(heads)
        self.spatial_conv = nn.Conv2d(heads, heads, kernel_size=3, padding=1)

        self.spectral_norm = nn.BatchNorm2d(1)
        self.spectral_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.to_qkv_spec = nn.Linear(num_patches, num_patches*3, bias=False)
        self.attend_spec = nn.Softmax(dim=-1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.spatial_conv(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        output = self.to_out(out)

        x = x.transpose(-2, -1)
        qkv_spec = self.to_qkv_spec(x).chunk(3, dim=-1)
        q_spec, k_spec, v_spec = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=1), qkv_spec)
        dots_spec = torch.matmul(q_spec, k_spec.transpose(-1, -2)) * self.scale
        attn = self.attend_spec(dots_spec)  # .squeeze(dim=1)
        attn = self.spectral_conv(attn).squeeze(dim=1)

        return torch.matmul(output, attn)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., num_patches=25):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.index = 0
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim)),
            ]))

    def forward(self, x):
        output = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            output.append(x)

        return x, output


class SubNet(nn.Module):
    def __init__(self, patch_size, num_patches, dim, emb_dropout, depth, heads, dim_head, mlp_dim, dropout):
        super(SubNet, self).__init__()
        self.to_patch_embedding = nn.Sequential(
            DEPTHWISECONV(in_ch=dim, out_ch=dim, kernel_size = patch_size, stride = patch_size, padding=0, is_fe=True),
            Rearrange('b c w h -> b (h w) c '),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches+1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, dropout=dropout, num_patches=num_patches)


def get_num_patches(ps, ks):
    return int((ps - ks)/ks)+1


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super(ViT, self).__init__()
        self.ournet = OurFE(channels, dim)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=dim, kernel_size=1)
        self.net = nn.ModuleList()
        self.mlp_head = nn.ModuleList()
        for ps in patch_size:
            num_patches = get_num_patches(image_size, ps) ** 2
            patch_dim = dim * num_patches
            sub_net = SubNet(ps, num_patches, dim, emb_dropout, depth, heads, dim_head, mlp_dim, dropout)
            self.net.append(sub_net)
            self.mlp_head.append(nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, num_classes)
            ))

        self.weight = torch.ones(len(patch_size))

    def forward(self, img):
        if len(img.shape) == 5: img = img.squeeze()
        img = self.ournet(img)
        img = self.pool(img)
        img = self.conv4(img)

        all_branch = []
        for sub_branch in self.net:
            spatial = sub_branch.to_patch_embedding(img)
            b, n, c = spatial.shape
            spatial = spatial + sub_branch.pos_embedding[:, :n]
            spatial = sub_branch.dropout(spatial)
            _, outputs = sub_branch.transformer(spatial)
            res = outputs[-1]
            all_branch.append(res)

        self.weight = F.softmax(self.weight, 0)
        res = 0
        for i, mlp_head in enumerate(self.mlp_head):
            out1 = all_branch[i].flatten(start_dim=1)
            cls1 = mlp_head(out1)
            res = res + cls1 * self.weight[i]
        return res