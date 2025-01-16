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


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


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

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


NUM_CLASS = 9



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
        self.feat_spe = nn.Linear(self.num_features, n_fc1)
        self.feat_ss = nn.Linear(n_fc1, n_fc2)
        # self.feat_ss2 = nn.Linear(n_fc2, 64)
        self.feat_ss2 = nn.Linear(n_fc1, 64)
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.feat_spe(y)
        y = self.relu(y)
        x = torch.cat([x, y], 1)
        x_re1 = y
        x_re1 = self.l2norm(x_re1)
        if self.dropout > 0:
            x = self.drop(x)

        x = self.classifier(x)

        return x, x_re1


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




class CCT_Net(nn.Module):
    def __init__(self, num_features=103, dropout=0, num_classes=0):
        super(CCT_Net, self).__init__()
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
        self.feat_spe = nn.Linear(self.num_features, n_fc1)
        self.feat_ss = nn.Linear(2624, n_fc2)
        self.feat_ss2 = nn.Linear(n_fc2, 64)
        self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.feat_ss3 = nn.Linear(n_fc2, 64)
        self.classifier = nn.Linear(2624, self.num_classes)
        self.l2norm = Normalize(2)
        self.decoder = decoder(num_features)

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

        x = self.avgpool(x) # 64, 5, 5



        x = x.view(x.size(0), -1)

        y = self.feat_spe(y)
        y = self.relu(y)

        fea1 = torch.cat([x, y], 1)
        fea = self.feat_ss(fea1)

        recon_y, recon_x = self.decoder(fea) # FEA 256

        return  fea1, fea1

class decoder(nn.Module):
    def __init__(self, num_features=103, dropout=0, num_classes=0):
        super(decoder, self).__init__()
        # self.conv00 = nn.Conv2d(103, 30, kernel_size=1, stride=1,
        #                        bias=True)
        # Append new layers
        n_fc1 = 1024
        self.recon_y1 = nn.Linear(256, 128)
        self.recon_y2 = nn.Linear(128, num_features)
        self.recon_x = nn.Linear(256, 1600)
        self.up_sample = nn.Upsample(4)
        self.RE_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.RE_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.up_sample2 = nn.Upsample(20)
        self.conv0 = nn.Conv2d(64, 60, kernel_size=1, stride=1,
                               bias=True)


    def forward(self, x):
        y_re = self.recon_y1(x)
        y_re = self.recon_y2(y_re)
        x = self.recon_x(x)
        x = x.reshape(x.size(0), 64, 5, 5)
        x = self.up_sample(x)
        x = self.RE_conv1(x)
        x = self.up_sample2(x)
        x = self.RE_conv2(x)
        x_re = self.conv0(x)

        return y_re, x_re

class classifier(nn.Module):
    def __init__(self, num_class):
        super(classifier, self).__init__()
        self.fc = nn.Linear(2624, num_class, bias=True)
        # self.f.load_state_dict(torch.load(pretrained_path), strict=False)

    def forward(self, x):
        out = self.fc(x)
        return out
