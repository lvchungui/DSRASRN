import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed

sys.path.append('./')
sys.path.append('../')
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead
from .model_transformer import FeatureEnhancer, ReasoningTransformer, FeatureEnhancerW2V

def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride==1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)


class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)

import torch
import torch.nn as nn

class TEAB(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(TEAB, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f*6, n_feats, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        
        self.conv3 = conv(f*6, f*6, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(f)

        self.horizontal_conv_weight = nn.Parameter(torch.randn(f, f, 1, 3))
        self.horizontal_conv_bias = nn.Parameter(torch.randn(f,))
        
        self.vertical_conv_weight = nn.Parameter(torch.randn(f, f, 3, 1))
        self.vertical_conv_bias = nn.Parameter(torch.randn(f,))
        
        self.conv_reduce = conv(f*6, f, kernel_size=3, padding=1)

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        res = x
        # 获取通道数维度的大小
        num_channels = x.size(1)
        num_h = x.size(2)
        num_w = x.size(3)
        # 打印通道数
        print("通道数 (num_channels) =", num_channels)
        print("高 =", num_h)
        print("宽 =", num_w)

        c1_ = (self.conv1(x))
        
        c1_v1 = self.relu(self.bn(F.conv2d(c1_, self.vertical_conv_weight, bias=self.vertical_conv_bias, stride=(2,2), padding=(1,0), dilation=1)))
        c1_v2 = self.relu(self.bn(F.conv2d(c1_, self.vertical_conv_weight, bias=self.vertical_conv_bias, stride=(2,2), padding=(2,0), dilation=2)))
        c1_v3 = self.relu(self.bn(F.conv2d(c1_, self.vertical_conv_weight, bias=self.vertical_conv_bias, stride=(2,2), padding=(3,0), dilation=3)))
        
        c1_h1 = self.relu(self.bn(F.conv2d(c1_, self.horizontal_conv_weight, bias=self.horizontal_conv_bias, stride=(2,2), padding=(0,1), dilation=1)))
        c1_h2 = self.relu(self.bn(F.conv2d(c1_, self.horizontal_conv_weight, bias=self.horizontal_conv_bias, stride=(2,2), padding=(0,2), dilation=2)))
        c1_h3 = self.relu(self.bn(F.conv2d(c1_, self.horizontal_conv_weight, bias=self.horizontal_conv_bias, stride=(2,2), padding=(0,3), dilation=3)))
        
        c1_fusion = torch.cat([c1_v1, c1_v2, c1_v3, c1_h1, c1_h2, c1_h3], dim=1)

        # 获取通道数维度的大小
        num_channels = c1_fusion.size(1)
        num_h = c1_fusion.size(2)
        num_w = c1_fusion.size(3)
        # 打印通道数
        print("通道数 (num_channels) =", num_channels)
        print("高 =", num_h)
        print("宽 =", num_w)

        v_max = F.max_pool2d(c1_fusion, kernel_size=4, stride=2)

        # 获取通道数维度的大小
        num_channels = v_max.size(1)
        num_h = v_max.size(2)
        num_w = v_max.size(3)
        # 打印通道数
        print("通道数 (num_channels) =", num_channels)
        print("高 =", num_h)
        print("宽 =", num_w)

        c3 = self.conv3(v_max)
        
        c3 = F.interpolate(c3, (res.size(2), res.size(3)), mode='bilinear', align_corners=False)
        
        c4 = self.conv_f(c3)
        m = self.sigmoid(c4)
        # 获取通道数维度的大小
        num_channels = m.size(1)
        num_h = m.size(2)
        num_w = m.size(3)
        # 打印通道数
        print("通道数 (num_channels) =", num_channels)
        print("高 =", num_h)
        print("宽 =", num_w)
        
        return res+m

class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



class OrthogonalAttention(nn.Module):
    def __init__(self, in_channels, out_channels, conv=default_conv):
        super(OrthogonalAttention, self).__init__()

        # Convolutional layers for horizontal edge features
        self.conv_h_edge_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv_h_edge_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv_h_edge_3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))

        # Convolutional layers for vertical edge features
        self.conv_v_edge_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv_v_edge_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv_v_edge_3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))

        # Convolutional layer for combining the horizontal and vertical edge features
        self.conv_combine = nn.Conv2d(2*out_channels, 1, kernel_size=1)

        # Batch normalization layers
        self.bn_h_edge_1 = nn.BatchNorm2d(out_channels)
        self.bn_h_edge_2 = nn.BatchNorm2d(out_channels)
        self.bn_h_edge_3 = nn.BatchNorm2d(out_channels)
        self.bn_v_edge_1 = nn.BatchNorm2d(out_channels)
        self.bn_v_edge_2 = nn.BatchNorm2d(out_channels)
        self.bn_v_edge_3 = nn.BatchNorm2d(out_channels)
        self.bn_combine = nn.BatchNorm2d(1)
        
        self.fs = eca_layer(channel=in_channels*2)

    def forward(self, x):
        
        # Compute horizontal edge features
        h_edge = self.conv_h_edge_1(x)
        h_edge = self.bn_h_edge_1(h_edge)
        h_edge = nn.functional.relu(h_edge)

        h_edge = self.conv_h_edge_2(h_edge)
        h_edge = self.bn_h_edge_2(h_edge)
        h_edge = nn.functional.relu(h_edge)

        h_edge = self.conv_h_edge_3(h_edge)
        h_edge = self.bn_h_edge_3(h_edge)
        h_edge = nn.functional.relu(h_edge)
        h_edge = x + h_edge

        # Compute vertical edge features
        v_edge = self.conv_v_edge_1(x)
        v_edge = self.bn_v_edge_1(v_edge)
        v_edge = nn.functional.relu(v_edge)

        v_edge = self.conv_v_edge_2(v_edge)
        v_edge = self.bn_v_edge_2(v_edge)
        v_edge = nn.functional.relu(v_edge)

        v_edge = self.conv_v_edge_3(v_edge)
        v_edge = self.bn_v_edge_3(v_edge)
        v_edge = nn.functional.relu(v_edge)
        v_edge = x + v_edge

        # Combine the horizontal and vertical edge features
        combined = self.fs(torch.cat([h_edge, v_edge], dim=1))
        combined = self.conv_combine(combined)
        combined = self.bn_combine(combined)
        combined = nn.functional.sigmoid(combined)

        # Apply the attention map to the input feature map
        output = x * combined

        return output
  
    
    

class TSRN(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, STN=False, srb_nums=5, mask=True, hidden_units=32):
        super(TSRN, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2*hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2*hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2*hidden_units, 2*hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2*hidden_units)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2*hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2*hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height//scale_factor, width//scale_factor]
        tps_outputsize = [height//scale_factor, width//scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):
        # embed()
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        output = torch.tanh(block[str(self.srb_nums + 3)])

        # print("block_keys:", block.keys())
        # print("output:", output.shape)
        return output


class InfoGen(nn.Module):
    def __init__(self, t_emb, output_size):
        super(InfoGen, self).__init__()
        
        self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=(1, 0), bias=False)
        self.bn4 = nn.BatchNorm2d(output_size)

    def forward(self, t_embedding):

        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        return x



# class SeparableConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
#         super(SeparableConv2d, self).__init__()

#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x


# class InfoGen(nn.Module):
#     def __init__(self, t_emb, output_size):
#         super(InfoGen, self).__init__()

#         self.tconv1 = SeparableConv2d(t_emb, 512, 3, 1, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(512)
#         self.att1 = eca_layer(512)

#         self.tconv2 = SeparableConv2d(512, 128, 3, 1, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.att2 = eca_layer(128)

#         self.tconv3 = SeparableConv2d(128, 64, 3, 1, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.att3 = eca_layer(64)

#         self.tconv4 = SeparableConv2d(64, output_size, 3, 1, 1, bias=False)
#         self.bn4 = nn.BatchNorm2d(output_size)

#     def forward(self, t_embedding):
#         x = F.relu(self.bn1(self.tconv1(t_embedding)))
#         a = self.att1(x)
#         x = x * a.expand_as(x)
#         x = F.relu(self.bn2(self.tconv2(x)))
#         a = self.att2(x)
#         x = x * a.expand_as(x)
#         x = F.relu(self.bn3(self.tconv3(x)))
#         a = self.att3(x)
#         x = x * a.expand_as(x)
#         x = F.relu(self.bn4(self.tconv4(x)))
        
#         return x

# class SeparableConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
#         super(SeparableConv2d, self).__init__()

#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x


# class InfoGen(nn.Module):
#     def __init__(self, t_emb, output_size):
#         super(InfoGen, self).__init__()

#         self.tconv1 = SeparableConv2d(t_emb, 64, 3, 1, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.att1 = eca_layer(64)

#         self.tconv2 = SeparableConv2d(t_emb + 64, 128, 3, 1, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.att2 = eca_layer(128)

#         self.tconv3 = SeparableConv2d(t_emb + 64 + 128, 256, 3, 1, 1, bias=False)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.att3 = eca_layer(256)

#         self.tconv4 = SeparableConv2d(t_emb + 64 + 128 + 256, output_size, 3, 1, 1, bias=False)
#         self.bn4 = nn.BatchNorm2d(output_size)

#     def forward(self, t_embedding):
#         x1 = F.relu(self.bn1(self.tconv1(t_embedding)))
#         a1 = self.att1(x1)
#         x1 = x1 * a1.expand_as(x1)

#         x2 = F.relu(self.bn2(self.tconv2(torch.cat([t_embedding, x1], dim=1))))
#         a2 = self.att2(x2)
#         x2 = x2 * a2.expand_as(x2)

#         x3 = F.relu(self.bn3(self.tconv3(torch.cat([t_embedding, x1, x2], dim=1))))
#         a3 = self.att3(x3)
#         x3 = x3 * a3.expand_as(x3)

#         x4 = F.relu(self.bn4(self.tconv4(torch.cat([t_embedding, x1, x2, x3], dim=1))))
        
#         return x4


# class InfoGen(nn.Module):
#     def __init__(self, t_emb, output_size):
#         super(InfoGen, self).__init__()
        
#         self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(512)

#         self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)

#         self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(64)

#         self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=(1, 0), bias=False)
#         self.bn4 = nn.BatchNorm2d(output_size)
        
#         self.spatial_attention = OrthogonalAttention(in_channels=64, out_channels=64)

#     def forward(self, t_embedding):

#         x = F.relu(self.bn1(self.tconv1(t_embedding)))
#         x = F.relu(self.bn2(self.tconv2(x)))
#         x = F.relu(self.bn3(self.spatial_attention(self.tconv3(x))))
#         x = F.relu(self.bn4(self.tconv4(x)))

#         return x




class TSRN_TL(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 word_vec_d=300,
                 text_emb=37, #26+26+1
                 out_text_channels=32):
        super(TSRN_TL, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlockTL(2 * hidden_units, out_text_channels))


        self.feature_enhancer = None #FeatureEnhancerW2V(
                                #        vec_d=300,
                                #        feature_size=2 * hidden_units,
                                #        head_num=4,
                                #        dropout=True)

        # From [1, 1] -> [16, 16]
        self.infoGen = InfoGen(text_emb, out_text_channels)
        self.emb_cls = text_emb

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2 * hidden_units)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none',
                input_size=self.tps_inputsize)
            
        # self.spatial_attention = SEAttention(channel=2*hidden_units)
        # self.spatial_attention = OrthogonalAttention(in_channels=2*hidden_units, out_channels=2*hidden_units)
        self.spatial_attention = TEAB(2*hidden_units)

    def forward(self, x, text_emb=None):
        # embed()

        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}


        if text_emb is None:
            N, C, H, W = x.shape
            text_emb = torch.zeros((N, self.emb_cls, 1, 26))

        spatial_t_emb = self.infoGen(text_emb)
        spatial_t_emb = F.interpolate(spatial_t_emb, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

#         for i in range(self.srb_nums + 1):
#             if i + 2 in [2, 3, 4, 5, 6]:
#                 block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], spatial_t_emb)
#             else:
#                 block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

#         block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
#             ((block['1'] + block[str(self.srb_nums + 2)]))
#         output = torch.tanh(block[str(self.srb_nums + 3)])


        # # 2个
        # x1 = self.block1(x)
        
        # x2 = self.block2(x1, spatial_t_emb)
        # x3 = self.block3(x2, spatial_t_emb)
        
        # x4 = self.block4(self.spatial_attention(x3))
        
        # output = torch.tanh(self.block5(x4+x1))

        # # 3个
        # x1 = self.block1(x)
        
        # x2 = self.block2(x1, spatial_t_emb)
        # x3 = self.block3(x2, spatial_t_emb)
        # x4 = self.block4(x3, spatial_t_emb)
        
        # x5 = self.block5(self.spatial_attention(x4))
        
        # output = torch.tanh(self.block6(x5+x1))
        
        # # 4个
        # x1 = self.block1(x)
        
        # x2 = self.block2(x1, spatial_t_emb)
        # x3 = self.block3(x2, spatial_t_emb)
        # x4 = self.block4(x3, spatial_t_emb)
        # x5 = self.block5(x4, spatial_t_emb)
        
        # x6 = self.block6(self.spatial_attention(x5))
        
        # output = torch.tanh(self.block7(x6+x1))
    
        # 5个
        x1 = self.block1(x)
        
        x2 = self.block2(x1, spatial_t_emb)
        x3 = self.block3(x2, spatial_t_emb)
        x4 = self.block4(x3, spatial_t_emb)
        x5 = self.block5(x4, spatial_t_emb)
        x6 = self.block6(x5, spatial_t_emb)
        
        x7 = self.block7(self.spatial_attention(x6))
        
        output = torch.tanh(self.block8(x7+x1))

        # # 6个
        # x1 = self.block1(x)
        
        # x2 = self.block2(x1, spatial_t_emb)
        # x3 = self.block3(x2, spatial_t_emb)
        # x4 = self.block4(x3, spatial_t_emb)
        # x5 = self.block5(x4, spatial_t_emb)
        # x6 = self.block6(x5, spatial_t_emb)
        # x7 = self.block6(x6, spatial_t_emb)
        
        # x8 = self.block8(self.spatial_attention(x7))
        
        # output = torch.tanh(self.block9(x8+x1))
        
        return output


class TSRN_C2F(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, STN=False, srb_nums=5, mask=True, hidden_units=32):
        super(TSRN_C2F, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2 * hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2 * hidden_units)
                ))

        self.coarse_proj = nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4)

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units + in_planes, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units + in_planes, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):
        # embed()
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        proj_coarse = self.coarse_proj(block[str(self.srb_nums + 2)])
        # block[str(srb_nums + 2)] =

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            (torch.cat([block['1'] + block[str(self.srb_nums + 2)], proj_coarse], axis=1))
        output = torch.tanh(block[str(self.srb_nums + 3)])

        # print("block_keys:", block.keys())

        return output, proj_coarse


class SEM_TSRN(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 word_vec_d=300):
        super(SEM_TSRN, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), ReasoningResidualBlock(2 * hidden_units))

        self.w2v_proj = ImFeat2WordVec(2 * hidden_units, word_vec_d)
        # self.semantic_R = ReasoningTransformer(2 * hidden_units)

        self.feature_enhancer = None #FeatureEnhancerW2V(
                                #        vec_d=300,
                                #        feature_size=2 * hidden_units,
                                #        head_num=4,
                                #        dropout=True)

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2 * hidden_units)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x, word_vecs=None):
        # embed()
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        all_pred_vecs = []

        # Reasoning block: [2, 3, 4, 5, 6]
        for i in range(self.srb_nums + 1):
            if i + 2 in [2, 3, 4, 5, 6]:
                pred_word_vecs = self.w2v_proj(block[str(i + 1)])
                all_pred_vecs.append(pred_word_vecs)
                if not self.training:
                    word_vecs = pred_word_vecs
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], self.feature_enhancer, word_vecs)
            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        output = torch.tanh(block[str(self.srb_nums + 3)])

        return output, all_pred_vecs


class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual)


class RecurrentResidualBlockTL(nn.Module):
    def __init__(self, channels, text_channels):
        super(RecurrentResidualBlockTL, self).__init__()
        
        # self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels)
        # self.gru1 = GruBlock(channels + text_channels, channels)
        # self.prelu = mish()
        # self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)
        # self.gru2 = GruBlock(channels, channels)
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = mish()

        self.conv2_w = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2_w = nn.BatchNorm2d(channels)
        self.conv2_h = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2_h = nn.BatchNorm2d(channels)
        
        self.gru1 = GruBlock(channels + text_channels, channels)
        self.gru2 = GruBlock(channels + text_channels, channels)
        
        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.prelu3 = mish()
        
        self.fs = eca_layer(channel=channels*2)

    def forward(self, x, text_emb):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        
        residual_w = self.conv2_w(residual)
        residual_w = self.bn2_w(residual_w)
        w_feat = torch.cat([residual_w, text_emb], 1)
        w_feat = self.gru1(w_feat)
        w_feat = residual + w_feat

        
        residual_h = self.conv2_h(residual)
        residual_h = self.bn2_h(residual_h)
        h_feat = torch.cat([residual_h, text_emb], 1)
        h_feat = self.gru2(h_feat.transpose(-1, -2)).transpose(-1, -2)
        h_feat = residual + h_feat
              
        
        fusion_feat = self.fs(torch.cat([h_feat, w_feat], dim=1))
        
        # residual = self.conv1(x)
        # residual = self.bn1(residual)
        # residual = self.conv2(residual)
        # residual = self.bn2(residual)

        # ############ Fusing with TL ############
        # cat_feature = torch.cat([residual, text_emb], 1)
        # ########################################

        # residual = self.gru1(cat_feature.transpose(-1, -2)).transpose(-1, -2)

        # return self.gru2(x + residual)
        
        return self.prelu3(self.bn3(self.conv3(fusion_feat)))



class ReasoningResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ReasoningResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        # self.gru2 = GruBlock(channels, channels)
        self.feature_enhancer = FeatureEnhancerW2V(
                                        vec_d=300,
                                        feature_size=channels,
                                        head_num=4,
                                        dropout=0.1)

    def forward(self, x, feature_enhancer, wordvec):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        size = residual.shape
        # residual: [N, C, H, W];
        # wordvec: [N, C_vec]
        residual = residual.view(size[0], size[1], -1)
        residual = self.feature_enhancer(residual, wordvec)
        residual = residual.resize(size[0], size[1], size[2], size[3])

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


class ImFeat2WordVec(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImFeat2WordVec, self).__init__()
        self.vec_d = out_channels
        self.vec_proj = nn.Linear(in_channels, self.vec_d)

    def forward(self, x):

        b, c, h, w = x.size()
        result = x.view(b, c, h * w)
        result = torch.mean(result, 2)
        pred_vec = self.vec_proj(result)

        return pred_vec


if __name__ == '__main__':
    # net = NonLocalBlock2D(in_channels=32)
    img = torch.zeros(7, 3, 16, 64)
    embed()
