import math

import torch

from utils import make_anchors
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)

class MixStructureBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm1 = torch.nn.BatchNorm2d(dim)
        self.norm2 = torch.nn.BatchNorm2d(dim)

        self.conv1 = torch.nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = torch.nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = torch.nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # Simple Channel Attention
        self.Wv = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim, 1),
            torch.nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(dim, dim, 1),
            torch.nn.Sigmoid()
        )

        # Channel Attention
        self.ca = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            torch.nn.GELU(),
            torch.nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            torch.nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            torch.nn.GELU(),
            torch.nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            torch.nn.Sigmoid()
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(dim * 3, dim * 4, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Conv2d(dim * 3, dim * 4, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        if self.dim < 512:
            identity = x
            #x = self.norm1(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
            x = self.mlp(x)
            x = identity + x
      
        identity = x
        #x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x
    
class MixStructureLayer(torch.nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = torch.nn.ModuleList(
            [MixStructureBlock(dim=dim) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))
class SKFusion(torch.nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SKFusion, self).__init__()
		
		self.height = height
		d = max(int(dim/reduction), 4)
		
		self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
		self.mlp = torch.nn.Sequential(
			torch.nn.Conv2d(dim, d, 1, bias=False), 
			torch.nn.ReLU(),
			torch.nn.Conv2d(d, dim*height, 1, bias=False)
		)
		
		self.softmax = torch.nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape
		
		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)
		
		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(self.avg_pool(feats_sum))
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out    


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2),]
        p2 = [Conv(width[1], width[2], 3, 2),
              CSP(width[2], width[2], depth[0]),
              MixStructureLayer(width[2],1)]
        p3 = [Conv(width[2], width[3], 3, 2),
              CSP(width[3], width[3], depth[1]),
              MixStructureLayer(width[3],1)]
        p4 = [Conv(width[3], width[4], 3, 2),
              CSP(width[4], width[4], depth[2]),
              MixStructureLayer(width[4],2)]
        p5 = [Conv(width[4], width[5], 3, 2),
              CSP(width[5], width[5], depth[0]),
              SPP(width[5], width[5]),]
        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return  p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)
        self.m6 = MixStructureLayer(width[5],1)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))

        return h2, h4, h6


class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Decoder(torch.nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Decoder, self).__init__()

        # Upsampling
        self.conv_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(input_nc, input_nc//2, 3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(input_nc//2),
            torch.nn.SiLU(),
            MixStructureLayer(input_nc//2,1)
        )

        self.conv_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(input_nc, (input_nc//2)//2, 3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d((input_nc//2)//2),
            torch.nn.SiLU(),
            MixStructureLayer((input_nc//2)//2,1)
        )

        self.conv_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(input_nc//2, ((input_nc//2)//2)//2, 3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(((input_nc//2)//2)//2),
            torch.nn.SiLU(),
            MixStructureLayer(((input_nc//2)//2)//2,1)
        )

        self.upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv_4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(64, output_nc, 7),
            torch.nn.Tanh()
         )


    def forward(self, x):

        feat_5,feat_4,feat_3 = x

        d1 = self.conv_1(feat_3)
        skip1_de = torch.cat((d1, feat_4), 1)

        d2 = self.conv_2(skip1_de)
        skip2_de = torch.cat((d2, feat_5), 1)

        d3 = self.conv_3(skip2_de)
        d4 = self.upsample(d3)
        output = self.conv_4(d4)
        
        return output

class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))

        self.dfl = DFL(self.ch)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c1, 3),
                                                           Conv(c1, c1, 3),
                                                           torch.nn.Conv2d(c1, self.nc, 1)) for x in filters)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c2, 3),
                                                           Conv(c2, c2, 3),
                                                           torch.nn.Conv2d(c2, 4 * self.ch, 1)) for x in filters)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
        temp =  x
        if self.training:
            return x
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.ch * 4, self.nc), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        return temp,torch.cat((box * self.strides, cls.sigmoid()), 1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)
        self.decoder = Decoder(512,3)
        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)[0]])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x= self.net(x)
        dehaze = self.decoder(x)
   
        x = self.fpn(x)
        return self.head(list(x)),dehaze

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v8_n(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, num_classes)


def yolo_v8_s(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, num_classes)


def yolo_v8_m(num_classes: int = 80):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(width, depth, num_classes)


def yolo_v8_l(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, num_classes)


def yolo_v8_x(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(width, depth, num_classes)