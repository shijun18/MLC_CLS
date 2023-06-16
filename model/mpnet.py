import sys
sys.path.append("..")
import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

import model.simplenet as simplenet
from model.swin_transformer import SwinTransformerBlock
import math

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, outdim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, outdim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8,  dropout=0.):
        super().__init__()
        project_out = not (heads == 1)

        self.heads = heads
        self.scale = (dim//heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots,dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SwinDenseTransformerBlock(nn.Module):
    def __init__(self, out_channels, growth_rate=32,  depth=4, heads=4,  dropout=0.5, input_h=None, input_w=None, Attention=SwinTransformerBlock):
        super().__init__()
        mlp_dim = growth_rate * 2
        cat_dim = out_channels + depth * growth_rate
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Linear(out_channels + i * growth_rate, growth_rate),
                PreNorm(growth_rate, 
                        Attention(dim=growth_rate, 
                                  num_heads=heads, 
                                  drop=dropout, 
                                  input_resolution=(input_h,input_w),
                                  window_size=7,
                                  shift_size=0 if (i % 2 == 0) else 7 // 2)
                ),
            ]))
        self.out_layer = FeedForward(dim=cat_dim, hidden_dim=mlp_dim, outdim=out_channels, dropout=dropout)
            
    def forward(self, x):
        features = [x]
        for liner, attn in self.layers:
            x = torch.cat(features, 2)
            x = liner(x)
            x = attn(x)
            features.append(x)
        x = torch.cat(features, 2)
        x = self.out_layer(x)
        return x


class DenseTransformerBlock(nn.Module):
    def __init__(self, out_channels, growth_rate=32,  depth=4, heads=8,  dropout=0.5, input_h=None, input_w=None, Attention=TransformerBlock):
        super().__init__()
        mlp_dim = growth_rate * 2
        cat_dim = out_channels + depth * growth_rate
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Linear(out_channels + i * growth_rate, growth_rate),
                PreNorm(growth_rate, Attention(dim=growth_rate, heads=heads, dropout=dropout)),
                PreNorm(growth_rate, FeedForward(dim=growth_rate, hidden_dim=mlp_dim, outdim=growth_rate, dropout=dropout))
            ]))
        self.out_layer = FeedForward(dim=cat_dim, hidden_dim=mlp_dim, outdim=out_channels, dropout=dropout)
            
    def forward(self, x):
        features = [x]
        for liner, attn, ff in self.layers:
            x = torch.cat(features, 2)
            x = liner(x)
            x = attn(x) + x
            # x = ff(x) + x
            x = ff(x)
            features.append(x)
        x = torch.cat(features, 2)
        x = self.out_layer(x)
        return x


class MultiPathAttention(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, growth_rate=32, patch_size=16, depth=6, dropout=0.5, attention=DenseTransformerBlock):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.outsize = (image_height // patch_size, image_width// patch_size)
        h = image_height // patch_height
        w = image_width // patch_width
        num_patches = (image_height // patch_height) * (image_width // patch_width)
       
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, out_channels))
        self.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.norm = nn.LayerNorm(out_channels)
        blocks = []
        for _ in range(depth):
            blocks.append(
                attention(out_channels, growth_rate=growth_rate, input_h=h, input_w=w)
            )
        self.blocks = nn.ModuleList(blocks)

        self.reshape = nn.Sequential(
            Rearrange('b (h w) c -> b c h w ', h=h,w=w)
        )
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def trunc_normal_(self, tensor, mean, std, a, b):
        # From PyTorch official master until it's in a few official releases - RW
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor
        
    def forward(self, img):
        x = self.patch_embeddings(img)  # (B, hidden, num_patches^(1/2), num_patches^(1/2))
        # print(x.size())
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, num_patches, hidden)
        x = self.norm(x)
        embeddings = x + self.position_embeddings
        x = self.dropout(embeddings)

        for block in self.blocks:
            x = block(x)
        
        x = self.reshape(x)
        return F.interpolate(x, self.outsize)



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

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class MultiPathNet(nn.Module):
    def __init__(self, in_channels=3, cnn_net='se_simplenet50', num_classes=5, n_filters=32, 
                 image_size=(224,224), transformer_depth=12, patch_size=8, drop_rate=0.,attention=DenseTransformerBlock):
        super(MultiPathNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.transformer_depth = transformer_depth
        self.n_filters = n_filters

        self.cnn_backbone = nn.ModuleList([simplenet.__dict__[cnn_net](depth=2,input_channels=1) for _ in range(self.in_channels)])
        self.cnn_out_feature = self.cnn_backbone[0].out_feature
        # print(self.cnn_out_feature)
        self.multi_attn = nn.ModuleList(
                [MultiPathAttention(in_channels=self.cnn_out_feature,
                                    out_channels=4*self.n_filters,
                                    image_size=image_size//4,
                                    patch_size=patch_size,
                                    depth=self.transformer_depth//4,
                                    attention=attention) for _ in range(self.in_channels)] 
            )

        
        
        cat_dim = 4 * n_filters * self.in_channels
        
        self.cls = nn.Sequential(
            SEBasicBlock(cat_dim,  4*n_filters, stride=1),
            SEBasicBlock(4*n_filters, n_filters, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else None
        self.fc = nn.Linear(n_filters, self.num_classes)


    def forward(self, x):
        attnall = []
        for i in range(self.in_channels):
            conv_out = self.cnn_backbone[i](x[:,i:i+1,:,:])
            att_out = self.multi_attn[i](conv_out[-1])
            attnall.append(att_out)
        attnall= torch.cat(attnall,1)
        x = self.cls(attnall)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)
        
        return x

def mpnet_12x32(in_channels, num_classes, image_size):
    return MultiPathNet(in_channels=in_channels, 
                        num_classes=num_classes, 
                        image_size=image_size, 
                        n_filters=32, 
                        transformer_depth=12,
                        attention=DenseTransformerBlock)



def mpnet_swin_12x32(in_channels, num_classes, image_size):
    return MultiPathNet(in_channels=in_channels, 
                        num_classes=num_classes, 
                        image_size=image_size, 
                        n_filters=32, 
                        transformer_depth=12,
                        attention=SwinDenseTransformerBlock)


if __name__ == "__main__":
  
  net = mpnet_swin_12x32(in_channels=3,num_classes=5,image_size=224)

  from torchsummary import summary
  import os 
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'
  net = net.cuda()
  summary(net,input_size=(3,224,224),batch_size=1,device='cuda')