import sys
sys.path.append("..")
import torch
import torch.nn as nn
# from . import simplenet as simplenet
# from . import vit as vit
import model.simplenet as simplenet
import model.swin_transformer as swin_transformer
import model.vit as vit

class HybridNet(nn.Module):
    def __init__(self,cnn_net,trans_net,img_size=(128,128),num_classes=2,input_channels=1,pretrained=False,out_index=4):
        super(HybridNet,self).__init__()
        self.out_index = out_index
        self.cnn_backbone = simplenet.__dict__[cnn_net](
            depth = out_index if not pretrained else 4,
            pretrained=pretrained,
            input_channels=input_channels
        )
        self.cnn_out_feature = self.cnn_backbone.out_feature if not pretrained else self.cnn_backbone.out_feature / (2**(4 - out_index))
        self.trans_input_size = (int(img_size[0]/2**out_index),int(img_size[1]/2**out_index))
        
        if trans_net.startswith('swin_transformer'):
            self.trans_net = swin_transformer.__dict__[trans_net](
                in_chans=self.cnn_out_feature,
                img_size=self.trans_input_size,
                patch_size=(4,4),
                num_classes=num_classes,
                depths=[2, 6]
            )
        elif trans_net.startswith('vit'):
            self.patch_size = (int(self.trans_input_size[0]/8),int(self.trans_input_size[1]/8))
            if min(self.patch_size) < 4:
                raise ValueError('path size must be larger than 4!')
            self.trans_net = swin_transformer.__dict__[trans_net](
                in_channels=self.cnn_out_feature,
                img_size=self.trans_input_size,
                patch_size=self.patch_size,
                num_classes=num_classes,
                spatial_dims=2
            )


    def forward(self,x):
        fea = self.cnn_backbone(x)
        x = fea[self.out_index]
        # print(x.size())
        x = self.trans_net(x)
        return x


def hybridnet_v1(**kwargs):
    net = HybridNet(cnn_net='simplenet18',
                    trans_net='swin_transformer',
                    out_index=2,
                    **kwargs)
    return net


def hybridnet_v2(**kwargs):
    net = HybridNet(cnn_net='simplenet18',
                    trans_net='vit_12x12',
                    out_index=2,
                    **kwargs)
    return net


if __name__ == "__main__":
  
  net = hybridnet_v1(img_size=(224,224),
                    input_channels=3,
                    num_classes=5
                    )

  from torchsummary import summary
  import os 
  os.environ['CUDA_VISIBLE_DEVICES'] = '2'
  net = net.cuda()
  summary(net,input_size=(3,224,224),batch_size=1,device='cuda')
