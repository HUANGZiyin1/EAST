import torch
import torch.nn as nn
import torch.nn.functional as F
import common


class EAST(nn.Module):
    def __init__(self, in_nc=1, nf=48, out_nc=3, base_ks=3,reduction = 16,BasicBlock=common.BasicBlock):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(DCnv5, self).__init__()
        self.radius = 3
        self.input_len = 2 * self.radius + 1

        self.in_conv1 = nn.Sequential(
            nn.Conv3d(in_nc,nf,kernel_size=base_ks, stride = (1,1,1), padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(nf, nf, kernel_size=(7,3,3), stride=(1, 1, 1), padding=(0,1,1)),
            nn.ReLU(inplace=True)
        )

        self.in_conv2 = nn.Sequential(
            nn.Conv3d(in_nc,nf,kernel_size=base_ks, stride = (1,1,1), padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(nf, nf, kernel_size=(4,3,3), stride=(1, 1, 1), padding=(0,1,1)),
            nn.ReLU(inplace=True)
        )

        self.in_conv3 = nn.Sequential(
            nn.Conv3d(in_nc,nf,kernel_size=base_ks, stride = (1,1,1), padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(nf, nf, kernel_size=(4,3,3), stride=(1, 1, 1), padding=(0,1,1)),
            nn.ReLU(inplace=True)
        ) #1


        self.deepFE = nn.Sequential(
            unet(nf*3, nf, 3, base_ks=3)
        )

        self.CA = RCB(nf*2)
#        self.CA2 = CALayer(nf,reduction=16)
        self.edge = REB(nf*3)

        self.GA = RGB(nf)

        self.rec= nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf,1,base_ks,padding = 1)
        )



    def forward(self, inputs):
        TG = inputs[:,:,3, ...]
        input_1 = inputs[:,:,0:4, ...]
        input_2 = inputs[:, :, 3:7, ...]
        o1 = self.in_conv1(inputs) #[32, 1, 48, 128, 128]
        o1 = o1.squeeze(2)#[32, 48, 128, 128]
        o2 = self.in_conv2(input_1)#[32, 1, 48, 128, 128]
        o2 = o2.squeeze(2)#[32, 48, 128, 128]
        o3 = self.in_conv3(input_2)#[32, 1, 48, 128, 128]
        o3 = o3.squeeze(2)#[32, 48, 128, 128]
        o = torch.cat([o2,o3],1)#[32, 64, 128, 128]
        o = self.CA(o)#[32, 64, 128, 128]
        o = torch.cat([o,o1],1)#[32, 96, 128, 128]
        o = self.edge(o,TG)#[32, 96, 128, 128]
        o = self.deepFE(o)
        o = self.GA(o)
        o = self.rec(o)
        o+=TG

        return o

class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCB(nn.Module):
    def __init__(self, channel):
        super(RCB, self).__init__()
        self.conv_bt = nn.Sequential(
            nn.Conv2d(channel, channel//2, 1, padding=1//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//2, channel, 3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.CAlayer = CALayer(channel,16)

    def forward(self, x):
        res = self.conv_bt(x)
        res = self.CAlayer(res)
        res = x+res
        return res

class SALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel // reduction, 3, padding=3//2, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x * y

class RGB(nn.Module):
    def __init__(self, channel):
        super(RGB, self).__init__()
        self.conv_bt = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=3//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.CA = CALayer(channel,16)
        self.SA = SALayer(channel,16)

    def forward(self, x):
        res = self.conv_bt(x)
        res = self.CA(res)
        res = self.SA(res)
        res = x+res
        return res

class unet(nn.Module):
    def __init__(self, in_nc, nf, nb, base_ks=3, ResBlock=common.ResBlock):
        super(unet,self).__init__()
        self.nb = nb
        self.in_nc = in_nc
        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )

        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
    def forward(self, inputs):
        nb = self.nb
        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )
        return out

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x
'''
class Sobel(nn.Module):
    def __init__(self):
        super(Sobel,self).__init__()
        self.sobel_x = nn.Parameter(torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        self.sobel_y = nn.Parameter(torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self,x):

        grad_x = F.conv2d(x,self.sobel_x)
        grad_y = F.conv2d(x,self.sobel_y)
        out_tensor = torch.sqrt(grad_x**2+grad_y**2)
        return out_tensor
'''
class edgeblock(nn.Module):
    def __init__(self,channel, reduction=16):
        super(edgeblock,self).__init__()
        # global average pooling: feature --> point
        self.Edge = Sobel()
        # feature channel downscale and upscale --> channel weight

        self.conv_in = nn.Sequential(
                nn.Conv2d(1, channel, 3, padding=0, bias=True),
                nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x, TG):
        edge = self.Edge(TG)
        edge = nn.functional.interpolate(edge, (x.size()[-2], x.size()[-1]))
        edge = self.conv_in(edge)
        y = self.avg_pool(edge)
        y = self.conv_du(y)
        return x*y

class REB(nn.Module):
    def __init__(self, channel):
        super(REB, self).__init__()
        self.conv_bt = nn.Sequential(
            nn.Conv2d(channel, channel//3, 1, padding=1//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//3, channel, 3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.EA = edgeblock(channel,16)

    def forward(self, x, TG):
        res = self.conv_bt(x)
        res = self.EA(res,TG)
        res = x+res
        return res



if __name__ == '__main__':
    model = DCnv5()
    net_input = torch.randn(32, 1, 7, 128, 128)
    net_output = model(net_input)
    print(net_output.shape)
