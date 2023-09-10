# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn
import torch.nn.functional as F

basic_dims = 8
num_modals = 1
patch_size = [2, 8, 8]


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    # elif norm == 'sync_bn':
    #     m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True,
                 act_type='lrelu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride,
                              padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class fusion_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(fusion_prenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d_prenorm(in_channel * num_modals, in_channel, k_size=1, padding=0, stride=1),
            general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
            general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=2, out_channels=basic_dims, kernel_size=3, stride=1, padding=1,
                                   padding_mode='zeros', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='zeros')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='zeros')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims * 2, stride=2, pad_type='zeros')
        self.e2_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='zeros')
        self.e2_c3 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='zeros')

        self.e3_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type='zeros')
        self.e3_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='zeros')
        self.e3_c3 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='zeros')

        self.e4_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type='zeros')
        self.e4_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='zeros')
        self.e4_c3 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='zeros')

        self.e5_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type='zeros')
        self.e5_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='zeros')
        self.e5_c3 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='zeros')

    def forward(self, x0,x1):
        x1 = self.e1_c1(torch.cat([x0,x1],1))
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5


class Decoder(nn.Module):
    def __init__(self, num_cls=1):
        super(Decoder, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='zeros')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='zeros')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='zeros')

        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='zeros')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='zeros')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='zeros')

        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='zeros')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='zeros')
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='zeros')

        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='zeros')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='zeros')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='zeros')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims * 16, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims * 8, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims * 4, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims * 2, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims * 16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims * 8, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims * 4, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims * 2, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims * 1, num_cls=num_cls)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        de_x5 = self.d4_c1(self.up2(de_x5))
        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = torch.sigmoid(logits)

        return pred


class RUNet_recon(nn.Module):
    def __init__(self,):
        super(RUNet_recon, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_cls=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x0,x1):

        z1, z2, z3, z4, z5 = self.encoder(x0,x1)
        recon = self.decoder(z1, z2, z3, z4, z5)

        return recon


class RUNet_seg(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(RUNet_seg, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        act1=nn.Sigmoid()

        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.res_1 = conv_block_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()

        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.res_2 = conv_block_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()

        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.res_3 = conv_block_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()

        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.res_4 = conv_block_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.res_bridge = conv_block_3d(self.num_filters * 8, self.num_filters * 16, activation)
        # Up sampling

        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.res_up2 = conv_block_3d(self.num_filters * 24, self.num_filters * 8, activation)

        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.res_up3 = conv_block_3d(self.num_filters * 12, self.num_filters * 4, activation)

        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.res_up4 = conv_block_3d(self.num_filters * 6, self.num_filters * 2, activation)

        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, act1)
    
    def forward(self, x0,x1):
        # Down sampling
        x = torch.cat([x0,x1-x0],1)
        down_1 = self.down_1(x) # -> [1, 4, 128, 128, 128]
        res_1 = self.res_1(x)
        pool_1 = self.pool_1(down_1+res_1) # -> [1, 4, 64, 64, 64]
        
        down_2 = self.down_2(pool_1) # -> [1, 8, 64, 64, 64]
        res_2 = self.res_2(pool_1)
        pool_2 = self.pool_2(down_2+res_2) # -> [1, 8, 32, 32, 32]
        
        down_3 = self.down_3(pool_2) # -> [1, 16, 32, 32, 32]
        res_3 = self.res_3(pool_2)
        pool_3 = self.pool_3(down_3+res_3) # -> [1, 16, 16, 16, 16]
        
        down_4 = self.down_4(pool_3) # -> [1, 32, 16, 16, 16]
        res_4 = self.res_4(pool_3)
        pool_4 = self.pool_4(down_4+res_4) # -> [1, 32, 8, 8, 8]

        
        # Bridge
        bridge = self.bridge(pool_4) # -> [1, 128, 4, 4, 4]
        res_bridge = self.res_bridge(pool_4)
        
        trans_2 = self.trans_2(bridge+res_bridge) # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4+res_4], dim=1) # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]
        res_up_2 = self.res_up2(concat_2)
        
        trans_3 = self.trans_3(up_2+res_up_2) # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3+res_3], dim=1) # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]
        res_up_3 = self.res_up3(concat_3)
        
        trans_4 = self.trans_4(up_3+res_up_3) # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2+res_2], dim=1) # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4) # -> [1, 8, 64, 64, 64]
        res_up_4 = self.res_up4(concat_4)
        
        trans_5 = self.trans_5(up_4+res_up_4) # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1+res_1], dim=1) # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5) # -> [1, 4, 128, 128, 128]
        
        # Output
        out = self.out(up_5) # -> [1, 3, 128, 128, 128]
        return out


if __name__ == '__main__':

    a = RUNet_recon()
    z1 = torch.zeros([2,1,128,128,128])
    z2 = torch.zeros([2,1,128,128,128])

    s = a(z1,z2)
    print(s.shape)