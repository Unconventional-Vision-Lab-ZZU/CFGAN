import torch
import torch.nn as nn
# from torch.nn import init
import functools
# from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

###############################################################################
# Functions
###############################################################################


res = []


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], use_parallel=True,
             learn_residual=False):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                               gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[],
             use_parallel=True):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2

        # 下采样
        # for i in range(n_downsampling): # [0,1]
        # 	mult = 2**i
        #
        # 	model += [
        # 		nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
        # 		norm_layer(ngf * mult * 2),
        # 		nn.ReLU(True)
        # 	]

        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        ]

        # 中间的残差网络
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            ]

        # 上采样
        # for i in range(n_downsampling):
        # 	mult = 2**(n_downsampling - i)
        #
        # 	model += [
        # 		nn.ConvTranspose2d(
        # 			ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
        # 			padding=1, output_padding=1, bias=use_bias),
        # 		norm_layer(int(ngf * mult / 2)),
        # 		nn.ReLU(True)
        # 	]
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
        ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(input + output, min=-1, max=1)
            print(len(res))
        return output


# Define a resnet block
class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()

        padAndConv = {
            'reflect': [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'replicate': [
                nn.ReplicationPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'zero': [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
        }

        try:
            blocks = padAndConv[padding_type] + [
                norm_layer(dim),
                nn.ReLU(True)
            ] + [
                         nn.Dropout(0.5)
                     ] if use_dropout else [] + padAndConv[padding_type] + [
                norm_layer(dim)
            ]
        except:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv_block = nn.Sequential(*blocks)

    # self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    # def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
    #     padAndConv = {
    #         'reflect': [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
    #         'replicate': [nn.ReplicationPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
    #         'zero': [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
    #     }
    #     try:
    #         blocks = [
    #             padAndConv[padding_type],
    #
    #             norm_layer(dim),
    #             nn.ReLU(True),
    #             nn.Dropout(0.5) if use_dropout else None,
    #
    #             padAndConv[padding_type],
    #
    #             norm_layer(dim)
    #         ]
    #     except:
    #         raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    #
    #     return nn.Sequential(*blocks)

    # blocks = []
    # if padding_type == 'reflect':
    # 	blocks += [nn.ReflectionPad2d(1),  nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
    # elif padding_type == 'replicate':
    # 	blocks += [nn.ReplicationPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
    # elif padding_type == 'zero':
    # 	blocks += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
    # else:
    # 	raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    #
    # blocks += [
    # 	norm_layer(dim),
    # 	nn.ReLU(True),
    # 	nn.Dropout(0.5) if use_dropout else None
    # ]
    #
    # if padding_type == 'reflect':
    # 	blocks += [nn.ReflectionPad2d(1),  nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
    # elif padding_type == 'replicate':
    # 	blocks += [nn.ReplicationPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
    # elif padding_type == 'zero':
    # 	blocks += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
    # else:
    # 	raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    #
    # blocks += [
    # 	norm_layer(dim)
    # ]
    #
    # return nn.Sequential(*blocks)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class sa(nn.Module):
    def __init__(self, channel):
        super(sa, self).__init__()
        self.gam = nn.Parameter(torch.zeros(1))
        self.key = nn.Conv2d(channel, channel, kernel_size=1, stride=1)
        self.query = nn.Conv2d(channel, channel, kernel_size=1, stride=1)
        self.value = nn.Conv2d(channel, channel, kernel_size=1, stride=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        b, c, h, w, = x1.size()
        # 送入MLP全连接神经网络, 得到权重
        k = self.key(x2).view(b, c, h * w)
        q = self.query(x2).view(b, c, h * w)
        v = self.value(x1).view(b, c, h * w)
        q = q.permute(0, 2, 1)
        atten = self.softmax(torch.bmm(q, k))
        out = torch.bmm(v, atten.permute(0, 2, 1)).view(b, c, h, w)

        return out + x1


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
            use_dropout=False, gpu_ids=[], use_parallel=True, learn_residual=False):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block
        self.conv1_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1024, 256, 1),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1024, 256, 1),
            nn.ReLU(inplace=True)
        )
        self.conv1_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1024, 256, 1),
            nn.ReLU(inplace=True)
        )
        self.conv1_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True)
        )
        self.conv1_5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True)
        )
        self.conv1_6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 256, 1),
            nn.ReLU(inplace=True)
        )
        self.conv1_7 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 256, 1),
            nn.ReLU(inplace=True)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(256 * 7, 1024, 1),
            nn.Sigmoid()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(256 * 7, 1024, 1),
            nn.Sigmoid()
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(256 * 7, 1024, 1),
            nn.Sigmoid()
        )
        self.conv2_4 = nn.Sequential(
            nn.Conv2d(256 * 7, 512, 1),
            nn.Sigmoid()
        )
        self.conv2_5 = nn.Sequential(
            nn.Conv2d(256 * 7, 256, 1),
            nn.Sigmoid()
        )
        self.conv2_6 = nn.Sequential(
            nn.Conv2d(256 * 7, 128, 1),
            nn.Sigmoid()
        )
        self.conv2_7 = nn.Sequential(
            nn.Conv2d(256 * 7, 3, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(3971, 3, 1),
            nn.ReLU(inplace=True)
        )

    def _upsample_add(self, x, y):  # 自上而下的采样模块 P5 ==> P4 ==> P3 ==> P2
        _, _, H, W = y.shape  # 获取 H W
        return F.upsample(x, size=(H, W), mode='bilinear')  # 双线性插值采样后 相加

    def forward(self, input):

        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            # print(len(res))
            output = self.model(input)
            # print(len(res))
            # for i in range (len(res)):
            #     print(res[i].shape)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        E1 = res[0]
        E2 = res[1]
        E3 = res[2]
        E4 = res[3]
        E5 = res[4]
        E6 = res[5]
        E7 = output
        x1_1 = self.conv1_1(E1)
        x1_2 = self.conv1_2(E2)
        x1_3 = self.conv1_3(E3)
        x1_4 = self.conv1_4(E4)
        x1_5 = self.conv1_5(E5)
        x1_6 = self.conv1_6(E6)
        x1_7 = self.conv1_7(E7)

        # x1_7 = self.conv1_7(x7)
        xx = torch.cat((x1_1, x1_2, x1_3, x1_4, x1_5, x1_6, x1_7), dim=1)

        x2_1 = self.conv2_1(xx)
        x2_2 = self.conv2_2(xx)
        x2_3 = self.conv2_3(xx)
        x2_4 = self.conv2_4(xx)
        x2_5 = self.conv2_5(xx)
        x2_6 = self.conv2_6(xx)
        x2_7 = self.conv2_7(xx)

        E1_new = E1 * x2_1
        E2_new = E2 * x2_2
        E3_new = E3 * x2_3
        E4_new = E4 * x2_4
        E5_new = E5 * x2_5
        E6_new = E6 * x2_6
        E7_new = E7 * x2_7

        E1 = self._upsample_add(E1_new, E7_new)
        E2 = self._upsample_add(E2_new, E7_new)
        E3 = self._upsample_add(E3_new, E7_new)
        E4 = self._upsample_add(E4_new, E7_new)
        E5 = self._upsample_add(E5_new, E7_new)
        E6 = self._upsample_add(E6_new, E7_new)
        E7 = self._upsample_add(E7_new, E7_new)

        output = torch.cat((E1, E2, E3, E4, E5, E6, E7), dim=1)
        output=self.conv(output)
        res.clear()
        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(
            self, outer_nc, inner_nc, submodule=None,
            outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.sa = sa(outer_nc)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        dConv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        dRelu = nn.LeakyReLU(0.2, True)
        dNorm = norm_layer(inner_nc)
        uRelu = nn.ReLU(True)
        uNorm = norm_layer(outer_nc)

        if outermost:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            dModel = [dConv]
            uModel = [uRelu, uConv, nn.Tanh()]
            try:
                model = dModel + submodule + uModel
            except:
                model = dModel
                model.append(submodule)
                model += uModel
        elif innermost:
            uConv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv]
            uModel = [uRelu, uConv, uNorm]
            model = dModel + uModel

        else:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv, dNorm]
            uModel = [uRelu, uConv, uNorm]

            try:
                model = dModel + submodule + uModel
            except:
                model = dModel
                model.append(submodule)
                model += uModel
            model += [nn.Dropout(0.5)] if use_dropout else []
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            n = self.model(x)
            res.append(n)
            # print(res[-1].shape)
            return n
        else:
            y = self.model(x)
            m = torch.cat([y, self.sa(x, y)], 1)
            res.append(m)
            return m

            # return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
