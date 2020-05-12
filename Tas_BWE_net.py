import torch as th
import torch.nn as nn

import torch.nn.functional as F
import os
import math, copy

EPS = 1e-8

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN",
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False):
        super(TemporalBlock, self).__init__()
        conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, bias=False)
        prelu = nn.PReLU()
        norm = build_norm(norm_type, out_channels)
        conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, bias=False)
        # Put together
        self.net = nn.Sequential(conv1, prelu, norm, conv2, prelu, norm)

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = self.residual_conv(x)
        out = self.net(x)
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        return out + residual  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)

class TCN(nn.Module):
    def __init__(self,
                 num_layers=4,
                 filter_size=9,
                 merge_filter_size=5,
                 nefilters=24,
                 filter=True,
                 norm_type="gLN",
                 causal=False):
        super(TCN, self).__init__()

        self.norm_type = norm_type
        self.causal = causal
        self.num_layers = num_layers
        self.filter_size = filter_size
        self.merge_filter_size = merge_filter_size
        self.nefilters = nefilters

        self.filter = filter

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if self.filter:
            print('change filter numbers and filter_size')

            echannelin = [1, 128, 256, 512]
            echannelout = [128, 256, 512, 512]
            dchannelout = echannelout[::-1]
            dchannelin = [1024, 1024, 768, 384]

            self.out = nn.Sequential(
                nn.Conv1d(128 + 1, 1, 1),
                nn.Tanh()
            )

            self.filter_size_en = [9, 9, 9, 9]
            self.filter_size_de = [9, 9, 9, 9]

        else:
            print('no change filter numbers and filter_size')
            self.inp = nn.Conv1d(1, 24, self.filter_size, padding=self.filter_size // 2)

            echannelin = [24] + [(i + 1) * self.nefilters for i in range(self.num_layers - 1)]
            echannelout = [(i + 1) * self.nefilters for i in range(self.num_layers)]
            dchannelout = echannelout[::-1]
            dchannelin = [dchannelout[0] * 2] + [(i) * self.nefilters + (i - 1) * self.nefilters for i in
                                                 range(self.num_layers, 1, -1)]

            self.out = nn.Sequential(
                nn.Conv1d(self.nefilters + 1, 1, 1),
                nn.Tanh()
            )

            self.filter_size_en = [filter_size for i in range(self.num_layers)]
            self.filter_size_de = [merge_filter_size for i in range(self.num_layers)]

        for i in range(self.num_layers):
            dilation_e = 2 ** i
            padding_e = (self.filter_size_en[i] - 1) * dilation_e // 2

            self.encoder.append(TemporalBlock(echannelin[i], echannelout[i], self.filter_size_en[i], stride=1,
                                              padding=padding_e,
                                              dilation=dilation_e,
                                              norm_type=norm_type,
                                              causal=causal))

            dilation_d = 2 ** (self.num_layers-1-i)
            padding_d = (self.filter_size_de[i] - 1) * dilation_d // 2

            self.decoder.append(TemporalBlock(dchannelin[i], dchannelout[i], self.filter_size_de[i], stride=1,
                                              padding=padding_d,
                                              dilation=dilation_d,
                                              norm_type=norm_type,
                                              causal=causal))


        self.middle = nn.Sequential(
            nn.Conv1d(echannelout[-1], echannelout[-1], self.filter_size_en[-1], padding=self.filter_size_en[-1] // 2),
            nn.BatchNorm1d(echannelout[-1]),
            nn.LeakyReLU(0.1))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, input_nb):
        encoder = list()
        # input_nb = input_nb.permute(0, 2, 1)
        x = input_nb
        if self.filter:
            x =x
        else:
            x = self.inp(x)

        for i in range(self.num_layers):
            x = self.encoder[i](x)
            encoder.append(x)
            x = x[:, :, ::2]

        x = self.middle(x)

        for i in range(self.num_layers):
            x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
            # M, N, T = encoder[self.num_layers - i - 1].size()
            encoder_tmp = encoder[self.num_layers - i - 1]
            stop = min(x.size(2), encoder_tmp.size(2))
            x = th.cat([x[:, :, :stop], encoder_tmp[:, :, :stop]], dim=1)
            x = self.decoder[i](x)

        stop = min(x.size(2), input_nb.size(2))
        op = th.cat([x[:, :, :stop], input_nb[:, :, :stop]], dim=1)

        op = self.out(op)

        return op


class TasBweNet(nn.Module):
    def __init__(self,
                 L=20,
                 N=256,
                 X=8,
                 R=4,
                 B=256,
                 H=512,
                 P=3,
                 norm="cLN",
                 num_spks=1,
                 non_linear="relu",
                 causal=False):
        super(TasBweNet, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "softmax": F.softmax
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.encoder_1d = Conv1D(1, N, L, stride=L // 2, padding=0)
        # keep T not change
        # T = int((xlen - L) / (L // 2)) + 1
        # before repeat blocks, always cLN
        self.ln = ChannelWiseLayerNorm(N)
        # n x N x T => n x B x T
        self.proj = Conv1D(N, B, 1)
        # repeat blocks
        # n x B x T => n x B x T
        self.repeats = self._build_repeats(
            R,
            X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)
        # output 1x1 conv
        # n x B x T => n x N x T
        # NOTE: using ModuleList not python list
        # self.conv1x1_2 = th.nn.ModuleList(
        #     [Conv1D(B, N, 1) for _ in range(num_spks)])
        # n x B x T => n x 2N x T
        self.mask = Conv1D(B, num_spks * N, 1)
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d = ConvTrans1D(
            N, 1, kernel_size=L, stride=L // 2, bias=True)
        self.num_spks = num_spks

        self.tcn = TCN(4, 9, 5, 24, filter=True)

    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2**b))
            for b in range(num_blocks)
        ]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        """
        Build Conv1D block repeats
        """
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)
        # n x 1 x S => n x N x T
        w = F.relu(self.encoder_1d(x))
        # n x B x T
        y = self.proj(self.ln(w))
        # n x B x T
        y = self.repeats(y)
        # n x 2N x T
        e = th.chunk(self.mask(y), self.num_spks, 1)
        # n x N x T
        if self.non_linear_type == "softmax":
            m = self.non_linear(th.stack(e, dim=0), dim=0)
        else:
            m = self.non_linear(th.stack(e, dim=0))
        # spks x [n x N x T]
        s = [w * m[n] for n in range(self.num_spks)]
        # spks x n x S
        out_enhan = [self.decoder_1d(x, squeeze=True) for x in s]
        out_enhan = out_enhan[0]

        #----------------------  BWE part----------------------------#
        out_final = th.squeeze(self.tcn(th.unsqueeze(out_enhan, 1)))

        return [out_final]


def foo_conv1d_block():
    nnet = Conv1DBlock(256, 512, 3, 20)
    print(param(nnet))


def foo_layernorm():
    C, T = 256, 20
    nnet1 = nn.LayerNorm([C, T], elementwise_affine=True)
    print(param(nnet1, Mb=False))
    nnet2 = nn.LayerNorm([C, T], elementwise_affine=False)
    print(param(nnet2, Mb=False))


def foo_TasBweNet():
    x = th.rand(4, 1000)
    nnet = TasBweNet(norm="cLN", causal=False)
    # print(nnet)
    print("ConvTasNet #param: {:.2f}".format(param(nnet)))
    x = nnet(x)
    s1 = x
    # s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    foo_TasBweNet()
    # foo_conv1d_block()
    # foo_layernorm()
