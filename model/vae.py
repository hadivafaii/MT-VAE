import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import *
from .model_utils import print_num_params


class VAE(nn.Module):
    def __init__(self, config, verbose=False):
        super(VAE, self).__init__()

        self.beta = 1.0
        self.config = config

        self.encoder = Encoder(config, verbose=verbose)
        self.decoder = Decoder(config, verbose=verbose)

        self.init_weights()
        self.apply(add_sn)
        # TODO: this will be the thing new new verison:
        # if config.conv_norm == 'spectral':
        #     self.apply(add_sn)
        # elif config.conv_norm == 'weight':
        #     self.apply(add_wn)

        if verbose:
            print_num_params(self)

    def forward(self, src, tgt):
        (x1, x2, x3, z1), (mu_x, logvar_x) = self.encoder(src)
        (y1, y2, y3, z2), (mu_z, logvar_z), (mu_xz, logvar_xz) = self.decoder(z1, x2)

        kl_x, kl_xz, recon_loss, loss = self._compute_loss(
            y3, tgt, mu_x, mu_xz, logvar_z, logvar_x, logvar_xz)

        return y3, (kl_x, kl_xz, recon_loss, loss)

    def update_beta(self, new_beta):
        assert 0.0 <= new_beta <= 1.0, "beta must be in [0, 1] interval"
        self.beta = new_beta

    # TODO: replace all logvar ---> 0.5 logvar
    # TODO: because: logvar = log \Sigma^2
    def _compute_loss(self, recon, tgt, mu_x, mu_xz, logvar_z, logvar_x, logvar_xz):
        kl_x = 0.5 * torch.sum(
            torch.pow(mu_x, 2) + torch.exp(logvar_x) - logvar_x - 1
        )
        kl_xz = 0.5 * torch.sum(
            torch.pow(mu_xz, 2) * torch.exp(-logvar_z) +
            torch.exp(logvar_xz) - logvar_xz - 1
        )

        kl_loss = self.beta * (kl_x + kl_xz)
        recon_loss = compute_endpoint_error(recon, tgt)
        loss = kl_loss + recon_loss

        return kl_x, kl_xz, recon_loss, loss

    def init_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """ Initialize the weights """
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        else:
            pass


class Decoder(nn.Module):
    def __init__(self, config, verbose=False):
        super(Decoder, self).__init__()

        self.z_dim = config.z_dim
        self.inplanes = config.nb_rot_kernels * config.nb_rotations * 2 ** config.nb_lvls

        self.init_size = tuple(config.decoder_init_grid_size)
        self.expand1 = nn.Sequential(
            nn.ConvTranspose3d(config.z_dim, self.inplanes, self.init_size, bias=False),
            nn.BatchNorm3d(self.inplanes), Swish(),)
        self.layer1 = self._make_layer(self.inplanes // 2, blocks=1, stride=2)
        self.proj1 = nn.Sequential(
            nn.ConvTranspose3d(self.inplanes, self.inplanes, 2, bias=False),
            nn.BatchNorm3d(self.inplanes), Swish(),)

        self.intermediate_size = tuple(item * 2 for item in config.decoder_init_grid_size)
        self.swish = Swish()
        self.expand2 = nn.Sequential(
            nn.ConvTranspose3d(config.z_dim, self.inplanes, self.intermediate_size, bias=False),
            nn.BatchNorm3d(self.inplanes),)
        self.layer2 = self._make_layer(self.inplanes // 2, blocks=1, stride=2)
        self.proj2 = deconv1x1x1(self.inplanes, 2, bias=True)

        self.condition_z = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), conv1x1x1(self.inplanes * 2, config.z_dim * 2, bias=True),)
        self.condition_xz = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), conv1x1x1(self.inplanes * 4, config.z_dim * 2, bias=True),)

        if verbose:
            print_num_params(self)

    def forward(self, z1, x2):
        y1 = z1.view(-1, self.z_dim, 1, 1, 1)
        y1 = self.expand1(y1)   # N x 256 x 4 x 4 x 3
        y1 = self.layer1(y1)    # N x 128 x 8 x 8 x 6
        y2 = self.proj1(y1)

        # side path
        mu_z, logvar_z = self.condition_z(y2).squeeze().chunk(2, dim=-1)
        xy = torch.cat([y2, x2], dim=1)
        mu_xz, logvar_xz = self.condition_xz(xy).squeeze().chunk(2, dim=-1)
        z2 = reparametrize(mu_z + mu_xz, logvar_z + logvar_xz)
        res = z2.view(-1, self.z_dim, 1, 1, 1)
        res = self.expand2(res)

        # second layer
        y2 = self.swish(y2 + res)
        y2 = self.layer2(y2)
        y3 = self.proj2(y2)

        return (y1, y2, y3, z2), (mu_z, logvar_z), (mu_xz, logvar_xz)

    def generate(self, device: torch.device, num_samples: int = 40, z1=None, z2=None):
        if z1 is None:
            z1 = torch.randn((num_samples, self.z_dim))
        z1 = z1.to(device)

        y1 = z1.view(-1, self.z_dim, 1, 1, 1)
        y1 = self.expand1(y1)
        y1 = self.layer1(y1)
        y2 = self.proj1(y1)

        if z2 is None:
            mu_z, logvar_z = self.condition_z(y2).squeeze().chunk(2, dim=-1)
            z2 = reparametrize(mu_z, logvar_z)
        z2 = z2.to(device)

        res = z2.view(-1, self.z_dim, 1, 1, 1)
        res = self.expand2(res)

        y2 = self.swish(y2 + res)
        y2 = self.layer2(y2)
        y3 = self.proj2(y2)

        return y3, (z1, z2)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                deconv1x1x1(self.inplanes, planes, stride),
                nn.BatchNorm3d(planes),
            )

        layers = [DeConvBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(DeConvBlock(self.inplanes, planes))

        return nn.Sequential(*layers)


class DeConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DeConvBlock, self).__init__()

        self.deconv1 = deconv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.swish1 = Swish()
        self.deconv2 = deconv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.swish2 = Swish()
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.swish1(out)

        out = self.deconv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.swish2(out)

        return out


class Encoder(nn.Module):
    def __init__(self, config, verbose=False):
        super(Encoder, self).__init__()

        self.rot_layer = RotationalConvBlock(config, verbose=verbose)
        self.inplanes = config.nb_rot_kernels * config.nb_rotations
        self.layer1 = self._make_layer(self.inplanes * 2, blocks=2, stride=2)
        self.layer2 = self._make_layer(self.inplanes * 2, blocks=2, stride=2)

        self.condition_x = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), conv1x1x1(self.inplanes, config.z_dim * 2, bias=True),)

        if verbose:
            print_num_params(self)

    def forward(self, x):
        x1 = self.rot_layer(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)    # N X (8 , 2)

        mu_x, logvar_x = self.condition_x(x3).squeeze().chunk(2, dim=-1)
        z1 = reparametrize(mu_x, logvar_x)

        return (x1, x2, x3, z1), (mu_x, logvar_x)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes, stride),
                nn.BatchNorm3d(planes),
            )

        layers = [ConvBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ConvBlock(self.inplanes, planes))

        return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ConvBlock, self).__init__()

        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.swish1 = Swish()
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SELayer(planes, reduction=16)
        self.downsample = downsample
        self.swish2 = Swish()
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.swish1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.swish2(out)

        return out


class RotationalConvBlock(nn.Module):
    def __init__(self, config, verbose=False):
        super(RotationalConvBlock, self).__init__()

        nb_units = config.nb_rot_kernels * config.nb_rotations
        padding = tuple(k - 1 for k in config.rot_kernel_size)
        self.chomp3d = Chomp(chomp_sizes=padding, nb_dims=3)
        self.conv1 = RotConv3d(
            in_channels=2,
            out_channels=config.nb_rot_kernels,
            nb_rotations=config.nb_rotations,
            kernel_size=config.rot_kernel_size,
            padding=padding,
            bias=False,)
        self.bn1 = nn.BatchNorm3d(nb_units)
        self.swish1 = Swish()
        self.conv2 = conv3x3x3(nb_units, nb_units)
        self.bn2 = nn.BatchNorm3d(nb_units)
        self.se = SELayer(nb_units, reduction=8)
        self.swish2 = Swish()

        if verbose:
            print_num_params(self)

    def forward(self, x):
        # x : N x 2 x grd x grd x tau
        x = self.chomp3d(self.conv1(x))  # N x nb_rot_kers*nb_rot x grd x grd x tau
        x = self.bn1(x)
        x = self.swish1(x)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.se(out)
        out = self.swish2(out + x)

        return out


class RotConv3d(nn.Conv3d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, ...]],
            nb_rotations: int = 8,
            stride: Union[int, Tuple[int, ...]] = 1,
            padding: Union[int, Tuple[int, ...]] = 0,
            dilation: Union[int, Tuple[int, ...]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
    ):
        super(RotConv3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)

        self.nb_rotations = nb_rotations
        rotation_mat = self._build_rotation_mat()
        self.register_buffer('rotation_mat', rotation_mat)

        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(out_channels * nb_rotations))

    def forward(self, x):
        # note: won't work when self.padding_mode != 'zeros'
        return F.conv3d(x, self._get_augmented_weight(), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

    def _build_rotation_mat(self):
        thetas = np.deg2rad(np.arange(0, 360, 360 / self.nb_rotations))
        c, s = np.cos(thetas), np.sin(thetas)
        rotation_mat = torch.tensor([[c, -s], [s, c]], dtype=torch.float).permute(2, 0, 1)
        return rotation_mat

    def _get_augmented_weight(self):
        w = torch.einsum('jkn, inlmo -> ijklmo', self.rotation_mat, self.weight)
        return w.flatten(end_dim=1)
