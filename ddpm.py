import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from os.path import join as pjoin
import sys
from motion_data import MotionNorm
from itertools import cycle
import numpy as np
import math
import time
from einops import rearrange, repeat
from config import Config as config
import random
from Quaternions_old import Quaternions
from scipy import linalg
import torchvision

from remove_fs import save_bvh_from_network_output

offsets = [[0., 0., 0.],
           [0., 0., 0.],
           [1.36306, -1.79463, 0.83929],
           [2.44811, -6.72613, 0.],
           [2.5622, -7.03959, 0.],
           [0.15764, -0.43311, 2.32255],
           [0., 0., 0.],
           [-1.30552, -1.79463, 0.83929],
           [-2.54253, -6.98555, 0.],
           [-2.56826, -7.05623, 0.],
           [-0.16473, -0.45259, 2.36315],
           [0., 0., 0.],
           [0.02827, 2.03559, -0.19338],
           [0.05672, 2.04885, -0.04275],
           [0., 0., 0.],
           [-0.05417, 1.74624, 0.17202],
           [0.10407, 1.76136, -0.12397],
           [0., 0., 0.],
           [3.36241, 1.20089, -0.31121],
           [4.983, -0., -0.],
           [3.48356, -0., -0.],
           [0., 0., 0.],
           [0.71526, -0., -0.],
           [0., 0., 0.],
           [0., 0., 0.],
           [-3.1366, 1.37405, -0.40465],
           [-5.2419, -0., -0.],
           [-3.44417, -0., -0.],
           [0., 0., 0.],
           [-0.62253, -0., -0.],
           [0., 0., 0.]]

parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 21, 20, 13, 24, 25, 26, 27, 28,
           27]

batch_size = 128
n_joints = 31
J = 32 * 4
T = 32
size = (batch_size, J, T)
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)



def get_loaders(batch_size, data_path=None, extra_data_dir=None):
    data_dir = pjoin(BASEPATH, 'data')
    data_filename = "xia.npz" 
    data_path = pjoin(data_dir, data_filename)
    extra_data_dir = pjoin(data_dir, data_filename.split('.')[-2].split('/')[-1] + "_norms")
    dataset = MotionNorm('train', data_path=data_path, extra_data_dir=extra_data_dir)
    testset = MotionNorm('test', data_path=data_path, extra_data_dir=extra_data_dir)
    trainloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return trainloader, testloader

class EMA:
    """
  Maintains (exponential) moving average of a set of parameters.
  """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the result of
        `model.parameters()`.
      decay: The exponential decay.
      use_num_updates: Whether to use number of updates when computing
        averages.
    """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
    Update currently maintained parameters.
    Call this every time the parameters are updated, such as the result of
    the `optimizer.step()` call.
    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the same set of
        parameters used to initialize this object.
    """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
    Copy current parameters into given collection of parameters.
    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored moving averages.
    """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
    Save the current parameters for restoring later.
    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        temporarily stored.
    """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
    Restore the parameters stored with the `store` method.
    Useful to validate the model with EMA parameters without affecting the
    original optimization process. Store the parameters before the
    `copy_to` method. After validation (or model saving), use this to
    restore the former parameters.
    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored parameters.
    """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b




def swish(x):
    return x * torch.sigmoid(x)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, hidden_dim, max_steps=500) -> None:
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(dim, max_steps))
        self.proj1 = nn.Linear(dim * 2, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t):
        x = self.embedding[t.long()]
        x = self.proj1(x)
        x = swish(x)
        x = self.proj2(x)
        x = swish(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(dim).unsqueeze(0)
        table = steps * 10.0 ** (dims * 4.0 / dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class Sinusoid(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Conv(kernel_size, in_channels, out_channels, dilation=1, stride=1, use_bias=True, kaiming=False, zeros=False):
    """
    returns a list of [pad, conv] => should be += to some list, then apply sequential
    """

    padding = int(dilation * (kernel_size - 1) / 2)

    conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                     padding=padding, padding_mode='circular', bias=use_bias)
    if kaiming:
        nn.init.kaiming_normal_(conv.weight)
    elif zeros:
        nn.init.zeros_(conv.weight)
    return conv

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, d_head=32):
        super().__init__()
        self.scale = d_head ** -0.5
        self.heads = heads
        hidden = heads * d_head
        self.qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        b, c, j, t = x.shape
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale
        k = k.softmax(dim=-1)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=j, y=t)
        return self.out(out)


class ResidualBlock2d(nn.Module):
    def __init__(self, temb_channels, in_channels, out_channels) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.diffusion_proj = nn.Linear(temb_channels, in_channels)
        self.conv_scale = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()


    def forward(self, x, t):
        x = self.norm1(x)
        t = self.diffusion_proj(t)
        y = x + t[:, :, None, None]
        y = self.conv1(y)

        y = swish(y)

        y = self.norm2(y)
        y = self.conv2(y)
        y = swish(y)
        y = (y + self.conv_scale(x)) / np.sqrt(2.)

        return y


class ResidualBlock(nn.Module):
    def __init__(self, temb_channels, in_channels, out_channels, dilation) -> None:
        super().__init__()
        self.norm1 = LayerNorm(in_channels)
        self.norm2 = LayerNorm(out_channels)
        self.conv1 = Conv(3, in_channels, out_channels, dilation=dilation)
        self.conv2 = Conv(3, out_channels, out_channels, dilation=dilation)
        self.diffusion_proj = nn.Linear(temb_channels, in_channels)
        self.conv_scale = Conv(1, in_channels, out_channels)

    def forward(self, x, t, adj=None):
        x = self.norm1(x)
        t = self.diffusion_proj(t)
        y = x + t[:, :, None]
        y = self.conv1(y)
        y = swish(y)
        y = self.norm2(y)

        y = self.conv2(y)
        y = swish(y)
        y = (y + self.conv_scale(x)) / np.sqrt(2.)
        return y, adj


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_1 = nn.Linear(dim, 4)
        self.layer_2 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.layer_1(x)
        x = swish(x)
        x = self.layer_2(x)
        x = swish(x)
        return x


class LayerNormNeXt(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.g[:, None, None] * x + self.b[:, None, None]
        return x

class FastConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim, 1e-6),
            nn.Linear(dim, dim * 4)
        )
        self.down = nn.Linear(dim * 4, dim)

        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)),
                                  requires_grad=True)

        self.t_linear = nn.Linear(96, dim)

    def forward(self, x, t):
        y = self.dw_conv(x)
        t = self.t_linear(gelu(t))
        y = y + t[:, :, None, None]
        y = y.permute(0, 2, 3, 1)
        y = self.net(y)
        y = self.down(gelu(y))
        y = self.gamma * y
        y = y.permute(0, 3, 1, 2)
        y = y + x
        return x


class ConvNext(nn.Module):
    def __init__(self, temb_dim=64, residual_hidden=64):
        super().__init__()
        self.diffusion_emb = DiffusionEmbedding(64, 96, 10000)
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 96, 7, padding=0),
            LayerNormNeXt(96, eps=1e-6)
        )
        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]
        self.down_layers = nn.Sequential(
            LayerNormNeXt(96, 1e-6),
            nn.Conv2d(96, 192, (3, 2), 2, (1, 0)),
            LayerNormNeXt(192, 1e-6),
            nn.Conv2d(192, 384, 2, (2, 2))
        )

        self.blocks = nn.Sequential(*[FastConvNeXtBlock(384) for _ in range(6)])

        self.up_blocks = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 2, 2),
            LayerNormNeXt(192, 1e-6),
            nn.ConvTranspose2d(192, 96, (3, 2), 2, (1, 0)),
            LayerNormNeXt(96, 1e-6)
        )
        self.b1 = FastConvNeXtBlock(384)
        self.b2 = FastConvNeXtBlock(384)
        self.b3 = FastConvNeXtBlock(384)
        self.b4 = FastConvNeXtBlock(384)
        self.b5 = FastConvNeXtBlock(384)
        self.b6 = FastConvNeXtBlock(384)

        self.conv_out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(96, 4, 7, padding=0),
            # nn.Tanh()
        )

        self.fc_content = MLP(6)

        self.fc_style = MLP(8)

        self.aux_conv = nn.Conv2d(96, 4, 3, 1, 1)

        self.root_out = Conv(3, 124, 4, zeros=True)

        self.foot_out = Conv(3, 124, 4, zeros=True)

    def forward(self, x, style, content, t):
        emb = self.diffusion_emb(t)
        b, c, t = x.shape
        x = x.view(b, 4, -1, t)
        content = self.fc_content(content).view(b, -1)
        style = self.fc_style(style).view(b, -1)

        x = x + content[:, :, None, None] + style[:, :, None, None]
        x = self.stem(x)

        x = x + gelu(emb[:, :, None, None])
        x = self.down_layers(x)
        x = self.b1(x, emb)
        x = self.b2(x, emb)
        x = self.b3(x, emb)
        x = self.b4(x, emb)
        x = self.b5(x, emb)
        x = self.b6(x, emb)
        x = self.up_blocks(x)
        x = x + gelu(emb[:, :, None, None])

        aux = self.aux_conv(x)
        aux = aux.view(b, -1, t)
        root = self.root_out(aux)
        foot = self.foot_out(aux)

        y = self.conv_out(x)

        y = y.view(b, c, t)
        angvel = None
        return y, root, foot, angvel


class RelativePositionBias(nn.Module):
    def __init__(
            self,
            heads=8,
            num_buckets=32,
            max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


class UNET2(nn.Module):
    def __init__(self, temb_dim=64, residual_hidden=64):
        super().__init__()
        self.use_angvel = False
        self.use_root = True
        self.use_foot_contacts = True

        self.diffusion_emb = DiffusionEmbedding(64, 64, 10000)
        features = [64, 128, 256, 512]
        self.in_conv = nn.Conv2d(4, 64, 3, 1, 1)
        self.down_layers = []
        prev_feat = 64
        for feat in features[1:]:
            self.down_layers.append(ResidualBlock2d(64, prev_feat, feat))
            prev_feat = feat
        self.down_layers = nn.ModuleList(self.down_layers)
        self.bridge = nn.ModuleList([ResidualBlock2d(64, 512, 512), ResidualBlock2d(64, 512, 512)])
        prev_feat = 512
        up_layers = []
        reversed_features = features[::-1]
        for feat in reversed_features[1:]:
            up_layers.append(nn.ConvTranspose2d(feat, feat, kernel_size=2, stride=2))
            up_layers.append(ResidualBlock2d(64, prev_feat * 2, feat))
            prev_feat = feat
        self.up_layers = nn.ModuleList(up_layers)

        self.aux_conv = nn.Conv2d(64, 4, 3, 1, 1)

        if self.use_root:
            self.root_out = nn.Sequential(
                nn.Conv2d(64, 32, (4, 3), (2, 1), 1),
                nn.MaxPool2d((2, 1), (2, 1)),
                nn.Conv2d(32, 4, (4, 3), (2, 1), 1),
                nn.MaxPool2d((2, 1), (2, 1)),
            )
        else:
            self.root_out = None

        if self.use_foot_contacts:
            self.foot_out = nn.Sequential(
                nn.Conv2d(64, 32, (4, 3), (2, 1), 1),
                nn.MaxPool2d((2, 1), (2, 1)),
                nn.Conv2d(32, 4, (4, 3), (2, 1), 1),
                nn.MaxPool2d((2, 1), (2, 1)),
            )
        else:
            self.foot_out = None
        self.conv_out = nn.Conv2d(64, 4, 3, 1, 1)

        self.angvel_out = None

        self.fc_content = MLP(6)

        self.fc_style = MLP(8)

        self.pool = nn.AvgPool2d(kernel_size=2)
        self.down = nn.Conv2d(128, 128, (3, 4), 2, 1)
        self.up = nn.ConvTranspose2d(64, 64, (3, 4), 2, 1)
        self.apply(init_weights)

    def forward(self, x, style, content, t):
        diffusion_step = self.diffusion_emb(t)
        b, c, t = x.shape
        x = x.view(b, 4, -1, t)
        content = self.fc_content(content).view(b, -1)
        style = self.fc_style(style).view(b, -1)


        x = x + content[:, :, None, None] + style[:, :, None, None]

        y = self.in_conv(x)
        skips = []
        is_first = True
        for i, layer in enumerate(self.down_layers):
            y = layer(y, diffusion_step)
            if is_first:
                y = self.down(y)

                is_first = False
            else:
                y = self.pool(y)

            skips.append(y)

        for layer in self.bridge:
            y = layer(y, diffusion_step)

        for i in range(0, len(self.up_layers), 2):

            skip = skips.pop()
            y = torch.cat([y, skip], dim=1)

            y = self.up_layers[i + 1](y, diffusion_step)
            if i < 4:
                y = self.up_layers[i](y)

            else:
                y = self.up(y)

        foot = self.foot_out(y)
        root = self.root_out(y)

        root = root.view(b, 4, t)
        foot = foot.view(b, 4, t)

        angvel = None
        y = self.conv_out(y).view(b, c, t)
        return y, root, foot, angvel


class UNET(nn.Module):
    def __init__(self, temb_dim=64, residual_hidden=256):
        super().__init__()

        self.diffusion_emb = DiffusionEmbedding(temb_dim, residual_hidden, 10000)

        self.in_conv = Conv(1, 132, 256, kaiming=True)
        self.conv_down_1 = ResidualBlock(256, 256, 512, dilation=1)

        self.conv_down_2 = ResidualBlock(256, 512, 1024, dilation=2)

        self.conv_down_3 = ResidualBlock(256, 1024, 2048, dilation=4)

        self.middle_conv = ResidualBlock(256, 2048, 2048, dilation=2)
        self.conv_up_1 = ResidualBlock(256, 2048 + 2048, 1024, dilation=4)

        self.conv_up_2 = ResidualBlock(256, 1024 + 1024, 512, dilation=2)

        self.conv_up_3 = ResidualBlock(256, 512 + 512, 256, dilation=1)

        self.root_out = Conv(3, 256, 4, zeros=True)

        self.foot_out = Conv(3, 256, 4, zeros=True)

        self.conv_out = Conv(3, 256, 124)

        self.angvel_out = None

        self.fc_content = MLP(6)

        self.fc_style = MLP(8)

        self.adj = None

    def construct_adj(self):
        graph = np.zeros((32, 32))
        lengths = np.sum(np.array(offsets) ** 2.0, axis=1) ** 0.5 + 0.001
        for i, p in enumerate(parents):
            if p == -1: continue
            if lengths[p] != 0:
                print(i, p)
            graph[i, p] = lengths[p]
            graph[p, i] = lengths[p]
        return torch.tensor(graph).unsqueeze(0)

    def _init(self):
        return

    def forward(self, x, style, content, t):

        content = self.fc_content(content)
        style = self.fc_style(style)
        content = content.unsqueeze(2)
        style = style.unsqueeze(2)
        style = style.repeat(1, 1, x.size(2))
        content = content.repeat(1, 1, x.size(2))

        x = torch.cat([x, style, content], dim=1)

        diffusion_step = self.diffusion_emb(t)
        adj = self.adj
        y = self.in_conv(x)

        c1, adj = self.conv_down_1(y, diffusion_step, adj)

        c2, adj = self.conv_down_2(c1, diffusion_step, adj)

        c3, adj = self.conv_down_3(c2, diffusion_step, adj)

        y, adj = self.middle_conv(c3, diffusion_step, adj)

        y = torch.cat([y, c3], dim=1)
        y, adj = self.conv_up_1(y, diffusion_step, adj)

        y = torch.cat([y, c2], dim=1)
        y, adj = self.conv_up_2(y, diffusion_step, adj)

        y = torch.cat([y, c1], dim=1)

        y, adj = self.conv_up_3(y, diffusion_step, adj)

        root = self.root_out(y)
        foot = self.foot_out(y)

        angvel = None
        y = self.conv_out(y)
        return y, root, foot, angvel


class ResidualBlockDisc(nn.Module):
    def __init__(self, temb_channels, in_channels, out_channels, dilation) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.conv1 = Conv(3, in_channels, out_channels, dilation=dilation)
        self.conv2 = Conv(3, out_channels, out_channels, dilation=dilation)
        self.conv_scale = Conv(1, in_channels, out_channels)

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = F.leaky_relu(y, 0.2)
        y = self.norm2(y)

        y = self.conv2(y)
        y = F.leaky_relu(y, 0.2)
        y = (y + self.conv_scale(x)) / np.sqrt(2.)
        return y


class StdDev(nn.Module):
    def __init__(self, group_size=4, new_feats=1):
        super().__init__()
        self.group_size = group_size
        self.new_feats = new_feats

    def forward(self, x):
        b, c, t = x.shape
        group_size = min(self.group_size, b)
        y = x.reshape([group_size, -1, self.new_feats, c // self.new_feats, t])
        y = y - y.mean(0, keepdim=True)
        y = (y ** 2).mean(0, keepdim=True)
        y = y + 1e-8 ** 0.5
        y = y.mean([3, 4], keepdim=True).squeeze(3)
        y = y.expand(group_size, -1, -1, t).clone().reshape(b, self.new_feats, t)
        z = torch.cat([x, y], dim=1)
        return z


class StyleClassifier(nn.Module):
    def __init__(self, dim=248):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 1, 1)
        self.time_dependent = nn.ModuleList([
            ResidualBlockDisc(256, dim, 256, 1),
            ResidualBlockDisc(256, 256, 512, 1),
            nn.AvgPool1d(2),
            ResidualBlockDisc(256, 512, 512, 1),
            nn.AvgPool1d(2),
            ResidualBlockDisc(256, 512, 512, 1),
            nn.AvgPool1d(2),

        ])
        self.std_dev = StdDev()
        self.emb = DiffusionEmbedding(64, 256)
        self.conv_style = nn.Conv1d(513, 256, 3, 1, 1)
        self.fc_style = nn.Linear(256, 8)


    def _init_weights(self, m):
        if type(m) in {nn.Conv1d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def features(self, x):
        y = self.conv(x)
        for layer in self.time_dependent:
            y = layer(y)
        y = self.std_dev(y)
        y = F.adaptive_avg_pool1d(y, 1)
        style = self.conv_style(y)
        style = style.reshape(x.size(0), -1)
        return style

    def forward(self, x):
        y = self.conv(x)
        for layer in self.time_dependent:
            y = layer(y)
        y = self.std_dev(y)
        y = F.adaptive_avg_pool1d(y, 1)
        style = self.conv_style(y)
        style = style.reshape(x.size(0), -1)
        style = self.fc_style(style)
        return F.softmax(style)


class Classifier(nn.Module):
    def __init__(self, dim=248):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 1, 1)
        self.time_dependent = nn.ModuleList([
            ResidualBlockDisc(256, dim, 256, 1),
            ResidualBlockDisc(256, 256, 512, 1),
            nn.AvgPool1d(2),
            ResidualBlockDisc(256, 512, 512, 1),
            nn.AvgPool1d(2),
            ResidualBlockDisc(256, 512, 512, 1),
            nn.AvgPool1d(2),

        ])
        self.std_dev = StdDev()
        self.emb = DiffusionEmbedding(64, 256)
        self.conv_content = nn.Conv1d(513, 256, 3, 1, 1)
        self.fc_content = nn.Linear(256, 6)

    def _init_weights(self, m):
        if type(m) in {nn.Conv1d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def features(self, x):
        y = self.conv(x)
        for layer in self.time_dependent:
            y = layer(y)
        y = self.std_dev(y)
        y = F.adaptive_avg_pool1d(y, 1)
        content = self.conv_content(y)
        content = content.reshape(x.size(0), -1)
        return content

    def forward(self, x):
        y = self.conv(x)
        for layer in self.time_dependent:
            y = layer(y)
        y = self.std_dev(y)
        y = F.adaptive_avg_pool1d(y, 1)
        content = self.conv_content(y)
        content = content.reshape(x.size(0), -1)
        content = self.fc_content(content)
        return F.softmax(content)


class DiscBlock(nn.Module):
    def __init__(self, dim, dim_out, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 4, stride, 1, padding_mode="reflect"),
            nn.InstanceNorm2d(dim_out),
        )

    def forward(self, x):
        x = gelu(self.block(x))
        return x


def init_weights(net, init_gain=0.02):
    def init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init)


class Classifier2d(nn.Module):
    def __init__(self, dim=4, style=False):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(dim, 64, 4, 2, 1, padding_mode="reflect"),
            nn.SiLU(),
        )
        self.d1 = DiscBlock(64, 128, 2)
        self.d2 = DiscBlock(128, 256, 2)
        self.d3 = DiscBlock(256, 512, 1)
        self.out = nn.Conv2d(512, 1, 4, 1, 1, padding_mode="reflect")
        self.apply(init_weights)

        self.fc_style = MLP(8)

        self.fc_content = MLP(6)

    def forward(self, x, content, style):
        b, j, t = x.shape
        x = x.view(b, 4, -1, t)
        c = self.fc_content(content).view(b, -1)
        s = self.fc_style(style).view(b, -1)
        x = x + c[:, :, None, None] + s[:, :, None, None]
        x = x.float()
        x = self.conv_in(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.out(x)
        return x


class Diffusion():
    def __init__(self, denoise_fn, device):
        self.num_diffusion_timesteps = 1000
        self.denoise_fn = denoise_fn
        self.device = device
        self.betas = torch.from_numpy(np.linspace(
            1e-4, 0.02, self.num_diffusion_timesteps, dtype=np.float64
        )).float().to(self.device)
        self.classification_loss = nn.NLLLoss()
        self.feature_loss = nn.MSELoss()

    @staticmethod
    def split_pos_glb(raw):
        return raw[:, :-4, :], raw[:, -4:, :]

    @staticmethod
    def merge_pos_glb(pos, glb):
        return torch.cat([pos, glb], dim=-2)

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
        return a

    def load_classifiers(self):
        state = torch.load('./classifier.pt')
        classifier_content = Classifier(128).to(self.device)
        classifier_content.load_state_dict(state['classifier-content'])

        state = torch.load('./style_classifier.pt')
        classifier_style = StyleClassifier(128).to(self.device)
        classifier_style.load_state_dict(state['classifier-style'])
        return classifier_content, classifier_style

    def loss(self, x0, t, style=None, content=None, angvel=None, keepdim=False):
        b = self.betas
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
        x_inp = x0
        x, root = self.split_pos_glb(x0)

        e = torch.randn_like(x, device=x.device)
        x1 = x * a.sqrt() + e * (1.0 - a).sqrt()

        x = x1

        output, out_root, foot_out, angvel_out = self.denoise_fn(x, style, content, t.float())

        loss = (e - output).square().sum(dim=(1, 2)).mean(dim=0)
        root_loss = torch.mean(torch.sum(torch.square(out_root - root), dim=(1, 2)), dim=0)
        x0 = x1 - output * (1.0 - a).sqrt()
        x0 = x0 / a.sqrt()
        out = x0
        recon = self.merge_pos_glb(out, out_root)
        with torch.no_grad():
            style_out = self.style_classifier(recon)
            style_in = self.style_classifier(x_inp)
            content_out = self.content_classifier(recon)
            content_in = self.content_classifier(x_inp)
            content_loss = self.feature_loss(content_out, content_in)
            style_loss = self.feature_loss(style_out, style_in)
        return loss, foot_out, root_loss, content_loss, style_loss

    def loss_adv(self, x0, t, style=None, content=None, angvel=None, keepdim=False):
        b = self.betas
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
        x_inp = x0
        x, root = self.split_pos_glb(x0)

        e = torch.randn_like(x, device=x.device)
        x1 = x * a.sqrt() + e * (1.0 - a).sqrt()

        x = x1

        output, out_root, foot_out, angvel_out = self.denoise_fn(x, style, content, t.float())

        loss = (e - output).square().sum(dim=(1, 2)).mean(dim=0)
        root_loss = torch.mean(torch.sum(torch.square(out_root - root), dim=(1, 2)), dim=0)
        x0 = x1 - output * (1.0 - a).sqrt()
        x0 = x0 / a.sqrt()
        out = x0
        recon = self.merge_pos_glb(out, out_root)

        return loss, foot_out, root_loss, recon

    def ddim_steps(self, x, seq, b, style, content):
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            x, _ = self.split_pos_glb(x)
            x0_preds = []
            xs = [x]
            betas = b
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = self.compute_alpha(betas, t.long())
                atm1 = self.compute_alpha(betas, next_t.long())
                xt = xs[-1].to('cuda')
                output, root, fc, _ = self.denoise_fn(x, style, content, t.float())
                e = output
                x0_t = (xt - e * (1 - at).sqrt()) / at.sqrt()
                c = (1 - atm1).sqrt()
                xt_next = atm1.sqrt() * x0_t + c * e
                xs.append(xt_next.to('cpu'))
            sample = xs[-1]
            root = root.to('cpu')
            sample = torch.cat([sample, root], dim=1)
            return sample, fc

    def ddpm_steps(self, x, seq, b, style, content):

        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            x, _ = self.split_pos_glb(x)
            xs = [x]
            x0_preds = []
            betas = b

            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = self.compute_alpha(betas, t.long())
                atm1 = self.compute_alpha(betas, next_t.long())
                beta_t = 1 - at / atm1
                x = xs[-1].to('cuda')
                output, root, fc, _ = self.denoise_fn(x, style, content, t.float())
                e = output

                x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
                x0_from_e = torch.clamp(x0_from_e, -1, 1)
                x0_preds.append(x0_from_e.to('cpu'))
                mean_eps = (
                                   (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
                           ) / (1.0 - at)

                mean = mean_eps
                noise = torch.randn_like(x)
                mask = 1 - (t == 0).float()
                mask = mask.view(-1, 1, 1)
                logvar = beta_t.log()
                sample = mean + mask * torch.exp(0.5 * logvar) * noise

                xs.append(sample.to('cpu'))

        sample = torch.cat([xs[-1], root.to('cpu')], dim=1)
        return [sample], fc

    def sample(self, batch_size, t, style, content):
        x = torch.randn((batch_size, 128, t), device=self.device)

        seq = range(0, self.num_diffusion_timesteps)
        start = time.time()
        samples, fc = self.ddpm_steps(x, seq, self.betas, style, content)
        print(time.time() - start)
        return samples[-1], fc

    def sample_from(self, x, t, style):
        seq = range(0, t)
        start = time.time()
        samples, _ = self.ddpm_steps(x, seq, self.betas, style)
        print(time.time() - start)
        print(samples[-1].shape)
        return samples[-1]

    def add_noise(self, x0, t):
        e = torch.randn_like(x0, device=x0.device)
        b = self.betas
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
        x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
        return x


class Trainer():
    def __init__(self):
        self.batch_size = 128
        trainloader, test_loader = get_loaders(self.batch_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNET2().to(self.device)
        print('Parameters: ', sum(p.numel() for p in self.model.parameters()))
        self.ema = EMA(self.model.parameters(), 0.9999)
        self.diffusion = Diffusion(self.model, self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=2e-4, betas=(0.5, 0.999), eps=1e-8)

        self.disc = Classifier2d(dim=4).to(self.device)
        self.disc_style = Classifier2d(dim=4, style=True).to(self.device)
        print('Disc Parameters: ', sum(p.numel() for p in self.disc.parameters()))
        self.optim_d = optim.Adam(self.disc.parameters(), lr=1e-4, betas=(0.5, 0.999), eps=1e-8)
        self.optim_style = optim.AdamW(self.disc_style.parameters(), lr=1e-4, betas=(0.5, 0.999), eps=1e-8)

        self.max_iter = 9000
        self.print_every = 1000
        self.step = 0
        self.trainloader = cycle(trainloader)
        self.testloader = cycle(test_loader)
        self.alpha = 100 / 180.0 * np.pi

    def load_weights(self):

        state = torch.load('./checkpoint.pt')
        self.model.load_state_dict(state['model'])
        self.model.eval()
        self.ema.load_state_dict(state['ema'])
        self.optim.load_state_dict(state['optimiser'])
        self.disc.load_state_dict(state['disc'])
        self.optim_d.load_state_dict(state['opt_d'])
        self.step = state['step']
        print("step: ", self.step)

    def to_onehot(self, x, n_classes):
        labels = torch.zeros(x.size(0), n_classes)
        labels[np.arange(x.size(0)), x.long()] = 1
        return labels.to(self.device)

    @staticmethod
    def convert_to_disc(
            raw): 
        vraw = raw[..., 1:] - raw[...,
                              :-1] 
        return torch.cat([raw[..., 0:1], vraw], dim=-1)

    def train_classifier(self):
        self.disc.train()
        start = time.time()
        loss = nn.MSELoss()
        running_loss = 0.0
        while self.step <= 6000:
            data = next(self.trainloader)
            x = data['content']
            content = self.to_onehot(data['content_label'], 6)
            style = self.to_onehot(data['label'], 8)
            y_c = self.disc(x, content, style)
            self.optim_d.zero_grad()
            loss_c = loss(y_c, content)
            loss_t = loss_c
            loss_t.backward()
            self.optim_d.step()
            running_loss += loss_t.item()
            if self.step % self.print_every == 0:
                test_data = next(self.testloader)
                x = test_data['content']
                content = self.to_onehot(test_data['content_label'], 6)
                with torch.no_grad():
                    c = self.disc(x)
                    c_loss = loss(c, content)
                print(self.step, time.time() - start)
                print('Classifier Loss: ', c_loss.item(), running_loss)
                running_loss = 0.0
                torch.save({'classifier-content': self.disc.state_dict(), 'opt_classifier': self.optim_d.state_dict(),
                            'step': self.step}, './classifier.pt')
            self.step += 1

    def train_style_classifier(self):
        self.disc_style.train()
        start = time.time()
        loss = nn.MSELoss()
        running_loss = 0.0
        while self.step <= 4100:
            data = next(self.trainloader)
            x = data['content']

            style = self.to_onehot(data['label'], 8)
            y_c = self.disc_style(x)
            self.optim_style.zero_grad()
            loss_c = loss(y_c, style)
            loss_t = loss_c
            loss_t.backward()
            self.optim_style.step()
            running_loss += loss_t.item()
            if self.step % self.print_every == 0:
                test_data = next(self.testloader)
                x = test_data['content']
                style = self.to_onehot(test_data['label'], 8)
                with torch.no_grad():
                    c = self.disc_style(x)
                    c_loss = loss(c, style)
                print(self.step, time.time() - start)
                print('Classifier Loss: ', c_loss.item(), running_loss)
                running_loss = 0.0
                torch.save(
                    {'classifier-style': self.disc_style.state_dict(), 'opt_classifier': self.optim_style.state_dict(),
                     'step': self.step}, './style_classifier.pt')
            self.step += 1

    def train(self):
        self.disc.train()
        self.model.train()

        start = time.time()
        running_loss = 0.0
        running_style = 0.0
        running_content = 0.0
        running_root = 0.0
        running_feet = 0.0
        while self.step < self.max_iter:
            data = next(self.trainloader)
            x = data['content']
            content = self.to_onehot(data['content_label'], 6)
            style = self.to_onehot(data['label'], 8)

            fc = data['foot_contact'].to(self.device)

            n = x.size(0)
            t = torch.randint(
                low=0, high=self.diffusion.num_diffusion_timesteps, size=(n // 2 + 1,)
            ).to(self.device)
            t = torch.cat([t, self.diffusion.num_diffusion_timesteps - t - 1], dim=0)[:n]
            g_loss, foot_contact, root_loss, content_loss, style_loss = self.diffusion.loss(x, t, style, content)
            l_feet = (foot_contact - fc).square().sum(dim=(1, 2)).mean(dim=0)

            self.optim.zero_grad()
            loss = g_loss + l_feet + root_loss  # + content_loss + style_loss
            loss.backward()
            self.optim.step()
            running_loss += loss.item()
            running_feet += l_feet.item()
            running_root += root_loss.item()
            running_content += content_loss.item()
            running_style += style_loss.item()
            self.ema.update(self.model.parameters())
            if self.step % self.print_every == 0:
                print(self.step, time.time() - start)
                print('Diffusion Loss: ', running_loss / 100.0)
                print('Content Loss: ', running_content / 100.0)
                print('Style Loss: ', running_style / 100.0)
                print('Root Loss: ', running_root / 100.0)
                print('Foot Contact Loss', running_feet / 100.0)
                running_loss = 0.0
                running_style = 0.0
                running_content = 0.0
                running_root = 0.0
                running_feet = 0.0
                torch.save({'model': self.model.state_dict(), 'optimiser': self.optim.state_dict(),
                            'ema': self.ema.state_dict(), 'step': self.step}, './ddpm-gcn-class-1000.pt')
            self.step += 1

    def train_adv(self):
        self.disc.train()
        self.model.train() 
        disc_loss = nn.MSELoss()
        start = time.time()
        while self.step < self.max_iter:
            data = next(self.trainloader) 
            x = data['content'] 
            content = self.to_onehot(data['content_label'], 6)

            style = self.to_onehot(data['label'], 8) 

            fc = data['foot_contact'].to(self.device) 

            n = x.size(0) 
            t = torch.randint(0, self.diffusion.num_diffusion_timesteps, (n,), device=self.device)
            if self.step % 1 == 0: 
                with torch.no_grad():
                    g_loss, foot_contact, root_loss, fake = self.diffusion.loss_adv(x, t, style, content)
                d_fake = self.disc(fake, content, style) 
                d_real = self.disc(x, content, style) 
                real_loss = disc_loss(d_real, torch.ones_like(d_real))
                fake_loss = disc_loss(d_fake, torch.zeros_like(d_fake)) 
                d_loss = real_loss + fake_loss

                self.optim_d.zero_grad()
                d_loss.backward()
                self.optim_d.step()

            self.optim.zero_grad()
            l_noise, foot_contact, l_root, fake = self.diffusion.loss_adv(x, t, style, content)
            l_feet = (foot_contact - fc).square().sum(dim=(1, 2)).mean(dim=0) 

            l_velocity = 0.0
            l_acceleration = 0.0
            for f in range(fake.shape[-1]-1):
                velocity = torch.sqrt(torch.square(x[:, :, f+1]-x[:, :, f])).sum()
                if f < 30:
                    acceleration = torch.sqrt(torch.square(x[:, :, f+2]-2 * x[:, :, f+1]+x[:, :, f])).sum()
                    l_acceleration = l_acceleration + acceleration
                l_velocity = l_velocity + velocity
            loss = l_noise + l_feet + l_root + 0.01 * l_velocity + 0.01 * l_acceleration
            loss.backward()
            self.optim.step()
            self.ema.update(self.model.parameters())
            if self.step % self.print_every == 0:
                print(f'Current Step: {str(self.step)}')
                print(f'Time already used: {str(time.time() - start)}')
                print(f'Total Loss: {str(loss.item())}')
                print(f'Discriminator Loss: {str(l_disc.item())}')
                print(f'Root loss: {str(l_root.item())}')
                print(f'Foot contact loss: {str(l_feet.item())}')
                print(f'Velocity loss: {str(l_velocity.item())}')
                print(f'Acceleration loss: {str(l_acceleration.item())}')

                torch.save({'model': self.model.state_dict(), 'optimiser': self.optim.state_dict(),
                            'ema': self.ema.state_dict(), 'step': self.step,
                            'disc': self.disc.state_dict(), 'opt_d': self.optim_d.state_dict()},
                           './checkpoint.pt')

            self.step += 1

def calculate_activation_statistics(activations):
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calc_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


class Generator:
    def __init__(self):
        self.batch_size = 1
        train_loader, test_loader = get_loaders(self.batch_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNET2().to(self.device)
        self.ema = EMA(self.model.parameters(), 0.9999)
        self.diffusion = Diffusion(self.model, self.device)
        self.testloader = cycle(test_loader)
        self.trainloader = cycle(train_loader)

    def load_weights(self):
        state = torch.load('./checkpoint.pt')
        self.model.load_state_dict(state['model'])
        self.model.eval()
        self.ema.load_state_dict(state['ema'])
        self.ema.copy_to(self.model.parameters())

    def to_onehot(self, x, n_classes):
        labels = torch.zeros(x.size(0), n_classes)
        labels[np.arange(x.size(0)), x.long()] = 1
        return labels.to(self.device)

    def generate(self, content_label, style_label, name):
        data = next(self.testloader)
        x = data['content']
        print(data['meta'])
        content = self.to_onehot(content_label, 6)
        style = self.to_onehot(style_label, 8)
        N, _, T = x.shape
        mean = data['mean'].cpu()
        std = data['std'].cpu()
        start = time.time()
        output, fc = self.diffusion.sample(N, 64, style, content)
        print(time.time() - start)
        output = output * std + mean
        output = output.squeeze(0)
        output = output.cpu().numpy()
        foot_contact = fc[0]
        save_bvh_from_network_output(output, output_path=f'./diff-gcn-{name}.bvh')

    def generate_data(self, size):
        state = torch.load('./classifier.pt')
        classifier_content = Classifier(128).to(self.device)
        classifier_content.load_state_dict(state['classifier-content'])
        self.classifier = classifier_content
        self.classifier.eval()
        content_labels = []
        style_labels = []
        k = 0
        while k < size:
            for i in range(8):
                for j in range(6):
                    if k == size:
                        break
                    content_label = torch.tensor([j])
                    style_label = torch.tensor([i])
                    content_labels.append(self.to_onehot(content_label, 6))
                    style_labels.append(self.to_onehot(style_label, 8))
                    k += 1

        content_labels = torch.stack(content_labels).squeeze(1)
        style_labels = torch.stack(style_labels).squeeze(1)
        N = content_labels.size(0)
        T = 32
        batch, fc = self.diffusion.sample(N, T, style_labels, content_labels)
        with torch.no_grad():
            act = self.classifier.features(batch.to(self.device))
        return act
        print(act.shape, batch.shape)
        mu, sigma = calculate_activation_statistics(act)
        print(mu.shape, sigma.shape)
        return mu, sigma

    def calc_fid(self):
        batch = next(self.trainloader)
        x = batch['content']
        size = x.size(0)
        acts_model = self.generate_data(size).cpu()
        with torch.no_grad():
            acts_truth = self.classifier.features(x).cpu()
        for i in range(11):
            batch = next(self.trainloader)
            x = batch['content']
            size = x.size(0)
            acts1 = self.generate_data(size).cpu()
            with torch.no_grad():
                acts2 = self.classifier.features(x).cpu()
            acts_model = torch.cat([acts_model, acts1], dim=0)
            acts_truth = torch.cat([acts_truth, acts2], dim=0)
            print(acts_model.shape)
            print(acts_truth.shape)
        mu1, sigma1 = calculate_activation_statistics(acts_model)
        mu2, sigma2 = calculate_activation_statistics(acts_truth)
        print(mu2.shape, sigma2.shape)
        fid = calc_fid(mu1, sigma1, mu2, sigma2)
        print(fid)


if __name__ == '__main__':

    t = Trainer()
    t.train_adv()

    g = Generator()
    g.load_weights()
    g.calc_fid()
