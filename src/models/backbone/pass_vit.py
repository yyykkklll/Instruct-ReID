import collections.abc as container_abcs
import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ckpt import checkpoint_wrapper


def _ntuple(n):
    """
    将输入转换为n维元组

    参数:
        n (int): 元组维度

    返回:
        function: 转换函数
    """

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)


class GeneralizedMeanPooling(nn.Module):
    """
    广义均值池化模块
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        """
        初始化广义均值池化

        参数:
            norm (float): 池化范数，默认为3
            output_size (int): 输出尺寸，默认为1
            eps (float): 最小值，防止除零，默认为1e-6
        """
        super().__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        """
        前向传播，执行广义均值池化

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 池化后的张量
        """
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool1d(x, self.output_size).pow(1. / self.p)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    按样本随机丢弃路径（随机深度）

    参数:
        x (torch.Tensor): 输入张量
        drop_prob (float): 丢弃概率，默认为0
        training (bool): 是否为训练模式，默认为False

    返回:
        torch.Tensor: 丢弃后的张量
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    按样本随机丢弃路径模块
    """

    def __init__(self, drop_prob=None):
        """
        初始化DropPath模块

        参数:
            drop_prob (float): 丢弃概率，默认为None
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        前向传播，应用随机丢弃路径

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 输出张量
        """
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    多层感知机模块
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        初始化MLP模块

        参数:
            in_features (int): 输入特征维度
            hidden_features (int): 隐藏层特征维度，默认为None
            out_features (int): 输出特征维度，默认为None
            act_layer (nn.Module): 激活函数，默认为GELU
            drop (float): Dropout比率，默认为0
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        前向传播，执行MLP计算

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 输出张量
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    注意力机制模块
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        初始化注意力模块

        参数:
            dim (int): 输入特征维度
            num_heads (int): 注意力头数，默认为8
            qkv_bias (bool): 是否添加偏置，默认为False
            qk_scale (float): 缩放因子，默认为None
            attn_drop (float): 注意力Dropout比率，默认为0
            proj_drop (float): 投影Dropout比率，默认为0
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        前向传播，执行注意力计算

        参数:
            x (torch.Tensor): 输入张量，形状[B, N, C]

        返回:
            torch.Tensor: 输出张量，形状[B, N, C]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer块模块
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        初始化Transformer块

        参数:
            dim (int): 输入特征维度
            num_heads (int): 注意力头数
            mlp_ratio (float): MLP隐藏层维度比例，默认为4
            qkv_bias (bool): 是否添加偏置，默认为False
            qk_scale (float): 缩放因子，默认为None
            drop (float): Dropout比率，默认为0
            attn_drop (float): 注意力Dropout比率，默认为0
            drop_path (float): 随机深度比率，默认为0
            act_layer (nn.Module): 激活函数，默认为GELU
            norm_layer (nn.Module): 归一化层，默认为LayerNorm
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        前向传播，执行Transformer块计算

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 输出张量
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class IBN(nn.Module):
    """
    实例归一化和批归一化组合模块
    """

    def __init__(self, planes):
        """
        初始化IBN模块

        参数:
            planes (int): 输入通道数
        """
        super().__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        """
        前向传播，执行IBN计算

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 输出张量
        """
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class PatchEmbed(nn.Module):
    """
    图像块嵌入模块
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768, stem_conv=False):
        """
        初始化PatchEmbed模块

        参数:
            img_size (int): 输入图像尺寸，默认为224
            patch_size (int): 块大小，默认为16
            stride_size (int): 步幅大小，默认为16
            in_chans (int): 输入通道数，默认为3
            embed_dim (int): 嵌入维度，默认为768
            stem_conv (bool): 是否使用卷积茎，默认为False
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        self.num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size

        self.stem_conv = stem_conv
        if self.stem_conv:
            hidden_dim = 64
            stem_stride = 2
            stride_size = patch_size = patch_size[0] // stem_stride
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride, padding=3, bias=False),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            in_chans = hidden_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        """
        前向传播，将图像分割为块并嵌入

        参数:
            x (torch.Tensor): 输入图像张量

        返回:
            torch.Tensor: 嵌入张量
        """
        if self.stem_conv:
            x = self.conv(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransReID(nn.Module):
    """
    Transformer-based ReID模型
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0,
                 view=0, drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), local_feature=False,
                 sie_xishu=1.0, hw_ratio=1, gem_pool=False, stem_conv=False):
        """
        初始化TransReID模型

        参数:
            img_size (tuple): 输入图像尺寸，默认为224
            patch_size (int): 块大小，默认为16
            stride_size (int): 步幅大小，默认为16
            in_chans (int): 输入通道数，默认为3
            num_classes (int): 分类数，默认为1000
            embed_dim (int): 嵌入维度，默认为768
            depth (int): Transformer层数，默认为12
            num_heads (int): 注意力头数，默认为12
            mlp_ratio (float): MLP隐藏层比例，默认为4
            qkv_bias (bool): 是否添加偏置，默认为False
            qk_scale (float): 缩放因子，默认为None
            drop_rate (float): Dropout比率，默认为0
            attn_drop_rate (float): 注意力Dropout比率，默认为0
            camera (int): 相机数，默认为0
            view (int): 视角数，默认为0
            drop_path_rate (float): 随机深度比率，默认为0
            norm_layer (nn.Module): 归一化层，默认为LayerNorm
            local_feature (bool): 是否使用局部特征，默认为False
            sie_xishu (float): SIE嵌入系数，默认为1.0
            hw_ratio (float): 高宽比，默认为1
            gem_pool (bool): 是否使用广义均值池化，默认为False
            stem_conv (bool): 是否使用卷积茎，默认为False
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.local_feature = local_feature
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
            embed_dim=embed_dim, stem_conv=stem_conv)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part_token3 = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part1_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part2_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.part3_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        self.in_planes = 768
        self.gem_pool = gem_pool
        if self.gem_pool:
            print('using gem pooling')
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            try:
                block = checkpoint_wrapper(block)
            except:
                print('fairscale checkpoint failed, use naive block')
                pass
            self.blocks.append(block)

        self.norm = norm_layer(embed_dim)

        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)
        self.gem = GeneralizedMeanPooling()

    def _init_weights(self, m):
        """
        初始化模型权重

        参数:
            m (nn.Module): 模型模块
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        """
        返回无需权重衰减的参数名

        返回:
            set: 参数名集合
        """
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        """
        获取分类器

        返回:
            nn.Module: 分类器模块
        """
        return self.fc

    def reset_classifier(self, num_classes, global_pool=''):
        """
        重置分类器

        参数:
            num_classes (int): 新的分类数
            global_pool (str): 全局池化方式，默认为空
        """
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id, view_id):
        """
        提取特征

        参数:
            x (torch.Tensor): 输入图像张量
            camera_id (torch.Tensor): 相机ID
            view_id (torch.Tensor): 视角ID

        返回:
            tuple: 全局特征和局部特征
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        part_tokens1 = self.part_token1.expand(B, -1, -1)
        part_tokens2 = self.part_token2.expand(B, -1, -1)
        part_tokens3 = self.part_token3.expand(B, -1, -1)
        x = torch.cat((cls_tokens, part_tokens1, part_tokens2, part_tokens3, x), dim=1)
        if self.cam_num > 0 and self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + torch.cat((self.cls_pos, self.part1_pos, self.part2_pos, self.part3_pos, self.pos_embed), dim=1)

        x = self.pos_drop(x)

        if self.local_feature:
            for blk in self.blocks[:-1]:
                x = blk(x)
            return x
        else:
            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)
        if self.gem_pool:
            gf = self.gem(x[:, 1:].permute(0, 2, 1)).squeeze()
            return x[:, 0] + gf
        return x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4:]

    def forward(self, x, cam_label=None, view_label=None):
        """
        前向传播

        参数:
            x (torch.Tensor): 输入图像张量
            cam_label (torch.Tensor): 相机标签，默认为None
            view_label (torch.Tensor): 视角标签，默认为None

        返回:
            tuple: 全局特征和局部特征
        """
        global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_all = self.forward_features(x, cam_label,
                                                                                                      view_label)
        return global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_all

    def load_param(self, model_path, hw_ratio):
        """
        加载预训练参数

        参数:
            model_path (str): 预训练模型路径
            hw_ratio (float): 高宽比
        """
        param_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        count = 0
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        if 'teacher' in param_dict:
            print('Convert dino model......')
            obj = param_dict["teacher"]
            newmodel = {}
            for k, v in obj.items():
                if k.startswith("module."):
                    k = k.replace("module.", "")
                if not k.startswith("backbone."):
                    continue
                old_k = k
                k = k.replace("backbone.", "")
                newmodel[k] = v
                param_dict = newmodel
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k or 'pre_logits' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x, hw_ratio)
            try:
                self.state_dict()[k].copy_(v)
                print(k, 'copied')
                count += 1
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape,
                                                                                                self.state_dict()[
                                                                                                    k].shape))
        print('Load %d / %d layers.' % (count, len(self.state_dict().keys())))


def resize_pos_embed(posemb, posemb_new, hight, width, hw_ratio):
    """
    调整位置嵌入大小

    参数:
        posemb (torch.Tensor): 原始位置嵌入
        posemb_new (torch.Tensor): 目标位置嵌入
        hight (int): 目标高度
        width (int): 目标宽度
        hw_ratio (float): 高宽比

    返回:
        torch.Tensor: 调整后的位置嵌入
    """
    ntok_new = posemb_new.shape[1]
    posemb_grid = posemb[0]
    gs_old_h = int(math.sqrt(len(posemb_grid) * hw_ratio))
    gs_old_w = gs_old_h // hw_ratio
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                posemb_new.shape, hight,
                                                                                                width))
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = posemb_grid
    return posemb


def vit_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0,
                                   local_feature=False, sie_xishu=1.5, **kwargs):
    """
    创建ViT-Base-Patch16-224 TransReID模型

    参数:
        img_size (tuple): 输入图像尺寸，默认为(256, 128)
        stride_size (int): 步幅大小，默认为16
        drop_path_rate (float): 随机深度比率，默认为0.1
        camera (int): 相机数，默认为0
        view (int): 视角数，默认为0
        local_feature (bool): 是否使用局部特征，默认为False
        sie_xishu (float): SIE嵌入系数，默认为1.5
        **kwargs: 其他参数

    返回:
        TransReID: 模型实例
    """
    model = TransReID(img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12,
                      mlp_ratio=4, qkv_bias=True, camera=camera, view=view, drop_path_rate=drop_path_rate,
                      sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)
    return model


def vit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0,
                                    local_feature=False, sie_xishu=1.5, **kwargs):
    """
    创建ViT-Small-Patch16-224 TransReID模型

    参数:
        img_size (tuple): 输入图像尺寸，默认为(256, 128)
        stride_size (int): 步幅大小，默认为16
        drop_path_rate (float): 随机深度比率，默认为0.1
        camera (int): 相机数，默认为0
        view (int): 视角数，默认为0
        local_feature (bool): 是否使用局部特征，默认为False
        sie_xishu (float): SIE嵌入系数，默认为1.5
        **kwargs: 其他参数

    返回:
        TransReID: 模型实例
    """
    model = TransReID(img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6,
                      mlp_ratio=4, qkv_bias=True, drop_path_rate=drop_path_rate, camera=camera, view=view,
                      sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)
    model.in_planes = 384
    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    无梯度截断正态初始化

    参数:
        tensor (torch.Tensor): 输入张量
        mean (float): 均值
        std (float): 标准差
        a (float): 最小值
        b (float): 最大值

    返回:
        torch.Tensor: 初始化后的张量
    """

    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.")

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    截断正态初始化

    参数:
        tensor (torch.Tensor): 输入张量
        mean (float): 均值，默认为0
        std (float): 标准差，默认为1
        a (float): 最小值，默认为-2
        b (float): 最大值，默认为2

    返回:
        torch.Tensor: 初始化后的张量
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
