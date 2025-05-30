import torch
import timm
from models_mae import (
    mae_vit_base_patch16 as _base16,
    mae_vit_large_patch16 as _large16,
    mae_vit_huge_patch14 as _huge14,
)

dependencies = ['torch', 'timm']

def mae_vit_base_patch16(pretrained: bool = True, **kwargs):
    model = _base16(**kwargs)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
            map_location="cpu"
        )
        model.load_state_dict(ckpt, strict=False)
    return model

def mae_vit_large_patch16(pretrained: bool = True, **kwargs):
    model = _large16(**kwargs)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
            map_location="cpu"
        )
        model.load_state_dict(ckpt, strict=False)
    return model

def mae_vit_huge_patch14(pretrained: bool = True, **kwargs):
    model = _huge14(**kwargs)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
            map_location="cpu"
        )
        model.load_state_dict(ckpt, strict=False)
    return model

__all__ = [
    'mae_vit_base_patch16',
    'mae_vit_large_patch16',
    'mae_vit_huge_patch14',
]