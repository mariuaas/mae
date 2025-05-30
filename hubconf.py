import torch
import timm
from models_mae import (
    mae_vit_base_patch16 as _base16,
    mae_vit_large_patch16 as _large16,
    mae_vit_huge_patch14 as _huge14,
)
from models_vit import (
    vit_base_patch16 as _vit_base16,
    vit_large_patch16 as _vit_large16,
    vit_huge_patch14 as _vit_huge14,
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

def mae_vit_base_patch16_in1k(pretrained: bool = True, **kwargs):
    # instantiate the ViT‐Base/16 classifier head
    model = _vit_base16(num_classes=1000, **kwargs)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth",
            map_location="cpu"
        )
        sd = ckpt.get('model', ckpt)
        model.load_state_dict(sd, strict=True)
    return model


def mae_vit_large_patch16_in1k(pretrained: bool = True, **kwargs):
    model = _vit_large16(num_classes=1000, **kwargs)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth",
            map_location="cpu"
        )
        sd = ckpt.get('model', ckpt)
        model.load_state_dict(sd, strict=True)
    return model


def mae_vit_huge_patch14_in1k(pretrained: bool = True, **kwargs):
    model = _vit_huge14(num_classes=1000, **kwargs)
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth",
            map_location="cpu"
        )
        sd = ckpt.get('model', ckpt)
        model.load_state_dict(sd, strict=True)
    return model


__all__ = [
    'mae_vit_base_patch16',
    'mae_vit_large_patch16',
    'mae_vit_huge_patch14',
    'mae_vit_base_patch16_in1k',
    'mae_vit_large_patch16_in1k',
    'mae_vit_huge_patch14_in1k',
]