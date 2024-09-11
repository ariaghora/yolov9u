from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    backbone: List[Any]
    head: List[Any]
    nc: int
    depth_multiple: float = 1.0
    width_multiple: float = 1.0
    ch: int = 3
    anchors: int = 3
    activation: Optional[str] = None
    inplace: bool = True


import contextlib  # TODO: WTF

from .blocks import (SPPELAN, ADown, CBFuse, CBLinear, Concat, Conv, DDetect,
                     RepConvN, RepNCSPELAN4, Silence, make_divisible)


def parse_model(config: ModelConfig, ch):  # model_dict, input_channels(3)
    # Parse a YOLO model based on some configuration
    anchors, nc, gd, gw, act = (
        config.anchors,
        config.nc,
        config.depth_multiple,
        config.width_multiple,
        config.activation,
    )
    if act:
        Conv.default_act = eval(act)  # TODO: make dict of class type instead of eval
        RepConvN.default_act = eval(act)
    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, skip_conn_indices, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(
        config.backbone + config.head
    ):  # from, number, module, args
        print(f"parsing {m}...")
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        if m in {
            Conv,
            ADown,
            RepNCSPELAN4,
            SPPELAN,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        # TODO: channel, gw, gd
        elif m in {
            DDetect,
            # Segment,
            # DSegment,
            # DualDSegment,
            # Panoptic,
        }:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            # TODO: handle the rest other than DDetect
        else:
            c2 = ch[f]

        m_ = m(*args)  # module

        # t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params

        # attach index, 'from' index, type, number params
        # TODO: this is problematic, as this setting arbitrary attribute to an object.
        # also, `type` is reserved keyword
        m_.i, m_.f, m_.np = (i, f, np)

        skip_conn_indices.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist: # TODO: WTF IS THIS
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(skip_conn_indices)


class BaseModel(nn.Module):
    # YOLO base model
    def forward(self, x):
        return self._forward_once(x)  # single-scale inference, train

    def _forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.skip_conn_indices else None)  # save output
        return x

    def _apply(self, fn, recurse: bool = True):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        m.stride = fn(m.stride)
        m.anchors = fn(m.anchors)
        m.strides = fn(m.strides)
        return self


class DetectionModel(BaseModel):
    # YOLO detection model
    def __init__(
        self, model_config: ModelConfig, ch=3, nc=None, anchors=None
    ):  # model, input channels, number of classes
        super().__init__()
        self.model_config = model_config
        self.model, self.skip_conn_indices = parse_model(
            self.model_config, ch=[ch]
        )  # model, savelist
        self.names = [str(i) for i in range(self.model_config.nc)]  # default names
        self.inplace = self.model_config.inplace

        # Build strides, anchors
        m: torch.nn.Module = self.model[-1]  # last module (Detect/DDetect/etc.)
        s = 256  # 2x min stride
        forward = lambda x: self.forward(x)

        m.stride = torch.tensor(
            [s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))]
        )
        self.stride = m.stride

    def forward(self, x: torch.Tensor, augment=False, profile=False, visualize=False):
        return self._forward_once(x)  # single-scale inference, train
