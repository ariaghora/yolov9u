import torch
import torch.nn as nn

from .blocks import (SPPELAN, ADown, CBFuse, CBLinear, Concat, Conv, DDetect,
                     RepConvN, RepNCSPELAN4, Silence)
from .config import ModelConfig

str_to_layer_type_dict = {
    "SPPELAN": SPPELAN,
    "ADown": ADown,
    "CBFuse": CBFuse,
    "CBLinear": CBLinear,
    "Concat": Concat,
    "Conv": Conv,
    "DDetect": DDetect,
    "RepConvN": RepConvN,
    "RepNCSPELAN4": RepNCSPELAN4,
    "Silence": Silence,
    "Upsample": nn.Upsample,
}


def parse_model(config: ModelConfig, input_channel_count: int):
    # initial input channels. This will be extended with each module's input channel
    # in the model.
    input_channels = [input_channel_count]

    # Parse a YOLO model based on some configuration
    act = config.activation
    if act:
        Conv.default_act = eval(act)  # TODO: make dict of class type instead of eval
        RepConvN.default_act = eval(act)

    layers, skip_conn_indices, out_ch = [], [], input_channel_count

    for i, (sources, module_type_str, args) in enumerate(
        config.backbone + config.head
    ):  # from, number, module, args
        ModuleType = str_to_layer_type_dict[module_type_str]

        if ModuleType in {
            Conv,
            ADown,
            RepNCSPELAN4,
            SPPELAN,
        }:
            inc_ch, out_ch = input_channels[sources], args[0]
            args = [inc_ch, out_ch, *args[1:]]
        elif ModuleType is Concat:
            out_ch = sum(input_channels[x] for x in sources)
        elif ModuleType is CBLinear:
            out_ch = args[0]
            inc_ch = input_channels[sources]
            args = [inc_ch, out_ch, *args[1:]]
        elif ModuleType is CBFuse:
            out_ch = input_channels[sources[-1]]
        elif ModuleType is DDetect:
            args.append([input_channels[x] for x in sources])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(sources)
        else:
            out_ch = input_channels[sources]

        module = ModuleType(*args)  # instantiate module

        # attach index, 'from' index, type, number params
        # TODO: this is problematic, as this setting arbitrary attribute to an object.
        # also, `type` is reserved keyword
        module.i, module.f = i, sources

        skip_conn_indices.extend(
            x % i
            for x in ([sources] if isinstance(sources, int) else sources)
            if x != -1
        )
        layers.append(module)

        if i > 0:
            input_channels.append(out_ch)
    return nn.Sequential(*layers), sorted(skip_conn_indices)


class BaseModel(nn.Module):
    # YOLO base model
    def forward(self, x):
        return self._forward_once(x)  # single-scale inference, train

    def _forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:  # if it has no previous layers...
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.skip_conn_indices else None)  # save output
        return x


class YOLODetector(BaseModel):
    # YOLO detection model
    def __init__(
        self, model_config: ModelConfig, input_channel_count: int = 3
    ):  # model, input channels, number of classes
        super().__init__()
        self.model_config = model_config
        self.inplace = self.model_config.inplace

        self.model, self.skip_conn_indices = parse_model(
            self.model_config, input_channel_count=input_channel_count
        )

        # Build strides, anchors
        m: torch.nn.Module = self.model[-1]  # last module (Detect/DDetect/etc.)
        s = 256  # 2x min stride
        forward = lambda x: self.forward(x)

        m.stride = torch.tensor([
            s / x.shape[-2] for x in forward(torch.zeros(1, input_channel_count, s, s))
        ])
        self.stride = m.stride

    def forward(self, x: torch.Tensor, augment=False, profile=False, visualize=False):
        return self._forward_once(x)  # single-scale inference, train
