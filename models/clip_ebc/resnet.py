from torch import nn, Tensor
import open_clip
from peft import get_peft_model, LoraConfig

from ..utils import ConvRefine, ConvUpsample, ConvAdapter
from ..utils import _get_norm_layer, _get_activation


resnet_names_and_weights = {
    "RN50": ["openai", "yfcc15m", "cc12m"],
    "RN101": ["openai", "yfcc15m", "cc12m"],
    "RN50x4": ["openai", "yfcc15m", "cc12m"],
    "RN50x16": ["openai", "yfcc15m", "cc12m"],
    "RN50x64": ["openai", "yfcc15m", "cc12m"],
}

refiner_channels = {
    "RN50": 2048,
    "RN101": 2048,
    "RN50x4": 2560,
    "RN50x16": 3072,
    "RN50x64": 4096,
}

refiner_groups = {
    "RN50": refiner_channels["RN50"] // 512,  # 4
    "RN101": refiner_channels["RN101"] // 512, # 4
    "RN50x4": refiner_channels["RN50x4"] // 512, # 5
    "RN50x16": refiner_channels["RN50x16"] // 512, # 6
    "RN50x64": refiner_channels["RN50x64"] // 512, # 8
}


class ResNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        weight_name: str,
        block_size: int = 16,
        adapter: bool = False,
        adapter_reduction: int = 4,
        norm: str = "none",
        act: str = "none"
    ) -> None:
        super(ResNet, self).__init__()
        assert model_name in resnet_names_and_weights, f"Model name should be one of {list(resnet_names_and_weights.keys())}, but got {model_name}."
        assert weight_name in resnet_names_and_weights[model_name], f"Pretrained should be one of {resnet_names_and_weights[model_name]}, but got {weight_name}."
        assert block_size in [32, 16, 8], f"block_size should be one of [32, 16, 8], got {block_size}"
        self.model_name, self.weight_name = model_name, weight_name
        self.block_size = block_size

        model = open_clip.create_model_from_pretrained(model_name, weight_name, return_transform=False).visual

        self.adapter = adapter
        if adapter:
            for param in model.parameters():
                param.requires_grad = False
        
        # Stem
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.conv2 = model.conv2
        self.bn2 = model.bn2
        self.act2 = model.act2
        self.conv3 = model.conv3
        self.bn3 = model.bn3
        self.act3 = model.act3
        self.avgpool = model.avgpool
        # Stem: reduction = 4

        # Layers
        for idx in range(1, 5):
            setattr(self, f"layer{idx}", getattr(model, f"layer{idx}"))
            if adapter:
                setattr(self, f"adapter{idx}", ConvAdapter(
                    in_channels=getattr(model, f"layer{idx}")[-1].conv3.out_channels,
                    bottleneck_channels=getattr(model, f"layer{idx}")[-1].conv3.out_channels // adapter_reduction,
                ) if idx < 4 else nn.Identity())  # No adapter for the last layer

        self.in_features = model.attnpool.c_proj.weight.shape[1]
        self.out_features = model.attnpool.c_proj.weight.shape[0]

        if norm == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm == "ln":
            norm_layer = nn.LayerNorm
        else:
            norm_layer = _get_norm_layer(model)

        if act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act == "gelu":
            activation = nn.GELU()
        else:
            activation = _get_activation(model)

        if block_size == 32:
            self.refiner = ConvRefine(
                in_channels=self.in_features,
                out_channels=self.in_features,
                norm_layer=norm_layer,
                activation=activation,
                groups=refiner_groups[self.model_name],
            )
        elif block_size == 16:
            self.refiner = ConvUpsample(
                in_channels=self.in_features,
                out_channels=self.in_features,
                norm_layer=norm_layer,
                activation=activation,
                groups=refiner_groups[self.model_name],
            )
        else:  # block_size == 8
            self.refiner = nn.Sequential(
                ConvUpsample(
                    in_channels=self.in_features,
                    out_channels=self.in_features,
                    norm_layer=norm_layer,
                    activation=activation,
                    groups=refiner_groups[self.model_name],
                ),
                ConvUpsample(
                    in_channels=self.in_features,
                    out_channels=self.in_features,
                    norm_layer=norm_layer,
                    activation=activation,
                    groups=refiner_groups[self.model_name],
                ),
            )
    
    def train(self, mode: bool = True):
        if self.adapter and mode:
            # training:
            self.conv1.eval()
            self.bn1.eval()
            self.act1.eval()
            self.conv2.eval()
            self.bn2.eval()
            self.act2.eval()
            self.conv3.eval()
            self.bn3.eval()
            self.act3.eval()
            self.avgpool.eval()

            for idx in range(1, 5):
                getattr(self, f"layer{idx}").eval()
                getattr(self, f"adapter{idx}").train()

            self.refiner.train()

        else:
            # evaluation:
            for module in self.children():
                module.train(mode)

    def stem(self, x: Tensor) -> Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        
        x = self.layer1(x)
        if self.adapter:
            x = self.adapter1(x)
        
        x = self.layer2(x)
        if self.adapter:
            x = self.adapter2(x)

        x = self.layer3(x)
        if self.adapter:
            x = self.adapter3(x)
        
        x = self.layer4(x)
        if self.adapter:
            x = self.adapter4(x)
    
        x = self.refiner(x)
        return x


def _resnet(
    model_name: str,
    weight_name: str,
    block_size: int = 16,
    adapter: bool = False,
    adapter_reduction: int = 4,
    lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.1,
    norm: str = "none",
    act: str = "none"
) -> ResNet:
    assert not (lora and adapter), "Lora and adapter cannot be used together."
    model = ResNet(
        model_name=model_name,
        weight_name=weight_name,
        block_size=block_size,
        adapter=adapter,
        adapter_reduction=adapter_reduction,
        norm=norm,
        act=act
    )

    if lora:
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                target_modules.append(name)
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

        # Unfreeze BN layers
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and "refiner" not in name:
                module.requires_grad_(True)

        # Unfreeze refiner
        for name, module in model.named_modules():
            if "refiner" in name:
                module.requires_grad_(True)
    
    return model
