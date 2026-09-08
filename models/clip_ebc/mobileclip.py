from torch import nn, Tensor
import open_clip
from peft import get_peft_model, LoraConfig 

from ..utils import ConvRefine, ConvUpsample, ConvAdapter
from ..utils import _get_norm_layer, _get_activation


mobileclip_names_and_weights = {
    "MobileCLIP-S1": ["datacompdr"],
    "MobileCLIP-S2": ["datacompdr"],
}


refiner_channels = {
    "MobileCLIP-S1": 1024,
    "MobileCLIP-S2": 1280,
}

refiner_groups = {
    "MobileCLIP-S1": 2,
    "MobileCLIP-S2": 2,
}


class MobileCLIP(nn.Module):
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
        super().__init__()
        assert model_name in mobileclip_names_and_weights, f"Model name should be one of {list(mobileclip_names_and_weights.keys())}, but got {model_name}."
        assert weight_name in mobileclip_names_and_weights[model_name], f"Pretrained should be one of {mobileclip_names_and_weights[model_name]}, but got {weight_name}."
        assert block_size in [32, 16, 8], f"block_size should be one of [32, 16, 8], got {block_size}"
        self.model_name, self.weight_name = model_name, weight_name
        self.block_size = block_size

        model = open_clip.create_model_from_pretrained(model_name, weight_name, return_transform=False).visual

        self.adapter = adapter
        if adapter:
            for param in model.parameters():
                param.requires_grad = False

        self.stem = model.trunk.stem
        self.stages = model.trunk.stages

        self.depth = len(model.trunk.stages)
        for idx, stage in enumerate(model.trunk.stages):
            if adapter:
                setattr(self, f"adapter{idx}", ConvAdapter(
                    in_channels=stage.blocks[-1].mlp.fc2.out_channels,
                    bottleneck_channels=stage.blocks[-1].mlp.fc2.out_channels // adapter_reduction,
                ))

        self.final_conv = model.trunk.final_conv

        self.in_features, self.out_features = model.trunk.head.fc.in_features, model.trunk.head.fc.out_features

        # refine_block = LightConvRefine if model_name == "MobileCLIP-S1" else ConvRefine
        # upsample_block = LightConvUpsample if model_name == "MobileCLIP-S1" else ConvUpsample

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
                groups=refiner_groups[model_name],
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
            self.stem.eval()
    
            for idx in range(self.depth):
                getattr(self, f"stage{idx}").eval()
                getattr(self, f"adapter{idx}").train()

            self.final_conv.eval()
            self.refiner.train()

        else:
            # evaluation:
            for module in self.children():
                module.train(mode)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        
        for idx in range(self.depth):
            x = self.stages[idx](x)
            if self.adapter:
                x = getattr(self, f"adapter{idx}")(x)
        
        x = self.final_conv(x)

        x = self.refiner(x)
        return x


def _mobileclip(
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
) -> MobileCLIP:
    assert not (lora and adapter), "Lora and adapter cannot be used together."
    model = MobileCLIP(
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

        # Unfreeze the BN layers
        for name, module in model.named_modules() and "refiner" not in name:
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(True)
        
        # Unfreeze refiner
        for name, module in model.named_modules():
            if "refiner" in name:
                module.requires_grad_(True)
    
    return model