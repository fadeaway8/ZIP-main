from torch import nn, Tensor
import open_clip
from peft import get_peft_model, LoraConfig 

from ..utils import ConvRefine, ConvAdapter
from ..utils import ConvUpsample, _get_norm_layer, _get_activation


convnext_names_and_weights = {
    "convnext_base": ["laion400m_s13b_b51k"],  # 107.49M
    "convnext_base_w": ["laion2b_s13b_b82k", "laion2b_s13b_b82k_augreg", "laion_aesthetic_s13b_b82k"],  # 107.75M
    "convnext_base_w_320": ["laion_aesthetic_s13b_b82k", "laion_aesthetic_s13b_b82k_augreg"],  # 107.75M
    "convnext_large_d": ["laion2b_s26b_b102k_augreg"],  # 217.46M
    "convnext_large_d_320": ["laion2b_s29b_b131k_ft", "laion2b_s29b_b131k_ft_soup"],  # 217.46M
    "convnext_xxlarge": ["laion2b_s34b_b82k_augreg", "laion2b_s34b_b82k_augreg_rewind", "laion2b_s34b_b82k_augreg_soup"]  # 896.88M
}

refiner_channels = {
    "convnext_base": 1024,
    "convnext_base_w": 1024,
    "convnext_base_w_320": 1024,
    "convnext_large_d": 1536,
    "convnext_large_d_320": 1536,
    "convnext_xxlarge": 3072,
}

refiner_groups = {
    "convnext_base": 1,
    "convnext_base_w": 1,
    "convnext_base_w_320": 1,
    "convnext_large_d": refiner_channels["convnext_large_d"] // 512,  # 3
    "convnext_large_d_320": refiner_channels["convnext_large_d_320"] // 512,  # 3
    "convnext_xxlarge": refiner_channels["convnext_xxlarge"] // 512,  # 6
}



class ConvNeXt(nn.Module):
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
        super(ConvNeXt, self).__init__()
        assert model_name in convnext_names_and_weights, f"Model name should be one of {list(convnext_names_and_weights.keys())}, but got {model_name}."
        assert weight_name in convnext_names_and_weights[model_name], f"Pretrained should be one of {convnext_names_and_weights[model_name]}, but got {weight_name}."
        assert block_size in [32, 16, 8], f"block_size should be one of [32, 16, 8], got {block_size}"
        self.model_name, self.weight_name = model_name, weight_name
        self.block_size = block_size

        model = open_clip.create_model_from_pretrained(model_name, weight_name, return_transform=False).visual

        self.adapter = adapter
        if adapter:
            self.adapter_reduction = adapter_reduction
            for param in model.parameters():
                param.requires_grad = False
        
        self.stem = model.trunk.stem
        self.depth = len(model.trunk.stages)
        for idx, stage in enumerate(model.trunk.stages):
            setattr(self, f"stage{idx}", stage)
            if adapter:
                setattr(self, f"adapter{idx}", ConvAdapter(
                    in_channels=stage.blocks[-1].mlp.fc2.out_features,
                    bottleneck_channels=stage.blocks[-1].mlp.fc2.out_features // adapter_reduction,
                ) if idx < self.depth - 1 else nn.Identity())  # No adapter for the last stage

        if self.model_name in ["convnext_base", "convnext_base_w", "convnext_base_w_320", "convnext_xxlarge"]:
            self.in_features, self.out_features = model.head.proj.in_features, model.head.proj.out_features
        else:  # "convnext_large_d", "convnext_large_d_320":
            self.in_features, self.out_features = model.head.mlp.fc1.in_features, model.head.mlp.fc2.out_features

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
            self.stem.eval()
    
            for idx in range(self.depth):
                getattr(self, f"stage{idx}").eval()
                getattr(self, f"adapter{idx}").train()

            self.refiner.train()

        else:
            # evaluation:
            for module in self.children():
                module.train(mode)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        for idx in range(self.depth):
            x = getattr(self, f"stage{idx}")(x)
            if self.adapter:
                x = getattr(self, f"adapter{idx}")(x)

        x = self.refiner(x)
        return x


def _convnext(
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
) -> ConvNeXt:
    assert not (lora and adapter), "Lora and adapter cannot be used together."
    model = ConvNeXt(
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
            if isinstance(module, (nn.Linear, nn.Conv2d)) and "refiner" not in name:
                target_modules.append(name)
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

        # Unfreeze refiner
        for name, module in model.named_modules():
            if "refiner" in name:
                module.requires_grad_(True)
    
    return model