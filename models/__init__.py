import os, torch
from typing import List, Tuple, Optional, Union, Dict

from .ebc import _ebc, EBC
from .clip_ebc import _clip_ebc, CLIP_EBC


def get_model(
    model_info_path: str,
    model_name: Optional[str] = None,
    block_size: Optional[int] = None,
    bins: Optional[List[Tuple[float, float]]] = None,
    bin_centers: Optional[List[float]] = None,
    zero_inflated: Optional[bool] = True,
    # parameters for CLIP_EBC
    clip_weight_name: Optional[str] = None,
    num_vpt: Optional[int] = None,
    vpt_drop: Optional[float] = None,
    input_size: Optional[int] = None,
    adapter: bool = False,
    adapter_reduction: Optional[int] = None,
    lora: bool = False,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    norm: str = "none",
    act: str = "none",
    text_prompts: Optional[List[str]] = None
) -> Union[EBC, CLIP_EBC]:
    if os.path.exists(model_info_path):
        model_info = torch.load(model_info_path, map_location="cpu", weights_only=False)        

        model_name = model_info["config"]["model_name"]
        block_size = model_info["config"]["block_size"]
        bins = model_info["config"]["bins"]
        bin_centers = model_info["config"]["bin_centers"]
        zero_inflated = model_info["config"]["zero_inflated"]

        clip_weight_name = model_info["config"].get("clip_weight_name", None)

        num_vpt = model_info["config"].get("num_vpt", None)
        vpt_drop = model_info["config"].get("vpt_drop", None)


        adapter = model_info["config"].get("adapter", False)
        adapter_reduction = model_info["config"].get("adapter_reduction", None)

        lora = model_info["config"].get("lora", False)
        lora_rank = model_info["config"].get("lora_rank", None)
        lora_alpha = model_info["config"].get("lora_alpha", None)
        lora_dropout = model_info["config"].get("lora_dropout", None)

        input_size = model_info["config"].get("input_size", None)
        text_prompts = model_info["config"].get("text_prompts", None)

        norm = model_info["config"].get("norm", "none")
        act = model_info["config"].get("act", "none")

        weights = model_info["weights"]

    else:
        assert model_name is not None, "model_name should be provided if model_info_path is not provided"
        assert block_size is not None, "block_size should be provided"
        assert bins is not None, "bins should be provided"
        assert bin_centers is not None, "bin_centers should be provided"
        weights = None

    if "ViT" in model_name:
        assert num_vpt is not None, f"num_vpt should be provided for ViT models, got {num_vpt}"
        assert vpt_drop is not None, f"vpt_drop should be provided for ViT models, got {vpt_drop}"

    if model_name.startswith("CLIP_") or model_name.startswith("CLIP-"):
        assert clip_weight_name is not None, f"clip_weight_name should be provided for CLIP models, got {clip_weight_name}"
        model = _clip_ebc(
            model_name=model_name[5:],
            weight_name=clip_weight_name,
            block_size=block_size,
            bins=bins,
            bin_centers=bin_centers,
            zero_inflated=zero_inflated,
            num_vpt=num_vpt,
            vpt_drop=vpt_drop,
            input_size=input_size,
            adapter=adapter,
            adapter_reduction=adapter_reduction,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            text_prompts=text_prompts,
            norm=norm,
            act=act
        )
        model_config = {
            "model_name": model_name,
            "block_size": block_size,
            "bins": bins,
            "bin_centers": bin_centers,
            "zero_inflated": zero_inflated,
            "clip_weight_name": clip_weight_name,
            "num_vpt": num_vpt,
            "vpt_drop": vpt_drop,
            "input_size": input_size,
            "adapter": adapter,
            "adapter_reduction": adapter_reduction,
            "lora": lora,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "text_prompts": model.text_prompts,
            "norm": norm,
            "act": act
        }
    
    else:
        assert not adapter, "adapter for non-CLIP models is not implemented yet"
        assert not lora, "lora for non-CLIP models is not implemented yet"
        model = _ebc(
            model_name=model_name,
            block_size=block_size,
            bins=bins,
            bin_centers=bin_centers,
            zero_inflated=zero_inflated,
            num_vpt=num_vpt,
            vpt_drop=vpt_drop,
            input_size=input_size,
            norm=norm,
            act=act
        )
        model_config = {
            "model_name": model_name,
            "block_size": block_size,
            "bins": bins,
            "bin_centers": bin_centers,
            "zero_inflated": zero_inflated,
            "num_vpt": num_vpt,
            "vpt_drop": vpt_drop,
            "input_size": input_size,
            "norm": norm,
            "act": act
        }

    model.config = model_config
    model_info = {"config": model_config, "weights": weights}

    if weights is not None:
        model.load_state_dict(weights)

    if not os.path.exists(model_info_path):
        torch.save(model_info, model_info_path)
    
    return model


__all__ = ["get_model"]
