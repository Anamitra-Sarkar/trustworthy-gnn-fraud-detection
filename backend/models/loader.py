import os
import json
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from config import HF_MODEL_REPO, HF_TOKEN, MODEL_CACHE_DIR
from .backbones import GraphSAGEBackbone, GATBackbone, GCNBackbone, EDLWrapper

BACKBONE_REGISTRY = {
    "graphsage": GraphSAGEBackbone,
    "gat": GATBackbone,
    "gcn": GCNBackbone,
}

FEATURE_DIMS = {
    "elliptic": 166,
    "dgraph": 17,
    "amazon": 25,
}


class ModelLoader:
    def __init__(self):
        self.models = {}
        self.config = None
        self.calibration_data = None

    def load_config(self) -> dict:
        if self.config is not None:
            return self.config
        try:
            path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename="model_config.json",
                token=HF_TOKEN or None,
                cache_dir=MODEL_CACHE_DIR,
            )
            with open(path) as f:
                self.config = json.load(f)
        except Exception:
            self.config = self._default_config()
        return self.config

    def load_calibration(self) -> dict:
        if self.calibration_data is not None:
            return self.calibration_data
        try:
            path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename="conformal_calibration.json",
                token=HF_TOKEN or None,
                cache_dir=MODEL_CACHE_DIR,
            )
            with open(path) as f:
                self.calibration_data = json.load(f)
        except Exception:
            self.calibration_data = {}
        return self.calibration_data

    def load_model(self, model_name: str) -> torch.nn.Module:
        if model_name in self.models:
            return self.models[model_name]

        config = self.load_config()
        if model_name not in config:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(config.keys())}")

        model_info = config[model_name]
        backbone_type = model_info["backbone"]
        dataset = model_info["dataset"]
        is_edl = model_info.get("edl", False)

        filename = f"{model_name}.safetensors"
        path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=filename,
            token=HF_TOKEN or None,
            cache_dir=MODEL_CACHE_DIR,
        )
        state_dict = load_file(path)

        in_channels = self._resolve_feature_dim(model_info, state_dict, backbone_type, is_edl)
        backbone_cls = BACKBONE_REGISTRY[backbone_type]
        if is_edl:
            backbone = backbone_cls(in_channels=in_channels, hidden_channels=128, out_channels=128)
            model = EDLWrapper(backbone, num_classes=2)
        else:
            model = backbone_cls(in_channels=in_channels, hidden_channels=128, out_channels=2)

        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name} with feature_dim={in_channels}: {e}") from e

        model.eval()
        self.models[model_name] = model
        return model

    def list_models(self) -> list:
        config = self.load_config()
        return list(config.keys())

    def _default_config(self) -> dict:
        configs = {}
        for dataset in ["elliptic", "dgraph", "amazon"]:
            for backbone in ["graphsage", "gat", "gcn"]:
                for topology in ["original", "similarity", "knn", "temporal", "augmented"]:
                    name = f"{backbone}_{topology}_{dataset}"
                    configs[name] = {
                        "backbone": backbone,
                        "dataset": dataset,
                        "topology": topology,
                        "edl": False,
                    }
                    edl_name = f"{backbone}_{topology}_{dataset}_edl"
                    configs[edl_name] = {
                        "backbone": backbone,
                        "dataset": dataset,
                        "topology": topology,
                        "edl": True,
                    }
        return configs

    def _resolve_feature_dim(self, model_info: dict, state_dict: dict, backbone_type: str, is_edl: bool) -> int:
        feature_dim = model_info.get("feature_dim")
        if isinstance(feature_dim, int) and feature_dim > 0:
            return feature_dim

        dataset = model_info.get("dataset")
        if dataset in FEATURE_DIMS:
            default_dim = FEATURE_DIMS[dataset]
        else:
            default_dim = 166

        prefixes = [""]
        if is_edl:
            prefixes = ["backbone."]

        key_candidates = {
            "graphsage": ["convs.0.lin_l.weight", "convs.0.lin_r.weight"],
            "gat": ["convs.0.lin_src.weight", "convs.0.lin_dst.weight"],
            "gcn": ["convs.0.lin.weight"],
        }.get(backbone_type, [])

        for prefix in prefixes:
            for suffix in key_candidates:
                key = f"{prefix}{suffix}"
                tensor = state_dict.get(key)
                if tensor is not None and getattr(tensor, "ndim", 0) == 2:
                    return int(tensor.shape[1])

        # Fallback: infer from the first non-output linear weight.
        for key, tensor in state_dict.items():
            if not key.endswith(".weight") or getattr(tensor, "ndim", 0) != 2:
                continue
            if tensor.shape[1] > 0 and tensor.shape[1] != tensor.shape[0]:
                return int(tensor.shape[1])

        return default_dim
