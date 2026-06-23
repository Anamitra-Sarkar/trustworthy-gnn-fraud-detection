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
        in_channels = FEATURE_DIMS.get(dataset, 166)

        backbone_cls = BACKBONE_REGISTRY[backbone_type]
        if is_edl:
            backbone = backbone_cls(in_channels=in_channels, hidden_channels=128, out_channels=128)
            model = EDLWrapper(backbone, num_classes=2)
        else:
            model = backbone_cls(in_channels=in_channels, hidden_channels=128, out_channels=2)

        try:
            filename = f"{model_name}.safetensors"
            path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=filename,
                token=HF_TOKEN or None,
                cache_dir=MODEL_CACHE_DIR,
            )
            state_dict = load_file(path)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load weights for {model_name}: {e}")

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
