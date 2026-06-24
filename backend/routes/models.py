from collections import defaultdict

from fastapi import APIRouter, Header

from firebase_auth import verify_token

router = APIRouter(prefix="/api")


@router.get("/models/summary")
async def model_summary(authorization: str = Header(...)):
    await verify_token(authorization)

    from app import model_loader

    config = model_loader.load_config()
    calibration = model_loader.load_calibration()

    models = []
    backbone_totals = defaultdict(lambda: {"count": 0, "f1": 0.0, "auc": 0.0})

    for model_name, info in config.items():
        metrics = info.get("metrics", {}) or {}
        if not metrics:
            continue

        f1 = float(metrics.get("f1_macro", 0.0))
        auc = float(metrics.get("auc_roc", 0.0))
        backbone = info.get("backbone", "unknown")
        entry = {
            "model_name": model_name,
            "backbone": backbone,
            "dataset": info.get("dataset", "elliptic"),
            "topology": info.get("topology", "original"),
            "edl": bool(info.get("edl", False)),
            "feature_dim": info.get("feature_dim"),
            "f1": f1,
            "auc": auc,
            "precision_fraud": float(metrics.get("precision_fraud", 0.0)),
            "recall_fraud": float(metrics.get("recall_fraud", 0.0)),
        }
        models.append(entry)

        bucket = backbone_totals[backbone]
        bucket["count"] += 1
        bucket["f1"] += f1
        bucket["auc"] += auc

    model_performance = []
    for backbone, bucket in backbone_totals.items():
        count = max(bucket["count"], 1)
        model_performance.append(
            {
                "backbone": backbone.upper(),
                "f1": round(bucket["f1"] / count, 4),
                "auc": round(bucket["auc"] / count, 4),
            }
        )
    model_performance.sort(key=lambda item: item["f1"], reverse=True)

    models.sort(key=lambda item: item["f1"], reverse=True)
    best_model = models[0] if models else None
    calibration_entries = len(calibration) if isinstance(calibration, dict) else 0

    return {
        "total_models": len(models),
        "calibration_entries": calibration_entries,
        "best_model": best_model,
        "model_performance": model_performance,
        "models": models,
    }
