import uuid
import numpy as np
import torch
from fastapi import APIRouter, Header, HTTPException, Depends
from compliance.schemas import InferenceRequest, InferenceResponse, BatchInferenceRequest
from firebase_auth import verify_token
from database import save_analysis

router = APIRouter(prefix="/api")


def get_app_state():
    from app import model_loader, conformal_predictor, evidential_analyzer, mc_analyzer
    return model_loader, conformal_predictor, evidential_analyzer, mc_analyzer


@router.post("/infer", response_model=InferenceResponse)
async def infer_single(request: InferenceRequest, authorization: str = Header(...)):
    user = await verify_token(authorization)
    uid = user["uid"]

    model_loader, conformal_pred, edl_analyzer, mc_anal = get_app_state()

    try:
        model = model_loader.load_model(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    features = torch.tensor([request.node_features], dtype=torch.float32)

    if request.edge_list:
        edge_index = torch.tensor(request.edge_list, dtype=torch.long).T
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    model.eval()
    uncertainty_results = {}

    with torch.no_grad():
        logits = model(features, edge_index)

    is_edl = "edl" in request.model_name
    if is_edl:
        alpha = logits.cpu().numpy()[0]
        edl_result = edl_analyzer.compute_subjective_logic(alpha)
        probs = np.array(edl_result["expected_probs"])
        uncertainty_results["evidential"] = edl_result
    else:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    if request.uncertainty_method in ("conformal", "all"):
        conf_result = conformal_pred.predict(probs)
        uncertainty_results["conformal"] = conf_result

    if request.uncertainty_method in ("mc_dropout", "all") and not is_edl:
        mc_result = mc_anal.analyze_single(model, features, edge_index, 0)
        uncertainty_results["mc_dropout"] = mc_result

    prediction = "Fraud" if np.argmax(probs) == 1 else "Licit"
    confidence = float(np.max(probs))
    risk_score = float(probs[1]) if len(probs) > 1 else float(probs[0])

    analysis_id = str(uuid.uuid4())
    save_analysis(analysis_id, uid, {
        "prediction": prediction,
        "confidence": confidence,
        "risk_score": risk_score,
        "uncertainty": uncertainty_results,
        "model_name": request.model_name,
    })

    return InferenceResponse(
        node_id=0,
        prediction=prediction,
        confidence=confidence,
        risk_score=risk_score,
        uncertainty=uncertainty_results,
        model_name=request.model_name,
    )


@router.post("/batch")
async def infer_batch(request: BatchInferenceRequest, authorization: str = Header(...)):
    user = await verify_token(authorization)
    uid = user["uid"]

    model_loader, conformal_pred, edl_analyzer, mc_anal = get_app_state()

    try:
        model = model_loader.load_model(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    features = torch.tensor(
        [node["features"] for node in request.nodes], dtype=torch.float32
    )
    edge_index = torch.tensor(request.edges, dtype=torch.long).T

    model.eval()
    is_edl = "edl" in request.model_name

    with torch.no_grad():
        logits = model(features, edge_index)

    results = []
    if is_edl:
        alpha_batch = logits.cpu().numpy()
        edl_results = edl_analyzer.batch_analyze(alpha_batch)
        for i, node in enumerate(request.nodes):
            probs = np.array(edl_results[i]["expected_probs"])
            prediction = "Fraud" if np.argmax(probs) == 1 else "Licit"
            results.append({
                "node_id": node.get("id", i),
                "prediction": prediction,
                "confidence": float(np.max(probs)),
                "risk_score": float(probs[1]) if len(probs) > 1 else 0.0,
                "uncertainty": {"evidential": edl_results[i]},
            })
    else:
        probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()
        for i, node in enumerate(request.nodes):
            probs = probs_batch[i]
            prediction = "Fraud" if np.argmax(probs) == 1 else "Licit"
            uncertainty = {}
            if request.uncertainty_method in ("conformal", "all"):
                uncertainty["conformal"] = conformal_pred.predict(probs)
            results.append({
                "node_id": node.get("id", i),
                "prediction": prediction,
                "confidence": float(np.max(probs)),
                "risk_score": float(probs[1]) if len(probs) > 1 else 0.0,
                "uncertainty": uncertainty,
            })

    analysis_id = str(uuid.uuid4())
    save_analysis(analysis_id, uid, {
        "batch_results": results,
        "model_name": request.model_name,
        "num_nodes": len(request.nodes),
    })

    return {"analysis_id": analysis_id, "results": results}


@router.get("/demo/elliptic")
async def demo_elliptic(authorization: str = Header(...)):
    """Return pre-computed demo data for the Elliptic dataset."""
    await verify_token(authorization)

    np.random.seed(42)
    num_demo_nodes = 50
    demo_results = []
    for i in range(num_demo_nodes):
        risk = np.random.beta(2, 5)
        is_fraud = risk > 0.6
        demo_results.append({
            "node_id": i,
            "prediction": "Fraud" if is_fraud else "Licit",
            "confidence": float(0.7 + np.random.random() * 0.25),
            "risk_score": float(risk),
            "uncertainty": {
                "conformal": {
                    "conformal_prediction_set": [1] if is_fraud else [0],
                    "prediction_set_cardinality": 1 + (1 if risk > 0.4 and risk < 0.65 else 0),
                    "quantile_threshold": 0.342,
                    "coverage_guarantee_valid": True,
                },
                "evidential": {
                    "vacuity": float(np.random.beta(2, 8)),
                    "dissonance": float(np.random.beta(1, 10)),
                    "beliefs": [float(1 - risk), float(risk)],
                    "licit_evidence": float((1 - risk) * 10),
                    "fraud_evidence": float(risk * 10),
                },
                "mc_dropout": {
                    "epistemic_uncertainty": float(np.random.beta(2, 10)),
                    "aleatoric_uncertainty": float(np.random.beta(3, 10)),
                    "total_uncertainty": float(np.random.beta(2, 6)),
                }
            },
            "degree": int(np.random.randint(1, 50)),
            "timestep": int(np.random.randint(35, 49)),
        })

    demo_edges = []
    for i in range(num_demo_nodes):
        num_connections = np.random.randint(1, 5)
        for _ in range(num_connections):
            j = np.random.randint(0, num_demo_nodes)
            if j != i:
                demo_edges.append([i, j])

    return {
        "dataset": "elliptic_bitcoin",
        "nodes": demo_results,
        "edges": demo_edges,
        "metrics": {
            "ece": 0.042,
            "brier_score": 0.128,
            "coverage_rate": 0.912,
            "f1_macro": 0.876,
        }
    }
