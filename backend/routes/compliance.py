import uuid
from fastapi import APIRouter, Header, HTTPException
from firebase_auth import verify_token
from database import get_analysis, save_escalation, get_escalations, update_escalation_status
from compliance.schemas import EscalationRequest
from compliance.agent import ComplianceAgent

router = APIRouter(prefix="/api/compliance")

_agent = None


def get_agent() -> ComplianceAgent:
    global _agent
    if _agent is None:
        _agent = ComplianceAgent()
    return _agent


@router.post("/escalate")
async def escalate(request: EscalationRequest, authorization: str = Header(...)):
    user = await verify_token(authorization)
    uid = user["uid"]

    analysis = get_analysis(request.analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    agent = get_agent()
    reports = []

    if "batch_results" in analysis:
        nodes = [r for r in analysis["batch_results"] if r["node_id"] in request.node_ids]
    else:
        nodes = [analysis]

    for node_data in nodes:
        payload = {
            "node_id": node_data.get("node_id", 0),
            "predictions": [1 - node_data.get("risk_score", 0.5), node_data.get("risk_score", 0.5)],
            "conformal_set": node_data.get("uncertainty", {}).get("conformal", {}).get("conformal_prediction_set", [0]),
            "evidential_evidence": [
                node_data.get("uncertainty", {}).get("evidential", {}).get("licit_evidence", 0),
                node_data.get("uncertainty", {}).get("evidential", {}).get("fraud_evidence", 0),
            ],
            "neighborhood_heterophily": 1.0 - node_data.get("uncertainty", {}).get("evidential", {}).get("vacuity", 0.5),
            "node_degree": node_data.get("degree", 10),
            "quantile_cutoff": node_data.get("uncertainty", {}).get("conformal", {}).get("quantile_threshold", 0.342),
        }
        report = agent.generate_report(payload)
        reports.append(report.model_dump())

    escalation_id = str(uuid.uuid4())
    save_escalation(escalation_id, uid, {
        "analysis_id": request.analysis_id,
        "node_ids": request.node_ids,
        "reason": request.reason,
        "reports": reports,
    })

    return {"escalation_id": escalation_id, "reports": reports}


@router.get("/escalations")
async def list_escalations(status: str = None, authorization: str = Header(...)):
    user = await verify_token(authorization)
    return {"escalations": get_escalations(user["uid"], status=status)}


@router.patch("/escalations/{escalation_id}")
async def update_escalation(escalation_id: str, status: str, authorization: str = Header(...)):
    await verify_token(authorization)
    update_escalation_status(escalation_id, status)
    return {"status": "updated"}
