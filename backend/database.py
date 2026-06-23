import datetime
from typing import Optional
from firebase_auth import get_db


def save_analysis(analysis_id: str, uid: str, data: dict) -> None:
    db = get_db()
    doc = {
        **data,
        "analysis_id": analysis_id,
        "uid": uid,
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    db.collection("analyses").document(analysis_id).set(doc)
    db.collection("users").document(uid).collection("history").document(analysis_id).set(
        {"analysis_id": analysis_id, "created_at": doc["created_at"]}
    )


def get_analysis(analysis_id: str) -> Optional[dict]:
    db = get_db()
    doc = db.collection("analyses").document(analysis_id).get()
    return doc.to_dict() if doc.exists else None


def save_escalation(escalation_id: str, uid: str, data: dict) -> None:
    db = get_db()
    doc = {
        **data,
        "escalation_id": escalation_id,
        "uid": uid,
        "status": "open",
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    db.collection("escalations").document(escalation_id).set(doc)


def get_escalations(uid: str, status: Optional[str] = None) -> list:
    db = get_db()
    query = db.collection("escalations").where("uid", "==", uid)
    if status:
        query = query.where("status", "==", status)
    return [doc.to_dict() for doc in query.stream()]


def update_escalation_status(escalation_id: str, status: str) -> None:
    db = get_db()
    db.collection("escalations").document(escalation_id).update({"status": status})


def get_user_analyses(uid: str, limit: int = 20) -> list:
    db = get_db()
    refs = (
        db.collection("users")
        .document(uid)
        .collection("history")
        .order_by("created_at", direction="DESCENDING")
        .limit(limit)
        .stream()
    )
    analysis_ids = [ref.to_dict().get("analysis_id") for ref in refs]
    results = []
    for aid in analysis_ids:
        data = get_analysis(aid)
        if data:
            results.append(data)
    return results
