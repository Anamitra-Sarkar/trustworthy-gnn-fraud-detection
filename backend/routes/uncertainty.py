from fastapi import APIRouter, Header, HTTPException
from firebase_auth import verify_token
from database import get_analysis, get_user_analyses

router = APIRouter(prefix="/api")


@router.get("/uncertainty-report/{analysis_id}")
async def get_uncertainty_report(analysis_id: str, authorization: str = Header(...)):
    user = await verify_token(authorization)
    data = get_analysis(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if data.get("uid") != user["uid"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return data


@router.get("/analyses")
async def list_analyses(limit: int = 20, authorization: str = Header(...)):
    user = await verify_token(authorization)
    return {"analyses": get_user_analyses(user["uid"], limit=limit)}
