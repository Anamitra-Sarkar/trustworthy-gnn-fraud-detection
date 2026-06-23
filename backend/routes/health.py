from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def healthcheck():
    return {"status": "healthy", "service": "trustworthy-gnn-fraud-detection"}
