import base64
import json
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="google")

import firebase_admin
from firebase_admin import credentials, auth, firestore

_app = None
_db = None


def init_firebase():
    global _app, _db
    if _app is not None:
        return

    key_json = os.getenv("FIREBASE_KEY_JSON", "")
    if key_json:
        cred_dict = json.loads(base64.b64decode(key_json))
        cred = credentials.Certificate(cred_dict)
    elif os.path.exists("firebase-key.json"):
        cred = credentials.Certificate("firebase-key.json")
    else:
        raise RuntimeError("No Firebase credentials found")

    _app = firebase_admin.initialize_app(cred)
    _db = firestore.client()


def get_db():
    if _db is None:
        init_firebase()
    return _db


async def verify_token(authorization: str) -> dict:
    if not authorization.startswith("Bearer "):
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[7:]
    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid or expired token")
