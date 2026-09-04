import os
import hmac
import hashlib
import time
import base64
import io
import re
from typing import List, Optional
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from rapidocr_onnxruntime import RapidOCR

SECRET_KEY = os.environ.get("SECRET_KEY", "pointcraft_secret_key_change_me_in_production")
MAX_TIMESTAMP_DIFF_SEC = 600

app = FastAPI(title="PointCraft Cloud OCR Server")

# High-Precision PC-Grade ONNX Engine (~90MB RAM, Full PC Accuracy)
ocr_engine = RapidOCR()

class TeamModel(BaseModel):
    id: str
    name: str
    players: List[str]

class ScanRequestModel(BaseModel):
    mode: str
    images: List[str]
    teams: Optional[List[TeamModel]] = []

def verify_hmac_signature(timestamp: str, body_bytes: bytes, signature: str):
    if not timestamp or not signature:
        raise HTTPException(status_code=401, detail="Missing X-Timestamp or X-Signature headers")
    try:
        req_time = int(timestamp)
        now = int(time.time() * 1000)
        if abs(now - req_time) > (MAX_TIMESTAMP_DIFF_SEC * 1000):
            raise HTTPException(status_code=401, detail="Request expired")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid X-Timestamp header")

    msg = f"{timestamp}:{body_bytes.decode('utf-8')}".encode('utf-8')
    computed_sig = hmac.new(SECRET_KEY.encode('utf-8'), msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(computed_sig, signature):
        raise HTTPException(status_code=403, detail="Invalid HMAC-SHA256 signature")

def base64_to_cv2(b64_str: str) -> np.ndarray:
    img_data = base64.b64decode(b64_str)
    image = Image.open(io.BytesIO(img_data)).convert('RGB')
    return np.array(image)

def normalize_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r'[^a-z0-9]', '', s.lower())

def levenshtein_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[0][j] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def is_name_match(ocr_name: str, roster_name: str) -> bool:
    a, b = normalize_text(ocr_name), normalize_text(roster_name)
    if not a or not b: return False
    if a == b: return True
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a): return True
    max_len = max(len(a), len(b))
    if max_len < 3: return False
    return levenshtein_distance(a, b) <= max(1, max_len // 4)

@app.post("/scan")
async def scan_endpoint(
    req_data: ScanRequestModel,
    request: Request,
    x_timestamp: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None)
):
    body_bytes = await request.body()
    verify_hmac_signature(x_timestamp, body_bytes, x_signature)

    matched_results = []
    unmatched_groups = []
    
    for img_b64 in req_data.images:
        cv_img = base64_to_cv2(img_b64)
        ocr_res, elapse = ocr_engine(cv_img)

        lines = []
        if ocr_res:
            for item in ocr_res:
                text = item[1]
                conf = float(item[2])
                if conf > 0.35 and text.strip():
                    lines.append(text.strip())

        current_placement = None
        current_kills = 0
        found_players = []

        for line in lines:
            m_rank = re.search(r'#?(\d{1,2})', line)
            if not current_placement and m_rank:
                val = int(m_rank.group(1))
                if 1 <= val <= 12:
                    current_placement = val

            m_elim = re.search(r'(\d{1,2})\s*(?:elim|kill|elimination)', line, re.IGNORECASE)
            if m_elim:
                current_kills += int(m_elim.group(1))
            else:
                if len(line) >= 2 and not line.isdigit():
                    found_players.append(line)

        matched_team = None
        best_overlap = 0.0

        if req_data.teams and found_players:
            for team in req_data.teams:
                common = 0
                for p_ocr in found_players:
                    for p_roster in team.players:
                        if is_name_match(p_ocr, p_roster):
                            common += 1
                            break
                overlap = common / max(1, len(found_players))
                if overlap > best_overlap:
                    best_overlap = overlap
                    matched_team = team

        if matched_team and best_overlap >= 0.25:
            matched_results.append({
                "teamId": matched_team.id,
                "teamName": matched_team.name,
                "placement": current_placement if current_placement else 1,
                "kills": current_kills
            })
        else:
            unmatched_groups.append({
                "placement": current_placement if current_placement else 1,
                "kills": current_kills,
                "players": found_players
            })

    return {
        "success": True,
        "matched": matched_results,
        "unmatchedCount": len(unmatched_groups),
        "unmatchedGroups": unmatched_groups,
        "usedFallback": False
    }

@app.get("/")
def health_check():
    return {"status": "ok", "service": "PointCraft Cloud OCR Server"}
