from fastapi import APIRouter, Request, Response, HTTPException
import os
import secrets
import time
import threading
from urllib.parse import urlencode
import requests

router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)

# In-memory session store (simple, for local dev). Production: use Redis or DB.
_SESSIONS = {}  # session_id -> {created, expires_at, status, tokens}
_LOCK = threading.Lock()

COGNITO_DOMAIN = os.getenv("COGNITO_DOMAIN_URL") or os.getenv("COGNITO_DOMAIN")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
# 여러 클라이언트의 시크릿을 지원하기 위한 맵 (형식: id1:secret1,id2:secret2)
CLIENT_SECRETS_RAW = os.getenv("CLIENT_SECRETS")
CLIENT_SECRETS_MAP = {}
if CLIENT_SECRETS_RAW:
    for pair in CLIENT_SECRETS_RAW.split(","):
        if ":" in pair:
            cid, secret = pair.split(":", 1)
            CLIENT_SECRETS_MAP[cid.strip()] = secret.strip()
# Back end host
BACKEND_HOST = os.getenv("BACKEND_HOST", "http://localhost:29001")
# Frontend callback URI (used when we want Cognito to redirect to the web app)
REDIRECT_URI_FRONTEND = os.getenv("REDIRECT_URI_FRONTEND") or os.getenv("FRONTEND_CALLBACK")

def _cleanup_sessions():
    now = time.time()
    with _LOCK:
        for k in list(_SESSIONS.keys()):
            if _SESSIONS[k]["expires_at"] < now:
                del _SESSIONS[k]


@router.post("/start")
async def auth_start(request: Request):
    """Create a short-lived session and return an authorize URL to open in browser."""
    if not COGNITO_DOMAIN or not CLIENT_ID:
        raise HTTPException(status_code=500, detail="Cognito configuration missing on server")

    # optional: caller may request a specific client_id (for desktop vs web). Accept only allowed ids.
    requested_client_id = None
    try:
        body = await request.json()
        requested_client_id = body.get("client_id")
    except Exception:
        # ignore if no json body
        requested_client_id = request.query_params.get("client_id")

    chosen_client_id = requested_client_id or CLIENT_ID
    # For safety, ensure chosen_client_id is within allowed list if ALLOWED_CLIENT_IDS env exists
    allowed_raw = os.getenv("ALLOWED_CLIENT_IDS")
    if allowed_raw:
        allowed = [c.strip() for c in allowed_raw.split(",") if c.strip()]
        if chosen_client_id not in allowed:
            raise HTTPException(status_code=400, detail="Requested client_id not allowed")

    session_id = secrets.token_urlsafe(16)
    state = session_id
    # Prefer redirecting to frontend callback if configured (single unified approach)
    redirect_uri = REDIRECT_URI_FRONTEND or f"{BACKEND_HOST}/api/auth/callback"

    params = {
        "response_type": "code",
        "client_id": chosen_client_id,
        "redirect_uri": redirect_uri,
        "scope": "openid profile email",
        "state": state
    }
    auth_url = f"{COGNITO_DOMAIN}/oauth2/authorize?{urlencode(params)}"

    # Debug: log the generated auth_url and session
    try:
        print(f"[auth_router] start session_id={session_id} redirect_uri={redirect_uri}")
        print(f"[auth_router] auth_url={auth_url}")
    except Exception:
        pass

    now = time.time()
    with _LOCK:
        _SESSIONS[session_id] = {
            "created": now,
            "expires_at": now + 120,  # 2 minutes
            "status": "pending",
            "tokens": None,
            "client_id": chosen_client_id
        }

    # quick cleanup
    _cleanup_sessions()

    return {"session_id": session_id, "auth_url": auth_url}


@router.get("/callback")
def auth_callback(request: Request):
    """Cognito will redirect here with code & state. Exchange code and store tokens in session."""
    params = dict(request.query_params)
    code = params.get("code")
    state = params.get("state")
    if not code or not state:
        return Response(content="Missing code or state", status_code=400)

    with _LOCK:
        sess = _SESSIONS.get(state)
    if not sess:
        return Response(content="Invalid or expired session state", status_code=400)

    # Exchange code for tokens
    token_url = f"{COGNITO_DOMAIN}/oauth2/token"
    # use client_id stored in session (allows desktop/web different clients)
    client_id_for_session = sess.get("client_id") or CLIENT_ID
    client_secret_for_session = CLIENT_SECRETS_MAP.get(client_id_for_session) or CLIENT_SECRET

    # The redirect URI used for token exchange must match the one used in the authorize request.
    redirect_uri_used = REDIRECT_URI_FRONTEND or f"{BACKEND_HOST}/api/auth/callback"

    data = {
        "grant_type": "authorization_code",
        "client_id": client_id_for_session,
        "client_secret": client_secret_for_session,
        "code": code,
        "redirect_uri": redirect_uri_used
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        # Debug logging
        print(f"[auth_router] callback received for state={state}, exchanging code at {token_url}")
        print(f"[auth_router] token request data keys: {list(data.keys())}")
        r = requests.post(token_url, data=data, headers=headers, timeout=10)
        print(f"[auth_router] token endpoint status={r.status_code}")
        try:
            print(f"[auth_router] token endpoint response: {r.text}")
        except Exception:
            pass
        r.raise_for_status()
        tokens = r.json()
    except Exception as e:
        return Response(content=f"Token exchange failed: {e}", status_code=500)

    with _LOCK:
        _SESSIONS[state]["status"] = "done"
        _SESSIONS[state]["tokens"] = tokens

    # Simple user-facing page
    html = "<html><body><h3>Authentication complete. You may close this window.</h3></body></html>"
    return Response(content=html, media_type="text/html")


@router.post("/callback/forward")
def auth_callback_forward(payload: dict):
    """Frontend can forward a code+state here (useful when Cognito redirected to web frontend).
    Expects JSON: {"code": "...", "state": "..."}
    The backend will perform the token exchange and store tokens under the session 'state'.
    """
    code = payload.get("code")
    state = payload.get("state")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state")

    with _LOCK:
        sess = _SESSIONS.get(state)
    if not sess:
        raise HTTPException(status_code=400, detail="Invalid or expired session state")

    token_url = f"{COGNITO_DOMAIN}/oauth2/token"
    client_id_for_session = sess.get("client_id") or CLIENT_ID
    client_secret_for_session = CLIENT_SECRETS_MAP.get(client_id_for_session) or CLIENT_SECRET

    redirect_uri_used = REDIRECT_URI_FRONTEND or f"{BACKEND_HOST}/api/auth/callback"

    data = {
        "grant_type": "authorization_code",
        "client_id": client_id_for_session,
        "client_secret": client_secret_for_session,
        "code": code,
        "redirect_uri": redirect_uri_used
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        print(f"[auth_router] forwarded callback exchange for state={state}")
        r = requests.post(token_url, data=data, headers=headers, timeout=10)
        r.raise_for_status()
        tokens = r.json()
    except Exception as e:
        print(f"[auth_router] forwarded token exchange failed: {e}")
        raise HTTPException(status_code=500, detail=f"Token exchange failed: {e}")

    with _LOCK:
        _SESSIONS[state]["status"] = "done"
        _SESSIONS[state]["tokens"] = tokens

    return {"status": "ok"}


@router.get("/result")
def auth_result(session_id: str):
    """Poll endpoint: returns tokens when ready."""
    _cleanup_sessions()
    with _LOCK:
        sess = _SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if sess["status"] != "done":
        # not ready yet
        return Response(status_code=202, content="pending")

    tokens = sess.get("tokens")
    # Optionally delete session after read
    with _LOCK:
        try:
            del _SESSIONS[session_id]
        except KeyError:
            pass

    return tokens
