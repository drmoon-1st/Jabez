# backend-api/auth_utils.py

# JWT 토큰 검증하고 사용자 ID 추출

import os
import time
import requests
from typing import Optional

from jose import jwt, jwk
from jose.exceptions import JWTError
from jose.utils import base64url_decode
from fastapi import Header, Cookie

# 환경변수 로드
COGNITO_REGION = os.getenv("COGNITO_REGION")
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
# 허용할 클라이언트 ID 목록 (콤마로 구분). 기본값은 `CLIENT_ID` 하나만 허용.
_ALLOWED_CLIENT_IDS_RAW = os.getenv("ALLOWED_CLIENT_IDS")
if _ALLOWED_CLIENT_IDS_RAW:
    ALLOWED_CLIENT_IDS = [c.strip() for c in _ALLOWED_CLIENT_IDS_RAW.split(",") if c.strip()]
else:
    ALLOWED_CLIENT_IDS = [CLIENT_ID] if CLIENT_ID else []

# 우선순위: 직접 지정(오버라이드)된 ISSUER/JWKS_URL 사용 가능
COGNITO_ISSUER = os.getenv("COGNITO_ISSUER")
JWKS_URL_OVERRIDE = os.getenv("JWKS_URL")

if COGNITO_ISSUER:
    ISSUER = COGNITO_ISSUER
elif COGNITO_REGION and COGNITO_USER_POOL_ID:
    ISSUER = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"
else:
    ISSUER = None

JWKS_URL = JWKS_URL_OVERRIDE or (f"{ISSUER}/.well-known/jwks.json" if ISSUER else None)

# JWKS 캐시
_JWKS_CACHE: Optional[dict] = None
_JWKS_CACHE_TS: float = 0
_JWKS_TTL = 60 * 60  # 1시간

def _fetch_jwks(force: bool = False) -> dict:
    global _JWKS_CACHE, _JWKS_CACHE_TS
    if not JWKS_URL:
        raise JWTError("JWKS URL 미설정: COGNITO_REGION/Cognito_USER_POOL_ID 또는 JWKS_URL 확인 필요")
    if _JWKS_CACHE and (time.time() - _JWKS_CACHE_TS) < _JWKS_TTL and not force:
        return _JWKS_CACHE
    try:
        print(f"[auth_utils] JWKS 조회: {JWKS_URL}")
        r = requests.get(JWKS_URL, timeout=5)
        r.raise_for_status()
        _JWKS_CACHE = r.json()
        _JWKS_CACHE_TS = time.time()
        return _JWKS_CACHE
    except requests.exceptions.RequestException as e:
        if _JWKS_CACHE:
            return _JWKS_CACHE
        raise JWTError(f"JWKS 조회 실패: {e}") from e

def _get_jwk_for_kid(kid: str) -> Optional[dict]:
    jwks = _fetch_jwks()
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            return k
    return None

def _verify_signature(token: str) -> dict:
    try:
        header = jwt.get_unverified_header(token)
    except Exception as e:
        raise JWTError("잘못된 토큰 헤더") from e

    kid = header.get("kid")
    if not kid:
        raise JWTError("토큰 헤더에 kid가 없습니다.")

    jwk_dict = _get_jwk_for_kid(kid)
    if not jwk_dict:
        # 강제 갱신 시도
        jwk_dict = _fetch_jwks(force=True).get("keys", [])
        jwk_dict = next((k for k in jwk_dict if k.get("kid") == kid), None)
        if not jwk_dict:
            raise JWTError("JWKS에서 해당 kid를 찾을 수 없습니다.")

    public_key = jwk.construct(jwk_dict)
    try:
        message = token.rsplit(".", 1)[0]
        encoded_sig = token.rsplit(".", 1)[1]
        decoded_sig = base64url_decode(encoded_sig.encode("utf-8"))
        if not public_key.verify(message.encode("utf-8"), decoded_sig):
            raise JWTError("서명 검증 실패")
    except Exception as e:
        raise JWTError("서명 검증 실패") from e

    claims = jwt.get_unverified_claims(token)
    return claims

def _validate_claims(claims: dict, audiences: Optional[list] = None) -> None:
    now = int(time.time())
    exp = claims.get("exp")
    if exp is None or int(exp) < now:
        raise JWTError("토큰이 만료되었습니다.")

    if ISSUER:
        iss = claims.get("iss")
        if iss != ISSUER:
            raise JWTError(f"iss 검증 실패 (got={iss} expected={ISSUER})")

    token_use = claims.get("token_use")  # 'id' 또는 'access' (Cognito)
    # audiences가 제공되지 않으면 전역 ALLOWED_CLIENT_IDS 사용
    if audiences is None:
        audiences = ALLOWED_CLIENT_IDS

    def _audience_matches(value):
        if not value:
            return False
        if isinstance(value, list):
            return any(v in audiences for v in value)
        return value in audiences

    if token_use == "id":
        aud = claims.get("aud")
        if audiences:
            if not _audience_matches(aud):
                raise JWTError(f"aud 검증 실패 (id_token) aud={aud} allowed={audiences}")
    elif token_use == "access":
        # access_token은 client_id 클레임을 사용하는 경우가 많음
        client_id_claim = claims.get("client_id")
        aud_claim = claims.get("aud")
        if audiences:
            if client_id_claim:
                if client_id_claim not in audiences:
                    raise JWTError(f"client_id 검증 실패 (access_token) client_id={client_id_claim} allowed={audiences}")
            elif aud_claim:
                if aud_claim not in audiences:
                    raise JWTError(f"aud 검증 실패 (access_token) aud={aud_claim} allowed={audiences}")
            else:
                raise JWTError("access_token에 client_id/aud가 없어 검증 불가")
    else:
        aud = claims.get("aud")
        if audiences and aud and (not _audience_matches(aud)):
            raise JWTError(f"aud 검증 실패 (unknown token_use) aud={aud} allowed={audiences}")

def get_current_user_id(
    Authorization: Optional[str] = Header(None, alias="Authorization"),
    access_token_cookie: Optional[str] = Cookie(None, alias="access_token"),
    id_token_cookie: Optional[str] = Cookie(None, alias="id_token")
) -> Optional[str]:
    token = None
    source = None
    if Authorization and Authorization.startswith("Bearer "):
        token = Authorization.split(" ", 1)[1]
        source = "header"
    elif access_token_cookie:
        token = access_token_cookie
        source = "access_cookie"
    elif id_token_cookie:
        token = id_token_cookie
        source = "id_cookie"

    if not token:
        return None

    try:
        claims = _verify_signature(token)
        # audience 검증은 token_use에 따라 유동적으로 수행
        _validate_claims(claims, audiences=ALLOWED_CLIENT_IDS if ALLOWED_CLIENT_IDS else None)
        user_id = claims.get("sub")
        if not user_id:
            raise JWTError("토큰에 sub가 없습니다.")
        return user_id
    except JWTError as e:
        print(f"JWT 검증 실패: {e} (source={source})")
        return None


def get_user_id_from_token(token: Optional[str]) -> Optional[str]:
    """
    Helper to validate a raw Bearer token string and return user_id (sub).
    Reuses existing verification logic.
    """
    if not token:
        return None
    if token.startswith("Bearer "):
        token = token.split(" ", 1)[1]
    try:
        claims = _verify_signature(token)
        _validate_claims(claims, audiences=ALLOWED_CLIENT_IDS if ALLOWED_CLIENT_IDS else None)
        return claims.get("sub")
    except JWTError as e:
        print(f"JWT 검증 실패 (token helper): {e}")
        return None