# backend-api/routers/token_router.py

import os
import requests
from fastapi import APIRouter, Form, Response, HTTPException
from urllib.parse import urlencode

router = APIRouter(
    prefix="/token", # 최종 URL: /api/token
    tags=["token"]
)

# .env에서 필요한 환경 변수 로드
COGNITO_DOMAIN_URL = os.getenv("COGNITO_DOMAIN_URL")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
# Next.js의 콜백 URL과 일치해야 합니다. .env의 REDIRECT_URI_FRONTEND로 설정하세요.
REDIRECT_URI = os.getenv("REDIRECT_URI_FRONTEND") or "http://localhost:29000/callback"


@router.post("/")
async def exchange_code_for_token(
    code: str = Form(..., description="프론트엔드로부터 받은 Cognito 인증 코드"),
    response: Response = Response() # FastAPI Response 객체 주입
):
    """
    인증 코드를 Cognito에 전송하여 토큰으로 교환하고, 
    ID Token과 Access Token을 HttpOnly 쿠키로 설정합니다.
    """
    
    # 1. 필수 환경 변수 확인
    if not COGNITO_DOMAIN_URL or not CLIENT_ID or not CLIENT_SECRET:
        raise HTTPException(
            status_code=500, 
            detail={"message": "백엔드 서버 내부 오류: Cognito 환경 변수가 설정되지 않았습니다."}
        )

    # 2. Cognito 토큰 엔드포인트 구성
    COGNITO_TOKEN_URL = f"{COGNITO_DOMAIN_URL}/oauth2/token"
    
    token_request_data = {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "redirect_uri": REDIRECT_URI
    }

    try:
        # 3. Cognito로 POST 요청 전송
        response_cognito = requests.post(
            COGNITO_TOKEN_URL,
            data=urlencode(token_request_data), # URL 인코딩 필수
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        response_cognito.raise_for_status() # HTTP 오류 발생 시 예외 발생
        token_data = response_cognito.json()

        # 4. 토큰을 HttpOnly 쿠키로 설정
        id_token = token_data.get("id_token")
        access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600)

        # ❗ HttpOnly Cookie 설정 ❗
        response.set_cookie(
            key="id_token", 
            value=id_token, 
            httponly=True, 
            secure=False, # 개발 환경(localhost)이므로 False, 운영 시 True
            samesite="Lax", 
            max_age=expires_in
        )
        
        response.set_cookie(
            key="access_token", 
            value=access_token, 
            httponly=True, 
            secure=False, # 개발 환경(localhost)이므로 False, 운영 시 True
            samesite="Lax",
            max_age=expires_in
        )
        
        # 5. 성공 메시지만 반환
        return {"status": "success", "message": "토큰이 HttpOnly 쿠키로 안전하게 설정되었습니다."}

    except requests.exceptions.HTTPError as e:
        # Cognito에서 오류 발생 시 (예: 유효하지 않은 코드)
        try:
            error_details = response_cognito.json()
            error_message = error_details.get("error_description", error_details.get("error", "Cognito 토큰 교환 오류"))
        except:
            error_message = f"Cognito 통신 오류: {e}"
            
        raise HTTPException(
            status_code=400,
            detail={"message": f"Cognito 토큰 교환 실패: {error_message}"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"message": f"백엔드 서버 내부 오류 발생: {str(e)}"}
        )

# ----------------------------------------------------
# 로그아웃 엔드포인트: /token/logout
# ----------------------------------------------------
# application의 토큰을 삭제시킴
@router.post("/logout")
def logout(response: Response):
    """
    HttpOnly 쿠키(id_token, access_token)를 삭제(만료)하여 서버측 세션을 종료합니다.
    프론트는 credentials: 'include'로 호출하세요.
    """
    # SameSite / Secure 등은 배포환경에 맞게 설정되어야 합니다.
    response.delete_cookie("id_token", path="/")
    response.delete_cookie("access_token", path="/")
    return {"status": "logged_out"}