# backend-api/main.py (통일성 확보 및 모든 설정 통합)

import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # CORS 임포트
from app.routers import upload_router
from app.routers import token_router
from app.routers import result_router
from app.routers import auth_router



app = FastAPI(
    title="Golf Analysis Upload API",
    version="1.0.0"
)

# -----------------------------------------------------------------
# 1. CORS 미들웨어 설정 (localhost:29000 허용)
# -----------------------------------------------------------------
origins = [
    "http://localhost:29000",
    "http://127.0.0.1:29000",
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,              
    allow_credentials=True,             
    allow_methods=["*"],                
    allow_headers=["*"],                
)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# 2. 라우터 등록 (통일된 /api Prefix 사용)
# -----------------------------------------------------------------
# token_router.py의 prefix=/token과 결합 -> 최종 경로: /api/token
app.include_router(token_router.router, prefix="/api") 

# upload_router.py의 prefix=/upload와 결합 -> 최종 경로: /api/upload
app.include_router(upload_router.router, prefix="/api") 
 
# result_router.py의 prefix=/result와 결합 -> 최종 경로: /api/result
app.include_router(result_router.router, prefix="/api") 
# auth router for backend-mediated login
app.include_router(auth_router.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Upload API"}