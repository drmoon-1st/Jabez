from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import tempfile, base64, os
import logging
import sys

# test.py와 동일한 구조를 사용하는 stgcn_tester 모듈
from modules.stgcn_tester import run_stgcn_test
from modules.utils import debug_log

app = FastAPI(title="mmaction-stgcn-api")

# configure logging early so startup logs are visible
logging.basicConfig(level=logging.INFO)

class CSVBase64Request(BaseModel):
    csv_base64: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/mmaction_stgcn_test")
def mmaction_stgcn_test_endpoint(payload: CSVBase64Request):
    """
    ST-GCN 모델을 사용하여 CSV 데이터를 평가하고 결과를 반환합니다.
    test.py와 동일한 구조로 Runner.from_cfg() -> runner.test() 호출
    """
    temp_csv = None
    try:
        if not payload.csv_base64:
            raise HTTPException(status_code=400, detail="csv_base64 is required")
        
        # CSV 디코딩 및 임시 파일 저장
        csv_text = base64.b64decode(payload.csv_base64).decode("utf-8")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write(csv_text)
            temp_csv = Path(tmp.name)
        
        debug_log(f"Received request, temp_csv={temp_csv}")
        
        # test.py와 동일한 구조로 테스트 실행
        result = run_stgcn_test(temp_csv)
        
        return JSONResponse(status_code=200, content={
            "message": "OK",
            "result": result
        })
    except Exception as e:
        debug_log(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_csv and temp_csv.exists():
            try:
                os.remove(temp_csv)
            except Exception:
                pass

@app.on_event("startup")
def init_mmaction_registry():
    import sys, importlib, logging, traceback
    logging.info("Initializing mmaction registry...")
    if '/mmaction2' not in sys.path: sys.path.insert(0, '/mmaction2')
    try:
        import mmengine, mmcv, mmaction
        try:
            from mmaction.utils import register_all_modules
        except Exception:
            from mmaction.utils.setup_env import register_all_modules
        register_all_modules(init_default_scope=True)
        # Ensure mmaction's registry entries are visible in mmengine global registries
        try:
            import mmengine.registry as _me_reg
            mma_reg = importlib.import_module('mmaction.registry')
            mma_models = getattr(mma_reg, 'MODELS', None)
            me_models = getattr(_me_reg, 'MODELS', None)
            if mma_models is not None and me_models is not None:
                mma_dict = getattr(mma_models, 'module_dict', {}) or {}
                me_dict = getattr(me_models, 'module_dict', {}) or {}
                added = 0
                for name, cls in mma_dict.items():
                    if name not in me_dict:
                        try:
                            me_models.register_module(module=cls, name=name, force=True)
                            added += 1
                        except Exception:
                            logging.exception(f"Failed to register model {name} into mmengine.MODELS")
                logging.info(f"Synchronized {added} mmaction models into mmengine.MODELS")
        except Exception:
            logging.exception('Failed to synchronize mmaction registry into mmengine')
    except Exception:
        logging.exception("Failed to init mmaction registry")
    # Log environment and module locations for debugging
    try:
        logging.info("CWD=%s", os.getcwd())
        logging.info("PYTHONPATH sample=%s", sys.path[:5])
        try:
            import mmaction as _mma
            logging.info("mmaction module file=%s", getattr(_mma, '__file__', None))
        except Exception:
            logging.exception("mmaction import failed at startup logging")
    except Exception:
        logging.exception("Failed to write startup debug info")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 19031))
    uvicorn.run("api_src.api_server:app", host="0.0.0.0", port=port, reload=False)