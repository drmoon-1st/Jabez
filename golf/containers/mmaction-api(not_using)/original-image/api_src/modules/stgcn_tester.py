"""
ST-GCN Tester Module
test.py와 완전히 동일한 구조로 작동하는 테스트 모듈
Runner.from_cfg() -> runner.test() 호출 -> result.pkl 파싱 및 반환
"""
import os
import os.path as osp
import pickle
import sys
import tempfile
import uuid
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner

# Do not import mmaction.registry at module import time. Import it later
# after we ensure the local /mmaction2 repo is on sys.path so the registry
# is populated from the correct package (avoids stale/site-package binding).
from modules.utils import debug_log, csv_to_pkl


def prepare_config_for_test(csv_path: Path):
    """
    test.py의 parse_args()와 merge_args() 역할을 수행
    CSV 파일을 받아서 config를 준비하고 필요한 설정을 오버라이드
    """
    # 1. Config 파일 경로 (my_stgcnpp.py)
    config_path = Path(__file__).parent / "my_stgcnpp.py"
    checkpoint_path = None
    
    # checkpoint 경로 찾기
    for p in [
        Path(__file__).parent.parent / "stgcn_70p.pth"
    ]:
        if Path(p).exists():
            checkpoint_path = str(p)
            break
    
    if checkpoint_path is None:
        raise FileNotFoundError("checkpoint file stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth not found")
    
    debug_log(f"Using config: {config_path}")
    debug_log(f"Using checkpoint: {checkpoint_path}")
    
    # 2. CSV를 ann.pkl로 변환
    unique_id = uuid.uuid4().hex[:8]
    ann_pkl_path = Path(tempfile.gettempdir()) / f"test_ann_{unique_id}.pkl"
    debug_log(f"Converting CSV to PKL: {csv_path} -> {ann_pkl_path}")
    csv_to_pkl(csv_path, ann_pkl_path)
    
    # 3. result.pkl 경로 설정
    result_pkl_path = Path(tempfile.gettempdir()) / f"test_result_{unique_id}.pkl"
    
    # Make sure local mmaction2 package is importable so model classes (e.g. RecognizerGCN)
    # are registered before loading the config. We try the repo-local path and the
    # container path '/mmaction2'.
    repo_dir = Path(__file__).parent
    candidate = repo_dir.parent / "mmaction2"
    try:
        if candidate.exists():
            sys.path.insert(0, str(candidate))
        else:
            # fallback to container path
            sys.path.insert(0, "/mmaction2")
    except Exception:
        pass

    # Ensure mmaction registers its modules (models, datasets, etc.).
    # Some mmaction registration happens when importing subpackages; call
    # register_all_modules if available to guarantee registries are populated.
    try:
        import mmaction  # noqa: F401
        try:
            # mmaction provides a helper to register all modules
            from mmaction.utils.setup_env import register_all_modules
            register_all_modules(init_default_scope=True)
            # Debug: list some registered model keys to verify GCN is present
            try:
                from mmaction.registry import MODELS
                model_keys = list(MODELS.module_dict.keys()) if hasattr(MODELS, 'module_dict') else []
                debug_log(f"Registered MODELS sample (len={len(model_keys)}): {model_keys[:50]}")
                debug_log(f"RecognizerGCN registered? {'RecognizerGCN' in model_keys}")
            except Exception as _e:
                debug_log(f"Failed to inspect MODELS registry: {_e}")
        except Exception:
            # Fallback: at least import models subpackage to trigger module imports
            try:
                import mmaction.models  # noqa: F401
                # Try to inspect MODELS registry even if register_all_modules unavailable
                try:
                    from mmaction.registry import MODELS
                    model_keys = list(MODELS.module_dict.keys()) if hasattr(MODELS, 'module_dict') else []
                    debug_log(f"Registered MODELS sample (fallback) (len={len(model_keys)}): {model_keys[:50]}")
                    debug_log(f"RecognizerGCN registered? {'RecognizerGCN' in model_keys}")
                except Exception as _e:
                    debug_log(f"Failed to inspect MODELS registry in fallback: {_e}")
            except Exception:
                # swallow; Config.fromfile() will still run and may raise a helpful error
                pass
    except Exception:
        # If mmaction cannot be imported at all, let Config.fromfile raise later
        pass

    # 4. Config 로드
    cfg = Config.fromfile(str(config_path))
    # Debug: what model type is requested in config
    try:
        cfg_model_type = cfg.model.type if hasattr(cfg, 'model') and isinstance(cfg.model, dict) and 'type' in cfg.model else getattr(cfg.model, 'type', None)
        debug_log(f"Config model.type -> {cfg_model_type}")
    except Exception as _e:
        debug_log(f"Failed to read cfg.model.type: {_e}")
    
    # 5. test.py의 merge_args와 동일한 설정 적용
    # work_dir 설정 (test.py와 동일한 우선순위)
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('/tmp/work_dirs', 
                               osp.splitext(osp.basename(str(config_path)))[0])
    
    # 6. checkpoint 로드 설정
    cfg.load_from = checkpoint_path
    
    # 7. test_dataloader의 ann_file 오버라이드
    cfg.test_dataloader.dataset.ann_file = str(ann_pkl_path)
    
    # 8. DumpResults 설정 (test.py의 --dump 옵션과 동일)
    dump_metric = dict(type='DumpResults', out_file_path=str(result_pkl_path))
    if isinstance(cfg.test_evaluator, (list, tuple)):
        cfg.test_evaluator = list(cfg.test_evaluator)
        # 기존 DumpResults가 있으면 제거
        cfg.test_evaluator = [e for e in cfg.test_evaluator if e.get('type') != 'DumpResults']
        cfg.test_evaluator.append(dump_metric)
    else:
        cfg.test_evaluator = [cfg.test_evaluator, dump_metric]
    
    # 9. launcher 설정
    cfg.launcher = 'none'
    
    # 10. 환경 설정 (안정성을 위해)
    if hasattr(cfg, 'env_cfg'):
        if hasattr(cfg.env_cfg, 'mp_cfg'):
            cfg.env_cfg.mp_cfg.mp_start_method = 'fork'
        if hasattr(cfg.env_cfg, 'dist_cfg'):
            cfg.env_cfg.dist_cfg.backend = 'gloo'
    
    # 11. visualization 비활성화 (서버 환경에서 불필요)
    if hasattr(cfg, 'default_hooks') and isinstance(cfg.default_hooks, dict):
        if 'visualization' in cfg.default_hooks:
            cfg.default_hooks.visualization.enable = False
    
    debug_log(f"Config prepared: work_dir={cfg.work_dir}")
    debug_log(f"Result will be saved to: {result_pkl_path}")
    
    return cfg, ann_pkl_path, result_pkl_path


def run_stgcn_test(csv_path: Path):
    """
    test.py의 main() 함수와 완전히 동일한 구조
    1. Config 로드 및 설정
    2. Runner 생성
    3. runner.test() 실행
    4. result.pkl 파싱 및 반환
    """
    debug_log(f"run_stgcn_test start: {csv_path}")
    
    ann_pkl_path = None
    result_pkl_path = None
    
    try:
        # 1. Config 준비 (test.py의 parse_args + merge_args)
        cfg, ann_pkl_path, result_pkl_path = prepare_config_for_test(csv_path)
        
        # 2. Run the test in a fresh subprocess to avoid registry/import issues
        debug_log("Invoking stgcn_subproc.py in a fresh Python process...")
        import subprocess
        subproc_script = Path(__file__).parent / "stgcn_subproc.py"
        # Build environment: ensure subprocess can import local mmaction2
        env = os.environ.copy()
        # candidate repo dir
        repo_dir = Path(__file__).parent.parent
        candidate = repo_dir / "mmaction2"
        if candidate.exists():
            env_pythonpath = str(candidate)
        else:
            env_pythonpath = "/mmaction2"
        # Prepend any existing PYTHONPATH
        if env.get('PYTHONPATH'):
            env['PYTHONPATH'] = env_pythonpath + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = env_pythonpath

        cmd = [
            sys.executable,
            str(subproc_script),
            '--config', str(Path(__file__).parent / 'my_stgcnpp.py'),
            '--checkpoint', str(cfg.load_from),
            '--ann', str(ann_pkl_path),
            '--out', str(result_pkl_path),
        ]
        # Forward CUDA_VISIBLE_DEVICES if set in current environment
        cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_env:
            env['CUDA_VISIBLE_DEVICES'] = cuda_env
            debug_log(f"Forwarding CUDA_VISIBLE_DEVICES={cuda_env} to subprocess")
        # If caller provided a desired device via environment variable MMACTION_DEVICE, forward it
        mma_device = os.environ.get('MMACTION_DEVICE')
        if mma_device:
            cmd += ['--device', mma_device]
            debug_log(f"Passing device='{mma_device}' to subprocess")
        debug_log(f"Running subprocess: {cmd} with PYTHONPATH={env['PYTHONPATH']}")
        # Redirect stdout/stderr to temporary files to avoid pipe buffer deadlocks
        import tempfile
        out_log = Path(tempfile.gettempdir()) / f"stgcn_subproc_{uuid.uuid4().hex[:8]}.stdout.log"
        err_log = Path(tempfile.gettempdir()) / f"stgcn_subproc_{uuid.uuid4().hex[:8]}.stderr.log"
        debug_log(f"Subprocess stdout redirected to: {out_log}")
        debug_log(f"Subprocess stderr redirected to: {err_log}")
        with open(out_log, 'wb') as _outf, open(err_log, 'wb') as _errf:
            proc = subprocess.Popen(cmd, stdout=_outf, stderr=_errf, env=env)
            try:
                # Wait for subprocess to finish. Increase timeout if your model loading
                # or data preparation is expected to take longer than this value.
                proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
                debug_log("Subprocess timed out and was killed")
                # Read whatever was written so far to help debugging
                try:
                    debug_log(f"subproc stdout (partial):\n{out_log.read_text(errors='replace')}")
                    debug_log(f"subproc stderr (partial):\n{err_log.read_text(errors='replace')}")
                except Exception:
                    pass
                raise RuntimeError("stgcn_subproc timed out")

        # After completion, log the captured output (tail)
        try:
            out_text = out_log.read_text(errors='replace')
            err_text = err_log.read_text(errors='replace')
            # Limit size to avoid huge logs
            max_len = 10000
            debug_log(f"subproc stdout (tail):\n{out_text[-max_len:]}")
            debug_log(f"subproc stderr (tail):\n{err_text[-max_len:]}")
        except Exception as _e:
            debug_log(f"Failed to read subproc logs: {_e}")

        if proc.returncode != 0:
            raise RuntimeError(f"stgcn_subproc failed (exit {proc.returncode}); see logs: {err_log}")

        debug_log("Subprocess completed successfully")
        
        # 4. result.pkl 파싱
        if not result_pkl_path.exists():
            raise FileNotFoundError(f"Result file not found: {result_pkl_path}")
        
        debug_log(f"Loading result from: {result_pkl_path}")
        with open(result_pkl_path, "rb") as f:
            result_data = pickle.load(f)
        
        # 5. 결과 파싱 및 반환
        parsed_result = parse_test_result(result_data)
        debug_log(f"Test completed successfully: {parsed_result}")
        
        return parsed_result
        
    finally:
        # 임시 파일 정리
        if ann_pkl_path and Path(ann_pkl_path).exists():
            try:
                Path(ann_pkl_path).unlink()
                debug_log(f"Cleaned up: {ann_pkl_path}")
            except Exception as e:
                debug_log(f"Failed to cleanup {ann_pkl_path}: {e}")
        
        if result_pkl_path and Path(result_pkl_path).exists():
            try:
                Path(result_pkl_path).unlink()
                debug_log(f"Cleaned up: {result_pkl_path}")
            except Exception as e:
                debug_log(f"Failed to cleanup {result_pkl_path}: {e}")


def parse_test_result(result_data):
    """
    DumpResults로 저장된 result.pkl을 파싱하여 평가 결과 추출
    result.pkl 구조:
    - list of dict, 각 dict는 하나의 샘플에 대한 예측 결과
    - 각 dict는 'pred_scores', 'pred_labels' 등을 포함
    """
    if not isinstance(result_data, list):
        debug_log(f"Unexpected result format: {type(result_data)}")
        return {
            "status": "error",
            "message": "Unexpected result format",
            "raw_type": str(type(result_data))
        }
    
    if len(result_data) == 0:
        return {
            "status": "success",
            "num_samples": 0,
            "predictions": []
        }
    
    # 결과 파싱
    predictions = []
    for idx, item in enumerate(result_data):
        pred_info = {
            "sample_index": idx,
        }
        
        # pred_scores 추출 (확률값)
        if 'pred_scores' in item:
            scores = item['pred_scores']
            if hasattr(scores, 'tolist'):
                pred_info['scores'] = scores.tolist()
            else:
                pred_info['scores'] = scores
        
        # pred_label 추출 (예측 클래스)
        if 'pred_label' in item:
            pred_info['predicted_class'] = int(item['pred_label'])
        elif 'pred_labels' in item:
            pred_info['predicted_class'] = int(item['pred_labels'])
        
        # gt_label 추출 (실제 클래스, 있는 경우)
        if 'gt_label' in item:
            pred_info['ground_truth_class'] = int(item['gt_label'])
        elif 'gt_labels' in item:
            pred_info['ground_truth_class'] = int(item['gt_labels'])
        
        predictions.append(pred_info)
    
    # 전체 결과 요약
    result = {
        "status": "success",
        "num_samples": len(result_data),
        "predictions": predictions
    }
    
    # 정확도 계산 (gt_label이 있는 경우)
    if predictions and 'ground_truth_class' in predictions[0]:
        correct = sum(1 for p in predictions 
                     if p.get('predicted_class') == p.get('ground_truth_class'))
        accuracy = correct / len(predictions)
        result['accuracy'] = accuracy
        result['correct_predictions'] = correct
    
    return result
