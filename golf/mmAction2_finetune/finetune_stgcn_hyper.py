# finetune_stgcn_hyper.py
#!/usr/bin/env python
"""
finetune_stgcn_hyper.py

Optuna-based Bayesian Optimization with K-Fold Cross-Validation (K=5)
for ST-GCN++ fine-tuning. Includes Modality tuning and CSV output.
"""
import argparse
import os
import sys
import pickle
import subprocess
import uuid
import numpy as np
import optuna
import pandas as pd  # ⭐️ pandas 라이브러리 추가
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# --- Global Configuration ---
# ⭐️ 경로를 실제 환경에 맞게 수정하세요.
MM_ROOT = r"D:\mmaction2" 
sys.path.append(MM_ROOT)

# Paths
DEFAULT_INPUT_PKL = r"D:\golfDataset\crop_pkl\combined_5class.pkl"
DEFAULT_CFG_BASE = r"configs\skeleton\stgcnpp\my_stgcnpp_hyper.py" 
DEFAULT_PRETRAINED = r"D:\mmaction2\checkpoints\stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth"
DEFAULT_WORK_DIR = r"D:\work_dirs\finetune_stgcn_optuna_modality" 

# K-Fold 설정 (기본값을 빠른 탐색용으로 낮춤 — 1일 내 완료 목표)
N_FOLDS = 3  # 기본을 3으로 줄임 (개발/초기 탐색용)
N_TRIALS = 20  # 기본 트라이얼 수를 20으로 줄임
K_FOLD_SEED = 42

# ----------------------------------------------------------------------
# PKL Manipulation Helper Functions for K-Fold
# ----------------------------------------------------------------------

def load_pkl_data(path: str) -> dict:
    """PKL 파일을 로드합니다."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_labels_and_indices(data: dict) -> tuple:
    """모든 annotation 인덱스와 해당 레이블을 가져옵니다."""
    anns = data.get('annotations', [])
    indices = list(range(len(anns)))
    labels = np.array([a.get('label') for a in anns if 'label' in a])
    if len(indices) != len(labels):
         raise ValueError("Annotation count and label count mismatch.")
    return indices, labels

def create_temp_pkl_with_split(base_data: dict, train_indices: list, val_indices: list, trial_dir: str) -> str:
    """임시 PKL 파일을 Optuna Trial Work Directory에 저장합니다."""
    # Make a shallow copy of the base data, then replace 'split' with
    # lists of annotation identifiers (filename or frame_dir) because
    # PoseDataset expects split values to match annotation identifiers.
    data = base_data.copy()
    anns = data.get('annotations', [])
    identifier = None
    if len(anns) > 0:
        identifier = 'filename' if 'filename' in anns[0] else 'frame_dir' if 'frame_dir' in anns[0] else None

    if identifier is None:
        # Fallback: if annotations have neither, treat indices as identifiers
        train_ids = train_indices
        val_ids = val_indices
    else:
        train_ids = [anns[i].get(identifier) for i in train_indices]
        val_ids = [anns[i].get(identifier) for i in val_indices]

    data['split'] = {
        'xsub_train': train_ids,
        'xsub_val': val_ids
    }

    os.makedirs(trial_dir, exist_ok=True)
    temp_pkl_path = os.path.join(trial_dir, f"fold_{uuid.uuid4().hex[:8]}_split.pkl")

    with open(temp_pkl_path, 'wb') as f:
        pickle.dump(data, f)

    return temp_pkl_path

# ----------------------------------------------------------------------
# Optuna Objective Function (K-Fold Cross-Validation)
# ----------------------------------------------------------------------

def objective(trial: optuna.Trial):
    """
    Optuna Trial을 실행하고 K-Fold 교차 검증 평균 정확도를 반환합니다.
    """
    
    # --- 1. 하이퍼 파라미터 제안 ---
    feats_type = trial.suggest_categorical('feats_type', ['j', 'b', 'jm', 'bm'])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 5e-3, log=True)
    optimizer_type = trial.suggest_categorical('optimizer_type', ['Adam', 'AdamW', 'SGD'])
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    # Reduce max_epochs search range for faster tuning (6-10 recommended)
    max_epochs = trial.suggest_int('max_epochs', 6, 10)
    warmup_epochs = trial.suggest_int('warmup_epochs', 1, 3)
    
    # --- 2. K-Fold 설정 ---
    trial_work_dir = os.path.join(DEFAULT_WORK_DIR, f"trial_{trial.number}")
    os.makedirs(trial_work_dir, exist_ok=True)
    
    base_data = load_pkl_data(DEFAULT_INPUT_PKL)
    indices, labels = get_labels_and_indices(base_data)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=K_FOLD_SEED)
    fold_scores = [] # ⭐️ 각 폴드별 점수를 저장할 리스트

    # --- 3. K-Fold 교차 검증 루프 ---
    for fold_idx, (train_index, val_index) in enumerate(skf.split(indices, labels)):
        print(f"\n--- [TRIAL {trial.number}] FOLD {fold_idx+1}/{N_FOLDS} (Modality: {feats_type}) ---")
        
        train_indices_fold = [indices[i] for i in train_index]
        val_indices_fold = [indices[i] for i in val_index]
        
        temp_pkl_path = create_temp_pkl_with_split(
            base_data, 
            train_indices_fold, 
            val_indices_fold, 
            trial_work_dir
        )

        cfg_path = os.path.join(MM_ROOT, DEFAULT_CFG_BASE.strip())
        
        # MMAction2 훈련 명령어 구성
        cmd = [
            sys.executable,
            os.path.join(MM_ROOT, 'tools', 'train.py'),
            cfg_path,
            '--work-dir', os.path.join(trial_work_dir, f"fold_{fold_idx}"),
            '--cfg-options',
            f"FEATS='{feats_type}'",
            f"train_dataloader.dataset.dataset.pipeline.1.feats=['{feats_type}']", 
            f"val_dataloader.dataset.pipeline.1.feats=['{feats_type}']", 
            f"model.backbone.init_cfg.checkpoint={DEFAULT_PRETRAINED}",
            f"train_dataloader.dataset.dataset.ann_file={temp_pkl_path}",
            f"val_dataloader.dataset.ann_file={temp_pkl_path}",
            f"train_dataloader.dataset.dataset.split=xsub_train",
            f"val_dataloader.dataset.split=xsub_val", 
            f"optim_wrapper.optimizer.lr={lr}",
            f"optim_wrapper.optimizer.weight_decay={weight_decay}",
            f"optim_wrapper.optimizer.type={optimizer_type}",
            f"train_dataloader.batch_size={batch_size}",
            f"val_dataloader.batch_size={batch_size}",
            f"train_cfg.max_epochs={max_epochs}",
            f"param_scheduler.0.end={warmup_epochs}",
            f"param_scheduler.1.begin={warmup_epochs}",
            f"param_scheduler.1.T_max={max_epochs}",
        ]
        # Ensure fold directory exists before launching training (and before writing logs)
        fold_dir = os.path.join(trial_work_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # 훈련 실행 및 결과 파싱
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Try primary parsing from stdout lines
            val_acc = 0.0
            for line in result.stdout.splitlines():
                if 'Epoch(val)' in line and 'top1_acc' in line:
                    try:
                        val_acc_str = line.split('top1_acc: ')[-1].split(' ')[0]
                        val_acc = float(val_acc_str)
                    except Exception:
                        continue

            # If primary parsing failed, try multiple regex patterns on stdout/stderr
            if val_acc == 0.0:
                import re
                combined = (result.stdout or '') + '\n' + (result.stderr or '')
                patterns = [
                    r"top1_acc[:=]\s*([0-9]*\.?[0-9]+)",
                    r"Acc@1[:=]?\s*([0-9]*\.?[0-9]+)",
                    r"top1[:\s]+([0-9]*\.?[0-9]+)%",
                    r"top1.*?([0-9]*\.?[0-9]+)%",
                ]
                for p in patterns:
                    m = re.search(p, combined)
                    if m:
                        try:
                            val_acc = float(m.group(1))
                            # If percentage like 85 -> convert to 0.85
                            if val_acc > 1.0 and val_acc <= 100.0:
                                val_acc = val_acc / 100.0
                            break
                        except Exception:
                            continue

            if val_acc > 0.0:
                print(f"[FOLD {fold_idx+1} RESULT] Validation Top1 Acc: {val_acc}")
                fold_scores.append(val_acc)
                # Report intermediate result to Optuna (mean over completed folds)
                current_mean = float(np.mean(fold_scores))
                trial.report(current_mean, step=fold_idx)
                # If pruner decides to prune, raise
                if trial.should_prune():
                    print(f"[TRIAL {trial.number}] Pruned after fold {fold_idx+1} (mean={current_mean:.4f})")
                    raise optuna.exceptions.TrialPruned()
            else:
                raise Exception("Validation score extraction failed.")
                
        except subprocess.CalledProcessError as e:
            # CalledProcessError: training script returned non-zero. Save stderr/stdout to fold log.
            error_path = os.path.join(fold_dir, "error_stderr.log")
            print(f"\n[ERROR] FOLD {fold_idx+1} training failed. Writing stderr to {error_path}")
            try:
                with open(error_path, "w", encoding="utf-8") as f:
                    f.write("COMMAND: \n" + " ".join(cmd) + "\n\n")
                    if e.stdout:
                        f.write("STDOUT:\n" + e.stdout + "\n\n")
                    if e.stderr:
                        f.write("STDERR:\n" + e.stderr + "\n\n")
                    f.write(repr(e))
            except Exception as write_err:
                print(f"Failed to write error log: {write_err}")
            raise optuna.exceptions.TrialPruned()
        except Exception as e:
            # Generic exception (e.g., FileNotFoundError when writing logs or other issues)
            error_path = os.path.join(fold_dir, "error_exception.log")
            print(f"\n[ERROR] FOLD {fold_idx+1} unexpected error. Writing exception to {error_path}")
            try:
                with open(error_path, "w", encoding="utf-8") as f:
                    f.write("COMMAND: \n" + " ".join(cmd) + "\n\n")
                    f.write("EXCEPTION:\n" + repr(e))
            except Exception as write_err:
                print(f"Failed to write exception log: {write_err}")
            raise optuna.exceptions.TrialPruned()
        finally:
            if os.path.exists(temp_pkl_path):
                os.remove(temp_pkl_path)
    
    # ⭐️ Optuna에 각 폴드별 점수를 저장 (분석 용이성 향상)
    trial.set_user_attr("fold_scores", fold_scores) 
    
    if fold_scores:
        avg_score = np.mean(fold_scores)
        print(f"\n[TRIAL {trial.number} SUMMARY] Avg K-Fold Acc: {avg_score:.4f}")
        return avg_score
    else:
        raise optuna.exceptions.TrialPruned()


def main():
    global N_FOLDS
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=None,
                        help='Override number of folds for K-Fold CV (default from script).')
    parser.add_argument('--study-name', type=str, default='stgcn_finetune_modality_study',
                        help='Optuna study name for resuming.')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS,
                        help='Number of optimization trials.')
    args = parser.parse_args()

    # Allow overriding number of folds from CLI
    if args.n_folds is not None:
        print(f"[SETUP] Overriding N_FOLDS: {N_FOLDS} -> {args.n_folds}")
        N_FOLDS = args.n_folds
    
    # 클래스 레이블 분포 확인 (Stratified K-Fold에 필요)
    try:
        data = load_pkl_data(DEFAULT_INPUT_PKL)
        cnt = Counter(get_labels_and_indices(data)[1])
        print(f"[SETUP] Label distribution in PKL: {dict(cnt)}")

        if len(cnt) < N_FOLDS:
            print(f"[WARN] Classes ({len(cnt)}) are fewer than N_FOLDS ({N_FOLDS}). Adjusting N_FOLDS to {len(cnt)}.")
            N_FOLDS = len(cnt)
    except Exception as e:
        print(f"[ERROR] Error during PKL data check: {e}")
        sys.exit(1)
        
    print(f"\n[SETUP] Starting Optuna with {N_TRIALS} trials and {N_FOLDS}-Fold Cross-Validation...")
        
    print(f"\n[SETUP] Starting Optuna with {N_TRIALS} trials and {N_FOLDS}-Fold Cross-Validation...")
    
    # Study DB 경로 만들기 전에 work dir이 존재하는지 보장
    os.makedirs(DEFAULT_WORK_DIR, exist_ok=True)
    db_path = os.path.join(DEFAULT_WORK_DIR, f"{args.study_name}.db")
    # Use a pruner to stop unpromising trials early (reduces total runtime)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        direction='maximize',
        study_name=args.study_name,
        storage=f'sqlite:///{db_path}',
        load_if_exists=True,
        pruner=pruner,
    )
    
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    # ⭐️ 튜닝 결과 CSV 파일로 저장 ⭐️
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'user_attrs', 'state'))
    output_path = os.path.join(DEFAULT_WORK_DIR, f"{args.study_name}_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n" + "="*50)
    print("✨ Optimization Finished ✨")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best K-Fold Validation Accuracy: {study.best_value:.4f}")
    print(f"Best Trial Number: {study.best_trial.number}")
    print(f"\n✅ All tuning results saved to: {output_path}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50)

if __name__ == '__main__':
    main()