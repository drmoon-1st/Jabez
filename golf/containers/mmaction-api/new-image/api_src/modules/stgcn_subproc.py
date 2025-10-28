import argparse
import os
import sys
import traceback
from pathlib import Path
import time


def ensure_repo_on_path():
    """Try inserting likely mmaction2 repo locations into sys.path so
    importing the local package registers modules correctly.
    """
    candidates = []
    here = Path(__file__).resolve()
    # container-standard location - prefer canonical path
    candidates.append(Path('/mmaction2'))
    # walk up available parents and append possible mmaction2 locations
    try:
        for parent in here.parents:
            candidates.append(parent / 'mmaction2')
    except Exception:
        # defensive: if parents iteration fails for any reason, skip
        pass

    # also try cwd as a fallback
    candidates.append(Path.cwd())

    # Insert the first valid candidate into sys.path (prefer earlier entries)
    for c in candidates:
        try:
            if c.exists() and c.is_dir():
                p = str(c)
                if p not in sys.path:
                    sys.path.insert(0, p)
                print(f"[stgcn_subproc] inserted {p} to sys.path", file=sys.stderr)
                # Do not return immediately; keep trying to ensure canonical path is present
        except Exception:
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ann', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cuda-devices', required=False,
                        help='CUDA_VISIBLE_DEVICES string (e.g. "0" or "0,1"). If set, it will be applied before importing torch/mmengine.')
    parser.add_argument('--device', required=False,
                        help='Device string to use (e.g. "cuda" or "cuda:0"). If not provided, will use torch.cuda if available.')
    args = parser.parse_args()

    try:
        # Ensure local repo is discoverable before importing mmengine/mmaction
        ensure_repo_on_path()

        # If user supplied CUDA_VISIBLE_DEVICES, set it before importing torch
        if args.cuda_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_devices)
            print(f"[stgcn_subproc] set CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", file=sys.stderr)

        # Import torch early to introspect CUDA availability and optionally log device info
        try:
            import torch
            cuda_avail = torch.cuda.is_available()
            print(f"[stgcn_subproc] torch.__version__={getattr(torch, '__version__', None)} cuda_available={cuda_avail}", file=sys.stderr)
            if cuda_avail:
                try:
                    dev = torch.cuda.current_device()
                    name = torch.cuda.get_device_name(dev)
                    print(f"[stgcn_subproc] current_cuda_device={dev} name={name}", file=sys.stderr)
                except Exception:
                    pass
        except Exception as _e:
            print(f"[stgcn_subproc] could not import torch: {_e}", file=sys.stderr)

        from mmengine.config import Config
        from mmengine.runner import Runner

        # ensure mmaction registration
        try:
            # preferred location
            from mmaction.utils import register_all_modules
        except Exception:
            try:
                from mmaction.utils.setup_env import register_all_modules
            except Exception:
                register_all_modules = None
        if register_all_modules:
            register_all_modules(init_default_scope=True)
        # Diagnostic info: versions, module files, sys.path
        try:
            import mmengine, mmcv, mmaction
            print(f"[stgcn_subproc] mmengine={getattr(mmengine,'__version__',None)} mmcv={getattr(mmcv,'__version__',None)}", file=sys.stderr)
            print(f"[stgcn_subproc] mmaction module file={getattr(mmaction,'__file__',None)}", file=sys.stderr)
        except Exception as _e:
            print(f"[stgcn_subproc] failed to import mmengine/mmcv/mmaction: {_e}", file=sys.stderr)

        try:
            import sys as _sys
            print(f"[stgcn_subproc] sys.path sample={_sys.path[:8]}", file=sys.stderr)
        except Exception:
            pass

        # Print MODELS registry keys to check RecognizerGCN presence
        try:
            from mmengine.registry import MODELS as _MODELS
            keys = list(getattr(_MODELS, 'module_dict', {}).keys())
            print(f"[stgcn_subproc] MODELS count: {len(keys)}", file=sys.stderr)
            # show a subset to avoid huge prints
            print(f"[stgcn_subproc] MODELS sample: {keys[:80]}", file=sys.stderr)
            print(f"[stgcn_subproc] RecognizerGCN in MODELS? {'RecognizerGCN' in keys}", file=sys.stderr)
        except Exception as _e:
            print(f"[stgcn_subproc] failed to inspect MODELS registry: {_e}", file=sys.stderr)

        cfg = Config.fromfile(args.config)
        # ensure work_dir exists in cfg (mmengine Runner.from_cfg expects cfg['work_dir'])
        if cfg.get('work_dir', None) is None:
            cfg.work_dir = str(Path('/tmp/work_dirs') / Path(args.config).stem)
        cfg.load_from = args.checkpoint
        cfg.test_dataloader.dataset.ann_file = args.ann

        # ensure DumpResults configured
        dump_metric = dict(type='DumpResults', out_file_path=str(args.out))
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator)
            cfg.test_evaluator = [e for e in cfg.test_evaluator if e.get('type') != 'DumpResults']
            cfg.test_evaluator.append(dump_metric)
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

        cfg.launcher = 'none'

        # Make environment settings safer for container execution
        if hasattr(cfg, 'env_cfg'):
            if hasattr(cfg.env_cfg, 'mp_cfg'):
                cfg.env_cfg.mp_cfg.mp_start_method = 'fork'
            if hasattr(cfg.env_cfg, 'dist_cfg'):
                cfg.env_cfg.dist_cfg.backend = 'gloo'
        # Try to resolve model type string to actual class object to avoid
        # registry scope mismatch issues (e.g., RecognizerGCN not found in
        # mmengine::model registry when DefaultScope isn't set to mmaction).
        try:
            model_type = None
            # support both dict-like and attribute access cfg.model
            try:
                mt = cfg.model.type
            except Exception:
                try:
                    mt = cfg.model.get('type') if isinstance(cfg.model, dict) else None
                except Exception:
                    mt = None

            if isinstance(mt, str) and mt:
                import importlib
                resolved = False
                try:
                    mod = importlib.import_module('mmaction.models.recognizers')
                    if hasattr(mod, mt):
                        cls = getattr(mod, mt)
                        try:
                            cfg.model.type = cls
                        except Exception:
                            try:
                                cfg.model['type'] = cls
                            except Exception:
                                pass
                        print(f"[stgcn_subproc] Resolved model type {mt} -> {cls}", file=sys.stderr)
                        resolved = True
                except Exception:
                    pass

                if not resolved:
                    try:
                        mod = importlib.import_module('mmaction.models')
                        if hasattr(mod, mt):
                            cls = getattr(mod, mt)
                            try:
                                cfg.model.type = cls
                            except Exception:
                                try:
                                    cfg.model['type'] = cls
                                except Exception:
                                    pass
                            print(f"[stgcn_subproc] Resolved model type {mt} -> {cls}", file=sys.stderr)
                            resolved = True
                    except Exception:
                        pass

                if not resolved:
                    print(f"[stgcn_subproc] Could not resolve model type {mt} to a class; will rely on registry", file=sys.stderr)
        except Exception as _e:
            print(f"[stgcn_subproc] model type resolution failed: {_e}", file=sys.stderr)

        # Synchronize mmaction registry into mmengine global registries to
        # ensure mmengine.MODELS.build can find classes registered by mmaction.
        try:
            import importlib
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
                        except Exception as _e:
                            print(f"[stgcn_subproc] failed to register {name}: {_e}", file=sys.stderr)
                print(f"[stgcn_subproc] synchronized {added} models into mmengine.MODELS", file=sys.stderr)
        except Exception as _e:
            print(f"[stgcn_subproc] registry sync failed: {_e}", file=sys.stderr)

        # Ensure cfg.model.type is a string (not a class object) so Config.pretty_text
        # and mmengine's formatting won't choke on invalid Python syntax like
        # "type=<class '...'>". If we earlier replaced the type with a class, convert
        # it back to the class name (string). Registry has been synced above so
        # mmengine.MODELS.build will find the class by name.
        try:
            # dict-like access
            if isinstance(cfg.model, dict):
                t = cfg.model.get('type')
                if isinstance(t, type):
                    cfg.model['type'] = t.__name__
                    print(f"[stgcn_subproc] converted cfg.model['type'] class -> '{t.__name__}'", file=sys.stderr)
            else:
                # attribute access
                try:
                    t = cfg.model.type
                except Exception:
                    t = None
                if isinstance(t, type):
                    try:
                        cfg.model.type = t.__name__
                    except Exception:
                        try:
                            cfg.model['type'] = t.__name__
                        except Exception:
                            pass
                    print(f"[stgcn_subproc] converted cfg.model.type class -> '{t.__name__}'", file=sys.stderr)
        except Exception as _e:
            print(f"[stgcn_subproc] failed to normalize cfg.model.type: {_e}", file=sys.stderr)

        # Determine which device to use: prefer explicit --device, then torch.cuda if available
        chosen_device = None
        try:
            if args.device:
                chosen_device = args.device
            else:
                try:
                    import torch
                    chosen_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                except Exception:
                    chosen_device = 'cpu'
        except Exception:
            chosen_device = 'cpu'

        # If cfg supports a device field, set it so Runner/model building uses the chosen device
        try:
            # mmengine configs sometimes have 'device' or runner/engine related settings
            if hasattr(cfg, 'device'):
                cfg.device = chosen_device
                print(f"[stgcn_subproc] set cfg.device='{chosen_device}'", file=sys.stderr)
            else:
                # fallback: set cfg.default_hooks.env_cfg or similar if present (best-effort)
                try:
                    cfg.device = chosen_device
                    print(f"[stgcn_subproc] set cfg.device='{chosen_device}' (fallback)", file=sys.stderr)
                except Exception:
                    pass
        except Exception as _e:
            print(f"[stgcn_subproc] failed to set cfg.device: {_e}", file=sys.stderr)

        print(f"[stgcn_subproc] chosen_device={chosen_device}", file=sys.stderr)

        # Try to reduce multiprocessing surface area: force dataloader workers to 0
        # and disable persistent_workers to avoid worker-process hangs in container
        try:
            td = cfg.get('test_dataloader', None)
            if td is not None:
                try:
                    # dict-like access
                    if isinstance(td, dict):
                        td['num_workers'] = 0
                        td['persistent_workers'] = False
                    else:
                        # attribute access
                        setattr(td, 'num_workers', 0)
                        setattr(td, 'persistent_workers', False)
                    print(f"[stgcn_subproc] forced test_dataloader num_workers=0 persistent_workers=False", file=sys.stderr)
                except Exception as _e:
                    print(f"[stgcn_subproc] failed to override dataloader workers: {_e}", file=sys.stderr)
        except Exception:
            pass

        # Instrumentation: log timestamps around runner creation and test
        try:
            print(f"[stgcn_subproc] [{time.strftime('%Y-%m-%d %H:%M:%S')}] about to call Runner.from_cfg", file=sys.stderr)
            sys.stderr.flush()
        except Exception:
            pass

        runner = Runner.from_cfg(cfg)

        try:
            print(f"[stgcn_subproc] [{time.strftime('%Y-%m-%d %H:%M:%S')}] Runner.from_cfg completed, about to call runner.test()", file=sys.stderr)
            sys.stderr.flush()
        except Exception:
            pass

        runner.test()

        try:
            print(f"[stgcn_subproc] [{time.strftime('%Y-%m-%d %H:%M:%S')}] runner.test() returned", file=sys.stderr)
            sys.stderr.flush()
        except Exception:
            pass
        return 0
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
