# debug_model_build.py
from mmengine.config import Config
from mmengine.runner import Runner
import torch, time
cfg = Config.fromfile('/mmaction2/api_src/modules/my_stgcnpp.py')
cfg.load_from = '/mmaction2/api_src/stgcn_70p.pth'
cfg.test_dataloader.dataset.ann_file = '/tmp/test_ann_bef913a9.pkl'
print('Config loaded')
# try to build model directly
try:
    from mmaction.models import build_model
    print('about to build model')
    t0 = time.time()
    model = build_model(cfg.model)
    print('model built in', time.time()-t0)
except Exception as e:
    print('model build failed:', e)
# try to load checkpoint
try:
    print('about to load checkpoint')
    t0 = time.time()
    ckpt = torch.load(cfg.load_from, map_location='cpu')
    print('checkpoint loaded in', time.time()-t0, 'keys:', list(ckpt.keys())[:20])
except Exception as e:
    print('checkpoint load failed:', e)