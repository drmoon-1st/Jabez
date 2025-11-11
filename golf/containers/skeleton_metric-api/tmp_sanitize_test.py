from pathlib import Path
p=Path(r'D:/Jabez/golf/containers/client/mmaction-client/skeleton2d.csv')
import pandas as pd
print('original cols:', list(pd.read_csv(p, nrows=0).columns)[:30])
# create a temp copy and sanitize
import sys
sys.path.insert(0, r'D:\Jabez\golf\containers\skeleton_metric-api')
from mmaction_client import sanitize_skeleton_csv
p2=Path(r'D:/Jabez/golf/containers/skeleton_metric-api/tmp_skel.csv')
import shutil
shutil.copy2(p, p2)
sanitize_skeleton_csv(p2)
print('sanitized cols:', list(pd.read_csv(p2, nrows=0).columns)[:60])
