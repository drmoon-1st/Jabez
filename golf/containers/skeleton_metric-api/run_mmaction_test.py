from pathlib import Path
import sys
sys.path.insert(0, r'D:\Jabez\golf\containers\skeleton_metric-api')
from mmaction_client import start_mmaction_from_csv

csvp = Path(r'D:\Jabez\golf\containers\client\mmaction-client\skeleton2d.csv')
res = start_mmaction_from_csv(csvp, r'D:\Jabez\golf\containers\skeleton_metric-api\test_dest', 'testjob123', '2d', {})
print('returned:', res)
