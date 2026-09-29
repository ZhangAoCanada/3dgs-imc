import os,json,sys,hashlib,platform,subprocess,importlib
from pathlib import Path
import torch,PIL
from PIL import features
record=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),torch=torch.__version__,torch_cuda=torch.version.cuda,pillow=PIL.__version__,jpeg=features.version_codec('jpg'),visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
record['extensions']={}
for name in ['diff_gaussian_rasterization','simple_knn']:
 module=importlib.import_module(name);base=Path(module.__file__).parent if module.__file__ else Path(next(iter(module.__path__)))
 record['extensions'][name]={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in base.rglob('*') if f.is_file() and f.suffix in ['.so','.py']}
print(json.dumps(record,indent=2))
