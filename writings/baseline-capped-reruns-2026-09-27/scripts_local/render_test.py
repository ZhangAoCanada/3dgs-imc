"""Render the complete prescribed test split with the method's native renderer."""
import argparse,json,sys,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--method',required=True);p.add_argument('--model',required=True);p.add_argument('--iteration',type=int,default=30000);a=p.parse_args()
sys.path.insert(0,a.repo)
import torch,torchvision
from argparse import Namespace
from arguments import ModelParams,PipelineParams
from scene import Scene,GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
safe_state(False)
args=Namespace(**json.loads((Path(a.model)/'full_args.json').read_text()))
parser=argparse.ArgumentParser();model=ModelParams(parser);pipe=PipelineParams(parser)
dataset=model.extract(args);pipeline=pipe.extract(args)
with torch.no_grad():
 if a.method=='scaffold':
  gaussians=GaussianModel(dataset.feat_dim,dataset.n_offsets,dataset.voxel_size,dataset.update_depth,dataset.update_init_factor,dataset.update_hierachy_factor,dataset.use_feat_bank,dataset.appearance_dim,dataset.ratio,dataset.add_opacity_dist,dataset.add_cov_dist,dataset.add_color_dist)
 else:gaussians=GaussianModel(dataset.sh_degree)
 scene=Scene(dataset,gaussians,load_iteration=a.iteration,shuffle=False)
 if a.method=='scaffold':
  gaussians.eval()
  # Novel test IDs need not be valid indices into training embeddings.
  # Use the same mean learned appearance for every test camera.
  class MeanAppearance(torch.nn.Module):
   def __init__(self,mean):super().__init__();self.register_buffer('mean',mean)
   def forward(self,indices):return self.mean.expand(indices.shape[0],-1)
  if dataset.appearance_dim>0:
   indices=torch.arange(len(scene.getTrainCameras()),device='cuda')
   mean=gaussians.get_appearance(indices).mean(0,keepdim=True)
   gaussians.embedding_appearance=MeanAppearance(mean)
  from gaussian_renderer import prefilter_voxel
 bg=torch.tensor([1,1,1] if dataset.white_background else [0,0,0],dtype=torch.float32,device='cuda')
 root=Path(a.model)/'test'/('ours_'+str(a.iteration));(root/'renders').mkdir(parents=True,exist_ok=True);(root/'gt').mkdir(exist_ok=True)
 names=[]
 for i,view in enumerate(scene.getTestCameras()):
  extra={}
  if a.method=='scaffold':extra['visible_mask']=prefilter_voxel(view,gaussians,pipeline,bg)
  rendered=render(view,gaussians,pipeline,bg,**extra)['render'].clamp(0,1)
  gt=view.get_image()[0] if a.method=='pgsr' else view.original_image[:3]
  assert rendered.shape==gt.shape,(rendered.shape,gt.shape)
  assert torch.isfinite(rendered).all() and torch.isfinite(gt).all()
  name=f'{i:05d}.png';torchvision.utils.save_image(rendered,root/'renders'/name);torchvision.utils.save_image(gt,root/'gt'/name)
  names.append(dict(file=name,camera=view.image_name));print(i+1,len(scene.getTestCameras()),view.image_name,flush=True)
 (root/'camera_map.json').write_text(json.dumps(names,indent=2))
