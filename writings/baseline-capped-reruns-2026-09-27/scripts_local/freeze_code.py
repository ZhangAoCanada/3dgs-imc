"""Run on hust2; preserve native repos and record isolated adaptations."""
from pathlib import Path
import shutil,json,hashlib,difflib
ROOT=Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')
repos={'3dgs':'/data1/zhangao/repos/gaussian_splatting','pgsr':'/data1/ZHANGAo_2024/PGSR','mcmc':'/data1/zhangao/repos/3dgs-mcmc','scaffold':'/data1/zhangao/repos/Scaffold-GS'}
for method,src in repos.items():
 src=Path(src); dst=ROOT/'code'/method
 if dst.exists():raise RuntimeError('Code snapshot already exists: '+str(dst))
 dst.mkdir(parents=True)
 # All top-level Python and required local packages; no assets, environments, outputs.
 for f in src.iterdir():
  if f.is_file() and (f.suffix in ['.py','.sh'] or f.name.startswith('LICENSE')):shutil.copy2(f,dst/f.name)
 for name in ['arguments','scene','utils','gaussian_renderer','lpipsPyTorch']:
  if (src/name).exists():shutil.copytree(src/name,dst/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 originals={str(p.relative_to(dst)):p.read_bytes() for p in dst.rglob('*') if p.is_file()}
 f=dst/'utils/general_utils.py';s=f.read_text();s='import os\n'+s
 for token in ['random.seed(0)','np.random.seed(0)','torch.manual_seed(0)']:
  assert '    '+token in s;s=s.replace('    '+token,'    '+token.replace('(0)','(int(os.environ["GS_LSS_RUN_SEED"]))'))
 f.write_text(s)
 f=dst/'train.py';s=f.read_text()
 anchor='    args.save_iterations.append(args.iterations)'
 assert anchor in s
 s=s.replace(anchor,anchor+'\n    import json, os\n    os.makedirs(args.model_path, exist_ok=True)\n    with open(os.path.join(args.model_path, "full_args.json"), "w") as record:\n        json.dump(dict(vars(args), run_seed=int(os.environ["GS_LSS_RUN_SEED"])), record, indent=2)')
 # No socket server is needed for unattended training.
 s=s.replace('network_gui.init(args.ip, args.port)','pass # campaign: no GUI socket')
 if method=='scaffold':
  anchor='    # rendering\n';assert anchor in s;s=s.split(anchor)[0]
  # Upstream capture refers to nonexistent _local; preserve actual optimizer state.
  s=s.replace('gaussians.capture()', '{"optimizer": gaussians.optimizer.state_dict(), "iteration": iteration, "format": "optimizer-only; model and networks in iteration directory"}')
 f.write_text(s)
 # Make split invariants explicit before any optimization.
 f=dst/'scene/__init__.py';s=f.read_text();anchor='        if not self.loaded_iter:'
 guard='''        assert scene_info.train_cameras and scene_info.test_cameras, "Empty training or test split"
        if args.test_selection_path:
            from pathlib import Path
            expected = {Path(x.strip()).stem for x in open(args.test_selection_path) if x.strip()}
            actual = {Path(c.image_name).stem for c in scene_info.test_cameras}
            assert expected == actual, ("Test split mismatch", expected - actual, actual - expected)
        import json
        from pathlib import Path
        split_record = {"train": [c.image_name for c in scene_info.train_cameras], "test": [c.image_name for c in scene_info.test_cameras]}
        split_file = Path(self.model_path) / "resolved_split.json"
        if split_file.exists():
            previous = json.loads(split_file.read_text())
            assert all(sorted(previous[k]) == sorted(split_record[k]) for k in split_record)
        else:
            split_file.write_text(json.dumps(split_record, indent=2))

'''
 assert anchor in s;s=s.replace(anchor,guard+anchor,1);f.write_text(s)
 changes=[]
 for rel,b in originals.items():
  now=(dst/rel).read_bytes()
  if now!=b: changes.extend(difflib.unified_diff(b.decode().splitlines(True),now.decode().splitlines(True),fromfile='native/'+rel,tofile='campaign/'+rel))
 (dst/'campaign.patch').write_text(''.join(changes))
 (dst/'source_hashes.json').write_text(json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in originals.items()},indent=2))
 print(method,len(originals),flush=True)
