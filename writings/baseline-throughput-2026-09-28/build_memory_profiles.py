"""CPU-only initial GPU budgets from completed runs; no model deserialization."""
import json
import math
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path('/data1/zhangao/outputs/GS-LSS-baseline-capped-reruns-20260927')
profiles = {}
for config in json.loads((ROOT/'configurations_local.json').read_text()):
    runs = [p for parent in ['attempts_local', 'attempts']
            for p in (ROOT/parent/config['id']).glob('seed_*')
            if (p/'complete.json').exists() or (p/'failure.json').exists()]
    states = [p/'chkpnt30000.pth' for p in runs if (p/'chkpnt30000.pth').exists()]
    state_bytes = max((p.stat().st_size for p in states), default=0)
    # For capped methods, reserve for the configured capacity, not just the
    # point count left after pruning in one historical run.
    args = config.get('extra_args', [])
    cap = max([int(args[i+1]) for i,v in enumerate(args)
               if v in ['--cap_max','--max_all_points']], default=0)
    state_bound = max(state_bytes, cap*768)
    estimated_mib = max(10240, math.ceil(2.5*state_bound/2**20 + 6144))
    sources = [{'kind':'saved_optimizer_size', 'path':str(p), 'bytes':p.stat().st_size} for p in states]
    recorded_peak_mib = 0
    if config['method_key'] == '3dgs':
        for run in runs:
            events = EventAccumulator(str(run), size_guidance={'scalars':0, 'images':1, 'histograms':1, 'tensors':1})
            events.Reload()
            tag = 'train_duration/peak_gpu_mem_GB'
            if tag in events.Tags()['scalars']:
                values = events.Scalars(tag)
                if values and values[-1].step == 30000:
                    peak = max(x.value for x in values)*1024
                    recorded_peak_mib = max(recorded_peak_mib, peak)
                    sources.append({'kind':'recorded_torch_peak_GiB', 'path':str(run),
                                    'tag':tag, 'peak_mib':peak, 'last_step':values[-1].step})
    budget = max(estimated_mib, math.ceil(recorded_peak_mib*1.25+1024))
    profiles[config['id']] = {'budget_mib':budget, 'initial_budget_mib':budget,
        'recorded_training_peak_mib':recorded_peak_mib, 'capacity_points':cap,
        'sources':sources, 'budget_is_estimate':True, 'exclusive':budget>24576}
    print(config['id'], 'budget_MiB', budget, 'training_peak_MiB', round(recorded_peak_mib), flush=True)
Path(__file__).with_name('memory_profiles.initial.json').write_text(json.dumps(profiles,indent=2)+'\n')
