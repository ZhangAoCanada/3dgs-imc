"""Admission rules for independent research pipelines on shared local GPUs."""
import math

HEADROOM_MIB = 4096
MAX_PIPELINES = 2


def admission(gpu, active, candidate, reserved=False):
    """Use peak budgets and live free memory; never share with foreign jobs."""
    if reserved:
        return False, 'reserved'
    if gpu.get('foreign_pids'):
        return False, 'other_compute_process'
    if len(active) >= MAX_PIPELINES:
        return False, 'two_pipeline_limit'
    if not active:
        if gpu['utilization'] >= 10 or gpu['used_mib'] >= 1024:
            return False, 'waiting_for_idle_device'
        # An oversized estimate requests exclusive execution. Historical
        # single-GPU runs already used this hardware; do not make these cases
        # permanently unschedulable merely because the safety budget exceeds it.
        return True, 'exclusive' if candidate.get('exclusive') else 'idle_device'
    if candidate.get('exclusive') or any(j.get('exclusive') for j in active):
        return False, 'exclusive_pipeline'
    need = candidate['budget_mib']
    budgets = [max(j['budget_mib'], j.get('peak_mib', 0)) for j in active]
    growth = sum(max(0, b-j.get('current_mib', 0)) for b,j in zip(budgets,active))
    if sum(budgets)+need+HEADROOM_MIB > gpu['total_mib']:
        return False, 'peak_budgets_exceed_capacity'
    if need+growth+HEADROOM_MIB > gpu['free_mib']:
        return False, 'insufficient_free_memory_for_growth'
    return True, 'shared_with_campaign_pipeline'


def updated_budget(profile, observed_mib):
    return max(profile.get('budget_mib',10240), math.ceil(observed_mib*1.25+1024))


def is_oom(text):
    text = text.lower()
    return any(s in text for s in ['cuda out of memory', 'cuda error: out of memory',
                                   'torch.outofmemoryerror', 'cublas_status_alloc_failed'])


def next_output(root, case_id, seed):
    """Preserve incomplete/failed attempts; retries never overwrite a path."""
    base = root/'attempts_local'/case_id/f'seed_{seed:06d}'
    if not base.exists():
        return base
    retry = 1
    while True:
        candidate = base.with_name(base.name+f'_retry_{retry:03d}')
        if not candidate.exists():
            return candidate
        retry += 1
