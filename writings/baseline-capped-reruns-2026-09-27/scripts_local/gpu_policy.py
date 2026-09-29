"""Admission rules for independent research pipelines on shared local GPUs."""
import math

HEADROOM_MIB = 4096
MAX_PIPELINES = 2


def admission(gpu, active, candidate, reserved=False, sharing=None):
    """Use peak budgets; foreign sharing requires a scoped process exception."""
    if reserved:
        return False, 'reserved'
    if gpu.get('foreign_pids'):
        return foreign_admission(gpu, active, candidate, sharing)
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


def foreign_admission(gpu, active, candidate, sharing):
    """One small pipeline beside explicitly identified existing processes."""
    if not isinstance(sharing, dict) or sharing.get('enabled') is not True:
        return False, 'other_compute_process'
    try:
        if sharing['gpu_uuid'] != gpu['uuid']:
            return False, 'sharing_gpu_identity_mismatch'
        permitted = {(int(p['pid']), int(p['uid']), int(p['start_ticks']))
                     for p in sharing['allowed_foreign_processes']}
        foreign = [p for p in gpu['apps'] if p['pid'] in gpu['foreign_pids']]
        if {p['pid'] for p in foreign} != set(gpu['foreign_pids']):
            return False, 'unidentified_foreign_process'
        observed = {(int(p['pid']), int(p['uid']), int(p['start_ticks'])) for p in foreign}
        if not observed or not observed <= permitted:
            return False, 'unapproved_foreign_process'
        if active:
            return False, 'one_pipeline_limit_with_foreign_job'
        if candidate.get('exclusive'):
            return False, 'exclusive_pipeline'
        need = int(candidate['budget_mib'])
        limit = min(10240, int(sharing.get('max_pipeline_budget_mib', 10240)))
        peak = candidate.get('observed_pipeline_peak_mib') or candidate.get('recorded_training_peak_mib', 0)
        if need <= 0 or need > limit or not 0 < peak <= need:
            return False, 'requires_small_profiled_pipeline'
        reserve = max(8192, int(sharing.get('headroom_mib', 8192)))
        if need + reserve > gpu['free_mib'] or gpu['used_mib'] + need + reserve > gpu['total_mib']:
            return False, 'insufficient_memory_for_foreign_sharing'
        return True, 'shared_with_authorized_existing_process'
    except (KeyError, TypeError, ValueError):
        return False, 'invalid_foreign_sharing_identity_or_policy'


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
