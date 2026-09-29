"""CPU checks for the explicitly authorized GPU 1 sharing exception."""
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'baseline-capped-reruns-2026-09-27/scripts_local'
sys.path.insert(0, str(SCRIPTS))
from gpu_policy import admission


class SharingPolicy(unittest.TestCase):
    def setUp(self):
        self.gpu = dict(uuid='gpu-one', total_mib=32607, used_mib=11070,
                        free_mib=21040, utilization=50, foreign_pids=[123],
                        apps=[dict(pid=123, uid=1003, start_ticks=456, memory_mib=11038,
                                   campaign_id=None)])
        self.policy = dict(enabled=True, gpu_uuid='gpu-one',
                           allowed_foreign_processes=[dict(pid=123, uid=1003, start_ticks=456)],
                           max_pipeline_budget_mib=10240, headroom_mib=8192)
        self.candidate = dict(budget_mib=10240, observed_pipeline_peak_mib=2800, exclusive=False)

    def check(self, allowed, active=None, reserved=False):
        result = admission(self.gpu, active or [], self.candidate, reserved, self.policy)
        self.assertEqual(result[0], allowed, result)

    def test_existing_job_with_memory_reserve(self):
        self.check(True)

    def test_default_still_excludes_foreign_jobs(self):
        self.assertEqual(admission(self.gpu, [], self.candidate),
                         (False, 'other_compute_process'))

    def test_birth_time_and_uid_are_required(self):
        for key in ['uid', 'start_ticks']:
            with self.subTest(key=key):
                saved = self.gpu['apps'][0][key]
                self.gpu['apps'][0][key] += 1
                self.check(False)
                self.gpu['apps'][0][key] = saved

    def test_new_or_unidentified_process_blocks_admission(self):
        self.gpu['foreign_pids'].append(789)
        self.check(False)
        self.gpu['apps'].append(dict(pid=789, uid=1003, start_ticks=456))
        self.check(False)

    def test_device_binding(self):
        self.gpu['uuid'] = 'another-gpu'
        self.check(False)

    def test_one_pipeline_only_and_reservation(self):
        self.check(False, active=[self.candidate])
        self.check(False, reserved=True)

    def test_large_exclusive_and_unprofiled_jobs_wait(self):
        for change in [dict(budget_mib=10241), dict(exclusive=True),
                       dict(observed_pipeline_peak_mib=0),
                       dict(observed_pipeline_peak_mib=12000)]:
            with self.subTest(change=change):
                previous = self.candidate.copy()
                self.candidate.update(change)
                self.check(False)
                self.candidate = previous

    def test_free_memory_and_total_capacity_both_checked(self):
        self.gpu['free_mib'] = 18431
        self.check(False)
        self.gpu['free_mib'] = 21040
        self.gpu['used_mib'] = 16000
        self.check(False)

    def test_reserve_cannot_be_lowered_below_eight_gib(self):
        self.policy['headroom_mib'] = 1
        self.gpu['free_mib'] = 18431
        self.check(False)

    def test_disabled_or_incomplete_policy_fails_closed(self):
        self.policy['enabled'] = False
        self.check(False)
        self.policy['enabled'] = True
        del self.policy['allowed_foreign_processes'][0]['start_ticks']
        self.check(False)

    def test_missing_process_metadata_fails_closed(self):
        self.gpu['apps'][0]['uid'] = None
        self.check(False)

    def test_normal_two_pipeline_policy_after_foreign_job_leaves(self):
        self.gpu.update(foreign_pids=[], apps=[], used_mib=0, free_mib=32607, utilization=0)
        self.check(True)
        self.gpu.update(used_mib=3000, free_mib=29600, utilization=90)
        self.check(True, active=[dict(budget_mib=10240, current_mib=3000, peak_mib=3000)])


if __name__ == '__main__':
    unittest.main()
