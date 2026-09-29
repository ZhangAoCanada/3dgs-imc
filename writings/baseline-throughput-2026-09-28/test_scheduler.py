"""CPU regression checks for GPU admission, process ownership and split validation."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]/'baseline-capped-reruns-2026-09-27/scripts_local'
sys.path.insert(0, str(SCRIPTS))
import coordinator_multi as scheduler
from gpu_policy import admission, is_oom, next_output, updated_budget
from secondary_evaluation import validate


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.gpu = dict(total_mib=32607, used_mib=12692, free_mib=19418,
                        utilization=90, foreign_pids=[])
        self.active = [dict(budget_mib=16407, current_mib=12668, peak_mib=12668)]
        self.candidate = dict(budget_mib=10240)

    def test_pair_and_future_growth(self):
        self.assertTrue(admission(self.gpu, self.active, self.candidate)[0])
        self.gpu['free_mib'] = 17000
        self.assertEqual(admission(self.gpu, self.active, self.candidate),
                         (False, 'insufficient_free_memory_for_growth'))

    def test_foreign_process_blocks_even_when_memory_fits(self):
        self.gpu['foreign_pids'] = [123]
        self.assertEqual(admission(self.gpu, self.active, self.candidate),
                         (False, 'other_compute_process'))

    def test_busy_then_free(self):
        self.assertFalse(admission(self.gpu, [], self.candidate)[0])
        self.gpu.update(used_mib=100, free_mib=32500, utilization=0)
        self.assertTrue(admission(self.gpu, [], self.candidate)[0])

    def test_no_third_pipeline(self):
        self.assertEqual(admission(self.gpu, self.active*2, self.candidate),
                         (False, 'two_pipeline_limit'))

    def test_exclusive_estimate_still_runs_alone(self):
        self.candidate.update(exclusive=True, budget_mib=40000)
        self.assertFalse(admission(self.gpu, self.active, self.candidate)[0])
        self.gpu.update(used_mib=100, free_mib=32500, utilization=0)
        self.assertTrue(admission(self.gpu, [], self.candidate)[0])

    def test_budget_and_reservation(self):
        self.candidate['budget_mib'] = 18000
        self.assertEqual(admission(self.gpu, self.active, self.candidate),
                         (False, 'peak_budgets_exceed_capacity'))
        self.assertEqual(admission(self.gpu, [], self.candidate, reserved=True),
                         (False, 'reserved'))
        self.assertEqual(updated_budget({'budget_mib':10000}, 12000), 16024)

    def test_retry_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            a=next_output(root, 'baseline_01', 3)
            a.mkdir(parents=True)
            (a/'sentinel').write_text('retain')
            b=next_output(root, 'baseline_01', 3)
            b.mkdir()
            c=next_output(root, 'baseline_01', 3)
            self.assertEqual(b.name, 'seed_000003_retry_001')
            self.assertEqual(c.name, 'seed_000003_retry_002')
            self.assertEqual((a/'sentinel').read_text(), 'retain')

    def test_oom_classification(self):
        self.assertTrue(is_oom('torch.OutOfMemoryError: CUDA out of memory.'))
        self.assertTrue(is_oom('CUDA error: CUBLAS_STATUS_ALLOC_FAILED'))
        self.assertFalse(is_oom('AssertionError: wrong camera split'))

    def test_process_birth_identity(self):
        identity=scheduler.proc(os.getpid())
        job=dict(runner_pid=os.getpid(), runner_start_ticks=identity['start_ticks'])
        self.assertTrue(scheduler.runner_alive(job))
        job['runner_start_ticks'] += 1
        self.assertFalse(scheduler.runner_alive(job))

    def test_gpu_ownership_by_session_not_username(self):
        jobs={'baseline_01':dict(id='baseline_01', runner_pid=100, runner_start_ticks=20,
                               gpu='0', peak_mib=0, output='/unused')}
        identities={100:dict(uid=os.getuid(), session=100, start_ticks=20),
                    101:dict(uid=os.getuid(), session=100, start_ticks=21),
                    102:dict(uid=os.getuid(), session=999, start_ticks=22)}
        responses=['0, uuid0, 32607, 11000, 21000, 80\n1, uuid1, 32607, 200, 32400, 0\n',
                   'uuid0, 101, 9000\nuuid0, 102, 2000\n']
        with patch.object(scheduler.subprocess, 'check_output', side_effect=responses), \
             patch.object(scheduler, 'proc', side_effect=lambda pid:identities.get(pid)):
            result=scheduler.gpu_snapshot(jobs)
        self.assertEqual(result['0']['foreign_pids'], [102])
        self.assertEqual(jobs['baseline_01']['peak_mib'], 9000)


class SplitTests(unittest.TestCase):
    def fixture(self, root, names, frozen=None):
        model=root/'model'; test=root/'input'
        (model/'test/ours_30000').mkdir(parents=True)
        test.mkdir()
        def put(path, value):path.write_text(json.dumps(value))
        put(test/'frozen.json', {'images':[{'name':n} for n in (frozen or ['0209.png','0210.png'])]})
        put(model/'resolved_split.json', {'test':names})
        mapping=[{'camera':name,'file':f'{i:05d}.png'} for i,name in enumerate(names)]
        put(model/'test/ours_30000/camera_map.json', mapping)
        put(model/'results.json', {'ours_30000':dict(SSIM=.3, PSNR=10., LPIPS=.6)})
        put(model/'per_view.json', {'ours_30000':{m:{x['file']:v for x in mapping}
                                                 for m,v in dict(SSIM=.3, PSNR=10., LPIPS=.6).items()}})
        return model,test

    def test_extensions_and_stems_both_valid(self):
        for names in [['0209.png','0210.png'], ['0209','0210']]:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                model,test=self.fixture(Path(directory), names)
                self.assertEqual(validate(model,test,30000,2)['PSNR'],10.)

    def test_wrong_image_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model,test=self.fixture(Path(directory), ['0209.png','9999.png'])
            with self.assertRaises(AssertionError):validate(model,test,30000,2)

    def test_ambiguous_filename_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model,test=self.fixture(Path(directory), ['0209.png','0209.jpg'], ['0209.png','0209.jpg'])
            with self.assertRaises(AssertionError):validate(model,test,30000,2)


if __name__=='__main__':
    unittest.main(verbosity=2)
