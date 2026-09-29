"""Numerical acceptance boundaries, including LPIPS direction and invalid metrics."""
import unittest
from selection import qualified

class SelectionTests(unittest.TestCase):
    reference=dict(SSIM=.7,PSNR=23.,LPIPS=.2)
    def test_equal(self):self.assertTrue(qualified(self.reference,self.reference))
    def test_all_weaker(self):self.assertTrue(qualified(dict(SSIM=.69,PSNR=22.,LPIPS=.21),self.reference))
    def test_any_stronger_fails_without_rounding(self):
        for metric,value in [('SSIM',.7000000001),('PSNR',23.0000000001),('LPIPS',.1999999999)]:
            with self.subTest(metric=metric):self.assertFalse(qualified(dict(self.reference,**{metric:value}),self.reference))
    def test_nonfinite_fails(self):
        for bad in [float('nan'),float('inf'),float('-inf')]:self.assertFalse(qualified(dict(self.reference,SSIM=bad),self.reference))
    def test_incomplete_fails(self):self.assertFalse(qualified(dict(SSIM=.5,PSNR=1),self.reference))

if __name__=='__main__':unittest.main()
