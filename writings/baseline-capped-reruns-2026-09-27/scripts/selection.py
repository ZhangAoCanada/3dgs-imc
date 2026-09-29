import math

def qualified(metrics,reference):
    """Require all three finite metrics from a validated complete model bundle."""
    if set(metrics) != {'SSIM','PSNR','LPIPS'} or set(reference) != set(metrics):
        return False
    if not all(math.isfinite(v) for v in [*metrics.values(),*reference.values()]):
        return False
    return (metrics['SSIM'] <= reference['SSIM'] and
            metrics['PSNR'] <= reference['PSNR'] and
            metrics['LPIPS'] >= reference['LPIPS'])
