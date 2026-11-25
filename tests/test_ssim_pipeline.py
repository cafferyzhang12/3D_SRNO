import math
import pytest
from vemgsbm.mri_ssim_pipeline import choose


def test_ssim_selection_never_trades_psnr_for_ssim():
    baseline=dict(name='baseline',psnr=32.5,ssim=.905)
    rows=[dict(name='worse_psnr',psnr=32.499999,ssim=.92),
          dict(name='eligible',psnr=32.5,ssim=.906),
          dict(name='worse_ssim',psnr=33.,ssim=.904)]
    assert choose(rows,baseline)['name']=='eligible'
    assert choose([rows[0],rows[2]],baseline) is baseline


def test_ssim_selection_requires_finite_results():
    with pytest.raises(ValueError,match='Nonfinite'):
        choose([dict(psnr=math.nan,ssim=.9)],dict(psnr=32.,ssim=.9))


def test_ssim_selection_maximizes_ssim_among_eligible_candidates():
    rows=[dict(name='psnr',psnr=33.,ssim=.906),dict(name='ssim',psnr=32.6,ssim=.91)]
    assert choose(rows,dict(psnr=32.5,ssim=.905))['name']=='ssim'
