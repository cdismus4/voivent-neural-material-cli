"""VNM1 learning CLI (productization P1).

The package is split so that the input layer (:mod:`vnm1.inputs`,
:mod:`vnm1.imaging`) depends only on numpy + Pillow and never imports
torch.  The training core (which needs torch) is added in implementation
step B and must stay behind that boundary.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
