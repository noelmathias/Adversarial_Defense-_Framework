# models/cnn.py
''' """
Backward-compatibility shim.

All existing imports of the form:
    from models.cnn import DefenseCNN
continue to resolve correctly to ResNet18.

Do not put any logic here. The canonical implementation is models/resnet.py.
"""

from models.resnet import ResNet18 as DefenseCNN  # noqa: F401

__all__ = ["DefenseCNN"]'''