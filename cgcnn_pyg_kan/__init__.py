"""CGCNN PyG and MODNet-style models with MLP and KAN blocks."""

from .modnet import MODNetKAN

__all__ = ["CGCNN", "MODNetKAN"]


def __getattr__(name):
    if name == "CGCNN":
        from .model import CGCNN

        return CGCNN
    raise AttributeError(name)
