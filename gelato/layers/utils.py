from lasagne.layers.base import Layer, MergeLayer


def islayersub(cls):
    return not issubclass(cls, MergeLayer) and issubclass(cls, Layer)


def ismergesub(cls):
    return issubclass(cls, MergeLayer)
