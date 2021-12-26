from __future__ import print_function, absolute_import
import numpy as np


def arrayCompat(*args, **kwargs):
    """ Make sure the given arrays have compatible dimensions

    Parameters
    ----------
    *args : np.array or list
        All of the arrays to convert
    nDim : int, optional
        The number of dimensions that the arrays should have
        Defaults to 2

    Returns
    -------
    list of np.array:
        All of the arguments converted into the proper number of dimensions
        and converted to numpy arrays if applicable

    Raises
    ------
    ValueError:
        If the number of dimensions of an array is greater than maxDim

    """
    nDim = kwargs.get("nDim", 2)

    pre = [1] * nDim
    ret = []
    for i, a in enumerate(args):
        a = asarray(a)
        if a.ndim < nDim:
            newShape = (pre + list(a.shape))[-nDim:]
            a = a.reshape(newShape)
            if hasattr(a, "getReturnType"):
                newTyp = a.getReturnType(newShape)
                a = a.view(newTyp)

        elif a.ndim > nDim:
            raise ValueError(
                "Input number {0} has more than {1} dimensions".format(i, nDim)
            )
        ret.append(a)
    if len(args) == 1:
        return ret[0]
    return ret


def asarray(ary):
    """ Return a numpy array of the given object
    If it is already a numpy array type (or any of the math3d subclasses)
    just return the object
    """
    if not isinstance(ary, np.ndarray):
        return np.asarray(ary)
    return ary


def toType(typ, *args):
    """ Cast all the args to the given type, but only if needed """
    args = [asarray(a) for a in args]
    return [a if isinstance(a, typ) else a.view(typ) for a in args]


def sliceLength(s):
    """ Return the count of values that a slice would create """
    start, stop, step = s.start, s.stop, s.step

    start = start or 0
    step = step or 1
    rng = stop - start

    r1 = rng // step
    r2 = 1 if rng % step else 0
    return r1 + r2
