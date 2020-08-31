import numpy as np


def arrayCompat(nDim=2, *args):
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
    pre = [1] * nDim
    ret = []
    for i, a in enumerate(args):
        a = np.asarray(a)
        if a.ndim < nDim:
            newShape = pre + list(a.shape)[:-nDim]
            a = a.reshape(newShape)
            if hasattr(a, 'getReturnType'):
                newTyp = a.getReturnType(newShape)
                a = a.view(newTyp)

        elif a.ndim > nDim:
            raise ValueError(
                "Input number {0} has more than {1} dimensions".format(i, nDim)
            )
        ret.append(a)
    return ret

def toType(typ, *args):
    """ Cast all the args to the given type, but only if needed """
    args = [np.asarray(a) for a in args]
    return [a if isinstance(a, typ) else a.view(typ) for a in args]
