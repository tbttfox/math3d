import numpy as np

def arrayCompat(minDim=2, maxDim=2, *args):
    """ Make sure the given arrays have compatible dimensions

    Parameters
    ----------
    *args : np.array or list
        All of the arrays to convert
    minDim : int, optional
        The minimum number of dimensions that the arrays should have
        Defaults to 2
    maxDim : int, optional
        The maximum number of dimensions that the arrays should have
        If None, the dimensions are unbounded
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
    pre = [1] * minDim
    ret = []
    for i, a in enumerate(args):
        a = np.asarray(a)
        if a.ndim < minDim:
            newShape = pre + list(a.shape)[:-minDim]
            a = a.reshape(newShape)
        elif maxDim is not None and a.ndim > maxDim:
            raise ValueError("Input number {0} has more than {1} dimensions".format(i, maxDim))
        ret.append(a)
    return ret


