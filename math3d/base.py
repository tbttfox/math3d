from __future__ import print_function, absolute_import
import numpy as np
from .utils import arrayCompat


class MathBase(np.ndarray):
    def asArray(self):
        """ Return the array type of this object

        Returns
        -------
        ArrayType:
            The current object up-cast into a length-1 array
        """
        return self[None, ...]

    def asNdArray(self):
        """ Return this object as a regular numpy array

        Returns
        -------
        ndarray:
            The current object as a numpy array
        """
        return self.view(np.ndarray)

    @classmethod
    def getReturnType(cls, shape, idx=None):
        """ Get the type for any return values based on the shape of the return value
        This is mainly for internal use

        Parameters
        ----------
        shape: tuple
            The shape of the output

        Returns
        -------
        type
            The type that the output should have
        """
        return np.ndarray

    def __getitem__(self, idx):
        ret = super(MathBase, self).__getitem__(idx)
        typ = self.getReturnType(ret.shape, idx)
        if typ is None:
            return ret
        return ret.view(typ)


class ArrayBase(MathBase):
    def __getitem__(self, idx):
        ret = super(ArrayBase, self).__getitem__(idx)
        # If we're getting columns from the array then we expect ndarrays back, not math3d types
        # However slicing matrices should give vectors, so only do this is ndim == 2
        if self.ndim == 2 and isinstance(idx, tuple):
            # If we're getting multiple indices And the second index isn't a ":"
            # Then we're getting columns, and therefore want an ndarray
            try:
                if len(idx) > 1:
                    if not isinstance(idx[-1], slice):
                        return ret.view(np.ndarray)
                    elif idx[-1] != slice(None, None, None):
                        return ret.view(np.ndarray)
            except ValueError:
                print(idx)
                raise
        typ = self.getReturnType(ret.shape, idx)
        if typ is None:
            return ret
        return ret.view(typ)

    def _convertToCompatibleType(self, value):
        """ Convert a value to a type compatible with
        Appending, extending, or inserting
        """
        # The default implementation just returns
        return value

    def appended(self, value):
        """ Return a copy of the array with the value appended

        Parameters
        ----------
        value: iterable
            An iterable to be appended as-is to the end of this array
        """
        value = self._convertToCompatibleType(value)
        newShp = list(self.shape)
        newShp[0] += 1
        ret = np.resize(self, newShp).view(type(self))
        ret[-1] = value
        return ret

    def extended(self, value):
        """ Return a copy of the array extended with the given values

        Parameters
        ----------
        value: iterable
            An iterable to be appended as-is to the end of this array
        """
        value = self._convertToCompatibleType(value)
        value = arrayCompat(value, nDim=self.ndim)
        newShp = list(self.shape)
        newShp[0] += len(value)
        ret = np.resize(self, newShp).view(type(self))
        ret[-len(value):] = value
        return ret

    def inserted(self, idx, value):
        """ Return a copy of the array with the value inserted at the given position

        Parameters
        ----------
        idx: int
            The index where value will be inserted
        value: iterable
            An iterable to be appended as-is to the end of this array
        """
        value = self._convertToCompatibleType(value)
        value = arrayCompat(value, nDim=self.ndim)
        newShp = list(self.shape)
        newShp[0] += len(value)
        ret = np.resize(self, newShp).view(type(self))
        ret[len(value) + idx:] = self[idx:]
        ret[idx: len(value) + idx] = value
        return ret.view(type(self))
