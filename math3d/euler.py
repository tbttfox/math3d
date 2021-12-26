from __future__ import print_function, absolute_import
import numpy as np
from .base import MathBase, ArrayBase


class Euler(MathBase):
    """ A 3d Euler rotation with arbitrary axis order

    Parameters
    ----------
    input_array : iterable, optional
        The input value to create the euler array. It must be an iterable of length 3
        If not given, defaults to (0, 0, 0)
    order: str, optional
        The order in which the axis rotations are applied
        It must be one of these options ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyz']
        Defaults to 'xyz'
    degrees: bool, optional
        Whether the angles are given in degrees or radians. Defaults to False (radians)
    """

    def __new__(cls, input_array=None, order="xyz", degrees=False):
        if input_array is None:
            ary = np.zeros(3)
        else:
            ary = np.asarray(input_array, dtype=float)
        if ary.size != 3:
            raise ValueError(
                "Initializer for Vector{0} must be of length {0}".format(3)
            )

        ret = ary.view(cls)
        ret.order = order.lower()
        ret._degrees = degrees
        return ret

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.order = getattr(obj, "order", "xyz")
        self._degrees = getattr(obj, "degrees", False)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        ret = super(Euler, self).__repr__()
        ret = ret[:-1] + ', "{0}", degrees={1})'.format(self.order, self._degrees)
        return ret

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, val):
        self[0] = val

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, val):
        self[1] = val

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, val):
        self[2] = val

    @property
    def degrees(self):
        return self._degrees

    def asRadians(self):
        """ Return a copy of this object converted to radians

        Returns
        -------
        Euler
            The current orientation in radians
        """
        if self.degrees:
            ret = np.deg2rad(self)
            ret._degrees = False
            return ret
        return self.copy()

    def asDegrees(self):
        """ Return a copy of this object as converted to degrees

        Returns
        -------
        Euler
            The current orientation in Degrees

        """
        if not self.degrees:
            ret = np.rad2deg(self)
            ret._degrees = True
            return ret
        return self.copy()

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
        if not shape:
            return None
        if len(shape) == 1:
            if shape[0] == 3:
                return cls
        elif len(shape) == 2:
            if shape[-1] == 3:
                return EulerArray
        return np.ndarray

    def asMatrix(self):
        """ Convert this euler object to a Matrix3

        Returns
        -------
        Matrix3
            The current orientation as a matrix

        """
        return self.asArray().asMatrixArray()[0]

    def asQuaternion(self):
        """ Convert this euler object to a Quaternion

        Returns
        -------
        Quaternion
            The current orientation as a quaternion
        """
        return self.asArray().asQuaternionArray()[0]

    def asNewOrder(self, order):
        """ Create a new euler object that represents the same orientation
        but composed with a different rotation order

        Returns
        -------
        Euler
            The same spatial orientation but with a different axis order
        """
        return self.asArray().asNewOrder(order)[0]


class EulerArray(ArrayBase):
    """ An array of 3d Euler rotations with a common axis order

    Parameters
    ----------
    input_array : iterable
        The input value to create the euler array. It must be an iterable with a length
        multiple of 3
    order: str, optional
        The order in which the axis rotations are applied
        It must be one of these options ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyz']
        Defaults to 'xyz'
    degrees: bool, optional
        Whether the angles are given in degrees or radians. Defaults to False (radians)
    """
    ORDER_PARITY = {
        "xyz": ((0, 1, 2), 0),
        "xzy": ((0, 2, 1), 1),
        "yxz": ((1, 0, 2), 1),
        "yzx": ((1, 2, 0), 0),
        "zxy": ((2, 0, 1), 0),
        "zyx": ((2, 1, 0), 1),
    }
    def __new__(cls, input_array=None, order="xyz", degrees=False):
        if input_array is None:
            input_array = np.array([])
        ary = np.asarray(input_array, dtype=float)
        ary = ary.reshape((-1, 3))
        ret = ary.view(cls)
        order = order.lower()
        if sorted(list(order)) != ["x", "y", "z"]:
            raise ValueError("Order is not a permutation of 'xyz'")
        ret.order = order
        ret._degrees = degrees
        return ret

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.order = getattr(obj, "order", "xyz")
        self._degrees = getattr(obj, "degrees", False)

    def __repr__(self):
        ret = super(EulerArray, self).__repr__()
        ret = ret[:-1] + ', "{0}", degrees={1})'.format(self.order, self._degrees)
        return ret

    def __str__(self):
        return repr(self)

    @property
    def x(self):
        return self[:, 0]

    @x.setter
    def x(self, val):
        self[:, 0] = val

    @property
    def y(self):
        return self[:, 1]

    @y.setter
    def y(self, val):
        self[:, 1] = val

    @property
    def z(self):
        return self[:, 2]

    @z.setter
    def z(self, val):
        self[:, 2] = val

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
        if not shape:
            return None
        if len(shape) == 2:
            if shape[1] == 3:
                return cls
        elif len(shape) == 1:
            if shape[0] == 3:
                return Euler
        return np.ndarray

    def _convertToCompatibleType(self, value):
        """ Convert a value to a type compatible with
        Appending, extending, or inserting
        """
        from .quaternion import Quaternion, QuaternionArray
        from .matrixN import MatrixN, MatrixNArray
        if isinstance(value, (MatrixNArray, QuaternionArray)):
            return value.asEulerArray(degrees=self._degrees, order=self.order)
        elif isinstance(value, (MatrixN, Quaternion)):
            return value.asEuler(degrees=self._degrees, order=self.order)
        return value

    @property
    def degrees(self):
        return self._degrees

    def asRadians(self):
        """ Return a copy of this array as converted to radians

        Returns
        -------
        EulerArray
            The current orientations in radians
        """
        if self.degrees:
            ret = np.deg2rad(self)
            ret._degrees = False
            return ret
        return self.copy()

    def asDegrees(self):
        """ Return a copy of this array as converted to degrees

        Returns
        -------
        EulerArray
            The current orientations in degrees
        """
        if not self.degrees:
            ret = np.rad2deg(self)
            ret._degrees = True
            return ret
        return self.copy()

    def asNewOrder(self, order):
        """ Create a new EulerArray object that represents the same orientations
        but composed with a different rotation order

        Returns
        -------
        EulerArray
            The same spatial orientations but with a different axis order
        """
        q = self.asQuaternionArray()
        return q.asEulerArray(order=order, degrees=self.degrees)

    @classmethod
    def zeros(cls, length):
        """ Alternate constructor to build an array of all-zero Euler orientations

        Parameters
        ----------
        length: int
            The length of the array to create
        """
        return cls(np.zeros((length, 3)))

    def asQuaternionArray(self):
        """ Convert this EulerArray object to a QuaternionArray

        Returns
        -------
        QuaternionArray
            The current orientations as a quaternionArray
        """
        from .quaternion import QuaternionArray

        (i, j, k), parity = self.ORDER_PARITY[self.order]

        cp = self.asRadians().copy()
        cp *= 0.5
        if parity:
            cp[:, j] *= -1

        c = np.cos(cp)
        s = np.sin(cp)

        cc = c[:, i] * c[:, k]
        cs = c[:, i] * s[:, k]
        sc = s[:, i] * c[:, k]
        ss = s[:, i] * s[:, k]

        a = np.zeros((len(self), 3))
        a[:, i] = c[:, j] * sc - s[:, j] * cs
        a[:, j] = c[:, j] * ss + s[:, j] * cc
        a[:, k] = c[:, j] * cs - s[:, j] * sc

        q = QuaternionArray.eye(len(self))
        q[:, :3] = a
        q[:, 3] = c[:, j] * cc + s[:, j] * ss

        if parity:
            q[:, j] *= -1

        # only return quaternions where the scalar value is positive
        negScalar = q[:, 3] < 0
        q[negScalar] *= -1

        return q

    def asMatrixArray(self):
        """ Convert this EulerArray object to a Matrix3Array

        Returns
        -------
        Matrix3Array
            The current orientations as a Matrix3Array
        """
        from .matrixN import Matrix3Array
        (i, j, k), parity = self.ORDER_PARITY[self.order]

        cp = self.asRadians().copy()
        if parity:
            cp *= -1

        c = np.cos(cp)
        s = np.sin(cp)

        cc = c[:, i] * c[:, k]
        cs = c[:, i] * s[:, k]
        sc = s[:, i] * c[:, k]
        ss = s[:, i] * s[:, k]

        out = Matrix3Array.eye(len(self))
        out[:, i, i] = c[:, j] * c[:, k]
        out[:, j, i] = s[:, j] * sc - cs
        out[:, k, i] = s[:, j] * cc + ss
        out[:, i, j] = c[:, j] * s[:, k]
        out[:, j, j] = s[:, j] * ss + cc
        out[:, k, j] = s[:, j] * cs - sc
        out[:, i, k] = -s[:, j]
        out[:, j, k] = c[:, j] * s[:, i]
        out[:, k, k] = c[:, j] * c[:, i]
        return out
