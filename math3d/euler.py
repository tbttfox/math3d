import numpy as np


class Euler(np.ndarray):
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
    def getReturnType(cls, shape):
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

    def __getitem__(self, idx):
        ret = super(Euler, self).__getitem__(idx)
        typ = self.getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

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


class EulerArray(np.ndarray):
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

    @classmethod
    def getReturnType(cls, shape):
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

    def __getitem__(self, idx):
        ret = super(EulerArray, self).__getitem__(idx)
        # If we're getting columns from the array
        # Then we expect arrays back, not math3d types
        if isinstance(idx, tuple):
            # If we're getting multiple indices
            # And the second index isn't a ":"
            # Then we're getting
            if len(idx) > 1 and idx[1] != slice(None, None, None):
                return ret.view(np.ndarray)
        typ = self.getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

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

    def appended(self, value):
        """ Return a copy of the array with the value appended
        Euler types will be converted to match degrees or radians
        Quaternion and Matrix types will be converted to Euler
        All other types will be appended as-is

        Parameters
        ----------
        value: iterable
            An iterable to be appended as-is to the end of this array
        """
        newShp = list(self.shape)
        newShp[0] += 1
        ret = np.resize(self, newShp).view(type(self))

        from .quaternion import Quaternion
        from .matrixN import MatrixN
        if isinstance(value, Euler):
            if self.degrees and not value.degrees:
                value = np.rad2deg(value)
            elif not self.degrees and value.degrees:
                value = np.deg2rad(value)
        elif isinstance(value, Quaternion):
            value = value.asEuler(order=self.order, degrees=self._degrees)
        elif isinstance(value, MatrixN):
            value = value.asEuler(order=self.order, degrees=self._degrees)

        ret[-1] = value
        return ret

    def extended(self, value):
        """ Return a copy of the array extended with the given values
        Euler types will be converted to match degrees or radians
        Matrix and Quaternion types will be converted to Euler
        All other types will be appended as-is

        Parameters
        ----------
        value: iterable
            An iterable to be appended as-is to the end of this array
        """
        from .quaternion import QuaternionArray
        from .matrixN import MatrixNArray
        newShp = list(self.shape)
        newShp[0] += len(value)
        ret = np.resize(self, newShp).view(type(self))

        if isinstance(value, EulerArray):
            if self.degrees and not value.degrees:
                value = value.asDegrees()
            elif not self.degrees and value.degrees:
                value = value.asRadians()
        elif isinstance(value, QuaternionArray):
            value = value.asEulerArray(order=self.order, degrees=self._degrees)
        elif isinstance(value, MatrixNArray):
            value = value.asEulerArray(order=self.order, degrees=self._degrees)

        ret[-len(value):] = value
        return ret

    def asNdArray(self):
        """ Return this object as a regular numpy array

        Returns
        -------
        ndarray:
            The current object as a numpy array
        """
        return self.view(np.ndarray)

    def asQuaternionArray(self):
        """ Convert this EulerArray object to a QuaternionArray

        Returns
        -------
        QuaternionArray
            The current orientations as a quaternionArray
        """
        from .quaternion import QuaternionArray

        # Convert multiple euler triples to quaternions
        result = QuaternionArray.alignedRotations(self.order[0], self[:, 0], degrees=self._degrees)
        for idx, axis in enumerate(self.order[1:], start=1):
            result = result * QuaternionArray.alignedRotations(axis, self[:, idx], degrees=self._degrees)
        return result

    def asMatrixArray(self):
        """ Convert this EulerArray object to a Matrix3Array

        Returns
        -------
        Matrix3Array
            The current orientations as a Matrix3Array
        """
        q = self.asQuaternionArray()
        return q.asMatrixArray()
