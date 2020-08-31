import numpy as np


class Euler(np.ndarray):
    """ A 3d Euler rotation with arbitrary axis order

    Parameters
    ----------
    inputArray : iterable, optional
        The input value to create the euler array. It must be an iterable of length 3
        If not given, defaults to (0, 0, 0)
    order: str, optional
        The order in which the axis rotations are applied
        It must be one of these options ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyz']
        Defaults to 'xyz'
    degrees: bool, optional
        Whether the angles are given in degrees or radians. Defaults to False (radians)
    """

    def __new__(cls, inputArray=None, order="xyz", degrees=False):
        if inputArray is None:
            ary = np.zeros(3)
        else:
            ary = np.asarray(inputArray, dtype=float)
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

    def toArray(self):
        """ Return the array type of this object

        Returns
        -------
        ArrayType:
            The current object up-cast into a length-1 array
        """
        return self[None, ...]

    @property
    def degrees(self):
        return self._degrees

    def toRadians(self):
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

    def toDegrees(self):
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
    def _getReturnType(cls, shape):
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
        typ = self._getReturnType(ret.shape)
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
        return self.toArray().asMatrixArray()[0]

    def asQuaternion(self):
        """ Convert this euler object to a Quaternion

        Returns
        -------
        Quaternion
            The current orientation as a quaternion
        """
        return self.toArray().asQuaternionArray()[0]

    def toNewOrder(self, order):
        """ Create a new euler object that represents the same orientation
        but composed with a different rotation order

        Returns
        -------
        Euler
            The same spatial orientation but with a different axis order
        """
        return self.toArray().toNewOrder(order)[0]


class EulerArray(np.ndarray):
    """ An array of 3d Euler rotations with a common axis order

    Parameters
    ----------
    inputArray : iterable
        The input value to create the euler array. It must be an iterable with a length
        multiple of 3
    order: str, optional
        The order in which the axis rotations are applied
        It must be one of these options ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyz']
        Defaults to 'xyz'
    degrees: bool, optional
        Whether the angles are given in degrees or radians. Defaults to False (radians)
    """

    def __new__(cls, inputArray, order="xyz", degrees=False):
        ary = np.asarray(inputArray, dtype=float)
        ary = ary.reshape((-1, 3))
        ret = ary.view(cls)
        ret.order = order.lower()
        ret._degrees = degrees
        return ret

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.order = getattr(obj, "order", "xyz")
        self._degrees = getattr(obj, "degrees", False)

    @classmethod
    def _getReturnType(cls, shape):
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
        typ = self._getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

    @property
    def degrees(self):
        return self._degrees

    def toRadians(self):
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
        return self

    def toDegrees(self):
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
        return self

    def toNewOrder(self, order):
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

    def append(self, value):
        """ Append an item to the end of this array
        Euler types will be converted to match degrees or radians

        Parameters
        ----------
        value: iterable
            A length-3 iterable to be appended to the end of this array
        """
        if isinstance(value, Euler):
            if self.degrees and not value.degrees:
                value = np.rad2deg(value)
            elif not self.degrees and value.degrees:
                value = np.deg2rad(value)

        self.resize((len(self) + 1, 3))
        self[-1] = value

    def extend(self, value):
        """ Extend this array with the given items
        Euler types will be converted to match degrees or radians

        Parameters
        ----------
        value: iterable
            A multiple-of-length-3 iterable to be appended to the end of this array
        """
        value = np.asarray(value, dtype=float)
        if isinstance(value, EulerArray):
            if self.degrees and not value.degrees:
                value = np.rad2deg(value)
            elif not self.degrees and value.degrees:
                value = np.deg2rad(value)

        self.resize((len(self) + len(value), 3))
        self[-len(value) :] = value

    def asQuaternionArray(self):
        """ Convert this EulerArray object to a QuaternionArray

        Returns
        -------
        QuaternionArray
            The current orientations as a quaternionArray
        """
        from .quaternion import QuaternionArray

        # Convert multiple euler triples to quaternions
        eulers = self
        if self.degrees:
            eulers = np.deg2rad(self)

        result = QuaternionArray.alignedRotations(self.order[0], eulers[:, 0])
        for idx, axis in enumerate(self.order[1:], start=1):
            result = result * QuaternionArray.alignedRotations(axis, eulers[:, idx])
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
