import numpy as np


class VectorN(np.ndarray):
    def __new__(cls, input_array=None):
        if input_array is None:
            ary = np.zeros(cls.N)
        else:
            ary = np.asarray(input_array)
        if ary.size != cls.N:
            raise ValueError(
                "Initializer for Vector{0} must be of length {0}".format(cls.N)
            )
        return ary.view(cls)

    def _getReturnType(self, shape):
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
            if shape[0] == self.N:
                return type(self)
        elif len(shape) == 2:
            # This could happen with fancy indexing
            if shape[-1] == self.N:
                return self.arrayType

        return np.ndarray

    def __getitem__(self, idx):
        ret = super(VectorN, self).__getitem__(idx)
        typ = self._getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

    def lengthSquared(self):
        """ Return the squared length of each vector

        Returns
        -------
        np.ndarray:
            The squared lengths of the vectors
        """
        return (self * self).sum()

    def length(self):
        """ Return the length of each vector

        Returns
        -------
        np.ndarray:
            The lengths of the vectors
        """
        return np.sqrt(self.lengthSquared())

    @classmethod
    def ones(cls):
        """ Alternate constructor to build a vector of all ones """
        return cls(np.ones(cls.N))

    @classmethod
    def full(cls, value):
        """ Alternate constructor to build a vector of a given value """
        return cls(np.full(cls.N, value))

    def toVectorSize(self, n, pad=1.0):
        """ Return a vector of a given size based on the current vector.
        Discard the ending items if the size is smaller
        Pad up if the size is bigger, filled with the pad value

        If the requested size is smaller, keep the upper left square of the matrix
        If the size is bigger, the new entries will the same as the identity matrix

        Parameters
        ----------
        n: int
            The new size of the vector
        pad: float
            The padding value for expanding the vector

        """
        if n == self.N:
            return self.copy()
        typ = VECTOR_BY_SIZE[n]
        ret = typ.full(pad)
        n = min(n, self.N)
        ret[:n] = self[:n]
        return ret

    def cross(self, other):
        """ Take Cross product with another vector

        Parameters
        ----------
        other: VectorN
            The value to cross with

        Returns
        -------
        VectorN:
            The result of the cross product
        """
        return np.cross(self, other)

    def __mul__(self, other):
        if isinstance(other, VectorN):
            if self.N != other.N:
                raise TypeError(
                    "Cannot compute the dot of vectors with different sizes"
                )
            return np.dot(self, other)
        elif isinstance(other, VectorNArray):
            if self.N != other.N:
                raise TypeError(
                    "Cannot compute the dot of vectors with different sizes"
                )
            return np.einsum("ij, ij -> i", self[None, ...], other)

        from .matrixN import MatrixN, MatrixNArray

        if isinstance(other, MatrixN):
            if other.N < self.N:
                raise TypeError("Can't mutiply a vector by a smaller matrix")
            exp = self.toVectorSize(other.N)
            ret = np.dot(self, other)
            return ret.toVectorSize(self.N)
        elif isinstance(other, MatrixNArray):
            if other.N < self.N:
                raise TypeError("Can't mutiply a vector by a smaller matrix")
            exp = self.toVectorSize(other.N)
            ret = np.einsum("ij, ijk -> ik", exp[None, ...], other)
            return ret.toVectorSize(self.N)

        from .quaternion import Quaternion, QuaternionArray

        if isinstance(other, Quaternion):
            exp = self.toVectorSize(3)
            ret = QuaternionArray.vectoquatproduct(exp[None, ...], other[None, ...])
            return ret[0].toVectorSize(self.N)

        elif isinstance(other, QuaternionArray):
            exp = self.toVectorSize(3)
            ret = QuaternionArray.vectoquatproduct(exp[None, ...], other)
            return ret.toVectorSize(self.N)


class VectorNArray(np.ndarray):
    def __new__(cls, input_array):
        ary = np.asarray(input_array)
        ary = ary.reshape((-1, cls.N))
        return ary.view(cls)

    def _getReturnType(self, shape):
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
            if shape[1] == self.N:
                return type(self)
            return np.ndarray
        elif len(shape) == 1:
            if shape[0] == self.N:
                return self.itemType
        return np.ndarray

    def __getitem__(self, idx):
        ret = super(VectorNArray, self).__getitem__(idx)
        typ = self._getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

    def lengthSquared(self):
        """ Return the squared length of each vector

        Returns
        -------
        np.ndarray:
            The squared lengths of the vectors
        """
        return np.einsum("ij,ij->i", self, self)

    def length(self):
        """ Return the length of each vector

        Returns
        -------
        np.ndarray:
            The lengths of the vectors
        """
        return np.sqrt(self.lengthSquared())

    def normal(self):
        """ Return the normalized quaternions

        Returns
        -------
        QuaternionArray:
            The normalized quaternions
        """
        return self / self.length()[..., None]

    def normalize(self):
        """ Normalize the quaternions in-place """
        self /= self.length()[..., None]

    @classmethod
    def ones(cls, length):
        """ Alternate constructor to build a vector of all ones

        Parameters
        ----------
        length: int
            The number of matrices to build
        """
        return cls(np.ones((length, cls.N)))

    @classmethod
    def full(cls, length, value):
        """ Alternate constructor to build a vector of a given value

        Parameters
        ----------
        length: int
            The number of matrices to build
        value: float
            The value to fill the vectors with
        """
        return cls(np.full((length, cls.N), value))

    def toVectorSize(self, n, pad=1.0, copy=False):
        """ Return a vector of a given size based on the current vector.
        Discard the ending items if the size is smaller
        Pad up if the size is bigger, filled with the pad value

        If the requested size is smaller, keep the upper left square of the matrix
        If the size is bigger, the new entries will the same as the identity matrix

        Parameters
        ----------
        n: int
            The new size of the vector
        pad: float
            The padding value for expanding the vector

        """
        if n == self.N:
            if copy:
                return self.copy()
            return self
        typ = VECTOR_ARRAY_BY_SIZE[n]
        ret = typ.full(len(self), pad)
        n = min(n, self.N)
        ret[:, :n] = self[:, :n]
        return ret

    def append(self, v):
        """ Append an item to the end of this array

        Parameters
        ----------
        value: iterable
            An iterable to be appended as-is to the end of this array
        """
        self.resize((len(self) + 1, self.N))
        self[-1] = v

    def extend(self, v):
        """ Extend this array with the given items

        Parameters
        ----------
        value: iterable
            An iterable to be added to the end of this array
        """
        self.resize((len(self) + len(v), self.N))
        self[-len(v):] = v

    def cross(self, other):
        """ Take Cross product with another array of vector

        Parameters
        ----------
        other: VectorNArray
            The values to cross with

        Returns
        -------
        VectorNArray:
            The result of the per-row cross product
        """
        return np.cross(self, other)

    def __mul__(self, other):
        if isinstance(other, VectorNArray):
            if other.N != self.N:
                raise TypeError("Can't dot vectors of different length")
            return np.einsum("ij,ij->i", self, other)
        elif isinstance(other, VectorN):
            if other.N != self.N:
                raise TypeError("Can't dot vectors of different length")
            return np.dot(self, other)

        from .matrixN import MatrixN, MatrixNArray

        if isinstance(other, MatrixN):
            if other.N < self.N:
                raise TypeError("Can't mutiply a vector by a smaller matrix")
            exp = self.toVectorSize(other.N)
            ret = np.dot(exp, other)
            return ret.toVectorSize(self.N)
        elif isinstance(other, MatrixNArray):
            if other.N < self.N:
                raise TypeError("Can't mutiply a vector by a smaller matrix")
            exp = self.toVectorSize(other.N)
            ret = np.einsum("ij, ijk -> ik", exp, other)
            return ret.toVectorSize(self.N)

        from .quaternion import Quaternion, QuaternionArray

        if isinstance(other, Quaternion):
            exp = self.toVectorSize(3)
            ret = QuaternionArray.vectoquatproduct(exp, other[None, ...])
            return ret.toVectorSize(self.N)

        elif isinstance(other, QuaternionArray):
            exp = self.toVectorSize(3)
            ret = QuaternionArray.vectoquatproduct(exp, other)
            return ret.toVectorSize(self.N)

    def angle(self, other):
        """ Get the angle between pairs of vectors

        Parameters
        ----------
        other: VectorNArray
            The array of vectors to get the angles between

        Returns
        -------
        np.ndarray:
            The angles between the paired vectors in radians
        """
        dots = self.normal() * other.normal()
        return np.acos(dots)


# Register the default sizes of array dynamically
VECTOR_BY_SIZE = {}
VECTOR_ARRAY_BY_SIZE = {}
glo = globals()
for n in [2, 3, 4]:
    name = "Vector{0}".format(n)
    aname = "Vector{0}Array".format(n)
    v = type(name, VectorN, {"N": n})
    va = type(aname, VectorNArray, {"N": n})
    v.arrayType = va
    va.itemType = v
    glo[name] = v
    glo[aname] = va
    VECTOR_BY_SIZE[n] = v
    VECTOR_ARRAY_BY_SIZE[n] = va
