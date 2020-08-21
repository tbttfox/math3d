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

    def getReturnType(self, shape):
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
        typ = self.getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

    def length_squared(self):
        return (self * self).sum()

    def length(self):
        return np.sqrt(self.length_squared())

    @classmethod
    def zeros(cls):
        return cls(np.zeros(cls.N))

    @classmethod
    def ones(cls):
        return cls(np.ones(cls.N))

    @classmethod
    def full(cls, value):
        return cls(np.full(cls.N, value))

    def toVectorSize(self, n, pad=1.0, copy=False):
        if n == self.N:
            if copy:
                return self.copy()
            return self
        typ = VECTOR_BY_SIZE[n]
        ret = typ.full(pad)
        n = min(n, self.N)
        ret[:n] = self[:n]
        return ret

    def cross(self, other):
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

    def getReturnType(self, shape):
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
        typ = self.getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

    def length_squared(self):
        return np.einsum("ij,ij->i", self, self)

    def length(self):
        return np.sqrt(self.length_squared())

    def normal(self):
        return self / self.length()[..., None]

    def normalize(self):
        self /= self.length()[..., None]

    @classmethod
    def zeros(cls, length):
        return cls(np.zeros((length, cls.N)))

    @classmethod
    def ones(cls, length):
        return cls(np.ones((length, cls.N)))

    @classmethod
    def full(cls, length, value):
        return cls(np.full((length, cls.N), value))

    def toVectorSize(self, n, pad=1.0, copy=False):
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
        self.resize((len(self) + 1, self.N))
        self[-1] = v

    def extend(self, v):
        self.resize((len(self) + len(v), self.N))
        self[-len(v):] = v

    def cross(self, other):
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
