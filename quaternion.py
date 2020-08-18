import numpy as np


class Quaternion(np.ndarray):
    def __new__(cls, input_array=None):
        if input_array is None:
            ary = np.zeros(4)
            ary[3] = 1
        else:
            ary = np.asarray(input_array)
        if ary.size != 4:
            raise ValueError(
                "Initializer for Vector{0} must be of length {0}".format(4)
            )
        return ary.view(cls)

    def getReturnType(self, shape):
        if not shape:
            return None
        if len(shape) == 1:
            if shape[0] == 4:
                return type(self)
        elif len(shape) == 2:
            # This could happen with fancy indexing
            if shape[-1] == 4:
                return self.arrayType

        return np.ndarray

    def __getitem__(self, idx):
        ret = super(Quaternion, self).__getitem__(idx)
        typ = self.getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

    def length_squared(self):
        return (self * self).sum()

    def length(self):
        return np.sqrt(self.length_squared())

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return QuaternionArray.quatquatProduct(self[None, ...], other[None, ...])[0]
        elif isinstance(other, QuaternionArray):
            return QuaternionArray.quatquatProduct(self[None, ...], other)

        from .vectorN import VectorN, VectorNArray
        if isinstance(other, VectorN, VectorNArray):
            raise NotImplementedError("Vectors must always be on the left side of the multiplication")

        return super(Quaternion, self).__mul__(other)

    def asMatrix(self):
        return self[None, :].asMatrixArray()[0]

    def asEuler(self, order='xyz', degrees=False):
        return self[None, :].asEulerArray(order=order, degrees=degrees)[0]

    @staticmethod
    def lookAt(look, up, axis="xy"):
        from .matrixN import MatrixNArray
        mats = MatrixNArray.lookAts([look], [up], axis=axis)
        return mats.asQuaternionArray()[0]


class QuaternionArray(np.ndarray):
    def __new__(cls, input_array):
        ary = np.asarray(input_array)
        ary = ary.reshape((-1, 4))
        return ary.view(cls)

    def getReturnType(self, shape):
        if not shape:
            return None
        if len(shape) == 2:
            if shape[1] == 4:
                return type(self)
        elif len(shape) == 1:
            if shape[0] == 4:
                return Quaternion
        return np.ndarray

    def __getitem__(self, idx):
        ret = super(QuaternionArray, self).__getitem__(idx)
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
        return cls(np.zeros((length, 4)))

    def append(self, v):
        self.resize((len(self) + 1, 4))
        self[-1] = v

    def extend(self, v):
        self.resize((len(self) + len(v), 4))
        self[-len(v):] = v

    @classmethod
    def alignedRotations(cls, axisName, angles, degrees=False):
        # where axisName is in 'xyz'
        angles = np.asarray(angles)
        if degrees:
            angles = np.deg2rad(angles)
        ind = 'xyz'.index(axisName.lower())
        ret = cls.zeros((len(angles), 4))
        ret[:, 3] = np.cos(angles / 2)
        ret[:, ind] = np.sin(angles / 2)
        return ret

    @classmethod
    def axisAngle(cls, axes, angles, degrees=False):
        count = len(axes)
        if degrees:
            angles = np.deg2rad(angles)
        sins = np.sin(angles)
        ret = cls.zeros(count)

        ret[:, :3] = axes * sins[:, None]
        ret[:, 3] = np.cos(angles)
        return ret

    @staticmethod
    def quatquatProduct(p, q):
        # This assumes the arrays are shaped correctly
        prod = np.empty((max(p.shape[0], q.shape[0]), 4))
        prod[:, 3] = p[:, 3] * q[:, 3] - np.sum(p[:, :3] * q[:, :3], axis=1)
        prod[:, :3] = (
            p[:, None, 3] * q[:, :3]
            + q[:, None, 3] * p[:, :3]
            + np.cross(p[:, :3], q[:, :3])
        )
        return prod

    @staticmethod
    def vectorquatproduct(v, q):
        # This assumes the arrays are shaped correctly
        qvec = q[:, :3]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        uv *= 2 * q[:, 3]
        uuv *= 2
        return v + uv + uuv

    def asMatrixArray(self):
        from .matrixN import Matrix3Array
        x = self[:, 0]
        y = self[:, 1]
        z = self[:, 2]
        w = self[:, 3]

        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w

        xy = x * y
        zw = z * w
        xz = x * z
        yw = y * w
        yz = y * z
        xw = x * w

        num_rotations = len(self)
        mats = Matrix3Array.zeros(num_rotations)

        mats[:, 0, 0] = x2 - y2 - z2 + w2
        mats[:, 1, 0] = 2 * (xy + zw)
        mats[:, 2, 0] = 2 * (xz - yw)

        mats[:, 0, 1] = 2 * (xy - zw)
        mats[:, 1, 1] = -x2 + y2 - z2 + w2
        mats[:, 2, 1] = 2 * (yz + xw)

        mats[:, 0, 2] = 2 * (xz + yw)
        mats[:, 1, 2] = 2 * (yz - xw)
        mats[:, 2, 2] = -x2 - y2 + z2 + w2

        return mats

    def asEulerArray(self, order='xyz', degrees=False):
        m = self.asMatrixArray()
        return m.asEulerArray(order=order, degrees=degrees)

    def __mul__(self, other):
        if isinstance(other, QuaternionArray):
            return self.quatquatProduct(self, other)
        elif isinstance(other, Quaternion):
            return self.quatquatProduct(self, other[None, ...])

        from .vectorN import VectorN, VectorNArray
        if isinstance(other, VectorN, VectorNArray):
            raise NotImplementedError("Vectors must always be on the left side of the multiplication")

        return super(QuaternionArray, self).__mul__(other)

    @staticmethod
    def lookAts(looks, ups, axis="xy"):
        from .matrixN import MatrixNArray
        mats = MatrixNArray.lookAts(looks, ups, axis=axis)
        return mats.asQuaternionArray()

