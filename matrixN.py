import numpy as np
from vectorN import VectorN, VECTOR_BY_SIZE, VECTOR_ARRAY_BY_SIZE


class MatrixN(np.ndarray):
    def __new__(cls, input_array=None):
        if input_array is None:
            ary = np.eye(cls.N)
        else:
            ary = np.asarray(input_array)
        if ary.size != cls.N ** 2:
            raise ValueError(
                "Initializer for Matrix{0} must be of length {0}".format(cls.N)
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
        ret = super(MatrixN, self).__getitem__(idx)
        typ = self.getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

    def length_squared(self):
        return (self * self).sum()

    def length(self):
        return np.sqrt(self.length_squared())

    def toMatrixSize(self, n, copy=False):
        if n == self.N:
            if copy:
                return self.copy()
            return self
        typ = MATRIX_BY_SIZE[n]
        ret = typ()
        n = min(n, self.N)
        ret[:n] = self[:n]
        return ret

    def asEuler(self, order="xyz", degrees=False):
        return self[None, ...].asEulerArray(order=order, degrees=degrees)[0]

    def asQuaternion(self):
        return self[None, ...].asQuaternionArray()[0]

    def inverse(self):
        return np.linalg.inv(self)

    def invert(self):
        self[:] = np.linalg.inv(self)

    def __mul__(self, other):
        if isinstance(other, VectorN):
            msg = "Cannot multiply matrix*vector. You must multiply vector*matrix\n"
            msg += "Make sure when you multiply it's in `child * parent * grandparent` order"
            raise TypeError(msg)

        if isinstance(other, MatrixN):
            if other.N != self.N:
                msg = "Cannot multiply matrices of different sizes. Got {0} and {1}"
                raise TypeError(msg.format(self.N, other.N))
            return np.dot(self.N, other.N)
        return super(MatrixN, self).__mul__(other)

    def __imul__(self, other):
        if isinstance(other, VectorN):
            msg = "Cannot multiply matrix*vector. You must multiply vector*matrix\n"
            msg += "Make sure when you multiply it's in `child * parent * grandparent` order"
            raise TypeError(msg)

        if isinstance(other, MatrixN):
            if other.N != self.N:
                msg = "Cannot multiply matrices of different sizes. Got {0} and {1}"
                raise TypeError(msg.format(self.N, other.N))
            self[:] = np.dot(self.N, other.N)
            return
        super(MatrixN, self).__imul__(other)

    @staticmethod
    def lookAt(look, up, axis="xy"):
        return MatrixNArray.lookAts([look], [up], axis=axis)[0]

    def changeUpAxis(self, oldUpAxis, newUpAxis):
        return self[None, ...].changeUpAxis(oldUpAxis, newUpAxis)[0]

    def decompose(self):
        t, r, s = self[None, ...].decompose()
        return t[0], r[0], s[0]

    def asScale(self):
        return self[None, ...].asScaleArray()[0]

    def asTranslation(self):
        return self[None, ...].asTranslationArray()[0]


class MatrixNArray(np.ndarray):
    def __new__(cls, input_array):
        ary = np.asarray(input_array)
        ary = ary.reshape((-1, cls.N, cls.N))
        return ary.view(cls)

    def getReturnType(self, shape):
        if not shape:
            return None
        if len(shape) == 3:
            if shape[1] == self.N and shape[2] == self.N:
                return type(self)
            return np.ndarray
        elif len(shape) == 2:
            if shape[1] == self.N:
                if shape[0] == self.N:
                    return type(self)
                else:
                    return self.vectorType
        elif len(shape) == 1:
            if shape[0] == self.N:
                return self.vectorType
        return np.ndarray

    def __getitem__(self, idx):
        ret = super(MatrixNArray, self).__getitem__(idx)
        typ = self.getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

    @classmethod
    def zeros(cls, length):
        return cls(np.zeros((length, cls.N, cls.N)))

    @classmethod
    def ones(cls, length):
        return cls(np.ones((length, cls.N, cls.N)))

    @classmethod
    def full(cls, length, value):
        return cls(np.full((length, cls.N, cls.N), value))

    @classmethod
    def eye(cls, length):
        ret = cls.zeros(length)
        ret[:] = np.eye(cls.N)
        return ret

    def toMatrixSize(self, n, copy=False):
        if n == self.N:
            if copy:
                return self.copy()
            return self
        typ = MATRIX_ARRAY_BY_SIZE[n]
        ret = typ.eye(len(self))
        n = min(n, self.N)
        ret[:, :n] = self[:, :n]
        return ret

    def append(self, v):
        self.resize((len(self) + 1, self.N, self.N))
        self[-1] = v

    def extend(self, v):
        self.resize((len(self) + len(v), self.N, self.N))
        self[-len(v) :] = v

    def setLength(self, length):
        self.resize((length, self.N, self.N))

    def inverse(self):
        return np.linalg.inv(self)

    def invert(self):
        self[:] = np.linalg.inv(self)

    def asEulerArray(self, order="xyz", degrees=False):
        from .euler import EulerArray

        # Taken almost directly from the scipy transforms rotations code
        # This algorithm doesn't actually assume the basis vectors are
        # perpendicular to each other.

        # The algorithm assumes intrinsic frame transformations. The algorithm
        # in the paper is formulated for rotation matrices which are transposition
        # rotation matrices used within Rotation.
        # Adapt the algorithm for our case by
        # Instead of transposing our representation, use the transpose of the
        # O matrix as defined in the paper, and be careful to swap indices

        order = order.lower()
        if self.ndim == 2:
            self = self[None, ...]

        self = self.toMatrixSize(3)
        num_rotations = self.shape[0]

        # Step 0
        # Algorithm assumes axes as column vectors, here we use 1D vectors
        bvs = []
        for axis in order:
            b = np.zeros(3)
            b["xyz".index(axis)] = 1
            bvs.append(b)
        n1, n2, n3 = b

        # Step 2
        # SL is the parity of the order
        sl = np.dot(np.cross(n1, n2), n3)
        cl = np.dot(n1, n3)

        # angle offset is lambda from the paper referenced in [2] from docstring of
        # `as_euler` function
        offset = np.arctan2(sl, cl)
        c = np.vstack((n2, np.cross(n1, n2), n1))

        # Step 3
        rot = np.array([[1, 0, 0], [0, cl, sl], [0, -sl, cl]])
        res = np.einsum("...ij,...jk->...ik", c, self)
        matrix_transformed = np.einsum("...ij,...jk->...ik", res, c.T.dot(rot))

        # Step 4
        angles = np.empty((num_rotations, 3))
        # Ensure less than unit norm
        positive_unity = matrix_transformed[:, 2, 2] > 1
        negative_unity = matrix_transformed[:, 2, 2] < -1
        matrix_transformed[positive_unity, 2, 2] = 1
        matrix_transformed[negative_unity, 2, 2] = -1
        angles[:, 1] = np.arccos(matrix_transformed[:, 2, 2])

        # Steps 5, 6
        eps = 1e-7
        safe1 = np.abs(angles[:, 1]) >= eps
        safe2 = np.abs(angles[:, 1] - np.pi) >= eps

        # Step 4 (Completion)
        angles[:, 1] += offset

        # 5b
        safe_mask = np.logical_and(safe1, safe2)
        angles[safe_mask, 0] = np.arctan2(
            matrix_transformed[safe_mask, 0, 2], -matrix_transformed[safe_mask, 1, 2]
        )
        angles[safe_mask, 2] = np.arctan2(
            matrix_transformed[safe_mask, 2, 0], matrix_transformed[safe_mask, 2, 1]
        )

        # For instrinsic, set third angle to zero
        # 6a
        angles[~safe_mask, 2] = 0
        # 6b
        angles[~safe1, 0] = np.arctan2(
            matrix_transformed[~safe1, 1, 0] - matrix_transformed[~safe1, 0, 1],
            matrix_transformed[~safe1, 0, 0] + matrix_transformed[~safe1, 1, 1],
        )
        # 6c
        angles[~safe2, 0] = np.arctan2(
            matrix_transformed[~safe2, 1, 0] + matrix_transformed[~safe2, 0, 1],
            matrix_transformed[~safe2, 0, 0] - matrix_transformed[~safe2, 1, 1],
        )

        # Step 7
        if order[0] == order[2]:
            # lambda = 0, so we can only ensure angle2 -> [0, pi]
            adjust_mask = np.logical_or(angles[:, 1] < 0, angles[:, 1] > np.pi)
        else:
            # lambda = + or - pi/2, so we can ensure angle2 -> [-pi/2, pi/2]
            adjust_mask = np.logical_or(
                angles[:, 1] < -np.pi / 2, angles[:, 1] > np.pi / 2
            )

        # Dont adjust gimbal locked angle sequences
        adjust_mask = np.logical_and(adjust_mask, safe_mask)

        angles[adjust_mask, 0] += np.pi
        angles[adjust_mask, 1] = 2 * offset - angles[adjust_mask, 1]
        angles[adjust_mask, 2] -= np.pi

        angles[angles < -np.pi] += 2 * np.pi
        angles[angles > np.pi] -= 2 * np.pi

        # Step 8
        # if not np.all(safe_mask):
        # warnings.warn(
        # "Gimbal lock detected. Setting third angle to zero since"
        # " it is not possible to uniquely determine all angles."
        # )
        if degrees:
            angles = np.deg2rad(angles)

        return EulerArray(angles)

    def asQuaternionArray(self):
        from .quaternion import QuaternionArray

        if self.ndim == 2:
            self = self[None, ...]
        self = self.toMatrixSize(3)
        num_rotations = self.shape[0]

        decision_matrix = np.empty((num_rotations, 4))
        decision_matrix[:, :3] = self.diagonal(axis1=1, axis2=2)
        decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
        choices = decision_matrix.argmax(axis=1)

        quats = np.empty((num_rotations, 4))

        ind = np.nonzero(choices != 3)[0]
        i = choices[ind]
        j = (i + 1) % 3
        k = (j + 1) % 3

        quats[ind, i] = 1 - decision_matrix[ind, -1] + 2 * self[ind, i, i]
        quats[ind, j] = self[ind, j, i] + self[ind, i, j]
        quats[ind, k] = self[ind, k, i] + self[ind, i, k]
        quats[ind, 3] = self[ind, k, j] - self[ind, j, k]

        ind = np.nonzero(choices == 3)[0]
        quats[ind, 0] = self[ind, 2, 1] - self[ind, 1, 2]
        quats[ind, 1] = self[ind, 0, 2] - self[ind, 2, 0]
        quats[ind, 2] = self[ind, 1, 0] - self[ind, 0, 1]
        quats[ind, 3] = 1 + decision_matrix[ind, -1]

        # normalize
        qlens = np.sqrt(np.einsum("ij,ij->i", quats, quats))
        quats = quats / qlens[:, None]
        return QuaternionArray(quats)

    @staticmethod
    def lookAts(looks, ups, axis="xy"):
        """ Set the upper 3x3 of these matrices to look along the look-vectors, oriented to the up-vectors

        Args:
            looks(Vector3Array): The pointing directions of the look axis
            ups(Vector3Array): The pointing directions of the up axis
            axis(string): The axes to align to the look and up vectors (ie: 'xy', 'yz', '-zy', 'x-z')
        """
        looks = VECTOR_ARRAY_BY_SIZE[3](looks)
        ups = VECTOR_ARRAY_BY_SIZE[3](ups)

        axis = axis.lower()

        looks = looks.normal()
        sides = np.cross(looks, ups.normal())
        sides.normalize()
        ups = np.cross(sides, looks)
        ups.normalize()

        if axis.startswith("-"):
            looks *= -1
        if axis[-2] == "-":
            ups *= -1

        pureAxes = axis.replace("-", "")
        lookAxis = "xyz".index(pureAxes[0])
        upAxis = "xyz".index(pureAxes[1])

        sideAxis = set([0, 1, 2]) - set([lookAxis, upAxis])
        sideAxis = sideAxis.pop()

        if (sideAxis + 1) % 3 != lookAxis:
            sides *= -1

        ret = MATRIX_ARRAY_BY_SIZE[3].eye(len(looks))
        ret[:, lookAxis] = looks
        ret[:, upAxis] = ups
        ret[:, sideAxis] = sides

        return ret

    def changeUpAxis(self, oldUpAxis, newUpAxis):
        reo = [0, 1, 2, 3]
        reo[oldUpAxis], reo[newUpAxis] = reo[newUpAxis], reo[oldUpAxis]

        ret = self.copy()
        ret = ret[:, :, reo]
        ret = ret[:, reo, :]
        ret[:, :, oldUpAxis] *= -1
        ret[:, oldUpAxis, :] *= -1
        return ret

    def asScaleArray(self):
        scale = self[:, :3, :3]
        scale = np.sqrt(np.einsum("...ij,...ij->...i", scale, scale))
        return VECTOR_ARRAY_BY_SIZE[3](scale)

    def asTranslationArray(self):
        return self[:, 3, :3]

    def decompose(self):
        tran = self.asTranslationArray()
        rot = self.asQuaternionArray()
        scale = self.asScaleArray()
        return tran, rot, scale


# Register the default sizes of array dynamically
MATRIX_BY_SIZE = {}
MATRIX_ARRAY_BY_SIZE = {}
glo = globals()
for n in [2, 3, 4]:
    name = "Matrix{0}".format(n)
    aname = "Matrix{0}Array".format(n)
    m = type(name, MatrixN, {"N": n})
    ma = type(aname, MatrixNArray, {"N": n})
    m.arrayType = ma
    ma.itemType = m
    m.vectorType = VECTOR_BY_SIZE[n]
    ma.vectorType = VECTOR_BY_SIZE[n]

    glo[name] = m
    glo[aname] = ma
    MATRIX_BY_SIZE[n] = m
    MATRIX_ARRAY_BY_SIZE[n] = ma
