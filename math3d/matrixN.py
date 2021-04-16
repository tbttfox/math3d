import numpy as np
from vectorN import VectorN, VectorNArray, VECTOR_BY_SIZE, VECTOR_ARRAY_BY_SIZE
from .utils import arrayCompat
from .base import MathBase, ArrayBase

class MatrixN(MathBase):
    """ An NxN matrix which can represent rotations and transforms

    Parameters
    ----------
    inputArray : iterable, optional
        The input value to create the euler array. It must be an iterable that can fill the
        matrix. If not given, defaults to the identity matrix.
    """

    def __new__(cls, inputArray=None):
        if inputArray is None:
            ary = np.eye(cls.N)
        else:
            ary = np.asarray(inputArray, dtype=float)
        if ary.size != cls.N ** 2:
            raise ValueError(
                "Initializer for Matrix{0} must be of length {0}".format(cls.N)
            )
        ary = ary.reshape((cls.N, cls.N))
        return ary.view(cls)

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
            if shape[0] == cls.N:
                return cls.vectorType
        elif len(shape) == 2:
            if shape == (cls.N, cls.N):
                return cls
            if shape[-1] == cls.N:
                return cls.vectorArrayType
        elif len(shape) == 3:
            if shape[-2:] == (cls.N, cls.N):
                return cls.arrayType
        return np.ndarray

    def asMatrixSize(self, n):
        """ Return a square matrix of a given size based on the current matrix.
        If the size is smaller, keep the upper left square of the matrix
        If the size is bigger, the new entries will the same as the identity matrix

        Parameters
        ----------
        n: int
            The new size of the matrix
        """

        if n == self.N:
            return self.copy()
        typ = MATRIX_BY_SIZE[n]
        ret = typ()
        n = min(n, self.N)
        ret[:n, :n] = self[:n, :n]
        return ret

    def asEuler(self, order="xyz", degrees=False):
        """ Convert the upper left 3x3 of this matrix to an Euler rotation

        Parameters
        ----------
        order: str, optional
            The order in which the axis rotations are applied
            It must be one of these options ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyz']
            Defaults to 'xyz'
        degrees: bool, optional
            Whether the angles are given in degrees or radians. Defaults to False (radians)
        """
        return self.asArray().asEulerArray(order=order, degrees=degrees)[0]

    def asQuaternion(self):
        """ Convert the upper left 3x3 of this matrix to an Quaternion rotation"""
        return self.asArray().asQuaternionArray()[0]

    def inverse(self):
        """ Return the inverse of the current matrix """
        return np.linalg.inv(self)

    def invert(self):
        """ Invert the current matrix in-place """
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
            return np.dot(self, other).view(type(self))

        if isinstance(other, MatrixNArray):
            if other.N != self.N:
                msg = "Cannot multiply matrices of different sizes. Got {0} and {1}"
                raise TypeError(msg.format(self.N, other.N))
            aa = self.asArray()
            return np.einsum('xij,xjk->xik', aa, other).view(type(aa))
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
            self[:] = np.dot(self, other)
            return
        super(MatrixN, self).__imul__(other)

    @staticmethod
    def lookAt(look, up, axis="xy"):
        """ Alternate constructor to create a 3x3 matrix looking at a point
        and twisted with the given up value
        Think of the 3x3 matrix resting at origin

        Parameters
        ----------
        look: iterable
            A length-3 vector that the primary axis will be pointed at
        up: iterable
            A length-3 vector that the secondary axis will be pointed at
        axis: str, optional
            Define the primary and secondary axes. Must be one of these options
            ['xy', 'xz', 'yx', 'yz', 'zx', 'zy']
        """
        return MatrixNArray.lookAts([look], [up], axis=axis)[0]

    def changeUpAxis(self, oldUpAxis, newUpAxis):
        """ Rotate the matrix so that the newUpAxis points where the oldUpAxis was

        Parameters
        ----------
        oldUpAxis: int
            The index of the old axis
        newUpAxis: int
            The index of the new axis
        """
        return self.asArray().changeUpAxis(oldUpAxis, newUpAxis)[0]

    def decompose(self):
        """ Decompose the matrix into Translation, Rotation, and Scale

        Returns
        -------
        Vector3:
            The translation
        Quaternion:
            The Rotation
        Vector3:
            The Scale
        """
        t, r, s = self.asArray().decompose()
        return t[0], r[0], s[0]

    def asScale(self):
        """ Return the scale part of the matrix

        Returns
        -------
        Vector3:
            The scale part of the matrix
        """
        return self.asArray().asScaleArray()[0]

    def asTranslation(self):
        """ Return the translation part of the matrix

        Returns
        -------
        Vector3:
            The translation part of the matrix
        """
        return self.asArray().asTranslationArray()[0]

    def asTransform(self):
        """ Convert this matrix to a Transform

        Returns
        -------
        Transform:
            The converted matrix
        """
        return self.asArray().asTransformArray()[0]

    def flattened(self):
        return self.reshape(self.N**2)

    def getHandedness(self):
        """ Return the handedness of each matrix in the array

        Returns
        -------
        float:
            1.0 if the matrix is right-handed, otherwise -1.0
        """
        return self.asArray().getHandedness()[0]

    def asRotScale(self):
        """ Get a normalized, right-handed rotation matrix along with the
        scale that was passed in

        Returns
        -------
        Vector3:
            The scale part of the matrix
        Matrix3:
            The normalized right-handed rotation matrix
        """
        r, s = self.asArray().asRotScaleArray()
        return r[0], s[0]


class MatrixNArray(ArrayBase):
    def __new__(cls, input_array=None):
        if input_array is None:
            input_array = np.array([])
        ary = np.asarray(input_array, dtype=float)
        ary = ary.reshape((-1, cls.N, cls.N))
        return ary.view(cls)

    def _convertToCompatibleType(self, value):
        """ Convert a value to a type compatible with
        Appending, extending, or inserting
        """
        from .quaternion import Quaternion, QuaternionArray
        from .euler import Euler, EulerArray
        from .transformation import Transformation, TransformationArray
        if isinstance(value, (EulerArray, QuaternionArray, TransformationArray)):
            return value.asMatrixArray()
        elif isinstance(value, (Euler, Quaternion, Transformation)):
            return value.asMatrix()
        return value

    def __getitem__(self, idx):
        ret = super(ArrayBase, self).__getitem__(idx)
        # If we're getting columns from the array then we expect ndarrays back, not math3d types
        # However slicing matrices should give vectors, so only do this is ndim == 2
        if self.ndim == 3 and isinstance(idx, tuple):
            # If we're getting multiple indices And the second index isn't a ":"
            # Then we're getting columns, and therefore want a vectorArray
            try:
                if len(idx) > 1:
                    if not isinstance(idx[-1], slice):
                        return ret.view(VECTOR_ARRAY_BY_SIZE[self.N])
                    elif idx[-1] != slice(None, None, None):
                        return ret.view(VECTOR_ARRAY_BY_SIZE[self.N])
            except ValueError:
                print idx
                raise
        typ = self.getReturnType(ret.shape, idx)
        if typ is None:
            return ret
        return ret.view(typ)


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
        if len(shape) == 3:
            if shape[-2:] == (cls.N, cls.N):
                return cls
        elif len(shape) == 2:
            if shape == (cls.N, cls.N):
                return cls.itemType
            elif shape[1] == cls.N:
                return cls.vectorArrayType
        elif len(shape) == 1:
            if shape[0] == cls.N:
                return cls.vectorType
        return np.ndarray

    @classmethod
    def zeros(cls, length):
        """ Alternate constructor to build an array of matrices that are all zero

        Parameters
        ----------
        length: int
            The number of matrices to build
        """
        return cls(np.zeros((length, cls.N, cls.N)))

    @classmethod
    def ones(cls, length):
        """ Alternate constructor to build an array of matrices that are all one

        Parameters
        ----------
        length: int
            The number of matrices to build
        """
        return cls(np.ones((length, cls.N, cls.N)))

    @classmethod
    def full(cls, length, value):
        """ Alternate constructor to build an array of matrices that are all a given value

        Parameters
        ----------
        length: int
            The number of matrices to build
        value: float
            The value to fill the arrays with
        """
        return cls(np.full((length, cls.N, cls.N), value))

    @classmethod
    def eye(cls, length):
        """ Alternate constructor to build an array of matrices that are all the identity matrix

        Parameters
        ----------
        length: int
            The number of matrices to build
        """
        ret = cls.zeros(length)
        ret[:] = np.eye(cls.N)
        return ret

    def asMatrixSize(self, n):
        """ Return a an array of square matrixes of a given size based on the current matrix.
        If the size is smaller, keep the upper left square of the matrix
        If the size is bigger, the new entries will the same as the identity matrix

        Parameters
        ----------
        n: int
            The new size of the matrix
        """
        if n == self.N:
            return self.copy()
        typ = MATRIX_ARRAY_BY_SIZE[n]
        ret = typ.eye(len(self))
        n = min(n, self.N)
        ret[:, :n, :n] = self[:, :n, :n]
        return ret

    def inverse(self):
        """ Return the inverse of the current matrixes

        Returns
        -------
        MatrixNArray
            The inverted matrices
        """
        return np.linalg.inv(self)

    def invert(self):
        """ Invert the matrices in-place """
        self[:] = np.linalg.inv(self)

    def __mul__(self, other):
        if isinstance(other, (VectorN, VectorNArray)):
            msg = "Cannot multiply matrix*vector. You must multiply vector*matrix\n"
            msg += "Make sure when you multiply it's in `child * parent * grandparent` order"
            raise TypeError(msg)

        if isinstance(other, MatrixN):
            if other.N != self.N:
                msg = "Cannot multiply matrices of different sizes. Got {0} and {1}"
                raise TypeError(msg.format(self.N, other.N))
            other = arrayCompat(other, nDim=3)
            return np.einsum('xij,xjk->xik', self, other).view(type(self))

        if isinstance(other, MatrixNArray):
            if other.N != self.N:
                msg = "Cannot multiply matrices of different sizes. Got {0} and {1}"
                raise TypeError(msg.format(self.N, other.N))
            return np.einsum('xij,xjk->xik', self, other).view(type(self))
        return super(MatrixN, self).__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (VectorN, VectorNArray)):
            msg = "Cannot multiply matrix*vector. You must multiply vector*matrix\n"
            msg += "Make sure when you multiply it's in `child * parent * grandparent` order"
            raise TypeError(msg)

        if isinstance(other, MatrixN):
            if other.N != self.N:
                msg = "Cannot multiply matrices of different sizes. Got {0} and {1}"
                raise TypeError(msg.format(self.N, other.N))
            other = arrayCompat(other, nDim=3)
            self[:] = np.einsum('xij,xjk->xik', self, other)

        if isinstance(other, MatrixNArray):
            if other.N != self.N:
                msg = "Cannot multiply matrices of different sizes. Got {0} and {1}"
                raise TypeError(msg.format(self.N, other.N))
            self[:] = np.einsum('xij,xjk->xik', self, other)
        return super(MatrixN, self).__mul__(other)

    def asEulerArray(self, order="xyz", degrees=False):
        """ Convert the upper left 3x3 of these matrixes to Euler rotations

        Parameters
        ----------
        order: str, optional
            The order in which the axis rotations are applied
            It must be one of these options ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyz']
            Defaults to 'xyz'
        degrees: bool, optional
            Whether the angles are given in degrees or radians. Defaults to False (radians)

        Returns
        -------
        EulerArray
            The array of orientations
        """
        from .euler import EulerArray

        # Taken almost directly from the scipy transforms rotations code
        # This algorithm doesn't actually assume the basis vectors are
        # perpendicular to each other.

        # ---------------------------------------------------------------
        # The algorithm assumes intrinsic frame transformations. The algorithm
        # in the paper is formulated for rotation matrices which are transposition
        # rotation matrices used within Rotation.
        # Adapt the algorithm for our case by
        # Instead of transposing our representation, use the transpose of the
        # O matrix as defined in the paper, and be careful to swap indices

        order = order.lower()
        if sorted(list(order)) != ["x", "y", "z"]:
            raise ValueError("The given order is not a permutation of 'xyz'")

        ma = self.copy()
        if ma.ndim == 2:
            ma = ma.asArray()

        ma = ma.asMatrixSize(3)
        num_rotations = ma.shape[0]

        # Step 0
        # Algorithm assumes axes as column vectors, here we use 1D vectors
        bvs = []
        for axis in order:
            b = np.zeros(3)
            b["xyz".index(axis)] = 1
            bvs.append(b)
        n1, n2, n3 = bvs[:3]

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
        res = np.einsum("...ij,...jk->...ik", c, ma)
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
            angles = np.rad2deg(angles)

        return EulerArray(-angles, degrees=degrees)

    def asQuaternionArray(self):
        """ Convert the upper left 3x3 of this matrix to an Quaternion rotation

        Returns
        -------
        EulerArray
            The array of orientations
        """

        from .quaternion import QuaternionArray

        m3 = self.asMatrixSize(3)
        m3, scales = m3.asRotScaleArray()

        num_rotations = m3.shape[0]

        decision_matrix = np.empty((num_rotations, 4))
        decision_matrix[:, :3] = m3.diagonal(axis1=1, axis2=2)
        decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
        choices = decision_matrix.argmax(axis=1)

        quats = np.empty((num_rotations, 4))

        ind = np.nonzero(choices != 3)[0]
        i = choices[ind]
        j = (i + 1) % 3
        k = (j + 1) % 3

        quats[ind, i] = 1 - decision_matrix[ind, -1] + 2 * m3[ind, i, i]
        quats[ind, j] = m3[ind, i, j] + m3[ind, j, i]
        quats[ind, k] = m3[ind, i, k] + m3[ind, k, i]
        quats[ind, 3] = m3[ind, j, k] - m3[ind, k, j]

        ind = np.nonzero(choices == 3)[0]
        quats[ind, 0] = m3[ind, 1, 2] - m3[ind, 2, 1]
        quats[ind, 1] = m3[ind, 2, 0] - m3[ind, 0, 2]
        quats[ind, 2] = m3[ind, 0, 1] - m3[ind, 1, 0]
        quats[ind, 3] = 1 + decision_matrix[ind, -1]

        # normalize
        qlens = np.sqrt(np.einsum("ij,ij->i", quats, quats))
        quats = quats / qlens[:, None]
        return QuaternionArray(quats)

    @staticmethod
    def lookAts(looks, ups, axis="xy"):
        """ Set the upper 3x3 of these matrices to look along the look-vectors, oriented to the up-vectors

        Parameters
        ----------
        looks: Vector3, Vector3Array, list
            The pointing directions of the look axis
        ups: Vector3, Vector3Array, list
            The pointing directions of the up axis
        axis: string
            The axes to align to the look and up vectors (ie: 'xy', 'yz', '-zy', 'x-z')

        Returns
        -------
        Matrix3Array:
            The looking matrices

        """
        looks, ups = arrayCompat(looks, ups)
        looks = VECTOR_ARRAY_BY_SIZE[3](looks)
        ups = VECTOR_ARRAY_BY_SIZE[3](ups)

        axis = axis.lower()

        looks = looks.normal()
        sides = looks.cross(ups.normal())
        sides.normalize()
        ups = sides.cross(looks)
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
        """ Rotate each of the matrixes so that the newUpAxis points where the oldUpAxis was

        Parameters
        ----------
        oldUpAxis: int
            The index of the old axis
        newUpAxis: int
            The index of the new axis

        Returns
        -------
        Matrix3Array:
            The rotated matrices
        """
        reo = range(self.N)
        reo[oldUpAxis], reo[newUpAxis] = reo[newUpAxis], reo[oldUpAxis]

        ret = self.copy()
        ret = ret[:, :, reo]
        ret = ret[:, reo, :]
        ret[:, :, oldUpAxis] *= -1
        ret[:, oldUpAxis, :] *= -1
        return ret

    def getHandedness(self):
        """ Return the handedness of each matrix in the array

        Returns
        -------
        NumpyArray:
            1.0 if the matrix at the index is right-handed, otherwise -1.0
        """
        # look for flipped matrices
        m33 = self.asMatrixSize(3)
        flips = m33[:, 0].cross(m33[:, 1]).dot(m33[:, 2])
        negs = np.ones(flips.shape)
        negs[flips < 0.0] = -1.0
        return negs

    def asRotScaleArray(self):
        """ Get a normalized, right-handed rotation matrix along with the
        scales that were passed in

        Returns
        -------
        Matrix3Array:
            The normalized right-handed rotation matrices
        Vector3Array:
            The scale part of the matrixes
        """
        m33 = self.asMatrixSize(3)
        scale = np.sqrt(np.einsum("...ij,...ij->...i", m33, m33))
        negs = m33.getHandedness()[..., None]
        scale *= negs

        # in numpy, transposing is basically free
        m33 = m33.transpose((0, 2, 1))
        m33 = m33 / scale[..., None]
        m33 = m33.transpose((0, 2, 1))

        return m33, VECTOR_ARRAY_BY_SIZE[3](scale)

    def asScaleArray(self):
        """ Return the scale part of the matrixes

        Returns
        -------
        Vector3Array:
            The scale part of the matrixes
        """
        m33 = self.asMatrixSize(3)
        scale = np.sqrt(np.einsum("...ij,...ij->...i", m33, m33))

        negs = m33.getHandedness()
        scale *= negs

        return VECTOR_ARRAY_BY_SIZE[3](scale)

    def asTranslationArray(self):
        """ Return the translation part of the matrixes

        Returns
        -------
        Vector3Array:
            The translation part of the matrixes
        """
        return self[:, 3, :3]

    def decompose(self):
        """ Decompose the matrix into Translation, Rotation, and Scale

        Returns
        -------
        Vector3Array:
            The translation array
        QuaternionArray:
            The Rotation array
        Vector3Array:
            The Scale array
        """
        tran = self.asTranslationArray()
        mrot, scale = self.asRotScaleArray()
        rot = mrot.asQuaternionArray()
        return tran, rot, scale

    def asTransformArray(self):
        """ Decompose the matrixes into a transform array

        Returns
        -------
        TransformArray:
            The transform array
        """
        from .transformation import TransformationArray
        t, r, s = self.asMatrixSize(4).decompose()
        return TransformationArray.fromParts(translation=t, rotation=r, scale=s)

    def flattened(self):
        return self.reshape((-1, self.N**2))


# Register the default sizes of array dynamically
MATRIX_BY_SIZE = {}
MATRIX_ARRAY_BY_SIZE = {}
glo = globals()
for n in [2, 3, 4]:
    name = "Matrix{0}".format(n)
    aname = "Matrix{0}Array".format(n)
    m = type(name, (MatrixN,), {"N": n})
    ma = type(aname, (MatrixNArray,), {"N": n})
    m.arrayType = ma
    ma.itemType = m
    m.vectorType = VECTOR_BY_SIZE[n]
    ma.vectorType = VECTOR_BY_SIZE[n]
    m.vectorArrayType = VECTOR_ARRAY_BY_SIZE[n]
    ma.vectorArrayType = VECTOR_ARRAY_BY_SIZE[n]

    glo[name] = m
    glo[aname] = ma
    MATRIX_BY_SIZE[n] = m
    MATRIX_ARRAY_BY_SIZE[n] = ma
