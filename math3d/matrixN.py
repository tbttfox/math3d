import numpy as np
from vectorN import VectorN, VectorNArray, VECTOR_BY_SIZE, VECTOR_ARRAY_BY_SIZE
from .utils import arrayCompat, sliceLength
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
            if shape[-1] in VECTOR_ARRAY_BY_SIZE:
                return VECTOR_ARRAY_BY_SIZE[shape[-1]]
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

    def normalized(self):
        """ return a normalized, rotation matrix

        Returns
        -------
        Matrix3:
            The normalized right-handed rotation matrix
        """
        return self.asArray().normalized()[0]

    def normalize(self):
        """ Normalize the rotation matrix in-place """
        self[:3, :3] = self.normalized()


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
                        num = sliceLength(idx[-1])
                        return ret.view(VECTOR_ARRAY_BY_SIZE[num])
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
        ret[:, lookAxis, :] = looks
        ret[:, upAxis, :] = ups
        ret[:, sideAxis, :] = sides

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

    def _scales(self):
        """ Get the scale of each matrix column """
        return np.sqrt(np.einsum("...ij,...ij->...j", self, self))

    def _handedness(self):
        """ Get the handedness of each matrix. -1 means left-handed """
        # look for flipped matrices
        flips = self[:, 0].cross(self[:, 1]).dot(self[:, 2])
        negs = np.ones(flips.shape)
        negs[flips < 0.0] = -1.0
        return negs

    def normalized(self):
        """ Return the normal of the current matrices

        Returns
        -------
        MatrixNArray
            The normalized matrices
        """
        return self / self._scales()[:, None, :]

    def normalize(self):
        """ Normalize the columns of the arrays in-place """
        self[:] = self.normalized()

    def getHandedness(self):
        """ Return the handedness of each matrix in the array

        Returns
        -------
        NumpyArray:
            1.0 if the matrix at the index is right-handed, otherwise -1.0
        """
        return self.asMatrixSize(3)._handedness()



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
        scale = m33._scales() * m33._handedness()[..., None]
        m33 = m33 / scale[:, None, :]
        return m33, VECTOR_ARRAY_BY_SIZE[3](scale)

    def asScaleArray(self):
        """ Return the scale part of the matrixes

        Returns
        -------
        Vector3Array:
            The scale part of the matrixes
        """
        m33 = self.asMatrixSize(3)
        scale = m33._scales() * m33._handedness()[..., None]
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

    def asQuaternionArray(self):
        """ Convert the upper left 3x3 of this matrix to an Quaternion rotation

        Returns
        -------
        QuaternionArray
            The array of orientations
        """

        from .quaternion import QuaternionArray

        # work on a copy */
        mat = self.normalized()

        # rotate z-axis of matrix to z-axis */
        nor = VECTOR_ARRAY_BY_SIZE[3].zeros(len(self))
        nor[:, 0] = mat[:, 2, 1]
        nor[:, 1] = -mat[:, 2, 0]
        nor.normalize()
        nor[np.isnan(nor)] = 0

        co = mat[:, 2, 2]
        co[co <= -1.0] = -1.0
        co[co >= 1.0] = 1.0
        angle = 0.5 * np.arccos(co)
        co = np.cos(angle)
        si = np.sin(angle)

        q1 = QuaternionArray.eye(len(self))
        q1[:, :3] = -nor * si[:, None]
        q1[:, 3] = co[None, :]

        # rotate back x-axis from mat, using inverse q1 */
        matr = q1.asMatrixArray()
        mat[:, 0, :3] = mat[:, 0, :3] * matr.inverse()

        # and align x-axes */
        angle = 0.5 * np.arctan2(mat[:, 0, 1], mat[:, 0, 0])
        co = np.cos(angle)
        si = np.sin(angle)
        q2 = QuaternionArray.eye(len(self))
        q2[:, 2] = si[None, :]
        q2[:, 3] = co[None, :]
        q = q1 * q2

        # only return quaternions where the scalar value is positive
        negScalar = q[:, 3] < 0
        q[negScalar] *= -1

        return q

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

        (i, j, k), parity = EulerArray.ORDER_PARITY[order]
        mat = self.asMatrixSize(3).normalized()

        # Get TWO differnt euler conversions then choose the best one
        cy = np.hypot(mat[:, i, i], mat[:, i, j])

        eul1 = EulerArray.zeros(len(self))
        eul2 = EulerArray.zeros(len(self))

        # use 16 because powers of two
        epsilon = 1.6e-8
        pos = cy > epsilon
        neg = ~pos

        eul1[pos, i] = np.arctan2(mat[pos, j, k], mat[pos, k, k])
        eul1[pos, j] = np.arctan2(-mat[pos, i, k], cy[pos])
        eul1[pos, k] = np.arctan2(mat[pos, i, j], mat[pos, i, i])

        eul2[pos, i] = np.arctan2(-mat[pos, j, k], -mat[pos, k, k])
        eul2[pos, j] = np.arctan2(-mat[pos, i, k], -cy[pos])
        eul2[pos, k] = np.arctan2(-mat[pos, i, j], -mat[pos, i, i])

        eul1[neg, i] = np.arctan2(-mat[neg, k, j], mat[neg, j, j])
        eul1[neg, j] = np.arctan2(-mat[neg, i, k], cy[neg])
        eul1[neg, k] = 0
        eul2[neg] = eul1[neg]

        if parity:
            eul1, eul2 = -eul1, -eul2

        # The "best" euler is the one with the smallest abs sum
        d1 = np.abs(eul1).sum(axis=-1)
        d2 = np.abs(eul2).sum(axis=-1)
        mx = d1 > d2
        eul1[mx] = eul2[mx]
        return eul1


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
