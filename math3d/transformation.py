from __future__ import print_function, absolute_import
import numpy as np
from .vectorN import Vector3, Vector3Array, VectorN, VectorNArray
from .quaternion import Quaternion, QuaternionArray
from .matrixN import Matrix4, Matrix4Array, Matrix3, Matrix3Array
from .euler import Euler, EulerArray
from .utils import asarray, arrayCompat
from .base import MathBase, ArrayBase


class Transformation(MathBase):
    def __new__(cls, input_array=None):
        if input_array is None:
            # Stored in SRT
            ary = np.zeros(10)
            ary[:3] = 1  # Scale
            ary[6] = 1  # Quat.w
        else:
            ary = np.asarray(input_array, dtype=float)
        if ary.size != 10:
            raise ValueError(
                "Initializer for Vector{0} must be of length {0}".format(10)
            )
        return ary.view(cls)

    def __repr__(self):
        return super(Transformation, self).__repr__()

    def __str__(self):
        return "[s:{0}, q:{1}, t:{2}]".format(
            self.scale, self.rotation, self.translation
        )

    @property
    def translation(self):
        return Vector3(self[7:])

    @translation.setter
    def translation(self, other):
        self[7:] = other

    @property
    def rotation(self):
        return Quaternion(self[3:7])

    @rotation.setter
    def rotation(self, other):
        self[3:7] = other

    @property
    def scale(self):
        return Vector3(self[:3])

    @scale.setter
    def scale(self, other):
        self[:3] = other

    @classmethod
    def getReturnType(cls, shape, idx=None):
        if not shape:
            return None
        if len(shape) == 1:
            if shape[0] == 10:
                return cls
        elif len(shape) == 2:
            # This could happen with fancy indexing
            if shape[-1] == 10:
                return TransformationArray

        return np.ndarray

    def asMatrix(self):
        ret = Matrix4()
        ret[3, :3] = self.translation
        rot = np.dot(np.diag(self.scale), self.rotation.asMatrix())
        ret[:3, :3] = rot
        return ret

    @classmethod
    def partCheck(cls, translation=None, rotation=None, scale=None):
        if translation is not None:
            translation = asarray(translation)
            if translation.ndim > 1:
                raise ValueError("Translation has too many dimensions")
            elif translation.shape[-1] != 3:
                raise ValueError("Provided translation is not 3d")

        if scale is not None:
            scale = asarray(scale)
            if scale.ndim > 1:
                raise ValueError("Scale has too many dimensions")
            elif scale.shape[-1] != 3:
                raise ValueError("Provided scale is not 3d")

        if rotation is not None:
            rotation = asarray(rotation)
            if rotation.ndim > 1:
                raise ValueError("Rotation has too many dimensions")
            elif isinstance(rotation, Euler):
                rotation = rotation.asQuaternion()
            elif rotation.shape[-1] != 4:
                raise ValueError(
                    "Provided rotation could not be converted to quaternions"
                )
        return translation, rotation, scale

    @classmethod
    def fromParts(cls, translation=None, rotation=None, scale=None):
        translation, rotation, scale = cls.partCheck(
            translation=translation, rotation=rotation, scale=scale
        )
        ret = cls()
        if translation is not None:
            ret.translation = translation
        if scale is not None:
            ret.scale = scale
        if rotation is not None:
            ret.rotation = rotation
        return ret

    def mirrored(self, axis="x"):
        """ Mirror the transformation along the given axis

        Parameters
        ----------
        axis: str, optional
            The axis to mirror along. Defaults to "x"

        Returns
        -------
        Transformation:
            The mirrored transformation
        """
        return self.asArray().mirrored(axis=axis)[0]

    @classmethod
    def lookAt(
        cls, position, look, normal, axis="xy", negativeSide=False,
    ):
        """
        Make a vector-oriented Transformation from a position, a major look
        axis, an a minor look axis

        Parameters
        ----------
        positions: Vector3
            The point that will be the translation of the output transform
        looks: Vector3
            The vector that the main axis will look along
        normals: Vector3
            The vector that are pointing in the normal direction
        axis: string
            Axis pointing to the target and to the normal (ie: 'xy', 'yz', '-zy', 'x-z')
        negativeSide: bool
            Flip the transform

        Returns
        -------
        TransformationArray:
            The resulting transformations
        """
        ta = TransformationArray.lookAts(
            position.asArray(),
            look.asArray(),
            normal.asArray(),
            axis=axis,
            negativeSide=negativeSide,
        )
        return ta[0]

    def copy(self, translation=None, rotation=None, scale=None):
        ta = self.asArray()
        ta = ta.copy(translation, rotation, scale)
        return ta[0]

    def __mul__(self, other):
        if isinstance(other, (VectorN, VectorNArray)):
            msg = "Cannot multiply transformation*vector. You must multiply vector*transformation\n"
            msg += "Make sure when you multiply it's in `child * parent * grandparent` order"
            raise TypeError(msg)

        if hasattr(other, 'asMatrix'):
            other = other.asMatrix()
        elif hasattr(other, 'asMatrixArray'):
            other = other.asMatrixArray()
        ret = self.asMatrix() * other
        return ret.asTransform()

class TransformationArray(ArrayBase):
    def __new__(cls, input_array=None):
        if input_array is None:
            input_array = np.array([])
        ary = np.asarray(input_array, dtype=float)
        ary = ary.reshape((-1, 10))
        return ary.view(cls)

    @classmethod
    def eye(cls, length):
        """ Alternate constructor to build an array of transformations that are all the identity

        Parameters
        ----------
        length: int
            The number of transformations to build
        """
        ret = np.zeros((length, 10), dtype=float)
        ret[:, :3] = 1.0
        ret[:, 6] = 1.0
        return cls(ret)

    @classmethod
    def partCheck(cls, translation=None, rotation=None, scale=None):
        if translation is not None:
            translation = asarray(translation)
            if translation.ndim == 1:
                translation = translation[None, ...]
            if translation.shape[-1] != 3:
                raise ValueError("Provided translation is not 3d")

        if scale is not None:
            scale = asarray(scale)
            if scale.ndim == 1:
                scale = scale[None, ...]
            if scale.shape[-1] != 3:
                raise ValueError("Provided scale is not 3d")

        if rotation is not None:
            rotation = asarray(rotation)
            if isinstance(rotation, Euler):
                rotation = rotation.asQuaternion()[None, ...]
            elif isinstance(rotation, EulerArray):
                rotation = rotation.asQuaternionArray()
            if rotation.ndim == 1:
                rotation = rotation[None, ...]
            if rotation.shape[-1] != 4:
                raise ValueError(
                    "Provided rotation could not be converted to quaternions"
                )

        return translation, rotation, scale

    @classmethod
    def fromParts(cls, translation=None, rotation=None, scale=None):
        translation, rotation, scale = cls.partCheck(
            translation=translation, rotation=rotation, scale=scale
        )

        count = [len(i) for i in (translation, rotation, scale) if i is not None]
        if not count:
            raise ValueError(
                "Nothing passed to the .fromParts constructor. Cannot infer length"
            )

        count = sorted(set(count))
        if len(count) == 2:
            if count[0] != 1:
                raise ValueError(
                    "Parts passed have different lengths, and cannot be broadcast"
                )
        elif len(count) != 1:
            raise ValueError(
                "Parts passed have different lengths, and cannot be broadcast together"
            )
        count = count[-1]

        ret = cls.eye(count)
        if translation is not None:
            ret.translation = translation
        if scale is not None:
            ret.scale = scale
        if rotation is not None:
            ret.rotation = rotation
        return ret

    def copy(self, translation=None, rotation=None, scale=None):
        ret = super(TransformationArray, self).copy()
        translation, rotation, scale = self.partCheck(
            translation=translation, rotation=rotation, scale=scale
        )
        if translation is not None:
            ret.translation = translation
        if scale is not None:
            ret.scale = scale
        if rotation is not None:
            ret.rotation = ret.rotation
        return ret

    def __repr__(self):
        return super(TransformationArray, self).__repr__()

    def __str__(self):
        if len(self) > 100:
            first3 = [str(i) for i in self[:3]]
            last3 = [str(i) for i in self[-3:]]
            lines = first3 + ["..."] + last3
        else:
            lines = [str(i) for i in self]
        lines = [" " + i for i in lines]
        lines[0] = "[" + lines[0][1:]
        lines[-1] = lines[-1] + "]"
        return "\n".join(lines)

    def __mul__(self, other):
        if isinstance(other, (VectorN, VectorNArray)):
            msg = "Cannot multiply transformation*vector. You must multiply vector*transformation\n"
            msg += "Make sure when you multiply it's in `child * parent * grandparent` order"
            raise TypeError(msg)

        if hasattr(other, 'asMatrix'):
            other = other.asMatrix()
        elif hasattr(other, 'asMatrixArray'):
            other = other.asMatrixArray()
        ret = self.asMatrixArray() * other
        return ret.asTransform()

    def _convertToCompatibleType(self, value):
        """ Convert a value to a type compatible with
        Appending, extending, or inserting
        """
        from .matrixN import Matrix4, Matrix4Array

        if isinstance(value, Matrix4Array):
            return value.asTransformArray()
        elif isinstance(value, Matrix4):
            return value.asTransform()
        return value

    @classmethod
    def getReturnType(cls, shape, idx=None):
        if not shape:
            return None
        if len(shape) == 1:
            if shape[0] == 10:
                return Transformation
        elif len(shape) == 2:
            # This could happen with fancy indexing
            if shape[-1] == 10:
                return cls
        return np.ndarray

    @property
    def translation(self):
        return Vector3Array(self[:, 7:])

    @translation.setter
    def translation(self, other):
        self[:, 7:] = other

    @property
    def rotation(self):
        return QuaternionArray(self[:, 3:7])

    @rotation.setter
    def rotation(self, other):
        self[:, 3:7] = other

    @property
    def scale(self):
        return Vector3Array(self[:, :3])

    @scale.setter
    def scale(self, other):
        self[:, :3] = other

    def asMatrixArray(self):
        size = len(self.translation)

        x = range(3)
        scaleArray = Matrix3Array.zeros(size)
        scaleArray[:, x, x] = self.scale

        rotArray = self.rotation.asMatrixArray()
        srArray = scaleArray * rotArray

        tranArray = Matrix4Array.eye(size)

        tranArray[:, 3, :3] = self.translation
        tranArray[:, :3, :3] = srArray

        return tranArray

    @classmethod
    def lookAts(
        cls, positions, looks, normals, axis="xy", negativeSide=False,
    ):
        """
        Make an array of vector-oriented Transformations
        positions, looks, and normals must all have the same length, or broadcast

        Parameters
        ----------
        positions: Vector3Array
            The points that will be the translations of the output transforms
        looks: Vector3Array
            The vectors that the main axis will look along per transform
        normals: Vector3Array
            The vectors that are pointing in the normal direction per transform
        axis: string
            Axis pointing to the target and to the normal (ie: 'xy', 'yz', '-zy', 'x-z')
        negativeSide: bool
            Flip the transforms for use on a mirrored chain

        Returns
        -------
        TransformationArray:
            The resulting transformations
        """
        positions, looks, normals = arrayCompat(positions, looks, normals)
        normals = normals.normal()
        looks = looks.normal()
        if bool(negativeSide):
            looks *= -1
            normals *= -1

        rots = Matrix3Array.lookAts(looks, normals, axis=axis).asQuaternionArray()

        out = cls.eye(len(positions))
        out.rotation = rots
        out.translation = positions
        return out

    @classmethod
    def chain(
        cls, positions, normal=None, axis="xy", negativeSide=False, endTransform=True,
    ):
        """Alternate constructor to create a chain of transforms based on a set of positions.

        The orientation of the last point is not well defined. So you can choose to either
        completely skip it, or re-use the orientation of the next-to-last point

        Parameters
        ----------
        positions: Vector3Array
            The points that will be the translations of the output transforms
        normal: Vector3, optional
            The normal vector for the first position. Subsequent normals will be created
            using Vector3.parallelTransport.
            If not given, the value Vector3((0, 0, 1)) will be used
        axis: string
            Axis pointing to the target and to the normal (ie: 'xy', 'yz', '-zy', 'x-z')
        negativeSide: bool
            Flip the transforms for use on a mirrored chain
        endTransform: bool
            Whether to include the guessed-at orientation of the last point in the output

        Returns
        -------
        TransformationArray:
            The resulting transformations
        """
        looks = positions[1:] - positions[:-1]

        if normal is None:
            normal = Vector3((0, 0, 1))
        if np.isclose(normal.lengthSquared(), 0.0):
            raise ValueError("Zero-length Normal Provided")
        # don't calculate the endTransform for the parallelTransport
        normals = positions.parallelTransport(normal, endTransform=False)

        ret = cls.lookAts(
            positions[:-1], looks, normals, axis=axis, negativeSide=negativeSide
        )
        if endTransform:
            end = ret[-1].copy()
            end.translation = positions[-1]
            ret = ret.appended(end)
        return ret

    def mirrored(self, axis="x"):
        """ Mirror the transformation along the given axis

        Parameters
        ----------
        axis: str, optional
            The axis to mirror along. Defaults to "x"

        Returns
        -------
        Transformation:
            The mirrored transformation
        """
        try:
            index = "xyz".index(axis.lower())
        except IndexError:
            raise ValueError('Axis must be "x", "y", or "z"')

        m = self.asMatrixArray()
        m[:, :, index] *= -1
        return m.asTransformArray()
