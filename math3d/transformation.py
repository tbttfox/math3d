import numpy as np
from .vectorN import Vector3, Vector3Array
from .quaternion import Quaternion, QuaternionArray
from .matrixN import Matrix4, Matrix4Array, Matrix3Array
from .euler import Euler, EulerArray
from .utils import asarray


class Transformation(np.ndarray):
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
    def getReturnType(cls, shape):
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

    def __getitem__(self, idx):
        ret = super(Transformation, self).__getitem__(idx)
        typ = self.getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

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


class TransformationArray(np.ndarray):
    def __new__(cls, input_array=None):
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
            raise ValueError("Nothing passed to the .fromParts constructor. Cannot infer length")

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
            ret.rotation = ret.rotation
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

    @classmethod
    def getReturnType(cls, shape):
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

    def __getitem__(self, idx):
        ret = super(TransformationArray, self).__getitem__(idx)
        typ = self.getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

    def toArray(self):
        """ Return the array type of this object

        Returns
        -------
        ArrayType:
            The current object up-cast into a length-1 array
        """
        return self[None, ...]

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
    def lookAts(cls, positions, targets, normals, axis="xy", negativeSide=False):
        """Alternate constructor to create a Transform from given positions

        Args:
            positions(Vector3Array): The transform translation and reference point
            targets(Vector3Array): The pointing direction of first axis
            normals(Vector3Array): The normal direction of the second axis
            axis(string): axis pointing to the target and to the normal (ie: 'xy', 'yz', '-zy', 'x-z')
            negativeSide(bool): Use mirror method to inverse the transformation (negative scaling or inversed rotation)
                mirror method can be set globally using setMirrorMethod()

        Returns:
            Transform: the resulting transformation
        """
        looks = targets - positions
        ups = normals.normal()

        if bool(negativeSide):
            looks *= -1
            ups *= -1

        rots = Matrix4.lookAts(looks, ups, axis=axis).asQuaternionArray()

        out = cls()
        out.rotation = rots
        out.translation = positions
        return out

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
