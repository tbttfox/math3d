import numpy as np
from .vectorN import Vector3, Vector3Array
from .quaternion import Quaternion, QuaternionArray
from .matrixN import Matrix4, Matrix4Array, Matrix3Array


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
        return "[s:{0}, q:{1}, t:{2}]".format(self.scale, self.rotation, self.translation)

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
    def _getReturnType(cls, shape):
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
        typ = self._getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

    def asMatrix(self):
        ret = Matrix4()
        ret[3, :3] = self.translation
        rot = np.diag(self.scale) * self.rotation.asMatrix()
        ret[:3, :3] = rot
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
    def _getReturnType(cls, shape):
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
        typ = self._getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

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


