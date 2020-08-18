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
            ary = np.asarray(input_array)
        if ary.size != 10:
            raise ValueError(
                "Initializer for Vector{0} must be of length {0}".format(10)
            )
        return ary.view(cls)

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

    def getReturnType(self, shape):
        if not shape:
            return None
        if len(shape) == 1:
            if shape[0] == 10:
                return type(self)
        elif len(shape) == 2:
            # This could happen with fancy indexing
            if shape[-1] == 10:
                return self.arrayType

        return np.ndarray

    def asMatrix(self):
        ret = Matrix4.eye()
        ret[3, :3] = self.translation
        rot = np.diag(self.scale) * self.rotation.asMatrix()
        ret[:3, :3] = rot
        return ret


class TransformationArray(np.ndarray):
    def __new__(cls, input_array=None):
        ary = np.asarray(input_array)
        ary = ary.reshape((-1, 10))
        return ary.view(cls)

    def getReturnType(self, shape):
        if not shape:
            return None
        if len(shape) == 1:
            if shape[0] == 10:
                return self.itemType
        elif len(shape) == 2:
            if shape[-1] == 10:
                return type(self)

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

