import numpy as np


class Euler(np.ndarray):
    def __new__(cls, input_array=None, order='xyz', degrees=False):
        if input_array is None:
            ary = np.zeros(3)
        else:
            ary = np.asarray(input_array, dtype=float)
        if ary.size != 3:
            raise ValueError(
                "Initializer for Vector{0} must be of length {0}".format(3)
            )

        ret = ary.view(cls)
        ret.order = order.lower()
        ret._degrees = degrees
        return ret

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.order = getattr(obj, 'order', 'xyz')
        self._degrees = getattr(obj, 'degrees', False)

    @property
    def degrees(self):
        return self._degrees

    def toRadians(self):
        if self.degrees:
            return np.deg2rad(self)
        return self

    def toDegrees(self):
        if not self.degrees:
            return np.rad2deg(self)
        return self

    def getReturnType(self, shape):
        if not shape:
            return None
        if len(shape) == 1:
            if shape[0] == 3:
                return type(self)
        elif len(shape) == 2:
            # This could happen with fancy indexing
            if shape[-1] == 3:
                return EulerArray

        return np.ndarray

    def __getitem__(self, idx):
        ret = super(Euler, self).__getitem__(idx)
        typ = self.getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

    def asMatrix(self):
        return self[None, ...].asMatrixArray()[0]

    def asQuaternion(self):
        return self[None, ...].asQuaternionArray()[0]

    def toNewOrder(self, order):
        return self[None, ...].toNewOrder(order)[0]


class EulerArray(np.ndarray):
    def __new__(cls, input_array, order='xyz', degrees=False):
        ary = np.asarray(input_array, dtype=float)
        ary = ary.reshape((-1, 3))
        ret = ary.view(cls)
        ret.order = order.lower()
        ret._degrees = degrees
        return ret

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.order = getattr(obj, 'order', 'xyz')
        self._degrees = getattr(obj, 'degrees', False)

    def getReturnType(self, shape):
        if not shape:
            return None
        if len(shape) == 2:
            if shape[1] == 3:
                return type(self)
        elif len(shape) == 1:
            if shape[0] == 3:
                return Euler
        return np.ndarray

    def __getitem__(self, idx):
        ret = super(EulerArray, self).__getitem__(idx)
        typ = self.getReturnType(ret.shape)
        if typ is None:
            return ret
        return ret.view(typ)

    @property
    def degrees(self):
        return self._degrees

    def toRadians(self):
        if self.degrees:
            return np.deg2rad(self)
        return self

    def toDegrees(self):
        if not self.degrees:
            return np.rad2deg(self)
        return self

    def toNewOrder(self, order):
        q = self.asQuaternionArray()
        return q.asEulerArray(order=order, degrees=self.degrees)

    @classmethod
    def zeros(cls, length):
        return cls(np.zeros((length, 3)))

    def append(self, v):
        self.resize((len(self) + 1, 3))
        self[-1] = v

    def extend(self, v):
        self.resize((len(self) + len(v), 3))
        self[-len(v):] = v

    def asQuaternionArray(self):
        from .quaternion import QuaternionArray
        # Convert multiple euler triples to quaternions
        eulers = self
        if self.degrees:
            eulers = np.deg2rad(self)

        result = QuaternionArray.alignedRotations(self.order[0], eulers[:, 0])
        for idx, axis in enumerate(self.order[1:], start=1):
            result = result * QuaternionArray.alignedRotations(axis, eulers[:, idx])
        return result

    def asMatrixArray(self):
        q = self.asQuaternionArray()
        return q.asMatrixArray()



