# Math3D
## A math library designed for vfx built on top of numpy

These vector/matrix libraries are a dime a dozen, but I felt the need to put my own hat into this ring because I usually see a couple of approaches, each with their own drawbacks.
The most common approach is to just write regular-old pure-python classes. And, while this is incredibly convenient, it's unfortunately not very performant.
The other one I see a lot is the c/c++ module.  Often, people just end up using Blender's very good mathutils library.  And while it is more performant, you lose out on options without having to either recompile all the time.
Then every once in a while, I see somebody try to use numpy, but they often only use it as a back-end. And while the algorithms already exist to do most of the work, the usual choice of abstraction can make it even slower than the pure python approach.

## Why Math3d?
Math3d subclasses numpy arrays to build the base objects. Since the objects are already numpy arrays, we can leverage all the power that comes with numpy (the fancy indexing, and all the pre-built numpy and scipy functions), on top of a more vfx-friendly convenient interface. But the real power of math3d is that *arrays* of the objects are special-cased to be a single numpy object. This means that the broadcasting and vectorized looping of numpy is also available.

The simple things are still simple (eg. dot two vectors, or convert quaternions to Euler rotations). But bigger and more complicated loops like multiplying thousands of 3d vectors by a 4x4 matrix, or inverting an array of matrices, or converting an array of Euler rotations into quaternions will leverage all of the performance that numpy has to offer, so long as you use the Math3d array types where possible.

This is performant enough to even test Maya deformer nodes in python ... especially if you use my fast mayaToNumpy library.


Slicing array objects also handles type conversions in most cases.  Got a Matrix4Array typed object?  If you were to slice that array with `array[:, -1, :3]` to get the translation data, it automatically changes the type to a Vector3Array.
The type detection is good, however it's not perfect. Some of the crazier fancy indexing can break the type detection. So watch out for that.
