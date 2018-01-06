from cpython cimport array
import numpy as np

cdef double NAN
NAN = float("NaN")
cdef int Py_EQ
Py_EQ = 2

cdef class LinearInterpolation:
    cdef readonly int size
    cdef readonly double[:] x
    cdef readonly double[:] y
    cdef readonly double lower_x
    cdef readonly double lower_y
    cdef readonly double upper_y
    cdef readonly double upper_x
    
    def __init__(LinearInterpolation self, double[:] x, double[:] y, double lower_x, double lower_y,
                double upper_x, double upper_y):
        self.size = x.shape[0]
        self.x = x
        self.y = y
        self.lower_x = lower_x
        self.lower_y = lower_y
        self.upper_x = upper_x
        self.upper_y = upper_y
    
    def __richcmp__(LinearInterpolation self, other, int op):
        if not isinstance(other, LinearInterpolation):
            return NotImplemented
        if op != Py_EQ:
            return NotImplemented
        if self is other:
            return True
        if self.size != other.size:
            return False
        if self.lower_x != other.lower_x:
            return False
        if self.lower_y != other.lower_y:
            return False
        if self.upper_x != other.upper_x:
            return False
        if self.upper_y != other.upper_y:
            return False
        cdef Py_ssize_t i
        if self.x is not other.x:
            for i in range(self.size):
                if self.x[i] != other.x[i]:
                    return False
        if self.y is not other.y:
            for i in range(self.size):
                if self.y[i] != other.y[i]:
                    return False
        return True
    
    cdef int _bisect_left(LinearInterpolation self, double v):
        cdef int lower = 0
        cdef int upper = self.size - 1
        if v >= self.x[upper]:
            return upper
        if v < self.x[lower]:
            return -1
        cdef int test
        cdef double test_value
        while lower < upper - 1:
            test = lower + ((upper - lower) / 2)
            test_value = self.x[test]
            if test_value <= v:
                lower = test
            else:
                upper = test
        return lower
    
    cpdef int bisect_left(LinearInterpolation self, double v):
        return self._bisect_left(v)
    
    cdef double call(LinearInterpolation self, double v):
        if v >= self.upper_x:
            return self.upper_y
        if v <= self.lower_x:
            return self.lower_y
        cdef int left = self._bisect_left(v)
        cdef double left_x, left_y, right_x, right_y
        if left == -1:
            left_x = self.lower_x
            left_y = self.lower_y
        else:
            left_x = self.x[left]
            left_y = self.y[left]
        if left == self.size - 1:
            right_x = self.upper_x
            right_y = self.upper_y
        else:
            right_x = self.x[left + 1]
            right_y = self.y[left + 1]
        return left_y + ((right_y - left_y) / (right_x - left_x)) * (v - left_x)
    
    def __call__(LinearInterpolation self, double v):
        return self.call(v)

cdef class LinearInterpolationArray:
    cdef readonly int size
    cdef readonly LinearInterpolation[:] f
     
    def __init__(LinearInterpolationArray self, LinearInterpolation[:] f):
        self.size = f.shape[0]
        self.f = f
     
    @classmethod
    def empty(cls, int size):
        f = np.empty(shape=(size,), dtype=object)
        return cls(f)
     
    def __len__(LinearInterpolationArray self):
        return self.size
     
    def __richcmp__(LinearInterpolationArray self, other, int op):
        if not isinstance(other, LinearInterpolationArray):
            return NotImplemented
        if op != Py_EQ:
            return NotImplemented
        if self is other:
            return True
        if self.size != other.size:
            return False
        cdef Py_ssize_t i
        if self.f is not other.f:
            for i in range(self.size):
                if self.f[i] != other.f[i]:
                    return False
        return True
    
    def __setitem__(LinearInterpolationArray self, key, value):
        if isinstance(value, LinearInterpolationArray):
            self.f.__setitem__(key, value.f)
        else:
            self.f.__setitem__(key, value)
     
    def __getitem__(LinearInterpolationArray self, key):
        item = self.f.__getitem__(key)
        if isinstance(item, LinearInterpolation):
            return item
        else:
            return LinearInterpolationArray(self.f.__getitem__(key))
     
#     def __getslice__(LinearInterpolationArray self, Py_ssize_t i, Py_ssize_t j):
#         return LinearInterpolationArray(self.f[i:j])
#      
#     def __setslice__(LinearInterpolationArray self, Py_ssize_t i, Py_ssize_t j, x):
#         if isinstance(x, LinearInterpolationArray):
#             self.f[i:j] = x.f
#         else:
#             self.f[i:j] = x
            
    cdef double[:] call_on_array(LinearInterpolationArray self, double[:] v):
        cdef double[:] result = np.empty(shape=self.size, dtype=float)
        cdef Py_ssize_t i, v_size
        v_size = v.shape[0]
        cdef LinearInterpolation f
        
        for i in range(self.size):
            f = self.f[i]
            if f is not None:
                result[i] = f.call(v[i % v_size])
            else:
                result[i] = NAN
        return result
    
    cdef double[:] call_on_scalar(LinearInterpolationArray self, double v):
        return self.call_on_array(np.array([v]))
    
    def __call__(LinearInterpolationArray self, v):
        if isinstance(v, (int, long, float)):
            return self.call_on_scalar(v)
        return self.call_on_array(v)
     
    
