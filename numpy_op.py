import numpy as np

a = 1
b = np.full((2, 2), 2, np.int32)
print(a / b)

a = 1
b = np.full((2, 2), 2, np.float32)
print(a / b)

a = np.full((2, 2), 1, np.int)
b = 2
print(a / b)

a = np.full((2, 2), 1, np.int)
b = 2.0
print(a / b)

a = 1
b = np.full((2, 2), 2, np.float32)
print(a - b)

###

a = np.full((2, 2), 2, np.int32)
b = 3
print(a ** b)

a = np.full((2, 2), 2, np.int32)
b = 3.0
c = a ** b
print(c)
print(c.dtype)

a = np.full((2, 2), 2, np.int32)
b = 3.5
print(a ** b)

a = 3
b = np.full((2, 2), 2, np.int32)
print(a ** b)


a = np.full((2, 2), 2.5, np.float32)
b = 2.0
c = a // b
print(c)

a = np.full((2, 2), 3, np.int32)
b = 2.0
c = a % b
print(c)

a = 1
b = np.ones((2, 2), np.float32)
print(a - b)

a = 2.5
b = np.full((2, 2), 2.0, np.float32)
c = a // b
print(c)

a = 2.5
b = np.full((2, 2), 2.0, np.float32)
c = a % b
print(c)