import paddle
import cloudpickle
import os
import subprocess
import multiprocessing as mp

class PaddleObject(paddle.nn.Layer):
    def __init__(self):
        super(PaddleObject, self).__init__()
        self.linear_1 = paddle.nn.Linear(784, 512)

class OtherObject(object):
    def __init__(self):
        self.x = 0

# def test_cloudpickle(cls):
#     def to_str(byte):
#         """ convert byte to string in pytohn2/3
#         """
#         return str(byte.decode())

#     # print(cls.__metaclass__)
#     encoded = cloudpickle.dumps(cls)
#     decoded_cls = cloudpickle.loads(encoded) # can be deserialized in the same process
#     obj = decoded_cls()

#     fname = "{}.cloudpickle".format(cls.__name__)
#     with open(fname, 'wb') as f:
#         f.write(encoded)

#     command = """python -c '
# import cloudpickle
# with open("{}", "rb") as f:
#     encoded = f.read()
#     decoded_cls = cloudpickle.loads(encoded)
# '
# """.format(fname)
#     # cannot be deserialized in another process
#     try:
#         subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
#     except subprocess.CalledProcessError as e:
#         print(str(e.output.decode()))

def load_cls(encoded)

def test_cloudpickle(cls):
    def to_str(byte):
        """ convert byte to string in pytohn2/3
        """
        return str(byte.decode())

    # print(cls.__metaclass__)
    encoded = cloudpickle.dumps(cls)
    decoded_cls = cloudpickle.loads(encoded) # can be deserialized in the same process
    obj = decoded_cls()

    fname = "{}.cloudpickle".format(cls.__name__)
    with open(fname, 'wb') as f:
        f.write(encoded)

    command = """python -c '
import cloudpickle
with open("{}", "rb") as f:
    encoded = f.read()
    decoded_cls = cloudpickle.loads(encoded)
'
""".format(fname)
    # cannot be deserialized in another process
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(str(e.output.decode()))

# test_cloudpickle(OtherObject)

test_cloudpickle(PaddleObject)