import paddle.nn as nn
from multiprocessing import Process, set_start_method
from paddle.incubate import multiprocessing as ms

ms.init_reductions()

class Model(nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, obs):
        x = nn.functional.relu(self.fc1(obs))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class Model(nn.Layer):
#     def __init__(self):
#         super(Model, self).__init__()

#     def forward(self, x):
#         return x

def func(model):
    print(model.state_dict())

if __name__ == "__main__":
    set_start_method('spawn')
    model = Model()
    p = Process(target=func, args=(model,))
    p.start()
    p.join()