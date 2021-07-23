import numpy as np
import paddle

class TensorDataset(paddle.io.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        arr = np.random.rand(3, 224, 224)
        # tensor = paddle.to_tensor(arr)
        # paddle.set_device("cpu")
        tensor = paddle.to_tensor(arr, place=paddle.CPUPlace())
        # tensor = tensor / 2

        return tensor

    def __len__(self):
        return 10

tensor_data = TensorDataset()

loader = paddle.io.DataLoader(tensor_data, batch_size=1, num_workers=2)

for data in loader:
    print("read")
    # print(data)
    # print(data[0].shape)