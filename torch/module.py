import torch

class MyLayer1(torch.nn.Module):
    def __init__(self):
        super(MyLayer1, self).__init__()
        self._linear = torch.nn.Linear(1, 1)
 
    def forward(self, input):
        return self._linear(input)
 
class MyLayer2(torch.nn.Module):
    def __init__(self):
        super(MyLayer2, self).__init__()
        self._linear = torch.nn.Linear(1, 1)
 
    def forward(self, input):
        x =  self._linear(input)
        return self.layer1(x)
 
    def set_layer1(self, layer):
        self.layer1 = layer
 
mylayer1 = MyLayer1()
mylayer2 = MyLayer2()
 
for name, param in mylayer2.named_parameters():
    print('mylayer2: ', name)
 
mylayer2.set_layer1(mylayer1)
for name, param in mylayer2.named_parameters():
    print('mylayer2 after set: ', name)