from models.classic import Modular, Net
from torch import randint

a = randint(10, size=(5,152))
net = Net()
print(net(a))

net2 = Modular(
            embedding_dim = 64, 
            batch_norm_conv=True,
            filter_sizes = [3,4,5,6],
            num_filters = [192, 192, 192, 192],
            hidden_neurons=[128,64], 
            dropout=0.7, 
            batch_norm_fcnn=True
        )
print(net2)
print(net2(a))
