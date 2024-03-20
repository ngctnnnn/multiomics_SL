import torch

a = torch.ones(1, 10)
b = torch.ones(1, 6)

avg_pool = torch.nn.AvgPool1d(stride=1, 
                              kernel_size=5)

print(a.shape)
a_reshape = avg_pool(a)
print(a_reshape.shape)