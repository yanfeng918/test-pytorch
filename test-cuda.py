import torch
flag = torch.cuda.is_available()
print(flag)



a= torch.rand([4,2])
print(a)
torch_mean = torch.mean(a)
torch_mean = torch.mean(a,dim=0)
mean1 = a.mean()
mean2 = a.mean(0)
print(a.mean(0).shape)
print(a.mean(0,keepdim=True).shape)