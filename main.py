# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import pylab
# x_data=torch.Tensor([[1.0],[2.0],[3.0]]) #输入数据集
# y_data=torch.Tensor([[1.0],[2.0],[3.0]]) #输出数据集
#
# class LinearModel(torch.nn.Module):  #定义训练模型
#     def __init__(self):
#         super(LinearModel,self).__init__();
#         self.linear = torch.nn.Linear(1,1);
#
#     def forward(self,x):
#         y_pred=self.linear(x);
#         return y_pred;
#
# model = LinearModel()
#
# criterion=torch.nn.MSELoss(size_average=False);
# optimizer=torch.optim.SGD(model.parameters(),lr=0.01);
#
# for epoch in range(1000):
#     y_pred = model(x_data);
#     loss=criterion(y_pred, y_data)
#     # print(epoch, loss.item());
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print('w = ', model.linear.weight.item());
# print('b = ', model.linear.bias.item());
#
# x_test = torch.Tensor([[4.0]])
# y_test = model(x_test)
#
# print('y_pred=', y_test.data);
#
# x=np.linspace(0,10,200);
# x_t=torch.Tensor(x).view((200,1))
# y_t=model(x_t);
# y=y_t.data.numpy();
# plt.plot(x,y);
# plt.plot([-15,15],[0.5,1.5],c='r');
# plt.xlabel('x')
# plt.ylabel('y');
# plt.grid();
# plt.show();
# pylab.show();

import numpy as np
y=np.array([1,0,0])
z=np.array([0.2,0.1,-0.1])
y_pred=np.exp(z)/np.exp(z).sum()
loss=(-y*np.log(y_pred)).sum()
print(np.exp(z).sum())
print(loss);