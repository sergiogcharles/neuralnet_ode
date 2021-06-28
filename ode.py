import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
import math

# X_train = torch.range(0, 11, 0.5, requires_grad=True).type(torch.float)
# X_train = X_train.reshape(len(X_train), -1)
# print(X_train.shape)
# y_train = 10 * torch.sin(X_train).type(torch.float) + 1
omega = 2 * math.pi
alpha = 1.0
# y_train = torch.cos(X_train)
# y_train = y_train.type(torch.float)

batch_size = 111
# Define model
model = nn.Sequential(
    nn.Linear(1, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)   

sgd = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.MSELoss(reduce='sum')

def F(x):
    return torch.exp(-x ** 2)

def F_proxy(x, F_0):
    # print(model(x).shape)
    return F_0 + x * model(x)

# def loss():
#     # for x in X_train:
#     return mse(F_proxy(X_train, 1), torch.autograd.grad(F(X_train), X_train))
#     # delta = torch.autograd.grad(F(X_train), X_train) - F_proxy(X_train, 1)
#     # loss = mse()

# def solve_ode(X):
#     X = Variable(X, requires_grad=True)
#     # output = X ** 2
#     output = model(X)
#     # loss = output.autograd.grad().autograd.grad() - 2
#     # print(X.shape)
#     # output = Variable(output, requires_grad=True)

#     loss = torch.autograd.grad(output, X, grad_outputs=torch.ones_like(output))

#     loss = Variable(loss[0].mean() - 2, requires_grad=True)
#     print(loss)
#     return loss

def training_step(F_0=1):
    X_train = torch.range(0, 11, 0.5, requires_grad=True).type(torch.float)
    X_train = X_train.reshape(len(X_train), -1)

    # Iterate through epochs
    num_epochs = 1000
    for epoch in range(num_epochs):
        # shuffle X_train
        idx_random = torch.randperm(len(X_train))
        X_train = X_train[idx_random]
        # X_train_batch = X_train
        # y_train_batch = y_train
        # for b in range(0, len(X_train), batch_size):
        #     input = X_train_batch[b: b + batch_size].reshape(batch_size, -1)
        #     correct = y_train_batch[b: b + batch_size].reshape(batch_size, -1)

        # output = model(input)
            
        #     # loss = mse(output, correct)
        #     # loss = solve_ode(input)
        # print(F_proxy(X_train, 1).shape)
        # for x in X_train:
        #     x = Variable(x, requires_grad=True)
            # print(model[0].weight.data)
        sgd.zero_grad()

        output = F(X_train)
        # dF = torch.ones_like(output)
        F_approx = F_proxy(X_train, F_0)
        dF = torch.autograd.grad(F_approx, X_train, grad_outputs=torch.ones_like(F_approx), create_graph=True)[0]
        # dF = dF.reshape(len(dF), -1)
    
        # loss = torch.sum(torch.abs(F_approx - dF), axis=0)

        loss = criterion(dF, output)
        loss.backward(retain_graph=True)
        # print(X_train.grad)
        sgd.step()

        # if epoch % 10 == 0:
            # print(f'F_approx {F_approx}')
            # print(f'dF {dF}')
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
        X_train = torch.range(0, 11, 0.5, requires_grad=True).type(torch.float)
        X_train = X_train.reshape(len(X_train), -1)
        with torch.no_grad():
            plt.plot(X_train.detach().numpy(), F_proxy(X_train.reshape(len(X_train), -1), F_0).detach().numpy())
            plt.plot(X_train.detach().numpy(), (2 - torch.exp(-X_train)).detach().numpy())
            plt.show(block=False)
            plt.pause(0.6)
            plt.close()


    with torch.no_grad():
        X_test = torch.range(0, 11, 0.5).type(torch.float)
        X_test = X_test.reshape(len(X_test), -1)
        # y_test = F(X_test)
        output = F_proxy(X_test, F_0)
        
        # loss = criterion(output)
        # print(f'Loss is {loss}')

        plt.plot(X_test.detach().numpy(), (2 - torch.exp(-X_test)).detach().numpy())
        plt.plot(X_test.detach().numpy(), output.detach().numpy())
        plt.show()


if __name__ == "__main__":
    training_step(1)