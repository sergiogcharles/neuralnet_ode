import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
import math

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

sgd = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss(reduce='sum')

def P(x):
    return torch.zeros_like(x)

def Q(x):
    return torch.zeros_like(x)

def R(x):
    return x

def NN_proxy(x, y_0, dy_0):
    return y_0 + x * (model(x) + dy_0 - y_0)

#y'' = x y' = x^2/2 + c, y = x^3/6 + 1x + 1 | y'(0)=c=1, y(0)=d=1
# Linear inhomogenous ODE: y''(x) + P(x)y'(x) + Q(x)y(x) = R(x)
def training_step(y_0=0, dy_0=0):
    X_train = torch.range(-1, 10, 0.1, requires_grad=True).type(torch.float)
    X_train = X_train.reshape(len(X_train), -1)

    # Iterate through epochs
    num_epochs = 1000
    for epoch in range(num_epochs):
        # shuffle X_train
        idx_random = torch.randperm(len(X_train))
        X_train = X_train[idx_random]
        sgd.zero_grad()

        X_train = Variable(X_train, requires_grad=True)
        y_approx = NN_proxy(X_train, y_0, dy_0)
        # y_approx = y_0 + dy_0 * X_train + X_train ** 2 / 2
        # print(y_approx.shape)
        dy_approx = torch.autograd.grad(y_approx, X_train, grad_outputs=torch.ones_like(y_approx), allow_unused=True, retain_graph=True, create_graph=True)[0]

        print(dy_approx)

        ddy_approx = torch.autograd.grad(dy_approx, X_train, grad_outputs=torch.ones_like(y_approx), allow_unused=True, retain_graph=True, create_graph=True)[0]

        Pdy_approx = P(X_train) * dy_approx
        Qy_approx = Q(X_train) * y_approx

        output = ddy_approx + Pdy_approx + Qy_approx
        target = R(X_train)

        # MSE(y''(x) + P(x)y'(x) + Q(x)y(x), R(x))
        loss = criterion(output, target)
        loss.backward(retain_graph=True)

        if epoch % 100 == 0:
            print(X_train.grad)

        sgd.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')
        X_train_plot = torch.range(-1, 10, 0.1, requires_grad=True).type(torch.float)
        X_train_plot = X_train_plot.reshape(len(X_train_plot), -1)
        with torch.no_grad():
            plt.plot(X_train_plot.detach().numpy(), NN_proxy(X_train_plot.reshape(len(X_train_plot), -1), y_0, dy_0).detach().numpy())
            plt.plot(X_train_plot.detach().numpy(), (1 + X_train_plot + X_train_plot ** 3 / 6).detach().numpy())
            plt.show(block=False)
            plt.pause(0.6)
            plt.close()


    with torch.no_grad():
        X_test = torch.range(-1, 10, 0.1).type(torch.float)
        X_test = X_test.reshape(len(X_test), -1)
        output = NN_proxy(X_test, y_0, dy_0)
        
        plt.plot(X_test.detach().numpy(), (1 + X_test + X_test ** 3 / 6).detach().numpy())
        plt.plot(X_test.detach().numpy(), output.detach().numpy())
        plt.show()


if __name__ == "__main__":
    training_step(1, 1)