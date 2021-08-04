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
    return 4 * x

def Q(x):
    return x ** 3

#y' + 4xy = x^3

def NN_proxy(x, F_0):
    return F_0 + x * model(x)

# Linear inhomogenous ODE: y'(x) + P(x)y(x) = Q(x)
# Solution: y(x) = e^{-int p(x)dx}[int e^{int p(x)dx}q(x)dx + C] with integration factor of I(x)=e^{int p(x)dx} so
# y(x) = 1/I(x)*[int I(x)q(x)dx + C]
# y(0)=1/I(0)*[S(0) + C]=y_0 where S(x)=int I(x)q(x)dx
def training_step(F_0=1):
    X_train = torch.range(-100, 100, 0.1, requires_grad=True).type(torch.float)
    X_train = X_train.reshape(len(X_train), -1)

    # Iterate through epochs
    num_epochs = 1000
    for epoch in range(num_epochs):
        # shuffle X_train
        idx_random = torch.randperm(len(X_train))
        X_train = X_train[idx_random]
        sgd.zero_grad()

        y_approx = NN_proxy(X_train, F_0)

        q = Q(X_train)
        py = P(X_train) * y_approx
        # Q(x) - P(x)y(x)
        output = q - py

        dy_approx = torch.autograd.grad(y_approx, X_train, grad_outputs=torch.ones_like(y_approx), create_graph=True)[0]

        loss = criterion(dy_approx, output)
        loss.backward(retain_graph=True)
        sgd.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')
        X_train_plot = torch.range(-100, 100, 0.1, requires_grad=True).type(torch.float)
        X_train_plot = X_train_plot.reshape(len(X_train_plot), -1)
        with torch.no_grad():
            plt.plot(X_train_plot.detach().numpy(), NN_proxy(X_train_plot.reshape(len(X_train_plot), -1), F_0).detach().numpy())
            plt.plot(X_train_plot.detach().numpy(), (1/8 * torch.exp(-2*X_train_plot ** 2) + X_train_plot ** 2 / 4 - 1/8).detach().numpy())
            plt.show(block=False)
            plt.pause(0.6)
            plt.close()


    with torch.no_grad():
        X_test = torch.range(-100, 100, 0.1).type(torch.float)
        X_test = X_test.reshape(len(X_test), -1)
        output = NN_proxy(X_test, F_0)
        
        plt.plot(X_test.detach().numpy(), (1/8 * torch.exp(-2*X_test ** 2) + X_test ** 2 / 4 - 1/8).detach().numpy())
        plt.plot(X_test.detach().numpy(), output.detach().numpy())
        plt.show()


if __name__ == "__main__":
    training_step(0)