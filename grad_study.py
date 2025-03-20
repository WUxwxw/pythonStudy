import torch


class LinearRegression:
    def __init__(self):
        self.w = torch.tensor([0.8], requires_grad=True)
        self.b = torch.tensor([0.2], requires_grad=True)

    def forward(self, x):
        y = self.w * x + self.b
        return y

    def parameters(self):
        return [self.w, self.b]

    def __call__(self, x):
        return self.forward(x)


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for para in self.parameters:
            para.data -= para.grad * self.lr

    def zero_grad(self):
        for para in self.parameters:
            para.grad.data.zero_()


def loss_func(y_true: torch.Tensor, y_pre: torch.Tensor):
    square = 1 / 2 * (y_true - y_pre) ** 2
    return square.mean()


def train(Epoch, lr):
    model = LinearRegression()
    opt = Optimizer(model.parameters(), lr)

    for epoch in range(Epoch):
        output = model(x_train)
        loss = loss_func(y_train, output)
        loss.backward()
        opt.step()
        opt.zero_grad()

        print('Epoch {}, loss is {:.4f}. 当前的权值: w = {:.2f}, b = {:.2f}'.format(epoch+1, loss.item(), model.w.item(), model.b.item()))


if __name__ == '__main__':
    # 用线性公式构造简单的训练样本
    x_train = torch.rand(100)
    y_train = x_train * 2 + 3

    # 设置学习率 α = 0.1
    alpha = 0.1

    # =====-----开始训练train-----=====
    train(Epoch=15, lr=alpha)

