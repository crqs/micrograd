# pyright: reportMissingParameterType=false

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###  MicroGrad demo
    """)
    return


@app.cell
def _():
    import random

    import matplotlib.pyplot as plt
    import numpy as np

    # '%matplotlib inline' command supported automatically in marimo
    return np, plt, random


@app.cell
def _():
    from micrograd import Tensor
    from micrograd.nn import MLP

    return MLP, Tensor


@app.cell
def _(np, random):
    np.random.seed(1337)
    random.seed(1337)
    return


@app.cell
def _(plt):
    # make up a dataset

    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=100, noise=0.1)

    y = y * 2 - 1  # make y be -1 or 1
    # visualize in 2D
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="jet")
    return X, y


@app.cell
def _(MLP):
    # initialize a model
    model = MLP(2, [16, 16], 1)  # 2-layer neural network
    # print(model)
    print("number of parameters", len(model.parameters))
    return (model,)


@app.cell
def _(Tensor, X, model, np, y):

    def loss(batch_size=None):
        if batch_size is None:
            Xb, yb = (Tensor(X), Tensor(y))
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = (Tensor(X[ri]), Tensor(y[ri]))

        inputs = Xb
        scores = model(inputs)
        losses = (1 + -yb * scores).relu()

        # inputs = [list(map(Tensor, xrow)) for xrow in Xb]
        # scores = list(map(model, inputs))
        # losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores, strict=True)]

        data_loss = sum(losses) * (1.0 / len(losses))
        alpha = 0.0001  # forward the model to get scores
        reg_loss = alpha * sum(p * p for p in model.parameters)

        _total_loss = data_loss + reg_loss

        # svm "max-margin" loss
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores, strict=True)]

        return (_total_loss, sum(accuracy) / len(accuracy))

    _total_loss, _acc = loss()
    print(_total_loss, _acc)  # L2 regularization  # also get accuracy
    return (loss,)


@app.cell
def _(loss, model):
    for epoch in range(10):
        _total_loss, _acc = loss()
        model.zero_grad()  # forward
        _total_loss.backward()

        learning_rate = 1.0 - 0.9 * epoch / 100

        for p in model.parameters():  # backward
            p.data -= learning_rate * p.grad

        if epoch % 1 == 0:
            print(f"step {epoch} loss {_total_loss.data}, accuracy {_acc * 100}%")  # update (sgd)
    return


@app.cell
def _(Tensor, X, model, np, plt, y):
    # visualize decision boundary

    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Tensor, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs))
    Z = np.array([s.data > 0 for s in scores])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    return


if __name__ == "__main__":
    app.run()
