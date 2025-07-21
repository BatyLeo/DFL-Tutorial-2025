import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pyepo
    from pyepo.model.opt import optModel
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    import copy
    return DataLoader, copy, nn, np, optModel, plt, pyepo, torch


@app.cell
def _(np):
    def build_polytope(N, shift=0.0):
        return [[np.cos((2 * k / N + shift) * np.pi), np.sin((2 * k / N + shift) * np.pi)] for k in range(N - 1)]
    return (build_polytope,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# CO algorithm""")
    return


@app.cell
def _(np, optModel):
    class myModel(optModel):
        def __init__(self, instance):
            self.vertices = instance
            super().__init__()

        def _getModel(self):
            return self.vertices, self.vertices

        def setObj(self, c):
            self.theta = c

        def solve(self):
            theta = self.theta
            index = np.argmin([np.dot(self.vertices[v], theta) for v in range(len(self.vertices))])
            sol = self.vertices[index].copy()
            obj = np.dot(self.vertices[index], theta)
            return sol, obj
    return (myModel,)


@app.cell
def _(build_polytope, myModel):
    polytope = build_polytope(6)
    optmodel = myModel(polytope) # ?? instance kwarg ?
    return (optmodel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""There seem to be not practical way to give custom instance information.""")
    return


@app.cell
def _(np, optmodel):
    theta = np.random.random(2)
    optmodel.setObj(theta) # set objective function
    sol, obj = optmodel.solve() # solve
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Dataset""")
    return


@app.cell
def _(DataLoader, np, optmodel, pyepo):
    N = 50
    x = [np.random.randn(5) for _ in range(N)]
    encoder = np.random.randn(2, 5)
    c = [encoder @ xi for xi in x]
    dataset = pyepo.data.dataset.optDataset(optmodel, x, c)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader, dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Statistical model""")
    return


@app.cell
def _(nn):
    class LinearRegression(nn.Module):
        def __init__(self):
            super(LinearRegression, self).__init__()
            self.linear = nn.Linear(5, 2, bias=False)

        def forward(self, x):
            out = self.linear(x)
            return out
    return (LinearRegression,)


@app.cell
def _(LinearRegression):
    initial_model = LinearRegression()
    print("Model weights:")
    print(f"Weights: {initial_model.linear.weight}")
    print(f"Bias: {initial_model.linear.bias}")
    return (initial_model,)


@app.cell
def _(np, optmodel):
    def compute_gap(dataset, model):
        gap = 0.0
        for data in dataset:
            x, c, w, z = data
            theta = model(x).detach().numpy()
            optmodel.setObj(theta)
            y, _ = optmodel.solve()
            gap += (np.dot(c, y) - z) / abs(z)
        return gap / len(dataset)
    return (compute_gap,)


@app.cell
def _(compute_gap, dataset, initial_model):
    compute_gap(dataset, initial_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Training""")
    return


@app.cell
def _(optmodel, pyepo):
    pfy = pyepo.func.perturbedFenchelYoung(optmodel, n_samples=200, sigma=0.1, processes=1)
    return (pfy,)


@app.cell
def _(copy, torch):
    def train_loop(initial_model, pfy, dataloader, num_epochs=100):
        loss_history = []
        model = copy.deepcopy(initial_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(num_epochs):
            total_loss = 0.0
            for data in dataloader:
                x, c, w, z = data
                cp = model(x)
                loss = pfy(cp, w)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_history.append(total_loss)
        return loss_history, model

    return (train_loop,)


@app.cell
def _(dataloader, initial_model, pfy, train_loop):
    loss_history, model = train_loop(initial_model, pfy, dataloader, num_epochs=100)
    return loss_history, model


@app.cell
def _(loss_history, plt):
    plt.plot(loss_history)
    plt.show()
    return


@app.cell
def _(loss_history):
    loss_history
    return


@app.cell
def _(compute_gap, dataset, model):
    compute_gap(dataset, model)
    return


if __name__ == "__main__":
    app.run()
