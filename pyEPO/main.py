import pyepo

# model for shortest path
grid = (5,5) # grid size
model = pyepo.model.grb.shortestPathModel(grid)

# generate data
num_data = 1000 # number of data
num_feat = 5 # size of feature
deg = 4 # polynomial degree
noise_width = 0 # noise width
x, c = pyepo.data.shortestpath.genData(num_data, num_feat, grid, deg, noise_width, seed=135)

# sklearn regressor
from sklearn.linear_model import LinearRegression
reg = LinearRegression() # linear regression

# build model
twostage_model = pyepo.twostage.sklearnPred(reg)

# training
twostage_model.fit(x, c)

# prediction
c_pred = twostage_model.predict(x)
