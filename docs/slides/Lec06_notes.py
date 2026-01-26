## Regression example

### Setup

import numpy as np

rng = np.random.default_rng(1234)
n = 1000

## Create Data

X = np.hstack(
      [np.ones((n,1)), 
       rng.random((n,5))]
    )
beta = np.array([3.5, 1.8, -7.5, 0, 3, 9.8])
err = rng.normal(0, 0.1, size = n)

y = X @ beta + err

### Fit regression model - beta_hat =  (X^T X)^-1 X^Ty

np.linalg.inv(X.T @ X) @ X.T @ y

np.linalg.solve(X.T @ X, X.T @ y)


## Exercise 1

x = np.arange(16).reshape((4,4)); x

x[1:3, 1:3]

x[[1,2], [1,2]] # incorrect


## Exercise 2

pts = np.linspace(-1,3, 5000)
x, y = np.meshgrid(pts, pts)

f = (1-x)**2 + 100*(y-x**2)**2


np.min(f)
x[f == np.min(f)]
y[f == np.min(f)]

min_i = np.argmin(f, axis=None)
x.reshape(-1)[min_i]
y.reshape(-1)[min_i]
