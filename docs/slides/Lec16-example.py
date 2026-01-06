import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import optax


def optax_optimize(params, X, y, loss_fn, predict, score, optimizer, steps=50, batch_size=1, seed=1234):
  n, k = X.shape
  res = {"loss": [], "score": [], "epoch": list(range(steps+1))}

  opt_state = optimizer.init(params)
  grad_fn = jax.grad(loss_fn)

  rng = np.random.default_rng(seed)
  batches = np.array(range(n))
  rng.shuffle(batches)

  for iter in range(steps):
    res["loss"].append(loss_fn(params, X, y).item())
    res["score"].append(score(predict(params, X), y).item())
    
    for batch in batches.reshape(-1, batch_size):
      grad = grad_fn(params, X[batch,:], y[batch])
      updates, opt_state = optimizer.update(grad, opt_state)
      params = optax.apply_updates(params, updates)
      
  res["loss"].append(loss_fn(params, X, y).item())
  res["score"].append(score(predict(params, X), y).item())
  res["params"] = params

  return(res)

## Load data

from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target

# Add intercept terms
X = (np.c_[[1]*X.shape[0], X])

X = X[:1700,:]
y = y[:1700]

n, k = X.shape
l = len(np.unique(y))

# Model matrix sizes
f"{n=},{k=},{l=}"

rng = np.random.default_rng(seed=1234)
beta = rng.normal(size=(k,l))


## Model

def predict(beta, X):
  return jnp.argmax(jax.nn.log_softmax(X @ beta), axis=1)

def accuracy(pred, y):
  return jnp.sum(pred == y) / len(y)

def loss_fn(beta, X, y):
  y_one_hot = jax.nn.one_hot(y, 10)
  preds = jax.nn.log_softmax(X @ beta)
  
  # Cross entropy loss function
  return -jnp.mean(jnp.sum(y_one_hot * preds, axis=1))

loss_fn(beta, X, y)
predict(beta, X)


## Fitting

res = optax_optimize(
  beta, X, y, loss_fn, predict, accuracy,
  optax.sgd(learning_rate=0.01), 
  steps=100, batch_size=100, seed=1234
)

## Results

plt.figure()
plt.plot(res["epoch"], res["loss"], "-k")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

plt.figure()
plt.plot(res["epoch"], res["score"], "-k")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()


