import numpy as np
import requests
import json
from pprint import pprint

rng = np.random.default_rng(seed=1234)

n = 100

X = rng.normal(size=(n,5))
b = np.array([5,0,0,3,-2]).reshape(-1,1)
y = (X @ b).squeeze() + rng.normal(scale=0.5, size=n)


## Fit the model
r = requests.post(
  'http://0.0.0.0:8000/fit',
  json = {
    "X": X.tolist(), "y": y.tolist()
  }
)
r.status_code
pretty_print(r.json())


## Model predictions
r = requests.post(
  'http://0.0.0.0:8000/predict',
  json = {
    "X": X.tolist()
  }
)
r.status_code
pretty_print(r.json())

## Plot endpoint
r = requests.get('http://0.0.0.0:8000/plot')
r.status_code
r.headers["content-type"]
from IPython.display import Image
Image(r.content)


## Other endpoints
pprint( requests.get('http://0.0.0.0:8000/coefs').json() )

pprint( requests.get('http://0.0.0.0:8000/reset').json() )

pprint( requests.get('http://0.0.0.0:8000/coefs').json() )
