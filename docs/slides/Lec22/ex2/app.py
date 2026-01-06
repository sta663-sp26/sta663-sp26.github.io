from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from pydantic import BaseModel

import numpy as np
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

class Data(BaseModel):
  X: list[list[float]]
  y: list[float] | None = None

app = FastAPI()

def get_coef():
  return {
    "intercept":  lm.intercept_.tolist() if hasattr(lm, "intercept_") else None,
    "coef":       lm.coef_.tolist()      if hasattr(lm, "coef_")      else None,
  }

# Redirect root requests to /docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url='/docs')

# Fit model based on the supplied data
@app.post("/fit")
def fit(data: Data):

    lm.fit(np.array(data.X), np.array(data.y))
    
    return get_coef()

# Predict from the fitted model
@app.post("/predict")
def predict(data: Data):
    return {
      "y_hat": lm.predict(np.array(data.X)).tolist()
    }

@app.get("/coefs")
async def coefs():
    return get_coef()

@app.get("/reset")
async def reset():
    global lm
    lm = LinearRegression()
    return {}

