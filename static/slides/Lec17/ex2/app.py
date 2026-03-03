from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse

from pydantic import BaseModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LinearRegression
import io

lm = LinearRegression()
X_train = None
y_train = None

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
    global X_train, y_train
    X_train = np.array(data.X)
    y_train = np.array(data.y)
    lm.fit(X_train, y_train)

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

@app.get("/plot")
async def plot():
    if X_train is None:
        raise HTTPException(status_code=400, detail="Model has not been fitted yet")

    y_hat = lm.predict(X_train)
    resid  = y_train - y_hat

    fig, ax = plt.subplots()
    ax.scatter(y_hat, resid, alpha=0.6)
    ax.axhline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return StreamingResponse(buf, media_type="image/png")

@app.get("/reset")
async def reset():
    global lm, X_train, y_train
    lm = LinearRegression()
    X_train = None
    y_train = None
    return {}
