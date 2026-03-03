import os

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from pins import board_folder
from vetiver import vetiver_pin_write, VetiverModel, prepare_docker
from vetiver.server import predict, vetiver_endpoint


digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, random_state=1234
)

m = LogisticRegression(
  penalty=None
).fit(
  X_train, y_train
)

# Create and pin model

v = VetiverModel(
  m, model_name = "mnist_log_reg", 
  prototype_data = X_train
)

board = board_folder("board", versioned = True, allow_pickle_read = True)
vetiver_pin_write(board, v)
board.pin_versions("mnist_log_reg")

# Pin Data

board.pin_write(X_train, "mnist_X_train", type = "joblib")
board.pin_write(y_train, "mnist_y_train", type = "joblib")
board.pin_write(X_test, "mnist_X_test", type = "joblib")
board.pin_write(y_test, "mnist_y_test", type = "joblib")


# Prepare Dockerfile

os.makedirs("docker/", exist_ok=True)
prepare_docker(board, "mnist_log_reg",  path="docker/")
