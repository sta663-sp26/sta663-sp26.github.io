import vetiver, pins


board = pins.board_local("board/")

# Load test data

X_test = board.pin_read("mnist_X_test")
y_test = board.pin_read("mnist_y_test")

# Test the API

endpoint = vetiver.server.vetiver_endpoint("http://127.0.0.1:8080/predict")
res = vetiver.server.predict(endpoint, pd.DataFrame(X_test[:10]))

print("y_hat = ",res.predict.values)
print("y     = ",y_test[:10])