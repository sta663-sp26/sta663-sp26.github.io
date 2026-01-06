from vetiver import VetiverAPI, VetiverModel
from pins import board_folder

board = board_folder("board", versioned = True, allow_pickle_read = True)
v = VetiverModel.from_pin(board, "mnist_log_reg")

app = VetiverAPI(v, check_prototype=True)
app.run(port = 8080)
