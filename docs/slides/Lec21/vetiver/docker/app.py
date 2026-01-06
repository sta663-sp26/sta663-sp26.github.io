from vetiver import VetiverModel
from dotenv import load_dotenv, find_dotenv
import vetiver
import pins

load_dotenv(find_dotenv())

b = pins.board_folder('board', allow_pickle_read=True)
v = VetiverModel.from_pin(b, 'mnist_log_reg', version = '20250331T101211Z-02741')

vetiver_api = vetiver.VetiverAPI(v)
api = vetiver_api.app
