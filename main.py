"""

OBJECT RECOGNITION USING A SPIKING NEURAL NETWORK.

* The main code script to run the model.

@author: atenagm1375

"""

# %% IMPORT MODULES

from utils.data import Data

# %% ENVIRONMENT CONSTANTS

PATH = "../101_ObjectCategories/"
CLASSES = ["car_side", "Faces", "Motorbikes", "panda"]
image_size = (100, 100)
DoG_params = {"sigma1": 3, "sigma2": 7}
test_ratio = 0.3

# %% LOAD DATA

data = Data(PATH, CLASSES, image_size)
data.apply_DoG(*DoG_params.values())

x_train, x_test, y_train, y_test = data.split_train_test(test_ratio)

# %%
