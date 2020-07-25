"""

OBJECT RECOGNITION USING A SPIKING NEURAL NETWORK.

* The main code script to run the model.

@author: atenagm1375

"""

# %% IMPORT MODULES

from utils.data import Data
from utils.model import DeepCSNN

# %% ENVIRONMENT CONSTANTS

PATH = "../101_ObjectCategories/"
CLASSES = ["Faces", "car_side", "Motorbikes", "watch"]
image_size = (100, 100)
DoG_params = {"sigma1": 3, "sigma2": 7}
test_ratio = 0.3

# %% LOAD DATA

data = Data(PATH, CLASSES, image_size)
data.apply_DoG(*DoG_params.values())

x_train, x_test, y_train, y_test = data.split_train_test(test_ratio)

# %% RUN DEEPCSNN MODEL

model = DeepCSNN(input_shape=(1, *image_size), n_classes=len(CLASSES))
model.compile()
model.fit(x_train, y_train, [500, 1000, 1500])
y_pred = model.predict(x_test)
model.classification_report(y_test, y_pred)

# %%
