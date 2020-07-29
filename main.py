"""

OBJECT RECOGNITION USING A SPIKING NEURAL NETWORK.

* The main code script to run the model.

@author: atenagm1375

"""

# %% IMPORT MODULES

import torch

from utils.data import CaltechDatasetLoader, CaltechDataset
from utils.model import DeepCSNN

# %% ENVIRONMENT CONSTANTS

PATH = "../101_ObjectCategories/"
CLASSES = ["Faces", "car_side", "Motorbikes", "watch"]
image_size = (100, 100)
DoG_params = {"size_low": 3, "size_high": 15}
test_ratio = 0.3
time = 100

# %% LOAD DATA

data = CaltechDatasetLoader(PATH, CLASSES, image_size)
data.split_train_test(test_ratio)
train_dataset = CaltechDataset(data, **DoG_params)
test_dataset = CaltechDataset(data, train=False, **DoG_params)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                          num_workers=4, pin_memory=False)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                         num_workers=4, pin_memory=False)

# %% RUN DEEPCSNN MODEL

model = DeepCSNN(input_shape=(1, *image_size), n_classes=len(CLASSES))
model.compile()

train_pred = model.fit(trainloader, time)
test_pred = model.predict(testloader)

# model.fit(x_train, y_train, [500, 1000, 1500])
# y_pred = model.predict(x_test)
# model.classification_report(y_test, y_pred)

# %%
