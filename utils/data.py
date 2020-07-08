"""

OBJECT RECOGNITION USING A SPIKING NEURAL NETWORK.

* The data preparation module.

@author: atenagm1375

"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


class Data:
    """Data class."""

    def __init__(self, path, classes, image_size=(100, 100)):
        self.classes = classes
        self.n_classes = len(classes)
        self.data_frame = pd.DataFrame()
        self.train_idx = []
        self.test_idx = []

        x = []
        y = []
        for obj in classes:
            cls_path = path + ("/" if path[-1] != "/" else "") + obj + "/"
            for img_path in os.listdir(cls_path):
                img = Image.open(cls_path).resize(image_size)
                x.append(np.asarray(img))
                y.append(obj)

        self.n_samples = len(self.y)

        self.data_frame = pd.DataFrame({"x": x, "y": y}, columns=["x", "y"])

        enc = pd.get_dummies(self.data_frame["y"])
        self.data_frame = pd.concat([self.data_frame, enc], axis=1)

    def split_train_test(self, test_ratio=0.3):
        """
        Split train and test samples.

        Parameters
        ----------
        test_ratio : float, optional
            The ratio of test samples. The default is 0.3.

        Returns
        -------
        x_train : pandas.Series
            Train image data.
        x_test : pandas.Series
            Test image data.
        y_train : pandas.DataFrame
            Train class labels.
        y_test : pandas.DataFrame
            Test class labels.

        """
        x_train, x_test, y_train, y_test = train_test_split(
                self.data_frame["x"], self.data_frame.iloc[:, -self.n_classes:],
                test_ratio=test_ratio, shuffle=True)
        self.train_idx = list(x_train.index)
        self.test_idx = list(x_test.index)
        return x_train, x_test, y_train, y_test
