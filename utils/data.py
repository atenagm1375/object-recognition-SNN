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
import cv2


class Data:
    """
    Data class.

    Attributes
    ----------
    path : str
        Path to image folders.
    classes: list of str
        List of classes.
    image_size: tuple, optional
        The input image size. All images are resized to the specified size.

    """

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
                img = Image.open(cls_path + img_path).convert("L").resize(
                    image_size)
                x.append(np.asarray(img).reshape((1, *image_size)))
                y.append(obj)

        self.n_samples = len(y)

        self.data_frame = pd.DataFrame({"x": x, "y": y}, columns=["x", "y"])

        enc = pd.get_dummies(self.data_frame["y"])
        self.data_frame = pd.concat([self.data_frame, enc], axis=1)

    def apply_DoG(self, sigma_low, sigma_high):
        """
        Apply DoG filter on input images.

        Parameters
        ----------
        sigma_low : int
            The sigma value for first GaussianBlur filter.
        sigma_high : int
            The sigma value for second GaussianBlur filter.

        Returns
        -------
        None.

        """
        s1, s2 = (sigma_low, sigma_low), (sigma_high, sigma_high)
        self.data_frame["x"] = self.data_frame.x.apply(
            lambda im: cv2.GaussianBlur(im.astype(np.float64), s1, 0) -
            cv2.GaussianBlur(im.astype(np.float64), s2, 0))

    def split_train_test(self, test_ratio=0.3):
        """
        Split train and test samples.

        Parameters
        ----------
        test_ratio : float, optional
            The ratio of test samples. The default is 0.3.

        Returns
        -------
        x_train : numpy.array
            Train image data.
        x_test : numpy.array
            Test image data.
        y_train : numpy.array
            Train class labels.
        y_test : numpy.array
            Test class labels.

        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.data_frame["x"], self.data_frame.iloc[:, -self.n_classes:],
            test_size=test_ratio, shuffle=True)
        self.train_idx = list(x_train.index)
        self.test_idx = list(x_test.index)
        x_train = np.stack(np.array(x_train))
        x_test = np.stack(np.array(x_test))
        y_train = np.stack(np.array(y_train))
        y_test = np.stack(np.array(y_test))
        return x_train, x_test, y_train, y_test
