"""

OBJECT RECOGNITION USING A SPIKING NEURAL NETWORK.

* The data preparation module.

@author: atenagm1375

"""

import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2


class CaltechDataset(Dataset):
    """
    CaltechDataset class.

    Attributes
    ----------
    caltech_dataset_loader : utils.data.CaltechDatasetLoader
        An instance of CaltechDatasetLoader.
    train : bool, optional
        Defines whether to load the train instances or the test. The default
        is True.

    Keyword Arguments
    -----------------
    size_low : int
        The size of first GaussianBlur filter.
    size_high : int
        The size of second GaussianBlur filter.

    """

    def __init__(self, caltech_dataset_loader, train=True, **kwargs):
        self._cdl = caltech_dataset_loader
        if kwargs:
            self._cdl.apply_DoG(kwargs.get("size_low", 0),
                                kwargs.get("size_high", 0))

        self.dataframe = self._cdl.data_frame.iloc[
            self._cdl.train_idx] if train else \
            self._cdl.data_frame.iloc[self._cdl.test_idx]

    def __len__(self):
        """
        Get number of instances in the dataset.

        Returns
        -------
        int
            number of instances in the dataset.

        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """
        Get value(s) at the described index.

        Returns the image matrix and one-hot encoded label of the instance(s)
        at location index.

        Parameters
        ----------
        index : int
            The index to return values of.

        Returns
        -------
        tuple of two numpy.arrays
            The tuple of image matrix and the label array.

        """
        return self.dataframe["x"].iloc[index].astype(np.float32), \
            self.dataframe[self._cdl.classes].iloc[index].values.astype(
                np.float32)


class CaltechDatasetLoader:
    """
    Loads the Caltech dataset.

    Attributes
    ----------
    path : str
        Path to Caltech image folders.
    classes: list of str
        List of classes.
    image_size: tuple, optional
        The input image size. All images are resized to the specified size.
        The default is (100, 100).

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
                img = cv2.imread(cls_path + img_path, 0)
                img = cv2.resize(img, image_size,
                                 interpolation=cv2.INTER_CUBIC)
                x.append(img.reshape((1, *image_size)))
                y.append(obj)

        self.n_samples = len(y)

        self.data_frame = pd.DataFrame({"x": x, "y": y}, columns=["x", "y"])

        enc = pd.get_dummies(self.data_frame["y"])
        self.data_frame = pd.concat([self.data_frame, enc], axis=1)

    def apply_DoG(self, size_low, size_high):
        """
        Apply DoG filter on input images.

        Parameters
        ----------
        size_low : int
            The size of first GaussianBlur filter.
        size_high : int
            The size of second GaussianBlur filter.

        Returns
        -------
        None.

        """
        try:
            s1, s2 = (size_low, size_low), (size_high, size_high)
            self.data_frame["x"] = self.data_frame.x.apply(
                lambda im: cv2.GaussianBlur(im, s1, 0) -
                cv2.GaussianBlur(im, s2, 0))
        except cv2.error:
            print("DoG failed to apply")
            pass

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
        train_df = pd.DataFrame(columns=["x", *self.classes])
        test_df = pd.DataFrame(columns=["x", *self.classes])

        for obj in self.classes:
            obj_df = self.data_frame[self.data_frame[obj] == 1]
            sub_df = obj_df.sample(frac=1 - test_ratio)
            train_df = train_df.append(sub_df)
            test_df = test_df.append(obj_df[~obj_df.isin(sub_df)].dropna())

        train_df = train_df.sample(frac=1)
        test_df = test_df.sample(frac=1)

        self.train_idx = list(train_df.index)
        self.test_idx = list(test_df.index)
        x_train = np.stack(np.array(train_df.x))
        x_test = np.stack(np.array(test_df.x))
        y_train = np.stack(np.array(train_df[self.classes]))
        y_test = np.stack(np.array(test_df[self.classes]))
        return x_train, x_test, y_train, y_test
