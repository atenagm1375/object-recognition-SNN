"""

OBJECT RECOGNITION USING A SPIKING NEURAL NETWORK.

* The proposed model module.

@author: atenagm1375

"""


import torch
from cv2 import filter2D, getGaborKernel

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Conv2dConnection, MaxPool2dConnection
from bindsnet.learning import PostPre
from bindsnet.encoding import rank_order
from bindsnet.network.monitors import NetworkMonitor
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import numpy as np

from utils.visualizations import heatmap


class ShallowRCSNN(Network):
    """
    Shallow rSTDP-based Convolutional Spiking Neural Network.

    Attributes
    ----------
    input_shape : TYPE
        DESCRIPTION.
    n_classes : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    """

    def __init__(self, input_shape=(1, 100, 100), n_classes=4, dt=1.0,
                 n_orientations=4, s1_rf_size=(5, 5),
                 c1_rf_size=(3, 3), s2_rf_size=(17, 17)):
        super().__init__(dt=dt)
        self.input_shape = input_shape
        self.n_classes = n_classes

        self.n_orientations = n_orientations
        self.s1_rf_size = s1_rf_size
        self.c1_rf_size = c1_rf_size
        self.s2_rf_size = s2_rf_size

        self.encoding_time = int(np.prod(input_shape))

    def __s1(self, x):
        x = torch.from_numpy(x).type(torch.double)
        feature_maps = []
        for i in range(self.n_orientations):
            theta = np.pi * i / self.n_orientations + np.pi / 8
            sigma = 2 / np.sqrt(2 * np.pi)
            kernel = getGaborKernel(self.s1_rf_size, sigma, theta, 2.5, 1)
            kernel /= np.sqrt(sum(kernel * kernel))
            kernel = torch.from_numpy(kernel).reshape(1, 1, *kernel.shape)

            f_imgs = torch.nn.functional.conv2d(x, kernel)
            feature_maps.append(f_imgs)

        feature_maps = np.stack(feature_maps)

        return torch.from_numpy(feature_maps)

    def __c1(self, x):
        pass

    def __s2(self, x):
        pass

    def __c2(self, x):
        pass

    def fit(self, x, y, time, encoding_time=None):
        """
        Build and train the model.

        Parameters
        ----------
        x : numpy.array
            Training images.
        y : numpy.array
            Training labels.
        time : int
            Training time.
        encoding_time : int, optional
            Interval of data encoding. The default is None.

        Returns
        -------
        self : object
            An instance of the model.

        """
        if encoding_time is not None:
            self.encoding_time = encoding_time

        s1 = self.__s1(x)

        # inp = Input(self.input_shape, traces=True)
        # self.add_layer(inp, "C1")

        # out = LIFNodes(shape=(2, 10))
        # self.add_layer(out, "S2")

        return self


class DeepCSNN(Network):
    """Model class."""

    def __init__(self, input_shape=(1, 100, 100), n_classes=4, dt=1):
        super().__init__(dt=dt)
        self.input_shape = input_shape
        self.n_classes = n_classes

        self.classifier = LinearSVC()
        self.encoding_time = np.prod(input_shape) * dt

        # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    def compile(self):
        """
        Build and compile the network.

        Returns
        -------
        DeepCSNN
            The self object.
        """
        m, n = self.input_shape[1], self.input_shape[2]

        inp = Input(shape=self.input_shape, traces=True)
        self.add_layer(inp, "DoG")

        s1 = LIFNodes(shape=(18, m, n), traces=True, )
        self.add_layer(s1, "conv_1")
        c1 = LIFNodes(shape=(18, m // 2, n // 2), traces=True)
        self.add_layer(c1, "pool_1")

        s2 = LIFNodes(shape=(24, m // 2, n // 2), traces=True)
        self.add_layer(s2, "conv_2")
        c2 = LIFNodes(shape=(24, m // 4, n // 4), traces=True)
        self.add_layer(c2, "pool_2")

        s3 = LIFNodes(shape=(32, m // 4, n // 4), traces=True)
        self.add_layer(s3, "conv_3")
        f = LIFNodes(shape=(32, 1), traces=True)
        self.add_layer(f, "global_pool")

        conv1 = Conv2dConnection(inp, s1, 5, padding=2, weight_decay=0.01,
                                 nu=0.01, update_rule=PostPre, decay=0.5)
        self.add_connection(conv1, "DoG", "conv_1")
        pool1 = MaxPool2dConnection(s1, c1, 2, 2, decay=0.5)
        self.add_connection(pool1, "conv_1", "pool_1")

        conv2 = Conv2dConnection(c1, s2, 3, padding=1, weight_decay=0.01,
                                 nu=0.01, update_rule=PostPre, decay=0.5)
        self.add_connection(conv2, "pool_1", "conv_2")
        pool2 = MaxPool2dConnection(s2, c2, 2, 2, decay=0.5)
        self.add_connection(pool2, "conv_2", "pool_2")

        conv3 = Conv2dConnection(c2, s3, 3, padding=1, weight_decay=0.01,
                                 nu=0.01, update_rule=PostPre, decay=0.5)
        self.add_connection(conv3, "pool_2", "conv_3")
        global_pool = MaxPool2dConnection(s3, f, (m // 4, n // 4), decay=0.5)
        self.add_connection(global_pool, "conv_3", "global_pool")

        monitor = NetworkMonitor(self, layers=["DoG", "conv_1", "pool_1",
                                               "conv_2", "pool_2",
                                               "conv_3", "global_pool"],
                                 connections=[("DoG", "conv_1"),
                                              ("pool_1", "conv_2"),
                                              ("pool_2", "conv_3")],
                                 state_vars=["w", "s"])
        self.add_monitor(monitor, "network_monitor")

        return self

    def __intensity_to_latency(self, x):
        x = torch.from_numpy(x.reshape((1, *x.shape)))
        x = x.type(torch.float)
        data = rank_order(x, self.encoding_time)
        return data

    def _get_features(self, x):
        n_samples = x.shape[0]

        self.layers["conv_1"].train(False)
        self.layers["conv_2"].train(False)
        self.layers["conv_3"].train(False)
        features = []
        for i in range(n_samples):
            encoded_x = self.__intensity_to_latency(x[i, :, :])
            inputs = {"DoG": encoded_x}
            self.run(inputs=inputs, time=self.encoding_time)
            features.append(self.global_pool.x)
        return features

    def fit(self, x, y, times, encoding_time=None):
        """
        Fit the model to training data.

        Parameters
        ----------
        x : numpy.array
            Train images.
        y : numpy.array
            True labels of train images.
        times : list of int
            A list containing time of running the network to train each layer.
        encoding_time : int, optional
            Duration in which intensities are coded. The default is None.

        Returns
        -------
        DeepCSNN
            The self object.

        """
        if encoding_time is not None:
            self.encoding_time = encoding_time

        n_samples = x.shape[0]
        features = []

        for _ in times:
            for t in range(3):
                print(f"layer {t}")
                truth = [False, False, False]
                truth[t] = True
                for i in range(n_samples):
                    print(f"sample {i}")
                    sample = self.__intensity_to_latency(x[i, :, :])
                    print(sample.shape)
                    self.layers["conv_1"].train(truth[0])
                    self.layers["conv_2"].train(truth[1])
                    self.layers["conv_3"].train(truth[2])
                    inputs = {"DoG": sample}
                    self.run(inputs=inputs, time=self.encoding_time)

        features = self._get_features(x)

        self.classifier.fit(features, y)

        return self

    def predict(self, x):
        """
        Predict model output for given data.

        Parameters
        ----------
        x : numpy.array
            Test images.

        Returns
        -------
        y_pred : numpy.array
            Predicted values.

        """
        features = self._get_features(x)

        y_pred = self.classifier.predict(features)

        return y_pred

    def classification_report(self, y_true, y_pred):
        # TODO
        """
        Confusion matrix.

        Parameters
        ----------
        y_true : numpy.array
            True labels.
        y_pred : TYPE
            Predicted labels.

        Returns
        -------
        None.

        """
        mat = confusion_matrix(y_true, y_pred)
        heatmap(mat)
