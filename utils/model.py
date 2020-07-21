"""

OBJECT RECOGNITION USING A SPIKING NEURAL NETWORK.

* The proposed model module.

@author: atenagm1375

"""


from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Conv2dConnection, MaxPool2dConnection
from bindsnet.learning import PostPre
from bindsnet.network.monitors import NetworkMonitor
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

from visualizations import heatmap


class DeepCSNN(Network):
    """Model class."""

    def __init__(self, input_shape=(1, 100, 100), n_classes=4):
        super(DeepCSNN, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes

        self.classifier = LinearSVC()
        self.encoding_time = 1

    def compile(self):
        """
        Build and compile the network.

        Returns
        -------
        DeepCSNN()
            The self object.

        """
        m, n = self.input_shape[1], self.input_shape[2]

        inp = Input(shape=self.shape)
        self.add_layer(inp, "DoG")

        s1 = LIFNodes(shape=(18, m, n))
        self.add_layer(s1, "conv_1")
        c1 = LIFNodes(shape=(18, m // 2, n // 2))
        self.add_layer(c1, "pool_1")

        s2 = LIFNodes(shape=(24, m // 2, n // 2))
        self.add_layer(s2, "conv_2")
        c2 = LIFNodes(shape=(24, m // 4, n // 4))
        self.add_layer(c2, "pool_2")

        s3 = LIFNodes(shape=(32, m // 4, n // 4))
        self.add_layer(s3, "conv_3")
        f = LIFNodes(shape=(32, 1))
        self.add_layer(f, "global_pool")

        conv1 = Conv2dConnection(inp, s1, 5, padding=2, weight_decay=0.01,
                                 nu=0.01, update_rule=PostPre)
        self.add_connection(conv1, "DoG", "conv_1")
        pool1 = MaxPool2dConnection(s1, c1, 2, 2)
        self.add_connection(pool1, "conv_1", "pool_1")

        conv2 = Conv2dConnection(c1, s2, 3, padding=1, weight_decay=0.01,
                                 nu=0.01, update_rule=PostPre)
        self.add_connection(conv2, "pool_1", "conv_2")
        pool2 = MaxPool2dConnection(s2, c2, 2, 2)
        self.add_connection(pool2, "conv_2", "pool_2")

        conv3 = Conv2dConnection(c2, s3, 3, padding=1, weight_decay=0.01,
                                 nu=0.01, update_rule=PostPre)
        self.add_connection(conv3, "pool_2", "conv_3")
        global_pool = MaxPool2dConnection(s3, f, (m // 4, n // 4))
        self.add_connection(global_pool, "conv_3", "global_pool")

        monitor = NetworkMonitor(self, layers=[s1, c1, s2, c2, s3, f],
                                 connections=[conv1, conv2, conv3],
                                 state_vars=["w", "s"])
        self.add_monitor(monitor, "network_monitor")

        return self

    def __intensity_to_latency(self, x):
        # TODO
        pass

    def __prepare_train_input(self, x, time):
        # TODO
        pass

    def fit(self, x, y, times, encoding_time=2):
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
            Duration in which intensities are coded. The default is 2.

        Returns
        -------
        DeepCSNN
            The self object.

        """
        self.encoding_time = encoding_time

        encoded_x = self.__intensity_to_latency(x)
        data = [self.__prepare_train_input(encoded_x, time) for time in times]

        self.conv1.train(True)
        self.conv2.train(False)
        self.conv3.train(False)
        self.run(inputs=data[0], time=times[0])

        self.conv1.train(False)
        self.conv2.train(True)
        self.conv3.train(False)
        self.run(inputs=data[1], time=times[1])

        self.conv1.train(False)
        self.conv2.train(False)
        self.conv3.train(True)
        self.run(inputs=data[2], time=times[2])

        self.conv1.train(False)
        self.conv2.train(False)
        self.conv3.train(False)
        self.run(inputs=encoded_x, time=encoding_time)

        self.classifier.fit(self.global_pool.x, y)

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
        encoded_x = self.__intensity_to_latency(x)

        self.conv1.train(False)
        self.conv2.train(False)
        self.conv3.train(False)
        self.run(inputs=encoded_x, time=self.encoding_time)

        y_pred = self.classifier.predict(self.global_pool.x)

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
