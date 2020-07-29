"""

OBJECT RECOGNITION USING A SPIKING NEURAL NETWORK.

* The proposed model module.

@author: atenagm1375

"""


import torch
from tqdm import tqdm

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Conv2dConnection, MaxPool2dConnection
from bindsnet.network.topology import Connection
from bindsnet.learning import WeightDependentPostPre, PostPre
from bindsnet.encoding import rank_order
from bindsnet.evaluation import assign_labels, all_activity
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes

from sklearn.metrics import confusion_matrix
from numpy import sqrt
from scipy.spatial.distance import euclidean

from utils.visualizations import heatmap
import matplotlib.pyplot as plt


def _lateral_inhibition_weights(n, inh_intencity=0.8, inh_bias=0.1):
    w = torch.ones(n, n) - torch.diag(torch.ones(n))
    n_sqrt = sqrt(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = i // n_sqrt, i % n_sqrt
                x2, y2 = j // n_sqrt, j % n_sqrt

                w[i, j] = sqrt(euclidean([x1, y1], [x2, y2]))
    w = w / w.max()
    w = w * inh_intencity + inh_bias
    print(w.shape)
    return w


class DeepCSNN(Network):
    """Model class."""

    def __init__(self, input_shape=(1, 100, 100), n_classes=4, dt=1):
        super().__init__(dt=dt)
        self.input_shape = input_shape
        self.n_classes = n_classes
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    def compile(self):
        """
        Build and compile the network.

        Returns
        -------
        DeepCSNN
            The self object.
        """
        ht, wdth = self.input_shape[1], self.input_shape[2]

        inp = Input(shape=self.input_shape, traces=True)
        self.add_layer(inp, "DoG")

        s1 = LIFNodes(shape=(18, ht, wdth), traces=True,
                      sum_input=True, thresh=-60)
        self.add_layer(s1, "conv_1")

        c1 = LIFNodes(shape=(18, ht // 2, wdth // 2), traces=True,
                      sum_input=True, thresh=-62)
        self.add_layer(c1, "pool_1")

        s2 = LIFNodes(shape=(24, ht // 2, wdth // 2), traces=True,
                      sum_input=True, thresh=-62)
        self.add_layer(s2, "conv_2")

        c2 = LIFNodes(shape=(24, ht // 4, wdth // 4), traces=True,
                      sum_input=True, thresh=-63)
        self.add_layer(c2, "pool_2")

        s3 = LIFNodes(shape=(4, ht // 4, wdth // 4), traces=True,
                      thresh=-63)
        self.add_layer(s3, "conv_3")

        f = LIFNodes(shape=(4, 1, 1), traces=True, thresh=-63)
        self.add_layer(f, "global_pool")

        conv1 = Conv2dConnection(
            source=inp,
            target=s1,
            kernel_size=5,
            padding=2,
            weight_decay=1.e-4,
            nu=[0.003, 0.004],
            update_rule=WeightDependentPostPre,
            wmin=0,
            wmax=1,
            decay=1,
            )
        self.add_connection(conv1, "DoG", "conv_1")

        pool1 = MaxPool2dConnection(
            source=s1,
            target=c1,
            kernel_size=2,
            stride=2,
            decay=1
            )
        self.add_connection(pool1, "conv_1", "pool_1")

        conv2 = Conv2dConnection(
            source=c1,
            target=s2,
            kernel_size=3,
            padding=1,
            weight_decay=1.e-4,
            nu=[0.003, 0.004],
            update_rule=WeightDependentPostPre,
            wmin=0,
            wmax=1,
            decay=1,
            )
        self.add_connection(conv2, "pool_1", "conv_2")

        pool2 = MaxPool2dConnection(
            source=s2,
            target=c2,
            kernel_size=2,
            stride=2,
            decay=1,
            )
        self.add_connection(pool2, "conv_2", "pool_2")

        conv3 = Conv2dConnection(
            source=c2,
            target=s3,
            kernel_size=3,
            padding=1,
            weight_decay=1.e-4,
            nu=[0.003, 0.004],
            update_rule=WeightDependentPostPre,
            wmin=0,
            wmax=1,
            decay=1,
            )
        self.add_connection(conv3, "pool_2", "conv_3")

        lateral_inh3 = Connection(
            source=s3,
            target=s3,
            w=_lateral_inhibition_weights(s3.n, 0.5, 0.01),
            update_rule=PostPre,
            nu=[5.e-5, 1.e-4],
            wmin=-1,
            wmax=1,
            decay=0.5,
            )
        self.add_connection(lateral_inh3, "conv_3", "conv_3")

        global_pool = MaxPool2dConnection(
            source=s3,
            target=f,
            kernel_size=(ht // 4, wdth // 4),
            decay=1,
            )
        self.add_connection(global_pool, "conv_3", "global_pool")

        recurrent_inh = Connection(
            source=f,
            target=f,
            w=0.0005 * (torch.eye(self.n_classes) - 1),
            decay=0.5,
            )
        self.add_connection(recurrent_inh, "global_pool", "global_pool")

        return self

    def _monitor_spikes(self):
        for name, layer in self.layers.items():
            monitor = Monitor(layer, ["s"], self.time)
            self.add_monitor(monitor, name)

    def evaluate(self, spikes, true_labels):
        assignments, _, _ = assign_labels(spikes, true_labels, self.n_classes)
        pred_labels = all_activity(spikes, assignments, self.n_classes)
        return pred_labels

    def fit(self, dataloader, time):
        print("start fit...")
        self.learning = True

        self.time = time
        n_samples = len(dataloader.dataset)

        self._monitor_spikes()

        output_spikes = torch.ones(n_samples, time, self.n_classes)
        true_labels = torch.ones(n_samples)
        for i, batch in enumerate(tqdm(dataloader)):
            print(f"iter {i}")
            true_labels[i] = batch[1].argmax()

            inputs = {"DoG": rank_order(batch[0], time).view(
                time, 1, *self.input_shape)}
            print("running network...")
            self.run(inputs, time)
            spikes = {}
            for name, layer in self.layers.items():
                spikes[name] = self.monitors[name].get("s")
            plt.ioff()
            plot_spikes(spikes)
            plt.show()

            output_spikes[i, :, :] = self.layers["global_pool"].s.view(
                1, time, self.n_classes)

        print("done running network")

        pred_labels = self.evaluate(output_spikes, true_labels)
        print("predicted")
        return self, pred_labels

    def predict(self, dataloader):
        self.learning = False

        n_samples = len(dataloader.dataset)

        output_spikes = torch.ones(n_samples, self.time, self.n_classes)
        true_labels = torch.ones(n_samples)
        for i, batch in enumerate(tqdm(dataloader)):
            true_labels[i] = batch[1]

            inputs = {"DoG": rank_order(batch[0], self.time).view(
                self.time, 1, *self.input_shape)}
            self.run(inputs, self.time)

            output_spikes[i, :, :] = self.layers["global_pool"].s.view(
                1, self.time, self.n_classes)

        pred_labels = self.evaluate(output_spikes, true_labels)
        return pred_labels

    # def __intensity_to_latency(self, x):
    #     x = torch.from_numpy(x.reshape((1, *x.shape)))
    #     x = x.type(torch.float)
    #     data = rank_order(x, self.encoding_time)
    #     return data

    # def _get_features(self, x):
    #     n_samples = x.shape[0]

    #     self.layers["conv_1"].train(False)
    #     self.layers["conv_2"].train(False)
    #     self.layers["conv_3"].train(False)
    #     features = []
    #     for i in range(n_samples):
    #         encoded_x = self.__intensity_to_latency(x[i, :, :])
    #         inputs = {"DoG": encoded_x}
    #         self.run(inputs=inputs, time=self.encoding_time)
    #         features.append(self.global_pool.x)
    #     return features

    # def fit(self, x, y, times, encoding_time=None):
    #     """
    #     Fit the model to training data.

    #     Parameters
    #     ----------
    #     x : numpy.array
    #         Train images.
    #     y : numpy.array
    #         True labels of train images.
    #     times : list of int
    #         A list containing time of running the network to train each layer.
    #     encoding_time : int, optional
    #         Duration in which intensities are coded. The default is None.

    #     Returns
    #     -------
    #     DeepCSNN
    #         The self object.

    #     """
    #     if encoding_time is not None:
    #         self.encoding_time = encoding_time

    #     n_samples = x.shape[0]
    #     features = []

    #     for _ in times:
    #         for t in range(3):
    #             print(f"layer {t}")
    #             truth = [False, False, False]
    #             truth[t] = True
    #             for i in range(n_samples):
    #                 print(f"sample {i}")
    #                 sample = self.__intensity_to_latency(x[i, :, :])
    #                 print(sample.shape)
    #                 self.layers["conv_1"].train(truth[0])
    #                 self.layers["conv_2"].train(truth[1])
    #                 self.layers["conv_3"].train(truth[2])
    #                 inputs = {"DoG": sample}
    #                 self.run(inputs=inputs, time=self.encoding_time)

    #     features = self._get_features(x)

    #     self.classifier.fit(features, y)

    #     return self

    # def predict(self, x):
    #     """
    #     Predict model output for given data.

    #     Parameters
    #     ----------
    #     x : numpy.array
    #         Test images.

    #     Returns
    #     -------
    #     y_pred : numpy.array
    #         Predicted values.

    #     """
    #     features = self._get_features(x)

    #     y_pred = self.classifier.predict(features)

    #     return y_pred

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
