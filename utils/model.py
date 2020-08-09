"""

OBJECT RECOGNITION USING A SPIKING NEURAL NETWORK.

* The proposed model module.

@author: atenagm1375

"""


import torch
from tqdm import tqdm

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, IFNodes
from bindsnet.network.topology import Conv2dConnection, MaxPool2dConnection
from bindsnet.network.topology import Connection
from bindsnet.learning import WeightDependentPostPre, PostPre, MSTDPET
from bindsnet.learning.reward import AbstractReward, MovingAvgRPE
from bindsnet.encoding import rank_order
from bindsnet.evaluation import assign_labels, all_activity
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_conv2d_weights

from sklearn.metrics import confusion_matrix, accuracy_score
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
        # TODO tune parameters
        """
        Build and compile the network.

        Returns
        -------
        object
        An instance of the model.
        """
        ht, wdth = self.input_shape[1], self.input_shape[2]

        inp = Input(shape=self.input_shape, traces=True)
        self.add_layer(inp, "DoG")

        s1 = LIFNodes(shape=(18, ht, wdth), traces=True, tc_decay=60,
                      thresh=-52, trace_scale=0.25)
        self.add_layer(s1, "conv_1")

        c1 = IFNodes(shape=(18, ht // 2, wdth // 2), traces=True, tc_decay=60,
                     thresh=-58, trace_scale=0.2)
        self.add_layer(c1, "pool_1")

        s2 = LIFNodes(shape=(24, ht // 2, wdth // 2), traces=True, tc_decay=80,
                      thresh=-55, trace_scale=0.4)
        self.add_layer(s2, "conv_2")

        c2 = IFNodes(shape=(24, ht // 4, wdth // 4), traces=True,
                     tc_decay=80, thresh=-62, trace_scale=0.4)
        self.add_layer(c2, "pool_2")

        s3 = LIFNodes(shape=(self.n_classes, ht // 4, wdth // 4), traces=True,
                      thresh=-57, trace_scale=0.8, tc_decay=100)
        self.add_layer(s3, "conv_3")

        c3 = IFNodes(shape=(self.n_classes, 1, 1), traces=True, tc_decay=100,
                     thresh=-62, trace_scale=0.8)
        self.add_layer(c3, "global_pool")

        d = LIFNodes(n=self.n_classes, traces=True, tc_decay=80,
                     thresh=-60, trace_scale=0.8)
        self.add_layer(d, "decision")

        conv1 = Conv2dConnection(
            source=inp,
            target=s1,
            kernel_size=5,
            padding=2,
            weight_decay=0.0002,
            nu=[0.003, 0.008],
            update_rule=WeightDependentPostPre,
            wmin=0,
            wmax=1,
            decay=0.2,
            )
        self.add_connection(conv1, "DoG", "conv_1")

        pool1 = MaxPool2dConnection(
            source=s1,
            target=c1,
            kernel_size=2,
            stride=2,
            decay=0.2
            )
        self.add_connection(pool1, "conv_1", "pool_1")

        conv2 = Conv2dConnection(
            source=c1,
            target=s2,
            kernel_size=3,
            padding=1,
            weight_decay=0.0006,
            nu=[0.008, 0.01],
            update_rule=WeightDependentPostPre,
            wmin=0,
            wmax=1,
            decay=0.5,
            )
        self.add_connection(conv2, "pool_1", "conv_2")

        pool2 = MaxPool2dConnection(
            source=s2,
            target=c2,
            kernel_size=2,
            stride=2,
            decay=0.5,
            )
        self.add_connection(pool2, "conv_2", "pool_2")

        conv3 = Conv2dConnection(
            source=c2,
            target=s3,
            kernel_size=3,
            padding=1,
            weight_decay=0.002,
            nu=[0.02, 0.06],
            update_rule=WeightDependentPostPre,
            wmin=0,
            wmax=1,
            decay=0.6,
            )
        self.add_connection(conv3, "pool_2", "conv_3")

        lateral_inh3 = Connection(
            source=s3,
            target=s3,
            w=_lateral_inhibition_weights(s3.n, 0.1, -0.1),
            decay=0.6,
            )
        self.add_connection(lateral_inh3, "conv_3", "conv_3")

        global_pool = MaxPool2dConnection(
            source=s3,
            target=c3,
            kernel_size=(ht // 4, wdth // 4),
            decay=0.5,
            )
        self.add_connection(global_pool, "conv_3", "global_pool")

        full = Connection(
            source=c3,
            target=d,
            update_rule=PostPre,
            weight_decay=0.0001,
            nu=[0.006, 0.05],
            wmin=0,
            wmax=1,
            decay=0.5,
            )
        self.add_connection(full, "global_pool", "decision")

        recurrent_inh = Connection(
            source=d,
            target=d,
            w=0.4 * (torch.eye(self.n_classes) - 1),
            decay=0.5,
            )
        self.add_connection(recurrent_inh, "decision", "decision")

        return self

    def _monitor_spikes(self):
        for name, layer in self.layers.items():
            monitor = Monitor(layer, ["s"], self.time)
            self.add_monitor(monitor, name)

    def _monitor_weights(self):
        for name, conn in self.connections.items():
            if not isinstance(conn, MaxPool2dConnection):
                monitor = Monitor(conn, ["w"], self.time)
                self.add_monitor(monitor, name)

    def evaluate(self, spikes, true_labels):
        """
        Evaluate the network.

        Parameters
        ----------
        spikes : torch.tensor
            Tensor of spikes.
        true_labels : torch.tensor
            Tensor of true image labels.

        Returns
        -------
        pred_labels : torch.tensor
            Predicted labels using bindsnet.evaluation.all_activity.

        """
        assignments, _, _ = assign_labels(spikes, true_labels, self.n_classes)
        pred_labels = all_activity(spikes, assignments, self.n_classes)
        return pred_labels

    def fit(self, dataloader, time, n_epochs=1, debug=True):
        """
        Train the network.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The train DataLoader instance.
        time : int
            Time to encode each image and run the network on.
        debug : bool, optional
            Whether to show debug logs and plots or not. The default is True.

        Returns
        -------
        object
            An instance of the model.

        """
        self.learning = True

        self.time = time
        n_samples = len(dataloader.dataset)

        self._monitor_spikes()
        self._monitor_weights()

        true_labels = torch.ones(n_samples)

        for episode in range(n_epochs):
            for i, batch in enumerate(tqdm(dataloader)):
                true_labels[i] = batch[1].argmax()

                inputs = {"DoG": rank_order(batch[0], time).view(
                    time, 1, *self.input_shape),
                    "conv_1": torch.bernoulli(0.005 * torch.ones(
                        time, *self.layers["conv_1"].shape)).byte(),
                    "conv_2": torch.bernoulli(0.002 * torch.ones(
                        time, *self.layers["conv_2"].shape)).byte(),
                    "conv_3": torch.bernoulli(0.001 * torch.ones(
                        time, *self.layers["conv_3"].shape)).byte(),
                    }
                self.run(inputs, time, one_step=True)

                if debug:
                    spikes = {}
                    for name, layer in self.layers.items():
                        spikes[name] = self.monitors[name].get("s")
                    plt.ioff()
                    plot_spikes(spikes)
                    plt.show()
                    conv1 = self.monitors[("DoG", "conv_1")].get("w")[:, :, 0, :, :]
                    conv2 = self.monitors[("pool_1", "conv_2")].get("w")[:, :, 0, :, :]
                    conv3 = self.monitors[("pool_2", "conv_3")].get("w")[:, :, 0, :, :]
                    plot_conv2d_weights(conv1)
                    plt.show()
                    plot_conv2d_weights(conv2)
                    plt.show()
                    plot_conv2d_weights(conv3)
                    plt.show()

        return self

    def predict(self, dataloader):
        """
        Predicts the output label.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The DataLoader instance.

        Returns
        -------
        pred_labels : torch.tensor
            The predicted labels of the given data.

        """
        self.learning = False

        n_samples = len(dataloader.dataset)

        output_spikes = torch.ones(n_samples, self.time, self.n_classes)
        true_labels = torch.ones(n_samples)
        for i, batch in enumerate(tqdm(dataloader)):
            true_labels[i] = batch[1].argmax()

            inputs = {"DoG": rank_order(batch[0], self.time).view(
                self.time, 1, *self.input_shape)}
            self.run(inputs, self.time, one_step=True)

            output_spikes[i, :, :] = self.monitors["decision"].get("s").view(
                1, self.time, self.n_classes)

        pred_labels = self.evaluate(output_spikes, true_labels)
        return pred_labels

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
        conv1 = self.monitors[("DoG", "conv_1")].get("w")[:, :, 0, :, :]
        conv2 = self.monitors[("pool_1", "conv_2")].get("w")[:, :, 0, :, :]
        conv3 = self.monitors[("pool_2", "conv_3")].get("w")[:, :, 0, :, :]
        plot_conv2d_weights(conv1)
        plt.show()
        plot_conv2d_weights(conv2)
        plt.show()
        plot_conv2d_weights(conv3)
        plt.show()
        mat = confusion_matrix(y_true, y_pred)
        heatmap(mat)
        print(accuracy_score(y_true, y_pred))


class Reward(AbstractReward):
    """Reward class."""

    def __init__(self, n_classes):
        self.n_classes = n_classes

        self.reward_predict = torch.tensor(0.0)
        self.reward_predict_episode = torch.tensor(0.0)
        self.rewards_predict_episode = ([])

    def compute(self, **kwargs):
        """
        Compute the reward prediction error using EMA.

        Keyword Arguments
        -----------------
        reward : float
            Current reward.
        """
        reward = kwargs["reward"]

        return reward - self.reward_predict


class RCSNN(Network):
    """RSTDP-based network."""

    # TODO
    def __init__(self, input_shape=(1, 100, 100),
                 n_classes=4, dt=1):
        super().__init__(dt, reward_fn=MovingAvgRPE)
        self.input_shape = input_shape
        self.n_classes = n_classes

    def compile(self):
        """
        Build and compile the network.

        Returns
        -------
        object
            An instance of the model.
        """
        ht, wdth = self.input_shape[1], self.input_shape[2]

        inp = Input(shape=self.input_shape, traces=True)
        self.add_layer(inp, "DoG")

        s1 = LIFNodes(shape=(18, ht, wdth), traces=True, tc_decay=50,
                      thresh=-52, trace_scale=0.2)
        self.add_layer(s1, "conv_1")

        c1 = LIFNodes(shape=(18, ht // 2, wdth // 2), traces=True, tc_decay=50,
                      thresh=-62, trace_scale=0.2)
        self.add_layer(c1, "pool_1")

        s2 = LIFNodes(shape=(24, ht // 2, wdth // 2), traces=True, tc_decay=50,
                      thresh=-58, trace_scale=0.4)
        self.add_layer(s2, "conv_2")

        c2 = LIFNodes(shape=(24, ht // 4, wdth // 4), traces=True,
                      tc_decay=50, thresh=-62, trace_scale=0.4)
        self.add_layer(c2, "pool_2")

        s3 = LIFNodes(shape=(self.n_classes, ht // 4, wdth // 4), traces=True,
                      thresh=-60, trace_scale=0.8, tc_decay=100)
        self.add_layer(s3, "conv_3")

        c3 = LIFNodes(shape=(self.n_classes, 1, 1), traces=True, tc_decay=100,
                      thresh=-63, trace_scale=0.8)
        self.add_layer(c3, "global_pool")

        d = LIFNodes(n=self.n_classes, traces=True, tc_decay=100,
                     thresh=-61, trace_scale=0.8)
        self.add_layer(d, "decision")

        conv1 = Conv2dConnection(
            source=inp,
            target=s1,
            kernel_size=5,
            padding=2,
            weight_decay=2.e-4,
            nu=[0.0006, 0.008],
            update_rule=MSTDPET,
            wmin=0,
            wmax=1,
            decay=0.2,
            tc_plus=100,
            tc_minus=125,
            tc_e_trace=1000,
            )
        self.add_connection(conv1, "DoG", "conv_1")

        pool1 = MaxPool2dConnection(
            source=s1,
            target=c1,
            kernel_size=2,
            stride=2,
            decay=0.2
            )
        self.add_connection(pool1, "conv_1", "pool_1")

        conv2 = Conv2dConnection(
            source=c1,
            target=s2,
            kernel_size=3,
            padding=1,
            weight_decay=1.e-4,
            nu=[0.0004, 0.008],
            update_rule=MSTDPET,
            wmin=0,
            wmax=1,
            decay=0.5,
            tc_plus=100,
            tc_minus=125,
            tc_e_trace=1000,
            )
        self.add_connection(conv2, "pool_1", "conv_2")

        pool2 = MaxPool2dConnection(
            source=s2,
            target=c2,
            kernel_size=2,
            stride=2,
            decay=0.5,
            )
        self.add_connection(pool2, "conv_2", "pool_2")

        conv3 = Conv2dConnection(
            source=c2,
            target=s3,
            kernel_size=3,
            padding=1,
            weight_decay=0.0,
            nu=[0.004, 0.006],
            update_rule=MSTDPET,
            wmin=0,
            wmax=1,
            decay=0.5,
            tc_plus=100,
            tc_minus=125,
            tc_e_trace=1000,
            )
        self.add_connection(conv3, "pool_2", "conv_3")

        lateral_inh3 = Connection(
            source=s3,
            target=s3,
            w=_lateral_inhibition_weights(s3.n, 0.1, 0.01),
            decay=0.5,
            )
        self.add_connection(lateral_inh3, "conv_3", "conv_3")

        global_pool = MaxPool2dConnection(
            source=s3,
            target=c3,
            kernel_size=(ht // 4, wdth // 4),
            decay=0.5,
            )
        self.add_connection(global_pool, "conv_3", "global_pool")

        full = Connection(
            source=c3,
            target=d,
            update_rule=MSTDPET,
            weight_decay=0,
            nu=[0.001, 0.005],
            wmin=0,
            wmax=1,
            decay=0.5,
            tc_plus=100,
            tc_minus=125,
            tc_e_trace=1000,
            )
        self.add_connection(full, "global_pool", "decision")

        return self

    def _monitor_spikes(self):
        for name, layer in self.layers.items():
            monitor = Monitor(layer, ["s"], self.time)
            self.add_monitor(monitor, name)

    def _monitor_weights(self):
        for name, conn in self.connections.items():
            if not isinstance(conn, MaxPool2dConnection):
                monitor = Monitor(conn, ["w"], self.time)
                self.add_monitor(monitor, name)

    def evaluate(self, spikes, true_labels):
        """
        Evaluate the network.

        Parameters
        ----------
        spikes : torch.tensor
            Tensor of spikes.
        true_labels : torch.tensor
            Tensor of true image labels.

        Returns
        -------
        pred_labels : torch.tensor
            Predicted labels using bindsnet.evaluation.all_activity.

        """
        assignments, _, _ = assign_labels(spikes, true_labels, self.n_classes)
        pred_labels = all_activity(spikes, assignments, self.n_classes)
        return pred_labels

    def fit(self, dataloader, time, n_epochs=1, debug=True):
        """
        Train the network.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The train DataLoader instance.
        time : int
            Time to encode each image and run the network on.
        debug : bool, optional
            Whether to show debug logs and plots or not. The default is True.

        Returns
        -------
        object
            An instance of the model.

        """
        self.learning = True

        self.time = time
        n_samples = len(dataloader.dataset)

        self._monitor_spikes()
        self._monitor_weights()

        true_labels = torch.ones(n_samples)

        for episode in range(n_epochs):
            for i, batch in enumerate(tqdm(dataloader)):
                true_labels[i] = batch[1].argmax()

                inputs = {"DoG": rank_order(batch[0], time).view(
                    time, 1, *self.input_shape)}
                self.run(inputs, time, reward=2)

                if debug:
                    spikes = {}
                    for name, layer in self.layers.items():
                        spikes[name] = self.monitors[name].get("s")
                    plt.ioff()
                    plot_spikes(spikes)
                    plt.show()

        return self

    def predict(self, dataloader):
        """
        Predicts the output label.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The DataLoader instance.

        Returns
        -------
        pred_labels : torch.tensor
            The predicted labels of the given data.

        """
        self.learning = False

        n_samples = len(dataloader.dataset)

        output_spikes = torch.ones(n_samples, self.time, self.n_classes)
        true_labels = torch.ones(n_samples)
        for i, batch in enumerate(tqdm(dataloader)):
            true_labels[i] = batch[1].argmax()

            inputs = {"DoG": rank_order(batch[0], self.time).view(
                self.time, 1, *self.input_shape)}
            self.run(inputs, self.time)

            output_spikes[i, :, :] = self.monitors["decision"].get("s").view(
                1, self.time, self.n_classes)

        pred_labels = self.evaluate(output_spikes, true_labels)
        return pred_labels

    def classification_report(self, y_true, y_pred):
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
        conv1 = self.monitors[("DoG", "conv_1")].get("w")[:, :, 0, :, :]
        conv2 = self.monitors[("pool_1", "conv_2")].get("w")[:, :, 0, :, :]
        conv3 = self.monitors[("pool_2", "conv_3")].get("w")[:, :, 0, :, :]
        plot_conv2d_weights(conv1)
        plt.show()
        plot_conv2d_weights(conv2)
        plt.show()
        plot_conv2d_weights(conv3)
        plt.show()
        mat = confusion_matrix(y_true, y_pred)
        heatmap(mat)
        print(accuracy_score(y_true, y_pred))
