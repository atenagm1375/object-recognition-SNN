"""

OBJECT RECOGNITION USING SPIKING NEURAL NETWORKS.

* The visualizations module.

@author: atenagm1375

"""


import matplotlib.pyplot as plt
import seaborn as sns


# TODO
def heatmap(mat, x_label=None, y_label=None, axes=None,
            title=None, save=False):
    """
    Plot heatmap of the given data matrix.

    Parameters
    ----------
    mat : TYPE
        DESCRIPTION.
    x_label : TYPE, optional
        DESCRIPTION. The default is None.
    y_label : TYPE, optional
        DESCRIPTION. The default is None.
    axes : TYPE, optional
        DESCRIPTION. The default is None.
    title : TYPE, optional
        DESCRIPTION. The default is None.
    save : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    sns.heatmap(mat)
    plt.show()
