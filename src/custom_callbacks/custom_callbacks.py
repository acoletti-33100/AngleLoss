# General information, notes, disclaimers
#
# author: A. Coletti
# 
#
#
#
#
#
#
#
# ==============================================================================
from typing import Dict
from typing import Union
from typing import Any
from typing import List

from os import sep
import numpy as np
import itertools
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils.utils_plot import pyplot_to_tf_image
from utils.hyper_sphere_plot import plot_angles
from utils.hyper_sphere_plot import scatter_plot
from utils.hyper_sphere_plot import sphere_4subplot
from utils.hyper_sphere_plot import sphere_4subplot_pca
from utils.hyper_sphere_plot import plot_tsne_2d


class RecorderAdaptiveSphereRadius(Callback):
    """
    Plots at the end of training all the values of the adaptive sphere radius
    on the y-axis, while the x-axis is the number of iterations.

    Attributes
    ---------

    - file_writer: 
    - sphere_radius_record: list, stores the mini-batch training values of the adaptive sphere radius.
    """
    def __init__(self, log_path: str, folder_name: str = 'sphere-radius-plot') -> None:
        super(RecorderAdaptiveSphereRadius, self).__init__()
        self.file_writer = tf.summary.create_file_writer(log_path + sep + folder_name)
        self._sphere_radius_record = [] 
    

    @property
    def sphere_radius_record(self):
        return self._sphere_radius_record
    

    @sphere_radius_record.setter
    def sphere_radius_record(self, sphere_radius_value):
        """
        Appends the current sphere radius to this class internal state to create a history of values
        to plot at the end of training.

        Parameters
        ---------

        - sphere_radius_value: float, single sphere radius value computed during a mini-batch training
            iteration.
        """
        self._sphere_radius_record.append(sphere_radius_value)


    # TODO
    # can't I somehow make it record the sphere radius value here?
    def on_train_batch_end(self, batch, logs=None):
        """
        Appends the current sphere radius to this class internal state to create a history of values
        to plot at the end of training.
        """
        #self.sphere_radius_record.append()
        pass


    def on_train_end(self, logs = None):
        x_ticks = np.arange(len(self.sphere_radius_record))
        fig, ax = plt.subplots(1)
        ax.plot(x_ticks, self.sphere_radius_record) 
        plt.xlabel('Num. of iterations')
        plt.ylabel('Sphere radius')
        plt.xticks([])
        plt.grid(True, linestyle='dotted', which='major', axis='y', color='grey', alpha=0.4)
        _image = pyplot_to_tf_image(fig)
        with self.file_writer.as_default():
            tf.summary.image('Changing process of angles - Training', _image, step=0)


class AngleAvgMedianCallback(Callback):
    """
    This Callback is responsible for showing a graph with the averages and medians of the angles during the training
    epochs.
    The x-axis shows the epochs, while the y-axis shows the angles (in degrees).


    Attributes
    ---------

    - log_path: str, path where to save the callback results.
    - epochs: int, total number of training epochs.
    - avg_theta_gt_classes: list of floats, each is the average of the angles of the corresponding (ground truth) classes
        during a specific ordered training epoch.
    - median_theta_gt_classes: list of floats, each is median of the angles of the corresponding (ground truth) classes
        during a specific ordered training epoch.
    - avg_theta_not_gt_classes: list of floats, each is average of the angles of the not-corresponding classes
        during a specific ordered training epoch.
    """
    def __init__(self, log_path: str):
        """
        Parameters
        ---------

        - log_path: str, path where to save the callback results.
        """
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_path + sep + 'theta-plot')
        self._hist_avg_theta_gt_classes = [] 
        self._hist_median_theta_gt_classes = []
        self._hist_avg_theta_not_gt_classes = [] 
    

    @property
    def hist_avg_theta_gt_classes(self) -> List[float]:
        if len(self._hist_avg_theta_gt_classes) == 0:
            raise ValueError('"self.hist_avg_theta_gt_classes" not initialized yet')
        return self._hist_avg_theta_gt_classes

    
    @hist_avg_theta_gt_classes.setter
    def hist_avg_theta_gt_classes(self, hist_avg_theta_gt_classes: List[float]):
        """
        Sets the history of the averages of the angles of the ground truth classes.
        Call this method only once at the end of training.

        Parameters
        ---------

        - hist_avg_theta_gt_classes: list of floats, recorded values of each training epoch of the averages of the angles of the 
            corresponding (ground truth) classes.
        """
        self._hist_avg_theta_gt_classes = hist_avg_theta_gt_classes


    @property
    def hist_median_theta_gt_classes(self) -> List[float]:
        if len(self._hist_median_theta_gt_classes) == 0:
            raise ValueError('self.hist_median_theta_gt_classes not initialized yet')
        return self._hist_median_theta_gt_classes
    

    @hist_median_theta_gt_classes.setter
    def hist_median_theta_gt_classes(self, hist_median_theta_gt_classes: List[float]):
        """
        Sets the history of the medians of the angles of the ground truth classes.
        Call this method only once at the end of training.

        Parameters
        ---------

        - hist_median_theta_gt_classes: list of floats, recorded values of each training epoch of the medians of the angles of the 
            corresponding (ground truth) classes.
        """
        self._hist_median_theta_gt_classes = hist_median_theta_gt_classes

    @property
    def hist_avg_theta_not_gt_classes(self):
        if len(self._hist_avg_theta_not_gt_classes) == 0:
            raise ValueError('self.hist_avg_theta_not_gt_classes not initialized yet')
        return self._hist_avg_theta_not_gt_classes 
    

    @hist_avg_theta_not_gt_classes.setter
    def hist_avg_theta_not_gt_classes(self, hist_avg_theta_not_gt_classes: List[float]):
        """
        Sets the history of the averages of the angles of the not-corresponding classes.
        Call this method only once at the end of training.

        Parameters
        ---------

        - hist_avg_theta_not_gt_classes: list of floats, recorded values of each training epoch of the averages of the angles of the 
            not-corresponding classes.
        """
        self._hist_avg_theta_not_gt_classes = hist_avg_theta_not_gt_classes


    def on_train_end(self, logs = None):
        """
        At the end of training plots:
        - the average angles in each mini-batch for the non-corresponding 
        classes;
        - median angles in each mini-batch for the corresponding classes;
        - average angles in each mini-batch for the corresponding classes.
        During training, at each mini-batch, the angular loss computes the angles between the input extracted
        features and the weights (angle(W_i, x_i)); then, the average and median from this list 
        of angle is computed and plotted with this callback.
        """
        deg_avg_theta_gt_classes = [np.rad2deg(x) for x in self.hist_avg_theta_gt_classes]
        deg_median_theta_gt_classes = [np.rad2deg(x) for x in self.hist_median_theta_gt_classes]
        deg_avg_theta_not_gt_classes = [np.rad2deg(x) for x in self.hist_avg_theta_not_gt_classes] # convert to deg for easier understanding
        _image = pyplot_to_tf_image(plot_angles(
            deg_median_theta_gt_classes, 
            deg_avg_theta_gt_classes,
            deg_avg_theta_not_gt_classes
            ))
        with self.file_writer.as_default():
            tf.summary.image('Changing process of angles - Training', _image, step=0)


# TODO 
# DEBUG how to scatter and what.
class ScatterPlot2DCallback(Callback):
    """
    Plots a scatter of an embeddings whose output dimension is equal to 2.
    """
    def __init__(
        self, 
        log_path: str, 
        class_ids: List[str], 
        flag_test: bool,
        counter_plot_every_n_epoch: int = 1) -> None:
        """
        Parameters
        ---------

        - ds: either tf.data.Dataset or list/numpy array of ints, labels of the training dataset.
        - log_path:
        - class_ids:
        - flag_test: bool
        - counter_plot_every_n_epoch: int, indicates after how many epochs to plot the 3D sphere.
            Example: counter_plot_every_n_epoch=10, total_epochs=30, it plots 3 times, at epochs: 10, 20, 30.
            If it's negative or zero than it is set to 1.
        """
        super().__init__()
        self._flag_test = flag_test
        self._class_ids = class_ids
        if counter_plot_every_n_epoch <= 0:
            counter_plot_every_n_epoch = 1
        self.counter_epoch_plot = counter_plot_every_n_epoch
        self.file_writer = tf.summary.create_file_writer(log_path + sep + 'scatter-plot')
        self.margins = None
        self.ks = None # concentrations parameters

    """
    def on_test_end(self, epoch, logs):
        pass
    """
    

    def on_epoch_end(self, epoch, logs):
        if epoch % self.counter_epoch_plot == 0:
            batch_embeddings = self.model(tf.constant(logs['currBatch']), training=False)
            labels = logs['currBatchLabels']
            fig = scatter_plot(
                batch_embeddings[:, 0], 
                batch_embeddings[:, 1], 
                labels,
                self._class_ids
            )
            _image = pyplot_to_tf_image(fig)
            with self.file_writer.as_default():
                if self._flag_test:
                    tf.summary.image('Test - Scatter plot', _image, step=epoch)
                else:
                    tf.summary.image('Scatter plot - epoch({})'.format(epoch), _image, step=epoch)


"""
# TODO
# OPEN QUESTION:
    Is it possible that this callback needs to be applied to "model_cl"?
    "model_cl" is the classifier model comprising the frozen embeddings trained with a DML loss
    and a Dense layer to classify the entero data. Then, during training, you will apply "model_cl" with this
    callback to a dataset with only 3 output classes (mandatory requirement for this class) and this callback shall show how
    the intra and inter class separation is achieved during training of the "model_cl" model???
    TODO: Check if the paper from SphereFace does this or not or if it makes sense.
    Answer:
    This doesn't sound so well to me because I would show the features of a trained frozen embeddings. So, I would show
    the features of the training process of the entero classifier, instead what I want to show is the inter/intra separation
    of the features achieved during the training of the DML embeddings (model to extract the DML features).
    TODO(2):
    Does it make sense to apply a PCA on an embeddings with shape (n, m),
    where 'n' is the number of samples and 'm' > 3 to reduce it to 3 components and then
    plot the 3 components on the unit sphere with this callback?
"""
class Plot3DSphereCallback(Callback):
    """
    Plots a sphere of an embeddings whose output dimension is equal to 3.
    This class implements image 5 from SphereFace, so it plots the intersection of the points
    created by the feature vectors (extracted by an embedding extractor with output 
    dimension equal to 3) and the unit sphere.
    The legend shows the concentration and margin scaler parameters (obviously together with the class).
    Plotting multiple perspective of the same sphere might be useful (different angles), on pyplot you can
    move the image, can you do it in tensorboard too?
    # TODO:
    This class simply plots the extracted features on a sphere and therefore requires an
    output feature dimension of 3. Does it plot the intersection of the extracted 3D features 
    with the unit sphere?
    I need to understand how to plot the intersection of the extracted 3D features with the unit sphere.

    References:
    ----------

    - For a reference on spherical coordinates, see: http://web.physics.ucsb.edu/%7Efratus/phys103/Disc/disc_notes_3_pdf.pdf
    """
    def __init__(
        self, 
        log_path: str, 
        class_ids: List[str], 
        flag_test: bool,
        counter_plot_every_n_epoch: int = 1) -> None:
        """
        Parameters
        ---------

        - ds: either tf.data.Dataset or list/numpy array of ints, labels of the training dataset.
        - log_path:
        - class_ids:
        - flag_test: bool
        - counter_plot_every_n_epoch: int, indicates after how many epochs to plot the 3D sphere.
            Example: counter_plot_every_n_epoch=10, total_epochs=30, it plots 3 times, at epochs: 10, 20, 30.
            If it's negative or zero than it is set to 1.
        """
        super().__init__()
        self._flag_test = flag_test
        self._class_ids = class_ids
        if counter_plot_every_n_epoch <= 0:
            counter_plot_every_n_epoch = 1
        self.counter_epoch_plot = counter_plot_every_n_epoch
        self.file_writer = tf.summary.create_file_writer(log_path + sep + 'sphere-distr')
        self.margins = None
        self.ks = None # concentrations parameters
    

    # TODO
    # might be better a different callback to take care of the margin and k.
    # not sure when to plot, at batch or epoh end?
    # for the margins I will have the plot in 3D of the features distribution like in figure 2 in "Large-margin Softmax loss for CNNs".
    def update_data(self, margins: List[float], ks: List[float]) -> None:
        """
        Updates both margins and concentration hat (k) for the vMF.

        Parameters
        ---------

        - margins:
        -ks:
        """
        self.margins = margins
        self.ks = ks


    """
    def on_test_end(self, epoch, logs):
        pass
    """
    

    def on_epoch_end(self, epoch, logs):
        if epoch % self.counter_epoch_plot == 0:
            """
            batch_embeddings = list(self._train_ds
                        .take(1)
                        .unbatch()
                        .map(lambda x, _: x)
                        .as_numpy_iterator()
                        ) # list of len=batch_size, each elem has shape=(1,1044)
            batch_embeddings = tf.constant(np.stack(batch_embeddings))
            """
            batch_embeddings = self.model(tf.constant(logs['currBatch']), training=False)
            labels = logs['currBatchLabels']
            if logs['currBatch'].shape[-1] == 3:
                fig = sphere_4subplot(
                    batch_embeddings[:, 0], 
                    batch_embeddings[:, 1], 
                    batch_embeddings[:, 2], 
                    labels,
                    self._class_ids
                )
            else:
                fig = sphere_4subplot_pca(
                    batch_embeddings, 
                    labels, 
                    self._class_ids)
            _image = pyplot_to_tf_image(fig)
            with self.file_writer.as_default():
                if self._flag_test:
                    tf.summary.image('Test - Unit sphere', _image, step=epoch)
                else:
                    tf.summary.image('Unit sphere - epoch({})'.format(epoch), _image, step=epoch)


# TODO
# switch from input array y_val to generator for sklearn.metrics.confusion_matrix
# or find another API to plot cm, which supports generators.
class ConfusionMatrixCallback(Callback):
    """
    Custom callback to plots the confusion matrix of the model.
    It does something only at the end of each epoch during training
    or at the end of the evaluation or validation of the model.
    Does nothing for predictions.

    External links
    -------------

    - https://www.tensorflow.org/guide/keras/custom_callback
    """
    def __init__(self, val_ds, y_val, class_names, file_writer, flag_test_cm=False):
        """
        Instantiates a Confusion Matrix callback.
        It overrides only the on_test_end method.

        Parameters
        ---------

        - x_val: tf.data.Dataset of (inputs, targets).
        - y_val: list of ints, labels of the model.
        - class_names: list of strings, each string is a human readable label.
        - file_writer: tf.summary.create_file_writer, object to write to file.
        - flag_test_cm: boolean flag (default=False), if true it indicates it is performing on test/val data.
        """
        super(ConfusionMatrixCallback, self).__init__()
        self.val_ds = val_ds
        self.y_val = y_val
        self.class_names = class_names 
        self.file_writer = file_writer
        self.flag_test_cm = flag_test_cm


    def on_test_end(self, logs=None):
        self.log_confusion_matrix(1)


    def on_epoch_end(self, epoch, logs):
        if not self.flag_test_cm:
            self.log_confusion_matrix(epoch)


    def log_confusion_matrix(self, epoch):
        """
        Helper function to log the confusion matrix at the end of each epoch of the training of the CNN.
        Call this function inside a callback for the fit loop of a net.
        This function is not supposed to be run alone, call it in a callback.
        From:
            - https://www.tensorflow.org/tensorboard/image_summaries

        Parameters
        ---------

        - epoch: number of epoch the training loop is currently in.
        """
        val_pred_raw = self.model.predict(self.val_ds)
        val_pred = np.argmax(val_pred_raw, axis=1)
        cm = confusion_matrix(self.y_val, val_pred) 
        fig = self.plot_confusion_matrix(cm, class_names=self.class_names)
        cm_image = pyplot_to_tf_image(fig)
        with self.file_writer.as_default():
            if self.flag_test_cm:
                tf.summary.image('Test - Confusion Matrix', cm_image, step=epoch)
            else:
                tf.summary.image('Confusion Matrix - epoch({})'.format(epoch), cm_image, step=epoch)
    

    # TODO
    # src\models\custom_callbacks.py:102: RuntimeWarning: invalid value encountered in true_divide
    # labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    # write image to buffer and return it.
    def plot_confusion_matrix(self, cm, class_names):
        """
        Saves a confusion matrix log at the end of each epoch to view it later in Tensorboard.
        From:
            - https://www.tensorflow.org/tensorboard/image_summaries
        Parameters
        ---------
        - cm: sklearn.metric.confusion_matrix(**).
        - class_names: list of strings, names of the CNN labels.
        """
        fig = plt.figure(figsize=(15, 15))
        plt.rc('xtick', labelsize=6) 
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = 'black'
            if cm[i, j] > threshold:
                color = 'white'
            if i == j:
                color = 'red'
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color, weight='bold', fontsize=7.0)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig


class PlotPsiMarginCallback(Callback):
    """
    Reports useful information regarding the margin scaler psi for Kappa based
    losses.
    This callback tasks are:
    - plot to a graph the different margin scaler psi value at the 
        end of training to show their evolution.
    - Save to a json file the margin scaler psi value
    """
    def __init__(self, class_names: List[str], save_folder_psi: str) -> None:
        """
        Parameters
        ---------

        - class_names: list of strings, names of the classes associated with 
            a particular margin scaler psi (ordered).
        - save_folder_psi: str, path of the folder where to save all the json files with 
            the margin scaler psi values. Each file will have name: "{EpochNumber}.json".
            Each file will have the following structure: "{ClassName1: Psi1}" (eg: "{'E. coli: 0.12}").
        """
        super().__init__()
    

    def on_train_end(self, logs):
        """
        Plots a bar graph with the behavior of the margin scaler psi.
        The x-axis presents the margin scaler psi values (the labels
        are the class names associated with a particular psi), while the
        y-axis are the epochs.
        """
        raise NotImplementedError('TODO')


    def on_epoch_end(self, epoch, logs):
        """
        Saves to a json file the margin scaler psi value, where the key is the 
        class name.
        """
        raise NotImplementedError('TODO')


# TODO
# status: pending
"""
Does it make sense to have a callback? I think I would get swarmed with images and 
if you only select a few, then it makes more sense to just do it with a trained model,
select the images you want and plot their features maps.
See this tutorial as a reference:
https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
Though this suggest to do it in Tnesorboard:
https://github.com/tensorflow/tensorflow/issues/908
"""
class FeatureMapsCallback(Callback):
    """
    Callback to plot the feature maps of a CNN.
    This class assumes there are convolutional layers in the net's architecture.
    It might make sense to plot the feature maps only at the end of the training to avoid 
    cerating too many output images.

    Background notes
    ----

    Remember that:
    - In CNNs, the filters are the weights of each layer in the network.
    - Each layer contains 2 sets of weights (biases and filters).
    - Feature maps are the result of applying filters to the input (either input signal or a feature map).

    External links
    -------------

    - CNN explanation and nice visualization: https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
    """


    def on_epoch_end(self, epoch, logs=None):
        return 0

    
    def log_filters(self):
        return 0
    

    def plot_filters(self):
        return 0


class TsneCallback(Callback):
    """
    """
    def __init__(
        self, 
        log_path: str, 
        class_ids: List[str], 
        flag_test: bool,
        counter_plot_every_n_epoch: int = 1) -> None:
        """
        Parameters
        ---------

        - log_path:
        - class_ids:
        - flag_test: bool
        - counter_plot_every_n_epoch: int, indicates after how many epochs to plot the tsne.
            Example: counter_plot_every_n_epoch=10, total_epochs=30, it plots 3 times, at epochs: 10, 20, 30.
            If it's negative or zero than it is set to 1.
        """
        super().__init__()
        self._flag_test = flag_test
        self._class_ids = class_ids
        if counter_plot_every_n_epoch <= 0:
            counter_plot_every_n_epoch = 1
        self.counter_epoch_plot = counter_plot_every_n_epoch
        self.file_writer = tf.summary.create_file_writer(log_path + sep + 'tsne')
        self.margins = None
        self.ks = None # concentrations parameters
    

    # TODO
    # might be better a different callback to take care of the margin and k.
    # not sure when to plot, at batch or epoh end?
    # for the margins I will have the plot in 3D of the features distribution like in figure 2 in "Large-margin Softmax loss for CNNs".
    def update_data(self, margins: List[float], ks: List[float]) -> None:
        """
        Updates both margins and concentration hat (k) for the vMF.

        Parameters
        ---------

        - margins:
        -ks:
        """
        self.margins = margins
        self.ks = ks


    """
    def on_test_end(self, epoch, logs):
        pass
    """
    
    def on_epoch_end(self, epoch, logs):
        if epoch % self.counter_epoch_plot == 0:
            batch_embeddings = self.model(tf.constant(logs['currBatch']), training=False)
            labels = logs['currBatchLabels']
            fig = plot_tsne_2d(batch_embeddings, self._class_ids, labels)
            _image = pyplot_to_tf_image(fig)
            with self.file_writer.as_default():
                if self._flag_test:
                    tf.summary.image('Test - t-SNE', _image, step=epoch)
                else:
                    tf.summary.image('t-SNE - epoch({})'.format(epoch), _image, step=epoch)


class DebugEmptyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._epoch = 0

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]=None):
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]=None):
        self._epoch = epoch
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]=None):
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]=None):
        pass
    
    # TRAIN

    def on_train_begin(self, logs: Dict[str, Any]=None):
        self._epoch = 0

    def on_train_batch_begin(self, batch: int, logs: Dict[str, Any]=None):
        pass

    def on_train_batch_end(self, batch: int, logs: Dict[str, Any]=None):
        batch_embeddings = self.model(tf.constant(logs['currBatch']), training=False)
        count_non_zeros = tf.math.count_nonzero(logs['currBatch']).numpy()
        count_zeros = np.prod(logs['currBatch'].shape[:]) - count_non_zeros
        print('\nhas {} non-zeros in batch, at epoch {}\n'.format(count_non_zeros, self._epoch))
        print('\nhas {} zeros in batch, at epoch {}\n'.format(count_zeros, self._epoch))

    def on_train_end(self, logs: Dict[str, Any]=None):
        pass

    # TEST

    def on_test_begin(self, logs: Dict[str, Any]=None):
        pass

    def on_test_batch_begin(self, batch: int, logs: Dict[str, Any]=None):
        pass

    def on_test_batch_end(self, batch: int, logs: Dict[str, Any]=None):
        pass

    def on_test_end(self, logs: Dict[str, Any]=None):
        pass

    # PREDICT

    def on_predict_begin(self, logs: Dict[str, Any]=None):
        pass

    def on_predict_batch_begin(self, batch: int, logs: Dict[str, Any]=None):
        pass

    def on_predict_batch_end(self, batch: int, logs: Dict[str, Any]=None):
        pass

    def on_predict_end(self, logs: Dict[str, Any]=None):
        pass