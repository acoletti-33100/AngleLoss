# General information, notes, disclaimers
#
# author: A. Coletti
# 
# Start by implementing ArcFace, then add MarginScaler class and you're done!
#
#
#
# Remember to convert the margin values from radians to degrees, eg: 0.5 rad => 28.6 degrees
# to make things easier on the user accept as input margin radians values in [0,1],
# then convert the value to degrees with "numpy.rad2deg(margin)".
#
# This file is based on the losses module in KevinMusgrave/pytorch-metric-learning
# repo in GitHub. It is not exaclty the same, but pytorch and tensorflow are similar
# and therefore I re-implemented their work with some tweaks (it would be more correct to say it is inspired
# by their work).
# Their repo is distributed under the Mit license, so we should be fine with this
# kind of usage (re-implementation in another language with modifications).
#
# Tensorflow Loss is not the same as a Pytorch loss, so I may implement it as a 
# tf regulizer or reducer and not a "keras.losses.Loss". 
#
# Also alpha needs to be converted to degrees according to pytorch:
# https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#angularloss
#
#
# pytorch applies one hot encoding, while we use sparse cce, careful how to handle.
# Either convert internally as one hot or deal with sparse internally or switch to 
# one hot.
# 
# Note that tf.math.cos expects the inputs in rad form in range[-1,1].
#
# The ArcFace and KappaSimple are the "Vanilla" versions from their respective papers. Therefore, do 
# not add any modifications to their original algorithms. Instead, add to trunc newton any kind of 
# modification you want (like from AdaCos).
#
# Not sure I'll try AdaCos too; there is no code online, only the paper so it may require some time.
#
# ==============================================================================

from __future__ import annotations

from typing import List
from typing import Tuple
from typing import Union
from typing import Dict
from typing import Optional
from typing import Any
from typing import Callable

import numpy as np
import abc
import math

import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from models.angle_margin.margins import VoidMarginScaler
from models.angle_margin.margins import KappaSimpleMarginScaler 
from models.angle_margin.margins import MarginScaler
from models.angle_margin.margins import KappaNewtonApproxMarginScaler

from exceptions.exception_manager import ExceptionManager

from utils.utils import invert_tf_mask 
from utils.utils import to_string_trim_parent_path
from utils.utils import to_string_trim_file_name
from utils.utils import select_tf_where_isnot_t
from utils.utils import apply_fun_every_n_elements
from utils.utils import tf_median


class CrossEntropyLossHistoryProbs(Loss):
    """
    Implementation of the Cross entropy loss function.
    This class stores the output softmax probabilities.
    Use this class when you need to plot the softmax probabilities.
    It support both sparse and not sparse labels.
    """
    def __init__(self, epsilon: float = 1.0, is_sparse: bool = False) -> None:
        """
        Parameters
        ---------

        - epsilon: float (default=1.0), epsilon hyperparameter for the polyloss.
        - is_sparse: bool (default=False), true if the labels are sparse (otherwise one hot encoding).
        """
        super().__init__()
        self.epsilon = epsilon
        self.is_sparse = is_sparse
        if self.is_sparse:
            self._pt_labels = self._sparse_labels_pt
        else:
            self._pt_labels = self._categorical_labels_pt
        self.hist_probs = tf.Variable(validate_shape=False, dtype=tf.float32)


    @property
    def probs(self):
        """
        Returns the current output softmax probabilities.
        """
        if self.probs is not None:
            return self.probs


    def _sparse_labels_pt(self, y_true: tf.Tensor, depth: int):
        return tf.one_hot(y_true, depth)


    def _categorical_labels_pt(self, y_true, _):
        return y_true

    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        labels = self._pt_labels(y_true, y_pred.shape[1])
        self.hist_probs.append(tf.reduce_sum(labels * tf.nn.softmax(y_pred), axis=-1)) # TODO: debug
        return tf.nn.softmax_cross_entropy_with_logits(labels, y_pred)


class PolyLoss(Loss):
    """
    PolyLoss implementation as subclass of "keras.losses.Loss".
    It support both sparse and not sparse labels.
    """
    def __init__(self, epsilon: float = 1.0, is_sparse: bool = False) -> None:
        """
        Parameters
        ---------

        - epsilon: float (default=1.0), epsilon hyperparameter for the polyloss.
        - is_sparse: bool (default=False), true if the labels are sparse (otherwise one hot encoding).
        """
        super().__init__()
        self.epsilon = epsilon
        self.is_sparse = is_sparse
        if self.is_sparse:
            self._pt_labels = self._sparse_labels_pt
        else:
            self._pt_labels = self._categorical_labels_pt


    def _sparse_labels_pt(self, y_true: tf.Tensor, depth: int):
        return tf.one_hot(y_true, depth)


    def _categorical_labels_pt(self, y_true, _):
        return y_true

    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        labels = self._pt_labels(y_true, y_pred.shape[1])
        pt = tf.reduce_sum(labels * tf.nn.softmax(y_pred), axis=-1)
        cce = tf.nn.softmax_cross_entropy_with_logits(labels, y_pred)
        return cce + self.epsilon * (1 - pt)
    

class BaseAngularLoss(abc.ABC, Loss):
    """
    Abstract class for the angular losses, this class provides a common
    interface, do not directly instantiate.
    This class aims at improving a features (embeddings) extractor model, not a classifier;
    ***Do not apply to a classifier model***.
    The subclasses are responsible for the computation of the margin penalty for the ground truth
    label and non-corresponding labels for each samples during mini-batch training.

    Attributes
    ---------

    - _gt_weights: tf.Tensor, tensor to emulate the last FC layer weights. This weights are used to compute 
        a linear transformation 'Wx + b'.
    - hist_theta_gt: list of floats, each value is the angle (theta only, not with the added margin) 
        between a specific couple of input features and ground truth Weights (x_i, W_i).
        This attribute is used to plot how the angles change during training.

    - hist_theta_not_gt: list of floats, each value is the angle (theta only, not with the added margin) 
        between a specific couple of input features and not corresponding Weights (x_i, W_i).
        This attribute is used to plot how the angles change during training.

    TODOs
    ----

    - Priority=Low:
        Currently each subclass inherits the Kappa based loss methods. I think it is not the best as an
        architecture choice, either add an intermediate subclass KappaBasedLoss or make it through 
        composition or find another cleaner way. 
    """
    def __init__(
        self, 
        angular_manager: BaseAngularLossBehavior,
        embedding_dim: int, 
        sphere_radius: Union[int, Callable[[Any], float]], 
        num_out_classes: int,
        name: str,
        margin: float,
        gt_weights_save_path: str,
        batch_size: int) -> None:
        """
        Parameters
        ---------

        - angular_manager: BaseAngularLossBehavior,
        - sphere_radius: int, radius of the hyperspere hyperparameter.
        - num_out_class: int, total number of output classes in the training dataset.
        - sphere_radius: int or Callable, radius of the hyperspere hyperparameter. 
        - name: str, name of this loss function object.
        - margin: float, margin to apply; valid values are in [0,1] radians values (radians). 
        - gt_weights_save_path: str, path where to save the gt_weights.
        - batch_size: int, number of elements in each batch during training.

        Raises
        -----

        - ValueError: if margin is not in [0,1].
        """
        if margin < 0 or margin > 1:
            raise ValueError('Invalid margin value, received: {}, margin must be in [0,1]'.format(margin))
        super(BaseAngularLoss, self).__init__(name=name)
        self._num_out_classes = num_out_classes
        self.embedding_dim = embedding_dim
        self._sphere_radius = sphere_radius
        self.angular_manager = angular_manager 
        self._gt_weights = tf.Variable(
            tf.random.Generator.from_seed(1234).normal((embedding_dim, num_out_classes)), 
            name='groundTruthW') 
        self._margin = margin # radians
        self._gt_weights_save_path = gt_weights_save_path 
        self.hist_theta_gt = [] # TODO: make it optional to record theta history
        self.hist_theta_not_gt = []  # TODO: make it optional to record theta history
        self.batch_size = batch_size
    

    @property 
    def sphere_radius(self):
        """
        Returns the sphere radius hyperparameter.
        """
        return self._sphere_radius 
    

    @sphere_radius.setter
    def sphere_radius(self, sphere_radius: Union[int, float]):
        """
        Sets the sphere_radius to a new value.
        """
        if sphere_radius <= 0 or \
        not (
            isinstance(sphere_radius, float) or 
            isinstance(sphere_radius, int) or
            ( # tensor with 1 element
                isinstance(sphere_radius, tf.Tensor) and \
                (
                    (
                        len(sphere_radius.shape) == 1 and \
                        sphere_radius.shape[0] == 1 # 1 element tensor
                    )
                    or len(sphere_radius.shape) == 0 # 1 element tensor
                )
            )):
            raise ValueError('sphere_radius must be an int or float or tensor bigger than 0; received: {}'.format(sphere_radius))
        self._sphere_radius = sphere_radius
    

    @property
    def avg_theta_gt(self) -> List[float]:
        """
        Gets the average (for each training batch) of the angles (theta)
        for the corresponding (ground truth) classes only.

        Returns
        ------

        list of float, average of theta + margins for each mini-batch during training.

        Raises
        -----

        - ValueError: if it attempts to fetch theta without computing its values during training.
            So, before training the first mini-batch of the first epoch.
        """
        if len(self.hist_theta_gt) > 0: # TODO not end of training though, needs improvement
            return apply_fun_every_n_elements(self.hist_theta_gt, np.average, 1)
        raise ValueError('Theta not computed in training yet')
    

    @property
    def avg_theta_not_gt(self) -> List[float]:
        """
        Gets the average (for each training batch) of the angles (theta + margins) 
        for the not-corresponding (ground truth) classes only.

        Returns
        ------

        list of float, average of theta + margins for each mini-batch during training.

        Raises
        -----

        - ValueError: if it attempts to fetch theta without computing its values during training.
            So, before training the first mini-batch of the first epoch.
        """
        if len(self.hist_theta_not_gt) > 0: # TODO not end of training though, needs improvement
            return apply_fun_every_n_elements(self.hist_theta_not_gt, np.average, 1)
        raise ValueError('Theta not computed in training yet')
    

    @property
    def median_theta_gt(self) -> List[float]:
        """
        Gets the median (for each training batch) of the angles (theta + margins) 
        for the corresponding (ground truth) classes only.

        Parameters
        ---------

        - mask: tf.Tensor, mask with zeros and ones. Ones indicate the position of ground truth classes, while
            zeros indicates not-corresponding classes.

        Returns
        ------

        list of float, median of theta + margins for each mini-batch during training.

        Raises
        -----

        - ValueError: if it attempts to fetch theta without computing its values during training.
            So, before training the first mini-batch of the first epoch.
        """
        if len(self.hist_theta_gt) > 0: # TODO not end of training though, needs improvement
            return apply_fun_every_n_elements(self.hist_theta_gt, np.median, 1)
        raise ValueError('Theta not computed in training yet')
    

    @property
    def rad_margin(self):
        """
        Gets the margin in radians.
        """
        return self._margin
    

    @property
    def margin(self):
        """
        Gets the margin in degrees.
        """
        return np.rad2deg(self._margin)
    

    @property
    def gt_weights(self):
        return self._gt_weights
    

    @property
    def gt_weights_save_path(self):
        return self._gt_weights_save_path
    

    @gt_weights_save_path.setter
    def gt_weights_save_path(self, gt_weights_save_path):
        ExceptionManager.is_npy_file(gt_weights_save_path, is_existing_file=False)
        self._gt_weights_save_path = gt_weights_save_path


    @property
    @abc.abstractmethod
    def mem_buf(self) -> tf.Tensor:
        """
        Returns the memory buffer for this class instance of Kappa based Loss.
        """
        pass


    @property
    @abc.abstractmethod
    def is_kappa(self) -> bool:
        """
        Returns true if the subclass implements a Kappa based Loss, otherwise
        returns false.
        """
        pass


    @abc.abstractmethod
    def _assign_mem_buf(self, new_mem_buf: np.ndarray) -> None:
        """
        Assign `new_mem_buf` to the memory buffer if this class is a Kappa
        based loss.

        Parameters
        ---------

        - new_mem_buf: numpy.ndarray, array with the values to assign to the memory buffer.

        Raises
        -----

        - ValueError: if `new_mem_buf.shape` is different from the memory buffer shape.
        """
        pass
    

    def _append_angles_to_hist(self, theta: tf.Tensor, mask: tf.Tensor) -> None:
        """
        Appends to self.hist_theta_gt and self.hist_not_gt the ground truth and not-corresponding
        angles, respectively.

        Parameters
        ---------

        - theta: tf.Tensor, tensor with the angles for each element in the mini-batch.
        - mask: tf.Tensor, mask of zeros and ones with shape=(batch_size, num_out_classes). 
            Ones indicates the ground truth labels.

        """
        inv_mask = invert_tf_mask(mask)
        self.hist_theta_gt.append(select_tf_where_isnot_t(theta * mask, 0.).numpy().tolist())
        self.hist_theta_not_gt.append(select_tf_where_isnot_t(theta * inv_mask, 0.).numpy().tolist())
    

    def update_mem_buf_on_epoch_end(self) -> None:
        """
        Updates the memory buffer. 
        ***This method should be called only during training at the end of each epoch***.
        """
        self._assign_mem_buf(self._gt_weights.numpy())
    

    @abc.abstractmethod
    def compute_margin_penalty(
        self, 
        dist_w_embeddings: tf.Tensor, 
        mask: tf.Tensor, 
        **kwargs: Dict[str, Any]) -> tf.Tensor:
        """
        This method computes the cosine of the additive angular margin penalty for each output class.

        Parameters
        ---------

        - dist_w_embeddings: tf.Tensor, distance (rank-2 tensor) between the 
            weights of the desired layer and the embeddings. This is the cos(theta_y), which is used to compute
            theta_y and finally obtain the cos(theta_y + margin). 
        - mask: tf.Tensor, mask of zeros and ones with shape=(batch_size, num_out_classes). 
            Ones indicates the ground truth labels. 
            The margin must be applied only to the angles associated with the ground truth labels.
        - kwargs: python dict, for extending the method in the children 
            if more parameters are needed.

        Notes
        ----

        - Computes the cosine of theta plus the margin (cos(theta + margin)).
        Different subclasses may implement different ways to compute this penalty (Kappa based
        losses multiply the margin by a scaler, for example).

        - Internally this method applies tensorflow tf.math.cos and tf.math.acos; both require the inputs
            in radians.

        Returns
        ------

        rank-2 tf.Tensor, each value is the additive angular margin penalty associated
            with each i-th sample in the batch depending on the sample's ground truth label.
            The result is expressed in radians.
        """
        pass
    

    @abc.abstractmethod
    def update_memory_buffer(self) -> None:
        """
        Updates the memory buffer as indicated in equation 7 in KappaFace.
        ***This method should only be called at the end of each training epoch***.
        """
        pass
    

    def call(
        self, 
        embedding: tf.Tensor, 
        labels: tf.Tensor) -> Union[float, tf.Tensor]:
        """
        Criterion to compute the angular margin loss between the embeddings and the labels.
        Note that the angular margin framework is the composition
        of a pipeline as: "embedding -> angular loss -> softmax -> cross-entropy loss".
        This loss aims at improving the angular margins on the hypersphere between classes, then
        transform the logits with the additive angular margin and finally apply the transformed
        logits to compute the cross-entropy.

        Parameters
        ---------

        embedding: tf.Tensor, output of the forward pass of a model which extracts the embeddings (features).
        labels: tf.Tensor, ground truth labels associated with the current batch input.
        - margin_penalty_func: Callable, function which computes the margin penalty (method
            from classes which inherits from BaseAngularLoss).
        - num_out_class: int, total number of output classes in the training dataset.

        Returns
        ------
        
        y_pred, as floats or tensors. In practice, it returns the labels and the
            computation value from the cross entropy or poly loss function.

        TODOS
        ----

        Ideally, this method should be called by the class itself when needed or maybe a custom callback.
        """
        return self.angular_manager.compute_loss(
            embedding, 
            labels, 
            self._gt_weights,
            self.compute_margin_penalty, 
            self.sphere_radius, # or callable to compute dynamic sphere radius
            self._num_out_classes)
    

    # TODO
    # make the class call this method internally somehow?
    def to_npy_gt_weights_on_train_end(self) -> None:
        """
        Saves to file the ground truth weights.
        This method needs to be called at the end of training.
        This method appends the loss name to the final .npy file.
        If the class instance is a Kappa based loss, then it also saves the associated memeory buffer.
        
        Examples
        -------

        self.__gt_weights_save_path = "p1/p2/gt_weights.npy"; the loss name is "arc_loss".
        Then, this method will save the ground truth weights to:
        "p1/p2/arc_loss_gt_weights.npy".
        While, the memory buffer is saved to:
        "p1/p2/mem_buf_arc_loss.npy".
        """
        ExceptionManager.is_npy_file(self._gt_weights_save_path,  is_existing_file=False)
        parent_path = to_string_trim_file_name(self._gt_weights_save_path)
        final_file_name = '{}{}_{}'.format(
            parent_path, 
            self.name, 
            to_string_trim_parent_path(self._gt_weights_save_path)
            )
        np.save(final_file_name, self._gt_weights.numpy()) # saves gt weights
        if self.is_kappa:
            mem_buf_save_path = '{}mem_buf_{}.npy'.format(
                to_string_trim_file_name(final_file_name), 
                self.name
                )
            np.save(mem_buf_save_path, self.mem_buf.numpy()) # saves memory buffer
    

    def load_gt_weights_from_npy(self, save_path: str) -> None:
        """
        Loads the ground truth weights from the specified file.
        Also, loads the memory buffer if this class instance is a Kappa based Loss.
        This method needs to be called before the beggining of training when you 
        want to resume a previous experiment training state.

        Parameters
        ---------

        - save_path: str, path where the .npy file with the saved weights is stored.

        Notes
        ----

        - This method assumes the memory buffer is stored in a file named:
            "mem_buf_{}".format(save_path)

        Raises
        -----

        - ValueError: if the npy file doesn't exist or it isn't an npy file or 
            the new array has an incopatible shape with the current ground truth weights.

        TODOS
        ----

        Ideally, this method should be called by the class itself when needed or maybe a custom callback.
        """
        ExceptionManager.is_npy_file(save_path)
        arr = np.load(save_path)
        if arr.shape != self.gt_weights.numpy().shape:
            raise ValueError('Incompatible shapes; file shape: {}, gt_weights requires: {}'.format(
                arr.shape, self.gt_weights.numpy().shape))
        self._gt_weights.assign(arr)
        if self.is_kappa:
            parent_path = to_string_trim_file_name(save_path)
            mem_buf_save_path = to_string_trim_file_name(save_path)
            mem_buf_save_path = '{}mem_buf_{}'.format(parent_path, mem_buf_save_path)
            ExceptionManager.is_npy_file(mem_buf_save_path)
            self._assign_mem_buf(np.load(mem_buf_save_path))
    

    def get_config(self) -> Dict[str, Union[str, int, float]]:
        """
        Returns a dictionary to instantiate a subclass of BaseAngularLoss.
        If a subclass contains a non JSON serializeable object, then you need to either
        make the object serializeable of make it possible to instatiate it in the constructor
        with serializeable types.
        """
        config = {
            'num_out_classes': self._num_out_classes,
            'gt_weights_save_path': self._gt_weights_save_path, 
            'is_poly_loss': self.angular_manager.is_poly_loss,
            'batch_size': self.batch_size
        }
        base_config = super(BaseAngularLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    @classmethod
    def from_config(cls, config: Dict[str, Union[float, str, int]]):
        """
        Instantiates a `BaseAngularLoss` from its config (output of `get_config()`).
        This class must always be implemented by subclasses.

        Parameters
        ---------

        - config: Output of `get_config()`.

        Returns
        ------

        A `Loss` instance.
        """
        pass


class BaseAngularLossBehavior(abc.ABC):
    """
    Abstract class; this is a common interface for the angular based loss
    behavior, do not directly implement this class.
    """
    

    @abc.abstractmethod
    def compute_loss(
        self, 
        embedding: tf.Tensor, 
        labels: tf.Tensor, 
        gt_weights: tf.Variable,
        margin_penalty_func: Callable,
        sphere_radius: int,
        num_out_class: int) -> Union[float, tf.Tensor]:
        """
        Criterion to compute the angular margin loss between the embeddings and the labels.
        Note that the angular margin framework is the composition
        of a pipeline as: "embedding -> angular loss -> softmax -> cross-entropy loss".
        This loss aims at improving the angular margins on the hypersphere between classes, then
        transform the logits with the additive angular margin and finally apply the transformed
        logits to compute the cross-entropy.

        Parameters
        ---------

        embedding: tf.Tensor, output of the forward pass of a model which extracts the embeddings (features).
        labels: tf.Tensor, ground truth labels associated with the current batch input.
        - gt_weights: tf.Variable, ground truth weights with shape (batch_size, num_out_classes)
            used to compute the angles between input features and gt_weights.
        - margin_penalty_func: Callable, function which computes the margin penalty (method
            from classes which inherits from BaseAngularLoss).
        - sphere_radius: int, radius of the hyperspere hyperparameter.
        - num_out_class: int, total number of output classes in the training dataset.

        Notes
        ----

        - This method assumes that the embeddings has already been L2 normalized and rescaled by the 
            sphere radius hyperparameter.

        - This method internally normalizes the weights of the last FC layer (last_fc_weights), so
        do not normalize the Tensor outside of this method.

        Returns
        ------
        
        - y_pred, it returns the computation value from the cross entropy or 
            poly loss function (forward pass on the loss function).
        """
        pass

    
    @property
    @abc.abstractmethod
    def is_poly_loss(self) -> bool:
        """
        Checks whether this loss internally applies a Poly Loss or not.
        """
        pass


    @abc.abstractmethod
    def get_name_inner_loss(self) -> str:
        pass

    
    @abc.abstractmethod
    def get_distance_embedding_weights(self, embedding: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        """
        Computes the distance between the input embedding and the weights.

        Parameters
        ---------

        - embedding: tf.Tensor, input embedding.
        - weights: tf.Tensor, weights to multiply with.

        Returns
        ------

        tf.Tensor as tf.matmul(embedding, tf.transpose(weights))
        """
        pass


    @abc.abstractmethod
    def scale_logits(
        self, 
        logits: tf.Tensor, 
        sphere_radius: int) -> tf.Tensor:
        """
        Rescales the logits by the sphere radius.

        Parameters
        ---------

        - logits: tf.Tensor, 
        - embedding: tf.Tensor, 
        - w: tf.Tensor, 
        - sphere_radius: int, radius of the hyperspere hyperparameter.

        Returns
        ------

        tf.Tensor, rescaled logits.
        """
        pass
    

# TODO - 1
# poly Loss is useful to improve classification. Not sure it could help
# angular margins. I could try it and leave it as a feature.
# TODO - 2
# add support for not sparse labels
# On this note, this means: instead of passing a boolean pass a loss object
# with the loss you desire or its name; eg: SparseCategoricalCrossEntropy,
# SparseCategoricalPolyLoss..
class AdditiveAngularLossBehavior(BaseAngularLossBehavior):
    """
    This class implements the common behavior to all the implementations
    of the additive angular based losses (aka: subclasses of BaseAngularLoss).
    This class should not be inherited, instead it should be composed inside the
    instance of the classes inheriting BaseAngularLoss.
    """
    def __init__(
        self, 
        is_poly_loss: bool = False) -> None:
        """
        Parameters
        ---------

        - is_poly_loss: bool (default=False), flag to indicate whether to apply a Poly or a
            categorical cross-entropy loss.
        """
        super(AdditiveAngularLossBehavior, self).__init__()
        if is_poly_loss:
            self.cce = PolyLoss(is_sparse=True)  
        else:
            self.cce = SparseCategoricalCrossentropy(from_logits=True) # y_pred is always a logits
        self._is_poly_loss = is_poly_loss
    

    @property
    def is_poly_loss(self) -> bool:
        return self._is_poly_loss


    def get_name_inner_loss(self) -> str:
        return self.cce.name


    def compute_loss(
        self, 
        embedding: tf.Tensor, 
        labels: tf.Tensor, 
        gt_weights: tf.Variable,
        margin_penalty_func: Callable,
        sphere_radius: int, # or Callable to compute dynamic sphere radius
        num_out_class: int) -> Union[float, tf.Tensor]:
        """
        Criterion to compute the angular margin loss between the embeddings and the labels.
        Note that the angular margin framework is the composition
        of a pipeline as: "embedding -> angular loss -> softmax -> cross-entropy loss".
        This loss aims at improving the angular margins on the hypersphere between classes, then
        transform the logits with the additive angular margin and finally apply the transformed
        logits to compute the cross-entropy.

        Parameters
        ---------

        embedding: tf.Tensor, output of the forward pass of a model which extracts the embeddings (features).
        labels: tf.Tensor, ground truth labels associated with the current input batch.
        - gt_weights: tf.Variable, ground truth weights with shape (batch_size, num_out_classes)
            used to compute the angles between input features and gt_weights.
        - margin_penalty_func: Callable, function which computes the margin penalty (method
            from classes which inherits from BaseAngularLoss).
        - sphere_radius: int, radius of the hyperspere hyperparameter.
        - num_out_class: int, total number of output classes in the training dataset.

        Notes
        ----

        - This method internally normalizes the embeddings (L2 norm and rescale by the 
            sphere radius hyperparameter).

        - This method internally normalizes the weights of the last FC layer (last_fc_weights), so
        do not normalize the Tensor outside of this method.

        Implementation details
        ---------------------

        - Why this method needs a mask:
            To each element in the batch I want to add ONLY the cosine of the 
            angular margin penalty associated with its class.
            that's why I need a mask: if the first element in the batch has ground
            truth label 2, then I want to add it only the cosine of the angular margin penalty
            associated with class 2, not also the values for other classes.
        
        - masked_distance is needed to make sure that, for the i-th sample, only the
            distance (and therefore the margin) associated with its ground truth label gets applied.
            Remember, all these computations are used in a loss framework; a loss objective is to
            compute something which is used to improve some existing parameters (net's weights).
            Therefore, since we want to improve the logits and the discriminative power 
            of the loss function, we add the angular margin properties to the frame. So, during training,
            we pass the ground truth information to make sure we can improve the margin toward a 
            specific class for each sample in every batch;
            or, more specifically in the case of the embedding extractor, to make sure that each 
            input feature will tend more toward its ground truth label.
            Finally, the mask is simply needed to ease the computation and apply matrices operations 
            instead of if walls or other unfriendly and costly method.
            For example, if the 0-th sample in the batch has ground truth label
            "2" (for a total of 3 output classes), then the masked_distance will
            have row 0-th set to [0., 0., X.] with "X." some float value, because
            the mask will be [0.,0., 1.] (shows only 0-th row). 

        - The following holds true in this method:
            - embedding.shape = (batch_size, dim_embedding) 
            - self.W.shape = (dim_embedding, num_out_classes)  
            - distance_w_embeddings.shape = (batch_size, num_out_classes) = 
                (embedding.shape[0], self.W.shape[1])
            - distance_w_embeddings = matmul(embedding, self.W)
        
        - logits_j = s * cos(theta_j + m) for all j in [0, .., num_classes - 1] -> ArcFace (Additive angular margin losses)

        Returns
        ------
        
        y_pred, it returns the computation value from the cross entropy or 
        poly loss function (forward pass on the loss function).

        References
        ---------

        - This method is inspired by:
            https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/large_margin_softmax_loss.py
            See the method:
            src/pytorch_metric_learning/losses/large_margin_softmax_loss.py::LargeMarginSoftmaxLoss.compute_loss
        """
        embedding = tf.math.l2_normalize(embedding, axis=1) 
        normed_last_fc_weights = tf.math.l2_normalize(gt_weights, axis=1) 
        distance_w_embeddings = self.get_distance_embedding_weights(embedding, normed_last_fc_weights) # cos(theta_y)
        mask = tf.one_hot(labels, num_out_class) # mask.shape=(batch_size, num_out_classes), 1 where the gt label is.
        logits = margin_penalty_func(distance_w_embeddings, mask)  # cos(theta_y + final_margin)
        logits = self.scale_logits(logits, sphere_radius) 
        return self.cce(labels, logits)
    

    def get_distance_embedding_weights(self, embedding: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        """
        Computes the distance, as the cosine similarity, between the input embedding and the weights.
        This method returns the cos(theta_y_i) which is the cosine of the angle between an input
        feature x_i and the W_j weight for a j-th output class.

        Parameters
        ---------

        - embedding: tf.Tensor, input embedding; embedding.shape[1] must be equal to 
            weights.shape[0].
        - weights: tf.Tensor, weights to multiply with.

        Returns
        ------

        tf.Tensor, matmul between embedding and the transpose of the weights.
            The output tensor will have shape: (embedding.shape[0], weights.shape[0]).
            This is because the matmul operation is equivalent to (from tf docs):
            out[.., i, j] = sum_k (a[.., i, k] * b[.., i, k]) for all i, j indices.
        """
        if embedding.shape[1] == weights.shape[0]: # already transposed weights
            return tf.matmul(embedding, weights)
        elif embedding.shape[1] == weights.shape[1]: # weights to transpose
            return tf.matmul(embedding, tf.transpose(weights))
        else:
            final_msg = 'embedding.shape[1] must be equal to either weights.shape[0] or weights.shape[1]'
            raise ValueError('Incompatible shapes; received (embedding - weights): {} - {};\n{}'.format(
                embedding.shape, 
                weights.shape,
                final_msg
                ))


    def scale_logits(
        self, 
        logits: tf.Tensor, 
        sphere_radius: int) -> tf.Tensor:
        """
        Rescales the logits by the sphere radius.

        Parameters
        ---------

        - logits: tf.Tensor, 
        - sphere_radius: int, radius of the hyperspere hyperparameter, used
            to rescale the logits.

        Returns
        ------

        tf.Tensor, rescaled logits as logits * sphere_radius.
        """
        return logits * sphere_radius


class ArcFaceLoss(BaseAngularLoss): 
    """
    """
    def __init__(
        self, 
        embedding_dim: int, 
        num_out_classes: int, 
        margin: float, 
        sphere_radius: int,
        gt_weights_save_path: str,
        batch_size: int,
        **kwargs) -> None:
        """
        Parameters
        ---------

        - embedding_dim: int, dimension of the output embeddings (z_i).
        - num_out_classes: int, number of the output classes used during training.
        - margin: float, fixed margin to apply in [0,1] radians values (radians). 
        - sphere_radius: int, radius of the sphere hyperparameter for this Loss framework.
        - gt_weights_save_path: str, path where to save the gt_weights.
        - batch_size: int, batch size to use during training.
        - **kwargs: python dict, it needs to contain 'is_poly_loss' and 'is_sparse' both as boolean flags.
        """
        super(ArcFaceLoss, self).__init__(
            AdditiveAngularLossBehavior(kwargs['is_poly_loss']),
            embedding_dim,
            sphere_radius,
            num_out_classes,
            'arc_loss',
            margin,
            gt_weights_save_path,
            batch_size)
    

    @property
    def is_kappa(self):
        return False
    

    @property
    def mem_buf(self) -> None:
        """
        This class doesn't have a memory buffer.

        TODOs
        ----

        Architecture incosistence? find better way to deal with different losses.
        """
        raise NotImplementedError('Does not have a memory buffer')
    

    def _assign_mem_buf(self, new_mem_buf: np.ndarray) -> None:
        """
        This method does not make sense for Arc based losses.

        TODOs
        ----

        Architecture incosistence? find better way to deal with different losses.
        """
        raise NotImplementedError('Does not have a memory buffer')
    

    def compute_margin_penalty(
        self, 
        dist_w_embeddings: tf.Tensor, 
        mask: tf.Tensor, 
        **kwargs: Dict[str, Any]) -> tf.Tensor:
        """
        This method computes the cosine of the additive angular margin penalty for each output class.
        It computes the cosine of theta plus the margin, as: cos(theta_y_i + margin).
        The angle (arccos(x_i * W)) is computed between the embedding and the weights W of this
        class (which are initialized to the weights of the model's embedding
        layer's weights).

        Parameters
        ---------

        - dist_w_embeddings: `tf.Tensor`, distance between the weights and the embeddings. 
        - mask: tf.Tensor, mask of zeros and ones with shape=(batch_size, num_out_classes). 
            Ones indicates the ground truth labels.
            The margin must be applied only to the angles associated with the ground truth labels.
        - kwargs: python dict, not needed for this subclass of `BaseAngularLoss`.

        Returns
        ------

        - rank-1 tf.Tensor, each value is the additive angular margin penalty associated
            with each i-th sample in the batch.
            This penalty is the original ArcFace penalty as: "cos(theta + margin)".
            The result is expressed in radians.
        """
        rad_theta = tf.math.acos(tf.clip_by_value(dist_w_embeddings, -1, 1)) # arccos(x): -1 <= x <= 1
        self._append_angles_to_hist(rad_theta, mask) # stores angles for plotting
        res = tf.math.cos(rad_theta + self.rad_margin) 
        diff = res - (mask * dist_w_embeddings) 
        return dist_w_embeddings + (mask * diff)
    

    def update_memory_buffer(self) -> None:
        pass
    

    def get_config(self):
        config = super().get_config()
        config['embedding_dim'] = self.embedding_dim,
        config['sphere_radius'] = self.sphere_radius,
        config['margin'] = self.rad_margin
        return config
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# TODO
# put common behavior here?
# or define common methods as external or static parents functions 
# and call them?
# need to refactor the architecture?
# class KappaBaseVmf(BaseAngularLoss):
# what if I use a Mixin?
# see:
# https://www.pythontutorial.net/python-oop/python-mixin/
# from: https://stackoverflow.com/a/533675  :
# "you want some generic means of ensuring that your type will do 
# this and it just works. You want code reuse"
# "if you just think of a mixin as a small base type designed to add a small 
# amount of functionality to a type without otherwise affecting that type, then you're golden.
# Hopefully. :)"

class KappaBehaviorMixin:
    """
    Mixin to define common Kappa based losses behavior to obtain better code reuse.
    As a design choice, keep this class stateless and only implement common Kappa
    behavior. Leave attributes to subclasses of BaseAngularLoss.

    See Also
    -------

    To better understand what a Mixin is in Python, see:

    - [1] https://www.pythontutorial.net/python-oop/python-mixin/
    - [2] https://stackoverflow.com/a/533675 

    [2] states: "you want some generic means of ensuring that your type will do 
    this and it just works. You want code reuse"
    "if you just think of a mixin as a small base type designed to add a small 
    amount of functionality to a type without otherwise affecting that type, then you're golden.
    Hopefully. :)"
    """
    def angle_margin_penalty(
        self, 
        margin_scaler_manager: MarginScaler,
        dist_w_embeddings: tf.Tensor, 
        theta_recorder_fun: Callable,
        mask: tf.Tensor,
        rad_margin: float) -> tf.Tensor:
        """
        This method computes the cosine of the additive angular margin penalty (logits) for each output class.
        It adds the margins only to the not corresponding labels.

        Parameters
        ---------

        - margin_scaler_manager: MarginScaler, object to compute the margin scaler (psi).
        - dist_w_embeddings: tf.Tensor, distance (rank-2 tensor) between the 
            weights of the desired layer and the embeddings. 
        - theta_recorder_fun: Callable, function with which to store the theta values for plotting
            with Callbacks during training.
        - mask: tf.Tensor, mask of zeros and ones with shape=(batch_size, num_out_classes). 
            Ones indicates the ground truth labels.
        - rad_margin: float, fixed margin parameter expressed in radians.

        Notes
        ----

        - Computes the cosine of theta plus the margin depending on mask:
            cos(theta_j + margin); where j=not corresponding labels.

        - Internally this method applies tensorflow tf.math.cos and tf.math.acos; both require the inputs
            in radians.
        
        Examples
        -------

        TODO: review example.

        >>> d=tf.constant([0.3,0.2,0.1,0.5])
        ... <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.3, 0.2, 0.1, 0.5], dtype=float32)>
        >>> # 2 output classes
        >>> # margins = [0.33, 0.11]
        >>> # ang_loss is the angular loss object which will call
        >>> # this method ("angle_margin_penalty") to compute the result.
        >>> ang_loss.compute_margin_penalty(d, mask)
        ...     <tf.Tensor: shape=(2, 4), dtype=float32, numpy=
                       TODO, dtype=float32)>
        >>> # each row is the margin penalty associated with a class
        >>> # first row corresponds to class 0
        >>> # each column corresponds to the i-th sample in the batch

        Returns
        ------

        rank-2 tf.Tensor (logits), 
            The output tensor will have shape=(num_out_classes, batch_size).
            In this case we have a different margin for each output class, so 
            each array will have batch_size elements, where
            each value is the additive angular margin penalty (for a specific class) associated
            with each i-th sample in the batch.
            The result is expressed in radians.
        """
        theta = tf.math.acos(tf.clip_by_value(dist_w_embeddings, -1, 1)) # arccos(x): -1 <= x <= 1
        theta_recorder_fun(theta, mask) # stores angles for plotting
        margins = rad_margin * margin_scaler_manager.compute_margin_scaler_psi() 
        theta += margins
        res = tf.math.cos(theta) # adds margins to all, then we need to subtract. 
        diff = res - (mask * dist_w_embeddings) 
        return dist_w_embeddings + (mask * diff)


    # TODO
    # make the history recorder variable more flexible to accept different variables where the input
    # parameters are the name of the variable, x-axis, y-axis.
    # In this way I will have a more general function to record a single or multiple variable values.
    # So, make it accept either a single variable or a matrix, where each row is a different variable.
    # EG: def variable_recorder_history(var: np.ndarray, names: List[str], x_axis: List[str], y_axis: List[str])
    def angle_margin_penalty_with_adaptive_sphere_radius(
        self, 
        margin_scaler_manager: MarginScaler,
        dist_w_embeddings: tf.Tensor, 
        theta_recorder_fun: Callable,
        mask: tf.Tensor,
        threshold_theta_med: tf.Tensor,
        curr_sphere_radius: Union[int, float],
        rad_margin: float,
        is_update_sphere_radius: bool,
        apply_margins: float = 1.0) -> Tuple[tf.Tensor, Union[None, tf.Tensor]]:
        """
        This method computes the cosine of the additive angular margin penalty (logits) for each output class
        as the combination of AdaCos and KappaFace loss. Therefore, this method applies a dynamically adaptive
        sphere radius hyperparameter, as well as a dynamically adpative margin scaler.

        Parameters
        ---------

        - margin_scaler_manager: MarginScaler, object to compute the margin scaler (psi).
        - dist_w_embeddings: tf.Tensor, distance (rank-2 tensor) between the 
            weights of the desired layer and the embeddings. 
        - theta_recorder_fun: Callable, function with which to store the theta values for plotting
            with Callbacks during training.
        - mask: tf.Tensor, mask of zeros and ones with shape=(batch_size, num_out_classes). 
            Ones indicates the ground truth labels.
        - threshold_theta_med: tf.Tensor, threshold value in equation 15 in AdaCos (pi/4 in
            the paper). This value needs to be the center of the range of possible theta values. Since, the
            angles are bounded in the range [0, pi/2] in AdaCos paper, then this parameter should be
            set to pi/4. So, this function sets the theta range is set to: [0, 2 * threshold_theta_med].
            The tensor contains the value expressed in radians.
        - curr_sphere_radius: int or float, current value of the sphere radius.
        - rad_margin: float, fixed margin parameter expressed in radians.
        - is_update_sphere_radius: bool, flag to indicate whether to update the sphere
            radius or not (True=update the sphere radius).
        - apply_margins: float (default=1.0), margins multiplier to apply the normal AdaCos algorithm
            pass "0.0", otherwise pass "1.0" (default value).

        Notes
        ----

        - Computes the cosine of theta plus the margin depending on mask:
            cos(theta_j + margin); where j=not corresponding labels.

        - Internally this method applies tensorflow tf.math.cos and tf.math.acos; both require the inputs
            in radians.
        
        Examples
        -------

        TODO: review example.

        >>> d=tf.constant([0.3,0.2,0.1,0.5])
        ... <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.3, 0.2, 0.1, 0.5], dtype=float32)>
        >>> # 2 output classes
        >>> # margins = [0.33, 0.11]
        >>> # ang_loss is the angular loss object which will call
        >>> # this method ("angle_margin_penalty") to compute the result.
        >>> ang_loss.compute_margin_penalty(d, mask)
        ...     <tf.Tensor: shape=(2, 4), dtype=float32, numpy=
                       TODO, dtype=float32)>
        >>> # each row is the margin penalty associated with a class
        >>> # first row corresponds to class 0
        >>> # each column corresponds to the i-th sample in the batch

        Returns
        ------

        - rank-2 tf.Tensor (logits), 
            The output tensor will have shape=(num_out_classes, batch_size).
            In this case we have a different margin for each output class, so 
            each array will have batch_size elements, where
            each value is the additive angular margin penalty (for a specific class) associated
            with each i-th sample in the batch.
            The result is expressed in radians.
        - tf.Tensor or None, new sphere radius or None.
        """
        theta = tf.math.acos(tf.clip_by_value(dist_w_embeddings, -1, 1)) # arccos(x): -1 <= x <= 1
        theta_recorder_fun(theta, mask) # stores angles for plotting
        margins = rad_margin * margin_scaler_manager.compute_margin_scaler_psi() * apply_margins
        res = tf.math.cos(theta + margins) # adds margins to all, then we need to subtract. 
        diff = res - (mask * dist_w_embeddings) 
        logits = dist_w_embeddings + (mask * diff) # output logits
        new_sphere_radius = None
        if is_update_sphere_radius: # update only at last mini-batch during each training epoch
            theta = tf.clip_by_value(theta, 0, 2 * threshold_theta_med) 
            theta_median = tf_median(select_tf_where_isnot_t(theta * mask, 0)) # only theta for corresponding classes
            mask_not_gt = invert_tf_mask(mask)
            threshold_theta_med = tf.reduce_mean(select_tf_where_isnot_t(theta * mask_not_gt, 0)) / 2 # center between 0 and theta mean for not-corresponding classes
            b_avg = tf.reduce_sum(tf.where( # B_avg in AdaCos paper
                mask_not_gt == 1, 
                tf.exp(curr_sphere_radius * logits), 
                0.))
            b_avg /= logits.shape[0] # B_avg /= mini-batch size
            new_sphere_radius = tf.math.log(b_avg) 
            tmp = tf.math.cos(tf.minimum(threshold_theta_med, theta_median))
            new_sphere_radius /= tmp
        return logits, new_sphere_radius


class KappaVmfSimple(KappaBehaviorMixin, BaseAngularLoss):
    """
    """
    # careful when you convert rad to deg, since I also need to compute wcs
    # in the ctor.
    def __init__(
        self, 
        margin: float, 
        sphere_radius: int,
        classes_freq: List[int],
        gamma: float, 
        temperature: float, 
        alpha: float,
        embedding_dim: int, 
        class_ids: List[str],
        gt_weights_save_path: str,
        batch_size: int,
        pop_weights_concentration_fun: str = "sigmoid", 
        beta: Optional[float] = None, 
        mem_buf: Optional[tf.Tensor] = None,
        **kwargs) -> None:
        """
        Parameters
        ---------

        - margin: float, fixed margin to apply in [0,1] radians values (radians). 
        - sphere_radius: int (default=64), radius of the sphere hyperparameter for this Loss framework.
        - classes_freq: list of ints, each value is the total number of samples for a specific class used in training.
        - gamma: float, gamma hyperparameter in [0,1] radians to apply in the scaling factor formula (EQ. 14).
            It balances the contribution of the sample and population concentration.
        - temperature: float, temperature hyperparameter in [0,1] radians.
        - alpha: float, alpha momentum hyperparameter in [0,1] radians needed in the memory buffer update.
        - embedding_dim: int, dimension of the output embeddings (z_i).
        - class_ids: list of str, each is the name of the output class associated with the classes_freq list.
        - gt_weights_save_path: str, path where to save the gt_weights.
        - batch_size: int, number of elements in each batch during training.
        - pop_weights_concentration_fun: str (default="sigmoid"), function to apply to compute the population
            weights concentration. Allowed values are: "sigmoid", "tanh".
            If the value is incorrect it rolls back to the default value ("sigmoid").
        - beta: float (default=None), regulizer hyperparameter in [0,1] radians to mupltiply to n_c in (EQ. 8).
            This is a modification to the original KappaFace function, set it to None if the classic behavior 
            is desired.
        - mem_buf: TODO
        """
        self._num_out_classes = len(class_ids)
        super(KappaVmfSimple, self).__init__(
            AdditiveAngularLossBehavior(kwargs['is_poly_loss']),
            embedding_dim, 
            sphere_radius,
            self._num_out_classes,
            'kappa_simple_vmf_loss',
            margin,
            gt_weights_save_path,
            batch_size
            )
        self.margin_scaler_manager = KappaSimpleMarginScaler(
            classes_freq, 
            gamma, 
            temperature, 
            alpha,
            embedding_dim,
            class_ids,
            pop_weights_concentration_fun,
            beta,
            mem_buf,
            **kwargs
            )
    

    @property
    def is_kappa(self):
        return True
    

    @property
    def mem_buf(self) -> tf.Tensor:
        return self.margin_scaler_manager.mem_buf
    

    def _assign_mem_buf(self, new_mem_buf: np.ndarray) -> None:
        self.margin_scaler_manager.assign_mem_buf(new_mem_buf)
    

    def compute_margin_penalty(
        self, 
        dist_w_embeddings: tf.Tensor, 
        mask: tf.Tensor, 
        **kwargs) -> tf.Tensor:
        """
        This method computes the cosine of the angular margin penalty (logits) for each output class.

        Parameters
        ---------

        - dist_w_embeddings: tf.Tensor, distance (rank-2 tensor) between the 
            weights of the desired layer and the embeddings. 
        - mask: tf.Tensor, mask of zeros and ones with shape=(batch_size, num_out_classes). 
            Ones indicates the ground truth labels.
        - kwargs: python dict, for future implementations needing extra parameters.

        Notes
        ----

        - Computes the cosine of theta plus the margin (cos(theta + margin)).
        Different subclasses may implement different ways to compute this penalty (Kappa based
        losses multiply the margin by a scaler, for example).

        - Internally this method applies tensorflow tf.math.cos and tf.math.acos; both require the inputs
            in radians.
        
        Examples
        -------

        >>> d=tf.constant([0.3,0.2,0.1,0.5])
        ... <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.3, 0.2, 0.1, 0.5], dtype=float32)>
        >>> # 2 output classes
        >>> # margins = [0.33, 0.11]
        >>> ang_loss.compute_margin_penalty(d)
        ...     <tf.Tensor: shape=(2, 4), dtype=float32, numpy=
                array([[0.63      , 0.53000003, 0.43      , 0.83000004],
                       [0.41000003, 0.31      , 0.21000001, 0.61      ]], dtype=float32)>
        >>> # each row is the margin penalty associated with a class
        >>> # first row corresponds to class 0
        >>> # each column corresponds to the i-th sample in the batch

        Returns
        ------

        rank-2 tf.Tensor (logits), 
            The output tensor will have shape=(num_out_classes, batch_size).
            In this case we have a different margin for each output class, so 
            each array will have batch_size elements, where
            each value is the additive angular margin penalty (for a specific class) associated
            with each i-th sample in the batch.
            The result is expressed in radians.
        """
        return self.angle_margin_penalty(
            self.margin_scaler_manager, 
            dist_w_embeddings, 
            self._append_angles_to_hist, 
            mask,
            self.rad_margin)
    

    def _compute_margin_penalty_add2ngt_sub2gt(
        self, 
        dist_w_embeddings: tf.Tensor, 
        mask: tf.Tensor, 
        **kwargs) -> tf.Tensor:
        """
        This method computes the cosine of the angular margin penalty (logits) for each output class.
        It subtracts the margins from the ground truth labels and add it to the non-corresponding classes.

        Parameters
        ---------

        - dist_w_embeddings: tf.Tensor, distance (rank-2 tensor) between the 
            weights of the desired layer and the embeddings. 
        - mask: tf.Tensor, mask of zeros and ones with shape=(batch_size, num_out_classes). 
            Ones indicates the ground truth labels.
        - kwargs: python dict, for future implementations needing extra parameters.

        Notes
        ----

        - Computes the cosine of theta plus or minus the margin depending on mask:
            cos(theta_y_i - margin) and cos(theta_j + margin); where y_i=ground truth label, j=not corresponding labels.

        - Internally this method applies tensorflow tf.math.cos and tf.math.acos; both require the inputs
            in radians.
        
        Examples
        -------

        >>> d=tf.constant([0.3,0.2,0.1,0.5])
        ... <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.3, 0.2, 0.1, 0.5], dtype=float32)>
        >>> # 2 output classes
        >>> # margins = [0.33, 0.11]
        >>> ang_loss.compute_margin_penalty(d)
        ...     <tf.Tensor: shape=(2, 4), dtype=float32, numpy=
                       TODO, dtype=float32)>
        >>> # each row is the margin penalty associated with a class
        >>> # first row corresponds to class 0
        >>> # each column corresponds to the i-th sample in the batch

        Returns
        ------

        rank-2 tf.Tensor (logits), 
            The output tensor will have shape=(num_out_classes, batch_size).
            In this case we have a different margin for each output class, so 
            each array will have batch_size elements, where
            each value is the additive angular margin penalty (for a specific class) associated
            with each i-th sample in the batch.
            The result is expressed in radians.
        """
        theta = tf.math.acos(tf.clip_by_value(dist_w_embeddings, -1, 1)) # arccos(x): -1 <= x <= 1
        self._append_angles_to_hist(theta, mask) # stores angles for plotting
        margins = self.rad_margin * self.margin_scaler_manager.compute_margin_scaler_psi() 
        theta = tf.clip_by_value(theta + margins, 0, np.deg2rad(180.0))
        res = tf.math.cos(theta) # adds margins to all, then we need to subtract.
        return res - (mask * 2 * margins) # subtract margin from corresponding classes.
    

    def update_memory_buffer(self) -> None:
        self.margin_scaler_manager.update_memory_buffer(self.gt_weights)
    

    def get_config(self):
        base_config = super().get_config()
        config = {
            'margin': self.rad_margin,
            'sphere_radius': self.sphere_radius,
            'classes_freq': self.margin_scaler_manager.classes_freq,
            'gamma': self.margin_scaler_manager.gamma,
            'temperature': self.margin_scaler_manager.temperature,
            'alpha': self.margin_scaler_manager.alpha,
            'embedding_dim': self.embedding_dim,
            'class_ids': self.margin_scaler_manager.class_ids,
            'pop_weights_concentration_fun': self.margin_scaler_manager.pop_weights_concentration_fun,
            'beta': self.margin_scaler_manager.beta,
            'mem_buf': None
        }
        return dict(list(base_config.items()) + list(config.items()))
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class KappaVmfTruncatedNewton(KappaBehaviorMixin, BaseAngularLoss):
    """
    """
    def __init__(
        self, 
        margin: float, 
        sphere_radius: int,
        classes_freq: List[int],
        gamma: float, 
        temperature: float, 
        alpha: float,
        embedding_dim: int, 
        class_ids: List[str],
        gt_weights_save_path: str,
        batch_size: int,
        pop_weights_concentration_fun: str = "sigmoid", 
        beta: Optional[float] = None, 
        tau: float = 0.0001,
        id_bessel_fun: int = 2,
        mem_buf: Optional[tf.Tensor] = None,
        **kwargs) -> None:
        """
        Parameters
        ---------

        - margin: float, fixed margin to apply in [0,1] radians values (radians). 
        - sphere_radius: int (default=64), radius of the sphere hyperparameter for this Loss framework.
        - classes_freq: list of ints, each value is the total number of samples in the training dataset
            for a specific class.
        - gamma: float, gamma hyperparameter in [0,1] radians to apply in the scaling factor formula (EQ. 14).
            It balances the contribution of the sample and population concentration.
        - temperature: float, temperature hyperparameter in [0,1] radians.
        - alpha_momentum: float, alpha momentum hyperparameter in [0,1] radians needed in the memory buffer update.
        - mem_buf_shape_zero: int, dimension zero of the memory buffer. it is equal to the output shape
            of the layer before L2 normalization.
        - hypersphere_radius: int (default=64), hypersphere radius hyperparameter (default value is the same as 
            the paper).
        - gt_weights_save_path: str, path where to save the gt_weights.
        - batch_size: int, number of elements in each batch during training.
        - pop_weights_concentration_fun: str (default="sigmoid"), function to apply to compute the population
            weights concentration. Allowed values are: "sigmoid", "tanh".
            If the value is incorrect it rolls back to the default value ("sigmoid").
        - beta: float (default=None), regulizer hyperparameter in [0,1] radians to mupltiply to n_c in (EQ. 8).
            This is a modification to the original KappaFace function, set it to None if the classic behavior 
            is desired.
        - tau: float, threshold error for the truncated newton method.
        - id_bessel_fun: int (default=2), id associated to how to compute the Bessel function.
            Allowed values are: 0 (implementation of (Sra,2011)), 1 (scipy.special.iv), 2 (scipy.special.ive).
            if id_bessel_fun value is not valid it rolls back to the default value.
        - mem_buf: TODO
        """
        self._num_out_classes = len(class_ids)
        super(KappaVmfTruncatedNewton, self).__init__(
            AdditiveAngularLossBehavior(kwargs['is_poly_loss']),
            embedding_dim, 
            sphere_radius,
            self._num_out_classes,
            'kappa_trunc_newton_vmf_loss',
            margin,
            gt_weights_save_path,
            batch_size
            )
        self.margin_scaler_manager = KappaNewtonApproxMarginScaler(
            classes_freq, 
            gamma, 
            temperature, 
            alpha,
            embedding_dim,
            class_ids,
            pop_weights_concentration_fun,
            beta,
            tau,
            id_bessel_fun,
            mem_buf=mem_buf,
            **kwargs)


    @property
    def is_kappa(self):
        return True
    

    @property
    def mem_buf(self) -> tf.Tensor:
        return self.margin_scaler_manager.mem_buf
    

    def _assign_mem_buf(self, new_mem_buf: np.ndarray) -> None:
        self.margin_scaler_manager.assign_mem_buf(new_mem_buf)


    def compute_margin_penalty(
        self, 
        dist_w_embeddings: tf.Tensor, 
        mask: tf.Tensor, 
        **kwargs: Dict[str, Any]) -> tf.Tensor:
        """
        This method computes the cosine of the additive angular margin penalty for each output class.

        Parameters
        ---------

        - dist_w_embeddings: tf.Tensor, distance (rank-2 tensor) between the 
            weights of the desired layer and the embeddings. 
        - mask: not used for this method. Kappa based losses compute a margin for each label.
        - kwargs: python dict, for future implementations needing extra parameters.

        Notes
        ----

        - Computes the cosine of theta plus the margin (cos(theta + margin)).
        Different subclasses may implement different ways to compute this penalty (Kappa based
        losses multiply the margin by a scaler, for example).

        - Internally this method applies tensorflow tf.math.cos and tf.math.acos; both require the inputs
            in radians.

        Examples
        -------

        >>> d=tf.constant([0.3,0.2,0.1,0.5])
        ... <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.3, 0.2, 0.1, 0.5], dtype=float32)>
        >>> # 2 output classes
        >>> # margins = [0.33, 0.11]
        >>> ang_loss.compute_margin_penalty(d)
        ...     <tf.Tensor: shape=(2, 4), dtype=float32, numpy=
                array([[0.63      , 0.53000003, 0.43      , 0.83000004],
                       [0.41000003, 0.31      , 0.21000001, 0.61      ]], dtype=float32)>
        >>> # each row is the margin penalty associated with a class
        >>> # first row corresponds to class 0
        >>> # each column corresponds to the i-th sample in the batch

        Returns
        ------

        - rank-2 tf.Tensor, 
            The output tensor will have shape=(num_out_classes, batch_size).
            In this case we have a different margin for each output class, so 
            each array will have batch_size elements, where
            each value is the additive angular margin penalty (for a specific class) associated
            with each i-th sample in the batch.
            The result is expressed in radians.
        """
        return self.angle_margin_penalty(
            self.margin_scaler_manager, 
            dist_w_embeddings, 
            self._append_angles_to_hist, 
            mask,
            self.rad_margin)


    def update_memory_buffer(self) -> None:
        self.margin_scaler_manager.update_memory_buffer(self.gt_weights)
    

    def get_config(self):
        base_config = super().get_config()
        config = {
            'margin': self.rad_margin,
            'sphere_radius': self.sphere_radius,
            'classes_freq': self.margin_scaler_manager.classes_freq,
            'gamma': self.margin_scaler_manager.gamma,
            'temperature': self.margin_scaler_manager.temperature,
            'alpha': self.margin_scaler_manager.alpha,
            'embedding_dim': self.margin_scaler_manager.embedding_dim,
            'class_ids': self.margin_scaler_manager.class_ids,
            'pop_weights_concentration_fun': self.margin_scaler_manager.pop_weights_concentration_fun,
            'beta': self.margin_scaler_manager.beta,
            'tau': self.margin_scaler_manager.tau,
            'id_bessel_fun': self.margin_scaler_manager.id_bessel_fun,
            'mem_buf': None
        }
        return dict(list(base_config.items()) + list(config.items()))
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UpdatableSphereRadiusMixin():
    """
    Mixin class to store a boolean flag to indicate whether to update the 
    sphere radius in the current iteration during mini-batch training.

    Attributes
    ---------

    - _flag_update_sphere_radius: bool (default=False), flag to indicate whether 
        to update the sphere radius value during training.
    """
    _flag_update_sphere_radius = False


    @property
    def flag_update_sphere_radius(self):
        """
        Returns the boolean flag of the sphere radius.
        """
        return self._flag_update_sphere_radius
    

    def switch_flag_update_sphere_radius(self):
        """
        Changes the sphere radius boolean flag state.
        Call this method at the end of each training epoch in the training
        loop and in the method where the sphere radius gets updated.
        """
        self._flag_update_sphere_radius ^= True 


class AdaCos(KappaBehaviorMixin, UpdatableSphereRadiusMixin, BaseAngularLoss):
    """
    """
    def __init__(
        self, 
        margin: float, 
        classes_freq: List[int],
        embedding_dim: int, 
        class_ids: List[str],
        gt_weights_save_path: str,
        batch_size: int,
        threshold_modulating_indicator: float = math.pi / 4,
        **kwargs) -> None:
        """
        Parameters
        ---------

        - margin: float, fixed margin to apply in [0,1] radians values (radians). 
        - classes_freq: list of ints, each value is the total number of samples for a specific class used in training.
        - embedding_dim: int, dimension of the output embeddings (z_i).
        - class_ids: list of str, each is the name of the output class associated with the classes_freq list.
        - gt_weights_save_path: str, path where to save the gt_weights.
        - batch_size: int, number of elements in each batch during training.
        - threshold_modulating_indicator: float (default=math.pi / 4), threshold of the modulating indicator variable (theta median)
            from equation 15 in AdaCos.
        """
        self._num_out_classes = len(class_ids)
        super(SimpleKappaAdaptiveSphere, self).__init__(
            AdditiveAngularLossBehavior(kwargs['is_poly_loss']),
            embedding_dim, 
            math.sqrt(2) * math.log(self._num_out_classes - 1), # see EQ. 15 from AdaCos
            self._num_out_classes,
            'kappa_simple_vmf_adaptive_sphere_loss',
            margin,
            gt_weights_save_path,
            batch_size
            )
        self.threshold_theta_med = tf.constant([threshold_modulating_indicator])
    

    @property
    def is_kappa(self):
        return False
    

    @property
    def mem_buf(self) -> tf.Tensor:
        raise NotImplementedError('Does not have a memory buffer')
    

    def _assign_mem_buf(self, new_mem_buf: np.ndarray) -> None:
        raise NotImplementedError('Does not have a memory buffer')
    

    def compute_margin_penalty(
        self, 
        dist_w_embeddings: tf.Tensor, 
        mask: tf.Tensor, 
        **kwargs) -> tf.Tensor:
        """
        This method computes the cosine of the angular margin penalty (logits) for each output class.

        Parameters
        ---------

        - dist_w_embeddings: tf.Tensor, distance (rank-2 tensor) between the 
            weights of the desired layer and the embeddings. 
        - mask: tf.Tensor, mask of zeros and ones with shape=(batch_size, num_out_classes). 
            Ones indicates the ground truth labels.
        - kwargs: python dict, for future implementations needing extra parameters.

        Notes
        ----

        - Computes the cosine of theta plus the margin (cos(theta + margin)).
        Different subclasses may implement different ways to compute this penalty (Kappa based
        losses multiply the margin by a scaler, for example).

        - Internally this method applies tensorflow tf.math.cos and tf.math.acos; both require the inputs
            in radians.
        
        Examples
        -------

        >>> d=tf.constant([0.3,0.2,0.1,0.5])
        ... <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.3, 0.2, 0.1, 0.5], dtype=float32)>
        >>> # 2 output classes
        >>> # margins = [0.33, 0.11]
        >>> ang_loss.compute_margin_penalty(d)
        ...     <tf.Tensor: shape=(2, 4), dtype=float32, numpy=
                array([[0.63      , 0.53000003, 0.43      , 0.83000004],
                       [0.41000003, 0.31      , 0.21000001, 0.61      ]], dtype=float32)>
        >>> # each row is the margin penalty associated with a class
        >>> # first row corresponds to class 0
        >>> # each column corresponds to the i-th sample in the batch

        Returns
        ------

        rank-2 tf.Tensor (logits), 
            The output tensor will have shape=(num_out_classes, batch_size).
            In this case we have a different margin for each output class, so 
            each array will have batch_size elements, where
            each value is the additive angular margin penalty (for a specific class) associated
            with each i-th sample in the batch.
            The result is expressed in radians.
        """
        new_logits, tmp_sphere_radius = self.angle_margin_penalty_with_adaptive_sphere_radius(
            VoidMarginScaler(), 
            dist_w_embeddings, 
            self._append_angles_to_hist, 
            mask,
            self.threshold_theta_med,
            self.sphere_radius,
            self.rad_margin,
            self.flag_update_sphere_radius,
            0.
        )
        if tmp_sphere_radius is not None: # end of training epoch => update sphere radius
            self.sphere_radius = tmp_sphere_radius
            self.switch_flag_update_sphere_radius() # resets flag update sphere radius
        return new_logits


    def update_memory_buffer(self) -> None:
        pass
    

    def get_config(self):
        config = super().get_config()
        if isinstance(self.sphere_radius, tf.Tensor):
            sphere_radius = float(self.sphere_radius.numpy()[0])
        else: 
            sphere_radius = self.sphere_radius
        config = {
            'margin': self.rad_margin,
            'sphere_radius': sphere_radius,
            'embedding_dim': self.embedding_dim,
            'mem_buf': None,
            'threshold_theta_med': float(self.threshold_theta_med.numpy()[0])
        }
        return config
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SimpleKappaAdaptiveSphere(KappaBehaviorMixin, UpdatableSphereRadiusMixin, BaseAngularLoss):
    """
    Custom combination of AdaCos and KappaFace losses. The idea is that AdaCos 
    takes care of dynamically adpting the sphere radius, while KappaFace loss dynamically
    adapts the margins. AdaCos does not apply margins and states it does not need them.
    But KappaFace loss is a later work so this class aims at combining the two works anyway.

    Attributes
    ---------

    - flag_update_sphere_radius: bool (initialized to False), flag to indicate whether or 
        not to update the sphere radius during training. 
        According to AdaCos equation 13, we need
        to update the sphere radius only at the end of each training epoch.
    """
    def __init__(
        self, 
        margin: float, 
        classes_freq: List[int],
        gamma: float, 
        temperature: float, 
        alpha: float,
        embedding_dim: int, 
        class_ids: List[str],
        gt_weights_save_path: str,
        batch_size: int,
        pop_weights_concentration_fun: str = "sigmoid", 
        beta: Optional[float] = None, 
        mem_buf: Optional[tf.Tensor] = None,
        threshold_modulating_indicator: float = math.pi / 4,
        **kwargs) -> None:
        """
        Parameters
        ---------

        - margin: float, fixed margin to apply in [0,1] radians values (radians). 
        - classes_freq: list of ints, each value is the total number of samples for a specific class used in training.
        - gamma: float, gamma hyperparameter in [0,1] radians to apply in the scaling factor formula (EQ. 14).
            It balances the contribution of the sample and population concentration.
        - temperature: float, temperature hyperparameter in [0,1] radians.
        - alpha: float, alpha momentum hyperparameter in [0,1] radians needed in the memory buffer update.
        - embedding_dim: int, dimension of the output embeddings (z_i).
        - class_ids: list of str, each is the name of the output class associated with the classes_freq list.
        - gt_weights_save_path: str, path where to save the gt_weights.
        - batch_size: int, number of elements in each batch during training.
        - pop_weights_concentration_fun: str (default="sigmoid"), function to apply to compute the population
            weights concentration. Allowed values are: "sigmoid", "tanh".
            If the value is incorrect it rolls back to the default value ("sigmoid").
        - beta: float (default=None), regulizer hyperparameter in [0,1] radians to mupltiply to n_c in (EQ. 8).
            This is a modification to the original KappaFace function, set it to None if the classic behavior 
            is desired.
        - mem_buf: TODO
        - threshold_modulating_indicator: float (default=math.pi / 4), threshold of the modulating indicator variable (theta median)
            from equation 15 in AdaCos.
        """
        self._num_out_classes = len(class_ids)
        super(SimpleKappaAdaptiveSphere, self).__init__(
            AdditiveAngularLossBehavior(kwargs['is_poly_loss']),
            embedding_dim, 
            math.sqrt(2) * math.log(self._num_out_classes - 1), # see EQ. 15 from AdaCos
            self._num_out_classes,
            'kappa_simple_vmf_adaptive_sphere_loss',
            margin,
            gt_weights_save_path,
            batch_size
            )
        self.margin_scaler_manager = KappaSimpleMarginScaler(
            classes_freq, 
            gamma, 
            temperature, 
            alpha,
            embedding_dim,
            class_ids,
            pop_weights_concentration_fun,
            beta,
            mem_buf,
            **kwargs
            )
        self.threshold_theta_med = tf.constant([threshold_modulating_indicator])
    

    @property
    def is_kappa(self):
        return True
    

    @property
    def mem_buf(self) -> tf.Tensor:
        return self.margin_scaler_manager.mem_buf
    

    def _assign_mem_buf(self, new_mem_buf: np.ndarray) -> None:
        self.margin_scaler_manager.assign_mem_buf(new_mem_buf)
    

    def compute_margin_penalty(
        self, 
        dist_w_embeddings: tf.Tensor, 
        mask: tf.Tensor, 
        **kwargs) -> tf.Tensor:
        """
        This method computes the cosine of the angular margin penalty (logits) for each output class.

        Parameters
        ---------

        - dist_w_embeddings: tf.Tensor, distance (rank-2 tensor) between the 
            weights of the desired layer and the embeddings. 
        - mask: tf.Tensor, mask of zeros and ones with shape=(batch_size, num_out_classes). 
            Ones indicates the ground truth labels.
        - kwargs: python dict, for future implementations needing extra parameters.

        Notes
        ----

        - Computes the cosine of theta plus the margin (cos(theta + margin)).
        Different subclasses may implement different ways to compute this penalty (Kappa based
        losses multiply the margin by a scaler, for example).

        - Internally this method applies tensorflow tf.math.cos and tf.math.acos; both require the inputs
            in radians.
        
        Examples
        -------

        >>> d=tf.constant([0.3,0.2,0.1,0.5])
        ... <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.3, 0.2, 0.1, 0.5], dtype=float32)>
        >>> # 2 output classes
        >>> # margins = [0.33, 0.11]
        >>> ang_loss.compute_margin_penalty(d)
        ...     <tf.Tensor: shape=(2, 4), dtype=float32, numpy=
                array([[0.63      , 0.53000003, 0.43      , 0.83000004],
                       [0.41000003, 0.31      , 0.21000001, 0.61      ]], dtype=float32)>
        >>> # each row is the margin penalty associated with a class
        >>> # first row corresponds to class 0
        >>> # each column corresponds to the i-th sample in the batch

        Returns
        ------

        rank-2 tf.Tensor (logits), 
            The output tensor will have shape=(num_out_classes, batch_size).
            In this case we have a different margin for each output class, so 
            each array will have batch_size elements, where
            each value is the additive angular margin penalty (for a specific class) associated
            with each i-th sample in the batch.
            The result is expressed in radians.
        """
        new_logits, tmp_sphere_radius = self.angle_margin_penalty_with_adaptive_sphere_radius(
            self.margin_scaler_manager, 
            dist_w_embeddings, 
            self._append_angles_to_hist, 
            mask,
            self.threshold_theta_med,
            self.sphere_radius,
            self.rad_margin,
            self.flag_update_sphere_radius
        )
        if tmp_sphere_radius is not None: # end of training epoch => update sphere radius
            self.sphere_radius = tmp_sphere_radius
            self.switch_flag_update_sphere_radius() # resets flag update sphere radius
        return new_logits


    def update_memory_buffer(self) -> None:
        self.margin_scaler_manager.update_memory_buffer(self.gt_weights)
    

    def get_config(self):
        config = super().get_config()
        if isinstance(self.sphere_radius, tf.Tensor) :
            if len(self.sphere_radius.shape) == 1:
                sphere_radius = float(self.sphere_radius.numpy()[0])
            else:
                sphere_radius = float(self.sphere_radius.numpy()) # zero length tensor's shape
        else: 
            sphere_radius = self.sphere_radius
        config = {
            'margin': self.rad_margin,
            'sphere_radius': sphere_radius,
            'classes_freq': self.margin_scaler_manager.classes_freq,
            'gamma': self.margin_scaler_manager.gamma,
            'temperature': self.margin_scaler_manager.temperature,
            'alpha': self.margin_scaler_manager.alpha,
            'embedding_dim': self.embedding_dim,
            'class_ids': self.margin_scaler_manager.class_ids,
            'pop_weights_concentration_fun': self.margin_scaler_manager.pop_weights_concentration_fun,
            'beta': self.margin_scaler_manager.beta,
            'mem_buf': None,
            'threshold_theta_med': float(self.threshold_theta_med.numpy()[0])
        }
        return config
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TruncNewtonKappaAdaptiveSphere(KappaBehaviorMixin, UpdatableSphereRadiusMixin, BaseAngularLoss):
    """
    Custom combination of AdaCos and KappaFace losses (with the truncated Newton method modification). 
    The idea is that AdaCos 
    takes care of dynamically adpting the sphere radius, while KappaFace loss dynamically
    adapts the margins. AdaCos does not apply margins and states it does not need them.
    But KappaFace loss is a later work so this class aims at combining the two works anyway.
    """
    def __init__(
        self, 
        margin: float, 
        classes_freq: List[int],
        gamma: float, 
        temperature: float, 
        alpha: float,
        embedding_dim: int, 
        class_ids: List[str],
        gt_weights_save_path: str,
        batch_size: int,
        pop_weights_concentration_fun: str = "sigmoid", 
        beta: Optional[float] = None, 
        mem_buf: Optional[tf.Tensor] = None,
        tau: float = 0.0001,
        id_bessel_fun: int = 2,
        threshold_modulating_indicator: float = math.pi / 4,
        **kwargs) -> None:
        """
        Parameters
        ---------

        - margin: float, fixed margin to apply in [0,1] radians values (radians). 
        - classes_freq: list of ints, each value is the total number of samples for a specific class used in training.
        - gamma: float, gamma hyperparameter in [0,1] radians to apply in the scaling factor formula (EQ. 14).
            It balances the contribution of the sample and population concentration.
        - temperature: float, temperature hyperparameter in [0,1] radians.
        - alpha: float, alpha momentum hyperparameter in [0,1] radians needed in the memory buffer update.
        - embedding_dim: int, dimension of the output embeddings (z_i).
        - class_ids: list of str, each is the name of the output class associated with the classes_freq list.
        - gt_weights_save_path: str, path where to save the gt_weights.
        - batch_size: int, number of elements in each batch during training.
        - pop_weights_concentration_fun: str (default="sigmoid"), function to apply to compute the population
            weights concentration. Allowed values are: "sigmoid", "tanh".
            If the value is incorrect it rolls back to the default value ("sigmoid").
        - beta: float (default=None), regulizer hyperparameter in [0,1] radians to mupltiply to n_c in (EQ. 8).
            This is a modification to the original KappaFace function, set it to None if the classic behavior 
            is desired.
        - mem_buf: TODO
        - tau: float, threshold error for the truncated newton method.
        - id_bessel_fun: int (default=2), id associated to how to compute the Bessel function.
            Allowed values are: 0 (implementation of (Sra,2011)), 1 (scipy.special.iv), 2 (scipy.special.ive).
            if id_bessel_fun value is not valid it rolls back to the default value.
        - threshold_modulating_indicator: float (default=math.pi / 4), threshold of the modulating indicator variable (theta median)
            from equation 15 in AdaCos.
        """
        self._num_out_classes = len(class_ids)
        super(TruncNewtonKappaAdaptiveSphere, self).__init__(
            AdditiveAngularLossBehavior(kwargs['is_poly_loss']),
            embedding_dim, 
            math.sqrt(2) * math.log(self._num_out_classes - 1), # see EQ. 15 from AdaCos
            self._num_out_classes,
            'kappa_trunc_newton_vmf_adaptive_sphere_loss',
            margin,
            gt_weights_save_path,
            batch_size
            )
        self.margin_scaler_manager = KappaNewtonApproxMarginScaler(
            classes_freq, 
            gamma, 
            temperature, 
            alpha,
            embedding_dim,
            class_ids,
            pop_weights_concentration_fun,
            beta,
            tau,
            id_bessel_fun,
            mem_buf=mem_buf,
            **kwargs)
        self.threshold_theta_med = tf.constant([threshold_modulating_indicator])
    

    @property
    def is_kappa(self):
        return True
    

    @property
    def mem_buf(self) -> tf.Tensor:
        return self.margin_scaler_manager.mem_buf
    

    def _assign_mem_buf(self, new_mem_buf: np.ndarray) -> None:
        self.margin_scaler_manager.assign_mem_buf(new_mem_buf)
    

    def compute_margin_penalty(
        self, 
        dist_w_embeddings: tf.Tensor, 
        mask: tf.Tensor, 
        **kwargs) -> tf.Tensor:
        """
        This method computes the cosine of the angular margin penalty (logits) for each output class.

        Parameters
        ---------

        - dist_w_embeddings: tf.Tensor, distance (rank-2 tensor) between the 
            weights of the desired layer and the embeddings. 
        - mask: tf.Tensor, mask of zeros and ones with shape=(batch_size, num_out_classes). 
            Ones indicates the ground truth labels.
        - kwargs: python dict, for future implementations needing extra parameters.

        Notes
        ----

        - Computes the cosine of theta plus the margin (cos(theta + margin)).
        Different subclasses may implement different ways to compute this penalty (Kappa based
        losses multiply the margin by a scaler, for example).

        - Internally this method applies tensorflow tf.math.cos and tf.math.acos; both require the inputs
            in radians.
        
        Examples
        -------

        >>> d=tf.constant([0.3,0.2,0.1,0.5])
        ... <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.3, 0.2, 0.1, 0.5], dtype=float32)>
        >>> # 2 output classes
        >>> # margins = [0.33, 0.11]
        >>> ang_loss.compute_margin_penalty(d)
        ...     <tf.Tensor: shape=(2, 4), dtype=float32, numpy=
                array([[0.63      , 0.53000003, 0.43      , 0.83000004],
                       [0.41000003, 0.31      , 0.21000001, 0.61      ]], dtype=float32)>
        >>> # each row is the margin penalty associated with a class
        >>> # first row corresponds to class 0
        >>> # each column corresponds to the i-th sample in the batch

        Returns
        ------

        rank-2 tf.Tensor (logits), 
            The output tensor will have shape=(num_out_classes, batch_size).
            In this case we have a different margin for each output class, so 
            each array will have batch_size elements, where
            each value is the additive angular margin penalty (for a specific class) associated
            with each i-th sample in the batch.
            The result is expressed in radians.
        """
        new_logits, tmp_sphere_radius = self.angle_margin_penalty_with_adaptive_sphere_radius(
            self.margin_scaler_manager, 
            dist_w_embeddings, 
            self._append_angles_to_hist, 
            mask,
            self.threshold_theta_med,
            self.sphere_radius,
            self.rad_margin,
            self.flag_update_sphere_radius
        )
        if tmp_sphere_radius is not None: # end of training epoch => update sphere radius
            self.sphere_radius = tmp_sphere_radius
            self.switch_flag_update_sphere_radius() # resets flag update sphere radius
        return new_logits


    def update_memory_buffer(self) -> None:
        self.margin_scaler_manager.update_memory_buffer(self.gt_weights)
    

    def get_config(self):
        config = super().get_config()
        if isinstance(self.sphere_radius, tf.Tensor):
            if isinstance(self.sphere_radius.numpy(), np.ndarray):
                sphere_radius = float(self.sphere_radius.numpy()[0])
            else:
                sphere_radius = float(self.sphere_radius.numpy())
        else: 
            sphere_radius = self.sphere_radius
        config = {
            'margin': self.rad_margin,
            'sphere_radius': sphere_radius,
            'classes_freq': self.margin_scaler_manager.classes_freq,
            'gamma': self.margin_scaler_manager.gamma,
            'temperature': self.margin_scaler_manager.temperature,
            'alpha': self.margin_scaler_manager.alpha,
            'embedding_dim': self.embedding_dim,
            'class_ids': self.margin_scaler_manager.class_ids,
            'pop_weights_concentration_fun': self.margin_scaler_manager.pop_weights_concentration_fun,
            'beta': self.margin_scaler_manager.beta,
            'tau': self.margin_scaler_manager.tau,
            'id_bessel_fun': self.margin_scaler_manager.id_bessel_fun,
            'mem_buf': None,
            'threshold_theta_med': float(self.threshold_theta_med.numpy()[0])
        }
        return config
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)