# General information, notes, disclaimers
#
# author: A. Coletti
# 
#
#
#
# What if the memory buffer were implemented as a custom callback in tesorflow?
# I need to apply it only during training and to compute it only at the end of each training epoch,
# very callback like behavior indeed!
#
# TODO
# decide what to move to a method in a class and where.
# Isn't the whole psi and margin computation reduceable to a custom regulizer layer?
# It only happens during training and aims at transforming the logits before passing them to the classic
# softmax function.
#
# Summary
# I want to compute the difference between the marginal target logits and the original target logits.
# The marginal target logits are the output of the AngularMargin class and to compute
# them, AngularMargin needs the original target logits and the margin scaler for the sphere radius.
# The original target logits are logits, first and foremost, therefore they are the inputs
# of a Dense layer applying a softmax. The softmax gets in input some logits and transforms 
# them to an output "probability distribution". Now, I am not sure if the original target logits
# should be a call to fc7.get_weights(), where fc7 is a Dense with a softmax as an activation,
# or the output of the L2 normalization of the z_i. But, in ArcFace (see the pseudo code, alg. 1)
# it clearly shows that the original target logits are not the l2 normed z_i but the 
# mx.sym.pick(fc7, gt, axis=1) which may be the output of fc7 or its weights.
#
# A side note about the softmax: the output layer of this model should include a 
# softmax activation function, thus the loss in model.compile() must have the parameter 
# from_logits set to false.
#
#
# ==============================================================================
from typing import Union
from typing import List
from typing import Tuple
from typing import Dict
from typing import Any

import math

import tensorflow as tf
from tensorflow.keras.initializers import Ones
from tensorflow.keras import layers
from tensorflow.keras.regularizers import Regularizer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.keras import Model
from tensorflow.keras import backend as kb

from src.utils.utils import sum_a_b_along_cols

from src.models.angle_margin.margins import MarginScaler


class AngularMargin(layers.Layer):
    """
    Trasforms the inputs logits into marginal logits.
    This class should be used as a Layer in a tf.keras.Model architecture.
    This class encapsulates the angular margin behaviour and needs to be
    inserted in the networks architecture inside a call() method of a class
    which inherits "tf.keras.Model".
    This class aims at transforming the original logits, thus
    obtaining the marginal logits which are then fed to the output layer.
    """

    def __init__(
            self,
            margin_scaler_manager: MarginScaler,
            margin: float,
            sphere_radius: int = 64,
            **kwargs):
        """
        Parameters
        ---------

        - margin_scaler: MarginScaler, instance of a class responsible of computing
            the margin scaler psi.
        - classes_freq: list of ints, list with the number of samples for each class in the 
            training set.
        - margin: float, margin to initialize this class with. 
        - alpha: int, momentum parameter for the memory buffer.
        - sphere_radius: int (default=64), hypersphere's radius hyperparameter.
        """
        super(AngularMargin, self).__init__(**kwargs)
        self.margin_scaler_manager = margin_scaler_manager
        self.margin = margin
        self.sphere_radius = sphere_radius

    def update_memory_buffer(self, z_i: tf.Tensor) -> None:
        """
        Wrapper method to update the memory buffer member in the MarginScaler object.

        Parameters
        ---------

        - z_i: Tensor, tensor of the extracted hidden features vector.
            This tensor must have shape (k,n), where each column represents a feature
            vector, where k depends on the dimension of the hidden extracted vector (512D in KappaFace),
            while n is the number of classes.
        """
        if self.has_memory_buffer():
            self.margin_scaler_manager.update_memory_buffer(z_i)

    def has_memory_buffer(self) -> bool:
        """
        Wrapper method to check if this model contains a memory buffer or not.

        Returns
        ------

        True if this model has a memory buffer (Kappa loss implementation), 
            false otherwise (eg: ArcFace implementation).
        """
        return self.margin_scaler_manager.has_memory_buffer()

    def call(self, inputs: tf.Tensor, sphere_radius: int = 1) -> tf.Tensor:
        """
        trasform the input logits to marginal logits.

        Parameters
        ---------

        - inputs: tf.Tensor, tensor with the current original logits extracted by the network's backbone.
        - sphere_radius: int (default=1), hypersphere's radius hyperparameter. If it's '1', then no rescaling
            is applied in the computation of the margin target logits. Values different from one cause
            a rescaling in the computation of the marginal target logits.

        Notes
        ----

        - For the ArcFace implementation the margin is fixed and margin_scaler_psi is equal to 1 (so fixed margin).
            Instead, for KappaFace or any other adaptive additive margin, margin_scaler_psi is a float.
        
        - The condition should be: 
            0 <= theta + (m*psi) <= math.pi; 
        which is equivalent to:
            -(m*psi) <= theta <= math.pi - (m*psi)

        - Theta should have shape (n, k), where k is the number of output classes
            self.margin_scaler_psi has shape (k,) (number of output classes)

        Returns
        ------

        tf.Tensor of the rescaled and transformed marginal logits.
        """
        theta = tf.math.acos(tf.math.cos(inputs))  # theta = arccos(original_target_logits)
        # print('theta.shape: {}'.format(theta.shape)) # DEBUG
        # print('self.margin_scaler_psi.shape: {}'.format(self.margin_scaler_psi.shape)) # DEBUG
        return tf.math.cos(
            sum_a_b_along_cols(theta, (self.margin * self.margin_scaler_psi))) * sphere_radius  # marginal target logits

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape[0])
        last_dim = input_shape[-1]
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `AngularMargin` '
                             'should be defined. Found `None`.')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.margin_scaler_psi = self.add_weight(  # are you sure it's a layer's variable?
            name='margin_scaler_psi',
            shape=(len(self.margin_scaler_manager.classes_freq),),
            dtype=tf.float32,
            initializer=Ones(),
            trainable=False  # update it with memory buffer during callback on_epoch_end
        )
        self.set_weights([self.margin_scaler_manager.compute_margin_scaler_psi()])  # init margin scaler psi
        # Set input spec
        self.built = True
        tf.print('built AngularMargin')

    def get_config(self):
        config = {
            'memory_buffer': self.margin_scaler_manager.mem_buf,
            'margin': self.margin,
            'sphere_radius': self.sphere_radius
        }
        base_config = super(AngularMargin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MarginalOutputDense(layers.Layer):
    """
    Output layer for additive angular margin loss frameworks (eg: ArcFace, KappaFace).
    This layer is the same as a keras.layers.Dense but it doesn't apply an activation function.
    This layer computes:
        "outputs = sphere_radius * (inputs + (dot(marginal_logits - inputs, kernel) + bias)))"
    """

    def __init__(
            self,
            data: Dict[str, Any],
            mem_buf_shape_zero: int,
            units: int,
            margin_scaler_sphere_radius: int = 1,
            use_bias: bool = True,
            **kwargs):
        """
        Parameters
        ---------

        - units: int, number of units for the dense layer.
        """
        super(MarginalOutputDense, self).__init__(**kwargs)
        margin_scaler, margin, sphere_radius = MarginScaler.create_margin_scaler(data, mem_buf_shape_zero)
        self.angular_margin = AngularMargin(margin_scaler, margin, sphere_radius)
        self.margin_scaler_sphere_radius = margin_scaler_sphere_radius
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True)
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True)

    def call(
            self,
            inputs: tf.Tensor,
            original_outputs: tf.Tensor,
            original_logits: tf.Tensor,
            sphere_radius: int) -> tf.Tensor:
        """
        Computes the difference between the marginal and the original logits, then linearly
        transforms this result and rescale everything by the sphere radius, as:
            "outputs = sphere_radius * (inputs + (dot(marginal_logits - inputs, kernel) + bias)))"

        Parameters
        ---------

        - inputs: tf.Tensor, input tensor. This is the output of previous layers (eg: Conv, LSTM, ..)
            computations. It should not be the output of a Dense(out_class_num) layer.
        - original_outputs: tf.Tensor, "probability distribution" of a Dense layer (fc7 from ArcFace paper).
        - original_logits: tf.Tensor, weights of the fc7 layer like in ArcFace.
        - fc7_layer: keras.layers.Layer, fc7 layer from ArcFace from which to extract the weights to 
            compute the marginal target logits.
        - sphere_radius: int, radius of the sphere by which to rescale the output logits.

        Returns
        ------

        tf.Tensor rescaled of the difference between the marginal and original logits.
        """
        marginal_logits = self.angular_margin(original_logits, self.margin_scaler_sphere_radius)
        self.kernel = marginal_logits - original_logits
        rank = inputs.shape.rank
        marginal_outputs = standard_ops.tensordot(inputs, self.kernel,
                                                  [[rank - 1], [0]])  # compute marginal output probs
        if self.use_bias:
            marginal_outputs = nn_ops.bias_add(marginal_outputs, self.bias)
        return tf.scalar_mul(sphere_radius,
                             original_outputs + marginal_outputs)  # sum output probs, original + marginal

    def get_config(self):
        config = {
            'units': self.units,
            'margin_scaler_sphere_radius': self.margin_scaler_sphere_radius,
            'use_bias': self.use_bias
        }
        base_config = super(MarginalOutputDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SimpleCl(Model):
    def __init__(
            self,
            inputs,
            output_class_shape,
            sphere_radius=64):
        """
        """
        super(SimpleCl, self).__init__(inputs)
        self._output_shape = output_class_shape
        # simple model
        self.simple_conv = layers.Conv1D(5, 2, padding='causal', activation='relu')
        self.l2_norm_sphere = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1) / sphere_radius)
        W = tf.Variable(kb.batch_get_value(self.l2_norm_sphere.weights), trainable=False, name='W_fc7')
        self.interim = layers.Dense(30, activation='relu')
        self.out = layers.Dense(output_class_shape, use_bias=False)
        self.count = 0

    def call(self, inputs, training: bool = None, mask=None) -> tf.Tensor:
        print('call {}'.format(self.count))
        self.count += 1
        if len(self.l2_norm_sphere.weights) > 0 and self.l2_norm_sphere.weights[0]:
            print('before')
            print('l2_norm_sphere.weights[0].shape: {}'.format(self.l2_norm_sphere.weights[0].shape))
            print('interim.weights[0].shape: {}'.format(self.interim.weights[0].shape))
        x = self.simple_conv(inputs)
        x = self.l2_norm_sphere(x)
        x = self.interim(x)
        if len(self.l2_norm_sphere.weights) > 0 and self.l2_norm_sphere.weights[0]:
            print('after')
            print('self.l2_norm_sphere.weights[0].shape: {}'.format(self.l2_norm_sphere.weights[0].shape))
            print('interim.weights[0].shape: {}'.format(self.interim.weights[0].shape))
        return self.out(x)

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}
