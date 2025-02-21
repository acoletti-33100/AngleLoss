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
from typing import Union
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras import Model
from tensorflow.keras import models
from tensorflow.keras import Input

from src.models.builder_common import compile_model
from src.models.builder_common import compile_embeddings_extractor


def normalize_input(
        train_ds: Union[tf.data.Dataset, np.ndarray],
        input_shape: Tuple[int, int],
        flag_l2_norm: bool = False) -> Tuple[preprocessing.Normalization, Input]:
    """
    Creates the input layer and normalizes the inputs.
    This function applies a scalar normalization thanks to "axis=None" in "preprocessing.Normalization".

    Parameters
    ---------

    - train_ds: tf.data.Dataset or numpy array of the samples.
    - input_shape: tuple of ints, input shape to assign to "keras.layers.Input".
    - flag_l2_norm: bool (default=False), whether to apply a L2_normalization to the train_ds data.

    Returns
    ------

    - tf.keras.layers created by a "preprocessing.Normalization" layer with "axis=None".
    - keras.Input layer (use it to instantiate a model).
    """
    normalizer = preprocessing.Normalization(axis=None)
    if isinstance(train_ds, tf.data.Dataset):
        normalizer.adapt(train_ds.map(lambda x, _: x))
    else:
        normalizer.adapt(train_ds)

    cnn_input = Input(shape=input_shape)
    x = normalizer(cnn_input)
    return x, cnn_input


# TODO
# load the trained layers of "build_dml_embedding_extractor" except for the last Dense and the stateless L2 norm final
# layer. Freeze and add: "x=Dense(3, use_bias=False)(frozen); outputs=L2Norm(..)(x)".
def build_dml_embedding_plot_on_unit_sphere(
        train_ds: Union[tf.data.Dataset, np.ndarray],
        out_class_count: int,
        input_shape: Tuple[int, int],
        data: Dict[str, Any]) -> Model:
    """
    Creates a model used ONLY to plot the data on the unit sphere.
    The embedding dimension (output shape of this model) is set to 3 and 
    each axis correspond to the x,y,z coordinates of the sphere.
    This model only makes sense for a low number of classes.
    Because if the output classes are too many, the image becomes unwatchable, it is 
    useful for analyzing the behaviour of [2, 8] classes which are deemed similiar with the 
    DML classifier (frozen embeddings extractor + Dense(num_entero_classes)) or we want 
    to further inspect their embedding formation.

    Parameters
    ---------

    train_ds: Union[tf.data.Dataset, np.ndarray], training dataset to normalize the input
        with a keras preprocessing layer.
    - out_class_count:
    input_shape: Tuple[int, int], input shape of the model.
    data: Dict[str, Any],

    Returns
    ------

    keras Model to extract the embeddings for classification tasks with a DML loss framework.
    """
    if data['preprocessingNorm']:  # to test different normalizations on the input
        preprocessed_x, cnn_input = normalize_input(train_ds, input_shape, True)
    else:
        cnn_input = Input(shape=input_shape)
        preprocessed_x = cnn_input

    y = layers.Conv2D(8, 3, padding='causal', dilation_rate=8, kernel_initializer=HeNormal(), trainable=False)(
        preprocessed_x)
    y = layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.11), trainable=False)(y)
    y = layers.Conv2D(16, 3, padding='causal', dilation_rate=2, kernel_initializer=HeNormal(), trainable=False)(y)
    y = layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.11), trainable=False)(y)
    y = layers.Conv2D(32, 3, padding='causal', dilation_rate=8, kernel_initializer=HeNormal(), trainable=False)(y)
    y = layers.BatchNormalization(epsilon=1e-6, momentum=0.95, trainable=False)(y)
    y = layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25), trainable=False)(y)
    y = layers.GlobalAveragePooling2D(trainable=False)(y)

    x = layers.LSTM(32, trainable=False)(preprocessed_x)

    x = layers.Concatenate(trainable=False)([x, y])
    outputs = layers.Dense(3, use_bias=False)(x)

    model = Model(inputs=cnn_input, outputs=outputs, name='AAM-LSTM-FCN-EMB-EXTRACTOR-3DS')
    compile_embeddings_extractor(model, data)
    # model.load_weights(data['embeddingsSavePath'], by_name=True, skip_mismatch=True)
    return model


def build_dml_embedding_extractor(
        train_ds: Union[tf.data.Dataset, np.ndarray],
        out_class_count: int,
        input_shape: Tuple[int, int],
        data: Dict[str, Any]) -> Model:
    """
    Backbone network to extract the embedding in the DML pipeline (trained with Angle Loss, this is the
    feature extractor).

    Parameters
    ---------

    train_ds: Union[tf.data.Dataset, np.ndarray], training dataset to normalize the input
        with a keras preprocessing layer.
    embedding_dim: int (a good value might be 64 for signals), dimension of the output embedding 
        (KappaFace, ArcFace, .. set it to 512 for IMAGES).
    input_shape: Tuple[int, int], input shape of the model.
    data: Dict[str, Any], contains keys: 
        - 'do_normalize': bool, apply normalization or not as a preprocessing layer.
        - 'AngularLoss': src.angle_margin.losses.BaseAngularLoss, angular loss object.

    Returns
    ------

    keras Model to extract the embeddings for classification tasks with a DML loss framework.
    """
    if data['do_normalize']:  # to test different normalizations on the input
        preprocessed_x, cnn_input = normalize_input(train_ds, input_shape, True)
    else:
        cnn_input = Input(shape=input_shape)
        preprocessed_x = cnn_input

    y = layers.Conv2D(64, 3, padding='causal', dilation_rate=8, kernel_initializer=HeNormal())(preprocessed_x)
    y = layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.11))(y)
    y = layers.Conv2D(128, 3, padding='causal', dilation_rate=2, kernel_initializer=HeNormal())(y)
    y = layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.11))(y)
    y = layers.Conv2D(256, 3, padding='causal', dilation_rate=8, kernel_initializer=HeNormal())(y)
    y = layers.BatchNormalization(epsilon=1e-6, momentum=0.95)(y)
    y = layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25))(y)
    y = layers.GlobalAveragePooling1D()(y)

    x = layers.LSTM(256)(preprocessed_x)

    x = layers.Concatenate()([x, y])  # load here for sphere embeddings
    outputs = layers.Dense(data['embeddingDim'], use_bias=False)(x)

    model = Model(inputs=cnn_input, outputs=outputs, name='AAM-LSTM-FCN-EMB-EXTRACTOR')
    compile_embeddings_extractor(model, data)
    return model


# FOR classification after creating DML ftrs extraction model.
def build_dml_classifier(
        embeddings: Model,
        train_ds: Union[tf.data.Dataset, np.ndarray],
        out_class_count: int,
        input_shape: Tuple[int, int],
        data: Dict[str, Any]) -> Model:
    """
    Actual method to train the DML classifier (trained with cross entropy loss).
    Freezes the embeddings model.
    Append at the end an output layer to classify data.

    Notes
    ----

    The embedding modules are already trained on the training dataset, 
    so the new model needs to freeze (training=False) these layers to avoid retraining them on 
    the same data, while the output layer still needs training on the same training dataset used
    for the embedding extractor model.
    """
    embeddings.trainable = False # the ftrs extractor has already been trained with an angle loss
    cnn_input = Input(shape=input_shape)
    if data['l2NormInputs']:  # to test different normalizations on the input
        preprocessed_x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(cnn_input)
    else:
        preprocessed_x = cnn_input

    x = embeddings(preprocessed_x, training=False)
    x = layers.Dropout(0.1)(x) # a little hope to avoid overfitting
    outputs = layers.Dense(out_class_count)(x)
    model = Model(inputs=cnn_input, outputs=outputs, name='AAM-LSTM-FCN-EMB-CLASSIFIER')
    return model
