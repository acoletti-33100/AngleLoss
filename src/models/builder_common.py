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
from typing import Any

import tensorflow as tf
from tensorflow import keras


def create_optim_for_dml(data: Dict[str, Any]):
    """
    Instantiates an optimizer depending on the information stored in data.
    Use this method for creating the optimier for the feature extraction (embeddings) model.
    """
    lr_schedule = data['learningRate']
    if data['exponentialDecayLr'] and data['optimizerDml'] == 'Adam':
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=data['initialLr'],
            decay_steps=int(data['trainSize'] / data['batchSize']),
            decay_rate=(data['finalLr'] / data['initialLr']) ** (1 / data['epochs']),
            staircase=True
        )
    elif data['hyperbolicDecayLr'] and data['optimizerDml'] == 'Adam':
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps= int(data['trainSize'] / data['batchSize']) * data['stepDecrease'],
            decay_rate=1,
            staircase=False
            )
    if data['optimizerDml'] == 'Adam':
        optim = tf.keras.optimizers.Adam(lr_schedule)
    else:
        optim = tf.keras.optimizers.Nadam(lr_schedule, beta_1=0.975, epsilon=1e-08)
    return optim
    

def create_optim(data: Dict[str, Any]):
    """
    Instantiates an optimizer depending on the information stored in data.
    Use this method for classification tasks (not for DML with feature extraction).
    """
    lr_schedule = data['learningRate']
    if data['optimizer'] == 'Adadelta': # do not use Adadelta for ftrs extraction
        optim = tf.keras.optimizers.Adadelta(lr_schedule, epsilon=1e-06)
    elif data['exponentialDecayLr'] and data['optimizer'] == 'Adam':
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=data['initialLr'],
            decay_steps=int(data['trainSize'] / data['batchSize']),
            decay_rate=(data['finalLr'] / data['initialLr']) ** (1 / data['epochs']),
            staircase=True
        )
    elif data['hyperbolicDecayLr'] and data['optimizer'] == 'Adam':
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=int(data['trainSize'] / data['batchSize']) * data['stepDecrease'],
            decay_rate=1,
            staircase=False
            )
    elif data['optimizer'] == 'Adam':
        optim = tf.keras.optimizers.Adam(lr_schedule)
    else:
        optim = tf.keras.optimizers.Nadam(lr_schedule, beta_1=0.975, epsilon=1e-08)
    return optim


def compile_embeddings_extractor(model, data):
    """
    Compiles a DML features vector extractor with an angular loss.
    """
    model.compile(
        optimizer=create_optim_for_dml(data),
        loss=data['angularLoss']
        )
    model.summary()


# TODO
# find alternative to ifs wall.
# move lr over time change to tf.keras.callbacks.LearningRateScheduler (so it works for Nadam too).
def compile_model(model, data):
    """

    Parameters
    ---------

    - model: tensorflow.keras model to compile.
    - data: python dict with the following keys:
        - optimizer: string, name of the tensorflow optimizer to apply.
        - learningRate: learning_rate parameter for the model's optimizer, applied if neither learning rate scheduler are used.
        - epochs: int, total number of epochs to run the model.
        - trainSize: int, number of training samples.
        - batchSize: int, batch size.
        - initialLr: float, initial learning rate for the exponential decay.
        - finalLr: float, final learning rate for the exponential decay.
        - stepDecrease: int, step number at which to apply the reduction of the learning rate during the hyperbolic learning rate.
        - exponentialDecayLr: boolean flag, if true it applies an exponential decay scheduler for the learning rate.
        - hyperbolicDecayLr: boolean flag, if true it applies a hyperbolic decay scheduler for the learning rate.
    """
    metrics = [
        keras.metrics.SparseCategoricalAccuracy()
    ]
    on_compile_model(model, data, metrics)


def on_compile_model(model, data, metrics):
    """
    Helper function to compile all created models with the same metrics and optimizer.
    The five metrics are: 
        - sparseCategoricalAccuracy 
        - categorical: true negatives/positives, false negatives/positives.
    It applies a SparseCategoricalCrossentropy as loss, it uses the default parameters for the optimizer.
    At the end it prints the summary of the model.

    Notes
    ----

    - if both flags are set to true then it only applies the exponential decay.

    - The hyperbolic learning rate schedulerer decrease of the learning rate to 1/2 of the base rate 
    at epoch number step_decrease=5, 1/3 at epoch number 10 and so on.
    

    Parameters
    ---------

    - model: tensorflow.keras model to compile.
    - data: python dict with the following keys:
        - optimizer: string, name of the tensorflow optimizer to apply.
        - learningRate: learning_rate parameter for the model's optimizer, applied if neither learning rate scheduler are used.
        - epochs: int, total number of epochs to run the model.
        - trainSize: int, number of training samples.
        - batchSize: int, batch size.
        - initialLr: float, initial learning rate for the exponential decay.
        - finalLr: float, final learning rate for the exponential decay.
        - stepDecrease: int, step number at which to apply the reduction of the learning rate during the hyperbolic learning rate.
        - exponentialDecayLr: boolean flag, if true it applies an exponential decay scheduler for the learning rate.
        - hyperbolicDecayLr: boolean flag, if true it applies a hyperbolic decay scheduler for the learning rate.
        - lossFun: string, name of the loss function to compile the model with.

    External links
    -------------

    - exponenatial decay example: https://stackoverflow.com/questions/61552475/properly-set-up-exponential-decay-of-learning-rate-in-tensorflow
    """
    optim = create_optim(data)
    if data['lossFun'] == 'sparse_categorical_crossentropy':
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) # last FC layer is Dense()
    else:
        raise NotImplementedError('still to implement other losses')

    model.compile(
        optimizer=optim, 
        loss=loss,
        metrics=metrics
        )
    model.summary()


def multi_input_concat_layers(input_shapes):
    """
    Instantiates multiple input layers and concatenates
    them together along "axis=1" (the shape must be the same except for 
    the first dimension).
    Each input layer is named: "input_{}x{}".format(shape[0], shape[1]),
    where "shape" is an element inside "input_shapes".

    Parameters
    ---------

    - input_shapes: list of (n,m,k), where n,m,k are ints and represent the input shape of the net as
        layers.Input(shape=input_shape). "k" is the number of channels, "n" is the height and "m" is
        the width of an image. All values must be equal along all axis for each input_shape except for
        axis 2 (eg: "[(None, 154, 22, 1), (None, 154, 232, 1)]").

    Returns
    ------

    - concatenation (along axis 2) of the input layers.
    - list of the input layers.
    """
    inputs = []
    for in_shape in input_shapes:
        inputs.append(keras.layers.Input(shape=in_shape, name='input_{}x{}'.format(in_shape[0], in_shape[1])))
    return keras.layers.Concatenate(axis=2)(inputs), inputs