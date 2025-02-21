# Example with ArcLoss with MNIST dataset.
# code for this example is partially inspired by: https://www.tensorflow.org/datasets/keras_example
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Dict
from typing import Any
from typing import List

from src.models.builder_model import build_dml_classifier
from src.models.builder_model import build_dml_embedding_extractor
from src.models.builder_model import build_dml_embedding_plot_on_unit_sphere

from src.utils.classification import fit_dml_ftrs_extractor
from src.utils.classification import do_classification
from src.utils.classification import predict_and_evaluate

from src.models.angle_margin.losses import ArcFaceLoss
from src.models.angle_margin.losses import KappaVmfSimple
from src.models.angle_margin.losses import SimpleKappaAdaptiveSphere
from src.models.angle_margin.losses import TruncNewtonKappaAdaptiveSphere
from src.models.angle_margin.losses import KappaVmfTruncatedNewton


def normalize_image(image, label):
    return tf.cast(image, tf.float32) / 255., label


def create_angular_loss(data: Dict[str, Any], ang_margin_name: str, class_ids: List[str], class_freq_train: List[int]) -> None:
    """
    Creates an Angular Loss object.

    Parameters
    ---------

    data: python dict,
    ang_margin_name: string, name of the angular loss to apply.
    class_ids: list of strings, each is the name of an output class.
    class_freq_train:list of ints, each value is the number of occurrences of each class in the training dataset.

    """
    if ang_margin_name == 'arc':
         return ArcFaceLoss(
             data['embeddingDim'],
             len(class_ids),
             data['margin'],
             data['sphereRadius'],
             data['embeddingsSavePath'],
             data['batchSize'],
             is_poly_loss=data['polyLoss']
         )
    elif ang_margin_name == 'simplekappa':
         return KappaVmfSimple(
             data['margin'],
             data['sphereRadius'],
             class_freq_train,
             data['gamma'],
             data['temperature'],
             data['alpha'],
             data['embeddingDim'],
             class_ids,
             data['embeddingsSavePath'],
             data['batchSize'],
             beta=data['beta'],
             is_poly_loss=data['polyLoss']
         )
    elif ang_margin_name == 'adacossimplekappa':
         return SimpleKappaAdaptiveSphere(
             data['margin'],
             class_freq_train,
             data['gamma'],
             data['temperature'],
             data['alpha'],
             data['embeddingDim'],
             class_ids,
             data['embeddingsSavePath'],
             data['batchSize'],
             beta=data['beta'],
             is_poly_loss=data['polyLoss']
         )
    elif ang_margin_name == 'adacostruncnewtonkappa':
         return TruncNewtonKappaAdaptiveSphere(
             data['margin'],
             class_freq_train,
             data['gamma'],
             data['temperature'],
             data['alpha'],
             data['embeddingDim'],
             class_ids,
             data['embeddingsSavePath'],
             data['batchSize'],
             beta=data['beta'],
             tau=data['tau'],
             id_bessel_fun=data['idBesselFun'],
             is_poly_loss=data['polyLoss']
         )
    else:
         return KappaVmfTruncatedNewton(
             data['margin'],
             data['sphereRadius'],
             class_freq_train,
             data['gamma'],
             data['temperature'],
             data['alpha'],
             data['embeddingDim'],
             class_ids,
             data['embeddingsSavePath'],
             data['batchSize'],
             beta=data['beta'],
             tau=data['tau'],
             id_bessel_fun=data['idBesselFun'],
             is_poly_loss=data['polyLoss']
         )


def main():
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test[:50%]', 'test[50%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    input_shape = ds_train.take(1).as_numpy_iterator()[0][0].shape
    data_loss = {
             'margin': 0.4,
             'sphereRadius': 2.9,
             'gamma': 0.3,
             'temperature': 0.42,
             'alpha': 0.45,
             'embeddingDim': 64,
             'embeddingsSavePath': 'logs/weights.npy',
             'batchSize': 64,
             'beta': None,
             'tau': 1e-07,
             'idBesselFun': 1,
             'polyLoss': False

    }
    class_ids = None
    class_freq_train = None
    ang_loss = create_angular_loss(data_loss, 'arc', class_ids, class_freq_train)
    data = {'do_normalize': True, 'AngularLoss': ang_loss}
    ds_train = ds_train.map(
        normalize_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_train.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    # val pipeline preparation
    ds_val = ds_val.map(
        normalize_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_val = ds_val.batch(128)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
    # test pipeline preparation
    ds_test = ds_test.map(
        normalize_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    # create embeddings model (features extractor).
    embeddings_model = build_dml_embedding_extractor(ds_train, ds_info.features['label'].num_classes, input_shape, data)
    # train embeddings model.
    # freeze the embeddings model, add it to a new model with a softmax to classify data and train it.
    # test the classification results.


main()