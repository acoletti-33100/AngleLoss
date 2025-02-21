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
from typing import Callable
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Dict
from typing import Union
from typing import Any
from typing import List

import os
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import ProgbarLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CallbackList
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adadelta

from sklearn.preprocessing import LabelEncoder

from src.utils.utils import to_string_trim_parent_path
from src.utils.utils import unique_csv_labels
from src.utils.utils import is_json_serializable
from src.utils.utils import find_object_index_in_list
from src.utils.utils import do_nothing

from src.custom_callbacks.custom_callbacks import DebugEmptyCallback
from src.custom_callbacks.custom_callbacks import ConfusionMatrixCallback
from src.custom_callbacks.custom_callbacks import Plot3DSphereCallback
from src.custom_callbacks.custom_callbacks import ScatterPlot2DCallback
from src.custom_callbacks.custom_callbacks import AngleAvgMedianCallback
from src.custom_callbacks.custom_callbacks import RecorderAdaptiveSphereRadius
from src.custom_callbacks.custom_callbacks import TsneCallback
from src.models.angle_margin.losses import BaseAngularLoss
from src.models.angle_margin.losses import SimpleKappaAdaptiveSphere
from src.models.angle_margin.losses import TruncNewtonKappaAdaptiveSphere


def _on_fit(val_ds, y_val, class_names, min_delta, log_path, tensorboard_flag):
    file_writer_cm = tf.summary.create_file_writer(log_path + os.sep + 'cm')
    callbacks = []
    if tensorboard_flag:
        callbacks.append(TensorBoard(log_dir=log_path, histogram_freq=1))
        if val_ds is not None:
            callbacks.append(ConfusionMatrixCallback(val_ds, y_val, class_names, file_writer_cm))
        callbacks.append(EarlyStopping(monitor='loss', min_delta=min_delta, patience=3))
    return log_path, file_writer_cm, callbacks


def do_classification(
            train_ds: tf.data.Dataset,
            val_ds: tf.data.Dataset,
            test_ds: tf.data.Dataset,
            y_val,
            model: Model,
            class_names: List[str],
            log_path: str,
            tensorboard_flag: bool,
            epochs: int,
            min_delta: int,
            callbacks: List[Callback] = []) -> None:
        """
        Internal helper function, common to all classification experiments,
         to perform the actual classification for signal spectra as input.

        Parameters
        ---------

        - train_ds: tf.data.Dataset, tensorflow dataset to train the classifier.
        - val_ds: tf.data.Dataset, tensorflow dataset to validate the classifier.
        - test_ds: tf.data.Dataset, tensorflow dataset to validate the classifier.
        - model: tensorflow model with which to run the current experiment.
        - model_path: string, name of the model's path in the logs directory.
            Example: model_path='shallowCnn', then it saves in
            logs\tf_logs\exp_rs\shallowCnn\current_time the accuracy and loss results.
        - time_path: string, current's experiment time. It indicates that all experiments will
            be run subsequently, therefore do not recompute the current time. This is needed to
            save all experiments with the same folder's name.
            If it's empty, recompute the current time.
            Valid format: "hh-mm-Ddd-mm-yyyy".
        """
        file_writer_cm = tf.summary.create_file_writer(log_path + os.sep + 'cm')
        if tensorboard_flag:
            callbacks.append(TensorBoard(log_dir=log_path, histogram_freq=1))
            if val_ds is not None:
                callbacks.append(ConfusionMatrixCallback(val_ds, y_val, class_names, file_writer_cm))
            callbacks.append(EarlyStopping(monitor='loss', min_delta=min_delta, patience=3))
        if len(tf.config.list_physical_devices('GPU')) > 0:
            device = '/GPU:0'
        else:
            device = '/CPU:0'
        with tf.device(device):
            """
            # <DEBUG>
            tf.debugging.set_log_device_placement(True) # debug, show where ops are done
            tf.debugging.experimental.enable_dump_debug_info(
                "debug-tf",
                tensor_debug_mode="NO_TENSOR", # 'FULL_HEALTH'
                circular_buffer_size=-1
                )
            # </DEBUG>
            """
            # _fit_with_frozen_embeddings(train_ds, val_ds, model, callbacks, log_path)
            # """
            model.fit(
                x=train_ds,
                epochs=epochs,
                validation_data=val_ds,
                callbacks=callbacks
            )
            # """
            dump_model_info(model, log_path)
            if test_ds is not None:
                predict_and_evaluate(model, test_ds, class_names, log_path)
            model.save(log_path + os.sep + 'saved-h5', save_format='h5')
            # model.save(log_path + os.sep + 'saved-tf')


def _fit_with_frozen_embeddings(

            train_ds: tf.data.Dataset,
            val_ds: tf.data.Dataset,
            model: Model,
            callbacks_list: CallbackList,
            tensorboard_flag: bool,
            epochs: int,
            log_path: str
    ):
        """
        Custom training with frozen embeddings to classify data.
        Note: do not compile the model with this method.
        """
        logs = {}
        optim = Adadelta(1.0, epsilon=1e-06)
        loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        train_acc = SparseCategoricalAccuracy()
        val_acc = SparseCategoricalAccuracy()
        callbacks = CallbackList(callbacks_list, model=model)
        if tensorboard_flag:
            hist_prog_callbacks = CallbackList(TensorBoard(log_dir=log_path, histogram_freq=1), add_history=True,
                                                   add_progbar=True, model=model)
        callbacks.on_train_begin(logs)
        hist_prog_callbacks.on_train_begin(logs)
        for epoch in range(epochs):  # loop over epochs
            if not model.stop_training:
                callbacks.on_epoch_begin(epoch, logs)
                hist_prog_callbacks.on_epoch_begin(epoch, logs)
                for step, (x_batch_train, y_batch_train) in enumerate(train_ds):  # loop over batches
                    callbacks.on_train_batch_begin(step, logs)
                    hist_prog_callbacks.on_train_batch_begin(step, logs)
                    with tf.GradientTape() as tape:
                        logits = model(x_batch_train, training=True)
                        loss_value = loss_fn(y_batch_train, logits)
                        grads = tape.gradient(loss_value, model.trainable_weights)
                        optim.apply_gradients(zip(grads, model.trainable_weights))
                        train_acc.update_state(y_batch_train, logits)
                        logs['loss'] = loss_value
                        logs['train_acc'] = train_acc.result()
                    callbacks.on_train_batch_end(step, logs)
                    # hist_prog_callbacks.on_train_batch_end(step, logs) # TODO: bugged
                for x_val, y_val in val_ds:
                    val_logits = model(x_val, training=False)
                    logs['val_{}'.format(loss_fn.name)] = loss_fn(val_logits, y_val)
                    val_acc.update_state(y_val, val_logits)
                    logs['val_acc'] = val_acc.result()
                val_acc.reset_states()
                hist_prog_callbacks.on_epoch_end(epoch, logs)
                callbacks.on_epoch_end(epoch, logs)
            train_acc.reset_states()
        callbacks.on_train_end(logs)
        hist_prog_callbacks.on_train_end(logs)


def dump_model_info(model: Model, log_path: str, labels: List[str], epochs: int, batch_size: int) -> None:
        """
        Writes to a file named "info.json" some useful information about the experiment.
        It writes: epochs, batch size, optimizer, optimizer parameters, batch size.

        Parameters
        ---------

        - model: tensorflow model used in the experiment.
        - log_path: str, path where to save the file without the directory symbol at the end.
         Example: log_path="something\\somethingPath", It will create a file at "something\\somethingPath\\info.json".
        """
        with open(log_path + os.sep + 'info_model.json', 'w') as out_f:
            json.dump(json.loads(model.to_json()), out_f, indent=2, separators=(',', ': '))
        with open(log_path + os.sep + 'info.json', 'w') as out_f:
            info = model.optimizer.get_config()
            info['label def'] = labels
            info['epochs'] = epochs
            info['batch size'] = batch_size
            for k in info.keys():
                if isinstance(info[k], np.float32):
                    info[k] = info[k].item()
            json.dump(info, out_f, indent=2, separators=(',', ': '))


def predict_and_evaluate(
            model: Model,
            test_ds: tf.data.Dataset,
            le: LabelEncoder,
            class_names: List[str],
            log_path: str) -> None:
        """
        Test the model by invoking model.evaluate(*) on both the test and permutation test dataset.
        Plots the confusion matrices for both datasets and prints the loss and accuracy for each sample.

        Parameters
        ---------

        - model: tensorflow trained model.
        - test_ds: tensorflow.data.Dataset, testing dataset.
        - class_names: list of str, each string is a human readeable label.
        - log_path: str, full path where to save an eperiment. For example:
            'logs\\exp_rsi\\DCNNSVGG15-conf0\\21-22-D01-03-21'.
        """
        y_test = []
        y_perm_test = []
        all_labels = le.transform(le.classes_)  # int encoded labels
        tmp_test_ds = test_ds.unbatch()
        """
        permutation test code. Done once it works, not really necessary.
        Uncomment if needed
        """
        for _, true_label in tmp_test_ds.as_numpy_iterator():
            """
            tmp_all_labels = all_labels
            tmp_all_labels = all_labels[all_labels != true_label]
            permutation_label = choice(tmp_all_labels)
            y_perm_test.append(permutation_label)
            """
            y_test.append(true_label)
        on_predict_and_evaluate(
            model,
            test_ds,
            y_test,
            class_names,
            tf.summary.create_file_writer(log_path + os.sep + 'cm_test'),
            '{} evaluation results'.format(model.name),
            log_path
        )
        """
        on_predict_and_evaluate(
            model, 
            test_ds, 
            y_perm_test, 
            class_names, 
            tf.summary.create_file_writer(log_path + os.sep + 'cm_permutation_test'),
            '{} permutation test, evaluation results'.format(model.name),
            )
        """


def on_predict_and_evaluate(
            model: Model,
            ds: tf.data.Dataset,
            y: List[int],
            class_names: List[str],
            fw_cm: Callable,
            title: str,
            logpath: str) -> None:
        """
        Helper method to generalize the evaluation of the test and permutation test sets evaluation
        with a tensorflow trained model.

        Parameters
        ---------

        - model: tensorflow trained model.
        - ds: tensorflow.data.Dataset of the test or permutation test datasets.
        - y: labels of the test or permutation set.
        - class_names: list of strings, each string is a human readeable label.
        - fw_cm: tf.summary.create_file_writer, object to write to file.
        - title: string, title to print before showing the loss and accuracy values for the evaluation.
        """
        callbacks = CallbackList(
            [
                TensorBoard(log_dir=logpath),
                ConfusionMatrixCallback(ds, y, class_names, fw_cm, True)
            ],
            model=model
        )
        res = model.evaluate(ds, callbacks=callbacks)
        with open(logpath + os.sep + 'info_test_model.json', 'w') as out_f:
            res_dict = {}
            for i in range(len(res)):
                res_dict['{}'.format(model.metrics_names[i])] = res[i]
            if isinstance(res, dict):
                for name, value in res.items():
                    res_dict['{}'.format(name)] = '{}'.format(value)
            json.dump(res_dict, out_f, indent=2, separators=(',', ': '))


def do_grads_for_dml_ftrs_extraction(
            inputs: Union[tf.data.Dataset, np.ndarray],
            model: Model,
            angular_loss: BaseAngularLoss,
            labels: tf.Tensor) -> Tuple[Any, Any]:
        """
        Performs a single step of optimization for a features extraction DML model.

        Parameters
        ---------

        - inputs: tf.Tensor, inputs of a single batch of training.
        - model: keras.Model, model which outputs an embeddings.
        - angular_loss: BaseAngularLoss, angular margin loss object to compute the loss for the embedding.
        - labels:  tf.Tensor, true labels of associated with inputs.

        Returns
        ------
        """
        with tf.GradientTape() as tape:
            embeddings = model(inputs, training=True)  # forward pass; shape=(batch_size, 64), embeddings := logits
            loss_value = angular_loss(embeddings, labels)
            grads = tape.gradient(
                loss_value,
                [model.trainable_variables, angular_loss.gt_weights]
            )
            model.optimizer.apply_gradients(zip(grads[0], model.trainable_variables))
            model.optimizer.apply_gradients(zip([grads[1]], [angular_loss.gt_weights]))
            return loss_value


def init_callbacks(
            model: Model,
            y_val,
            log_path: str,
            flags: List[bool],
            data: Dict[str, Any],
            class_names: List[str],
            tensorboard_flag: bool,
            min_delta: int,
            is_custom_fit_loop: bool = True,
            val_ds: Union[None, tf.data.Dataset] = None) -> Union[CallbackList, List[Callback], None]:
        """
        Inits the necessary callbacks.

        Parameters
        ---------

        - model: keras Model to apply the callbacks to.
        - model_path: str,
        - time_path: str,
        - flags: list of bools,
        - data: python dict,
        - is_custom_fit_loop: bool,
        - val_ds: Union[None, tf.data.Dataset]=None) -> List[Callback]:
        """
        _callbacks = []
        file_writer_cm = tf.summary.create_file_writer(log_path + os.sep + 'cm')
        if tensorboard_flag and val_ds is not None:
            _callbacks.append(ConfusionMatrixCallback(val_ds, y_val, class_names, file_writer_cm))
        _callbacks.append(EarlyStopping(
                monitor=data['lossToMonitor'],
                min_delta=min_delta,
                patience=2,
                mode='min',
                # verbose=1,
                # restore_best_weights=True
            ))
        if flags[0]:
            _callbacks.append(
                Plot3DSphereCallback(log_path, class_names, False, data['plotSphereEveryNEpoch']))
        if flags[1]:
            _callbacks.append(
                ScatterPlot2DCallback(log_path, class_names, False, data['plotSphereEveryNEpoch']))
        if flags[2]:
            _callbacks.append(AngleAvgMedianCallback(log_path))
        if flags[3]:
            _callbacks.append(RecorderAdaptiveSphereRadius(log_path))
        if flags[4]:
            _callbacks.append(
                TsneCallback(log_path, class_names, False, data['plotSphereEveryNEpoch']))
        # _callbacks.append(DebugEmptyCallback())
        if len(_callbacks) > 0:
            if is_custom_fit_loop:
                return CallbackList(
                    _callbacks,
                    # add_history=True,
                    # add_progbar=True,
                    model=model)
            else:
                return _callbacks


def fit_dml_ftrs_extractor(
            train_ds: tf.data.Dataset,
            val_ds: tf.data.Dataset,
            model: Model,
            log_path: str,
            data: Dict[str, Any],
            callback_flags: List[str]) -> None:
        """
        Trains a features extractor model (outputs an embedding/z_i).

        Parameters
        ---------

        - train_ds: tf.data.Dataset,
        - val_ds: tf.data.Dataset, here it is called "val_ds" to use the same terminology as tensorflow documentation.
            But this is not a validation dataset but a test dataset, since it is used, during training, to evaluate the model,
            but it "val_ds" does not influence the training parameters of the model. So, do not confuse it with a validation
            dataset, used to optimize the model's hyperparameters. Also, if you decide to keep 2 separate test
            dataset, one to use here and another to evluate on after training and saving to file the trained model, be careful
            to make sure both dataset contain at least 1 sample for each output class used during training, otherwise the
            confusion matrix callback will present some NaN rows (since it doesn't have the data to predict with).
        - model: keras model,
        - model_path: str,
        - time_path: str,
        - data: python dict,
        - callback_flags: list of strings,
        """
        data['lossToMonitor'] = data['angularLoss'].name
        callbacks = init_callbacks(
            model,
            log_path,
            callback_flags,
            data,
            val_ds=None)  # do not want cm during training
        angular_loss = data['angularLoss']  # BaseAngularLoss
        angular_loss.gt_weights_save_path = '{}{}{}'.format(
            log_path,
            os.sep,
            to_string_trim_parent_path(data['embeddingsSavePath'])
        )
        is_sparse = data['isSparse']
        logs = {}
        """
        devices = tf.config.list_physical_devices('GPU')
        if len(devices) > 0:
            device_name = '/GPU:0' 
        else: 
            device_name = '/CPU:0'

        #for d in devices:
            #tf.config.experimental.set_memory_growth(d, True)

        with tf.device(device_name):
            # <DEBUG>
            tf.debugging.set_log_device_placement(True) # debug, show where ops are done
            tf.debugging.experimental.enable_dump_debug_info(
                "debug-tf",
                #tensor_debug_mode='NO_TENSOR', 
                tensor_debug_mode='FULL_HEALTH',
                circular_buffer_size=-1
                )
            # </DEBUG>
        """
        if callbacks is not None:
            """
            print('\n\n--------------\n\n')
            print('<\DEBUG>\n\n')
            for x, y in train_ds:
                print(tf.math.count_nonzero(x))
            print('\n\n--------------\n\n')
            """
            _fit_dml_ftrs_extractor_with_callbacks(
                train_ds,
                val_ds,
                model,
                angular_loss,
                callbacks,
                log_path
            )
        else:
            _fit_dml_ftrs_extractor_without_callbacks(
                train_ds,
                model,
                angular_loss,
                logs,
                is_sparse,
                log_path
            )
        with open(log_path + os.sep + 'config.json', 'w') as out_f:
            info = {}
            for k in data.keys():
                if is_json_serializable(data[k]):
                    info[k] = data[k]
                    if isinstance(info[k], np.float32):
                        info[k] = info[k].item()
            json.dump(info, out_f, indent=2, separators=(',', ': '))


def _fit_dml_ftrs_extractor_with_callbacks(
            train_ds: tf.data.Dataset,
            val_ds: tf.data.Dataset,
            model: Model,
            angular_loss: BaseAngularLoss,
            callbacks: CallbackList,
            tensorboard_flag: bool,
            epochs: int,
            log_path: str
    ):
        """
        Custom training DML features extraction loop with tensorflow callbacks.
        """
        if isinstance(angular_loss, SimpleKappaAdaptiveSphere) or \
                isinstance(angular_loss, TruncNewtonKappaAdaptiveSphere):
            fn_flag_update = angular_loss.switch_flag_update_sphere_radius
        else:
            fn_flag_update = do_nothing
        logs = {}  # for custom callbacks which needs the current batch as parameter
        logs_to_console = {}  # not to show the current batch to stdout
        if tensorboard_flag:
            hist_prog_callbacks = CallbackList(TensorBoard(log_dir=log_path, histogram_freq=1), add_history=True,
                                                   add_progbar=True, model=model)
        callbacks.on_train_begin(logs)
        hist_prog_callbacks.on_train_begin(logs_to_console)
        for epoch in range(epochs):  # loop over epochs
            has_store_first_batch = False  # get only the first batch to test on the callbacks
            if not model.stop_training:
                callbacks.on_epoch_begin(epoch, logs)
                hist_prog_callbacks.on_epoch_begin(epoch, logs_to_console)
                for step, (x_batch_train, y_batch_train) in enumerate(train_ds):  # loop over batches
                    if not has_store_first_batch:
                        logs['currBatch'] = np.array(
                            x_batch_train.numpy())  # to pass to callbacks the current batch data
                        logs['currBatchLabels'] = np.array(
                            y_batch_train.numpy())  # now callbacks can access current batch data
                        has_store_first_batch = True
                    callbacks.on_train_batch_begin(step, logs)
                    hist_prog_callbacks.on_train_batch_begin(step, logs_to_console)
                    loss_value = do_grads_for_dml_ftrs_extraction(x_batch_train, model, angular_loss,
                                                                       y_batch_train)
                    logs[angular_loss.name] = loss_value  # for EarlyStopping callback
                    logs_to_console[angular_loss.name] = loss_value  # To show to stdout
                    for c in callbacks:
                        if isinstance(c, RecorderAdaptiveSphereRadius):
                            c.sphere_radius_record = angular_loss.sphere_radius
                    callbacks.on_train_batch_end(step, logs)
                    hist_prog_callbacks.on_train_batch_end(step, logs_to_console)
                fn_flag_update()  # do nothing if not adacos; else update sphere radius
                if angular_loss.is_kappa:
                    angular_loss.update_memory_buffer()
                    angular_loss.update_mem_buf_on_epoch_end()
                # validation loop at the end of each epoch
                for x_val, y_val in val_ds:
                    val_logits = model(x_val, training=False)
                    logs_to_console['val_{}'.format(angular_loss.name)] = angular_loss(val_logits, y_val)
                hist_prog_callbacks.on_epoch_end(epoch, logs_to_console)
                callbacks.on_epoch_end(epoch, logs)
        for c in callbacks:
            if isinstance(c, AngleAvgMedianCallback):
                c.hist_avg_theta_gt_classes = angular_loss.avg_theta_gt
                c.hist_median_theta_gt_classes = angular_loss.median_theta_gt
                c.hist_avg_theta_not_gt_classes = angular_loss.avg_theta_not_gt
        callbacks.on_train_end(logs)
        hist_prog_callbacks.on_train_end(logs_to_console)
        angular_loss.to_npy_gt_weights_on_train_end()  # TODO, switch to tensorflow format (.tf or .HD5)
        model.save(log_path + os.sep + 'saved-h5', save_format='h5')
        dump_model_info(model, log_path)


def _fit_dml_ftrs_extractor_without_callbacks(
            train_ds: tf.data.Dataset,
            model: Model,
            angular_loss: BaseAngularLoss,
            logs: Dict[str, Any],
            is_sparse: bool,
            epochs: int,
            log_path: str
    ):
        """
        Custom training DML features extraction loop without tensorflow callbacks.
        """
        logs = {}  # TODO: can't I use the History callback dict?
        for _ in range(epochs):
            curr_batch_index = 0
            for x, y in train_ds:
                loss_value = do_grads_for_dml_ftrs_extraction(x, model, angular_loss, y)
                logs[angular_loss.name] = loss_value
                curr_batch_index += 1
            angular_loss.update_memory_buffer()
            angular_loss.update_mem_buf_on_epoch_end()
            # TODO: print info
        angular_loss.to_npy_gt_weights_on_train_end()
        model.save(log_path + os.sep + 'saved-h5', save_format='h5')
        dump_model_info(model, log_path)