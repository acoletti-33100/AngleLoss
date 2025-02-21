# General information, notes, disclaimers
#
# author: A. Coletti
# 
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
# I think margins and math.pi need to be converted from radians to degrees, instead the scalers
# temperature, alpha, gamma need not. Also, be careful to compute degs with degs and rads with rads
# so do NOT compute deg_pi with rad_margin, CAREFUL!
# 
# TODO:
# check which Bessel function approximation is better between mine, scipy iv(x,v), scipy ive(x,v) (more stable than iv(..)).
#
# ==============================================================================
from __future__ import annotations

from typing import Union
from typing import List
from typing import Optional
from typing import Tuple
from typing import Dict
from typing import Any

import numpy as np
import math
import warnings
from scipy.special import iv as bessel
from scipy.special import ive as bessel_exp

import tensorflow as tf
from abc import ABC
from abc import abstractmethod

from tensorflow_probability.python.math import bessel_ive

from utils.utils import get_class_freq_from_csv


class MarginScaler(ABC):
    """
    Abstract class to implement the common Kappa Loss framework behavior.
    Do not directly instantiate this class, instead instantiate the subclasses.
    The subclasses implements different version of how to compute the concentration
    "\hat{k}".

    Notes
    ----

    - "k_c" is the concentration parameter of a vMF (von Mises-Fisher) distribution
        for a specific class c. "\hat{k}_c" is the approximation of k for a specific class
        c. So, "khat" is a tensor with all the concentration values for each class.
        "k_tilde"({k_c}^~) is the normalization of "khat" for a specific class c; the normalization
        is computed as "k_hat" minus its mean and all divided by its standard deviation.
        See equations 9, 10, 11 from KappaFace.
    
    - The memory buffer is not a layer variable (weight) because it is only needed during training
        and, its goal, is simply to help in the computation of the margin scaler psi, which, instead,
        is a layer's weight.
        Therefore, the memory buffer is not instantiated in build() with an "add_weight()" call.
    
    - Each different output class has its own specific margin scaler psi. This makes sense
        since the Kappa loss framework aims at creating an adaptive angular margin between
        the inputs and the feature vectors.
    
    - For the Kappa loss framework, remember that the memory buffer is updated only at the end
        of each training epoch. As a consequence, the "k_tilde" (which is the normalization of 
        "k_hat") is also updated only at the end of each training epoch. Therefore, it follows 
        that the margin scaler psi (it's a tensor where each element is the margin scaler
        for a particular class) is also updated only at the end of each training epoch.
    """
    def __init__(
        self,
        classes_freq: List[int],
        gamma: float, 
        temperature: float, 
        alpha: float,
        mem_buf_shape_zero: int, 
        class_ids: List[str],
        pop_weights_concentration_fun: str = "sigmoid", 
        beta: Union[float, None]=None, 
        mem_buf: tf.Tensor=None,
        **kwargs
        ) -> None:
        """
        Creates an instance of MarginScaler class.

        Parameters
        ---------

        - classes_freq: list of ints, each value is the total number of samples for a specific class used in training.
        - gamma: float, gamma hyperparameter in [0,1] radians to apply in the scaling factor formula (EQ. 14).
            It balances the contribution of the sample and population concentration.
        - temperature: float, temperature hyperparameter in [0,1] radians.
        - alpha: float, alpha momentum hyperparameter in [0,1] radians needed in the memory buffer update.
        - mem_buf_shape_zero: int, dimension zero of the memory buffer. it is equal to the batch size during training.
        - class_ids: list of str, each is the name of the output class associated with the classes_freq list.
        - pop_weights_concentration_fun: str (default="sigmoid"), function to apply to compute the population
            weights concentration. Allowed values are: "sigmoid", "tanh".
            If the value is incorrect it rolls back to the default value ("sigmoid").
        - beta: float (default=None), regulizer hyperparameter in [0,1] radians to mupltiply to n_c in (EQ. 8).
            This is a modification to the original KappaFace function, set it to None if the classic behavior 
            is desired.
        """
        super(MarginScaler, self).__init__()
        self._gamma = gamma
        self._temperature = temperature
        self._alpha = alpha
        self._beta = beta
        self._class_ids = class_ids
        allowed_pop_weights_concentration_fun = ['sigmoid', 'tanh']
        actual_pop_weights_concentration_fun = [tf.math.sigmoid, tf.math.tanh]
        if pop_weights_concentration_fun is None or \
            not isinstance(pop_weights_concentration_fun, str) or \
            pop_weights_concentration_fun not in allowed_pop_weights_concentration_fun:
                self.pop_weights_concentration_fun = actual_pop_weights_concentration_fun[0]
        elif pop_weights_concentration_fun == 'sigmoid':
            self.pop_weights_concentration_fun = actual_pop_weights_concentration_fun[0]
        else: # tanh 
            self.pop_weights_concentration_fun = actual_pop_weights_concentration_fun[1]
        self._mem_buf_shape = (mem_buf_shape_zero, len(classes_freq)) # old
        # self._mem_buf_shape = (len(classes_freq),) # mem buf shape depends on W which is the output of fc7
        self._mem_buf = None
        self._classes_freq = classes_freq
        self._max_num_train_samples = max(classes_freq)
        self._fn_kwargs = kwargs
        # wcs := (cos(pi * n_c/K)) / 2, K:= max num train samples, n_c= num train sample for class c
        n_div_by_k = [x / self._max_num_train_samples for x in self._classes_freq]
        deg_pi = np.rad2deg(math.pi) # margin and pi must either both be degs or rads 
        self._wcs = (tf.math.cos(tf.math.scalar_mul(deg_pi, tf.constant(n_div_by_k))) + 1) / 2
        self._init_memory_buffer()
    

    @staticmethod
    def create_margin_scaler(
        conf: Dict[str, Any], 
        mem_buf_shape_zero: int) -> Tuple[MarginScaler, float, int]:
        """
        Instantiates a MarginScaler class depending on the information in conf. 
        This function assumes that conf contains the following keys:

            - "margin"
            - "sphereRadius"
            - "tau"
            - "temperature"
            - "gamma"
            - "beta"
            - "alpha"
            - "csvFrequencyTrainOnly"
            - "l2NormInputs"

        Note that, conf["additiveAngularMargin"] may only contain the following values (case insensitive):

            - "simpleKappa" 
            - "TruncNewtonKappa" 

        Parameters
        ---------

        - conf: python dict, dict of the configuration file for this experiment.
        - mem_buf_shape_zero:

        Returns
        ------

        Tuple, consisting of 3 elements: MarginScaler, margin and sphere radius.

        Raises
        -----

        - ValueError: if conf["additiveAngularMargin"] is not a legal value, None or not a key in conf.
            Also, if conf doesn't contains the keys: "margin", "sphereRadius", 
            "beta", "tau", "temperature", "gamma" (case sensitive).
        """
        conf_keys = conf.keys()
        keys_needed = [
            'additiveAngularMargin', 
            'csvFrequencyTrainOnly',
            'gamma',
            'temperature',
            'alpha',
            'beta',
            'tau',
            'margin',
            'sphereRadius'
            ]
        for k in keys_needed:
            if k not in conf_keys:
                raise ValueError('conf misses key: "{}"'.format(k))
        target = conf['additiveAngularMargin'].lower()
        class_ids, class_freq = get_class_freq_from_csv(conf['csvFrequencyTrainOnly'])
        if target == 'simplekappa':
            return KappaSimpleMarginScaler(
                class_freq,
                conf['gamma'],
                conf['temperature'],
                conf['alpha'],
                mem_buf_shape_zero,
                beta=conf['beta'],
                class_ids=class_ids
            ), conf['margin'], conf['sphereRadius']
        elif target == 'truncnewtonkappa':
            return KappaNewtonApproxMarginScaler(
                class_freq,
                conf['gamma'],
                conf['temperature'],
                conf['alpha'],
                mem_buf_shape_zero,
                beta=conf['beta'],
                tau=conf['tau'],
                class_ids=class_ids
            ), conf['margin'], conf['sphereRadius']
        else: 
            raise ValueError('Invalid name for margin scaler, received: {}'.format(conf['additiveAngularMargin']))


    @property
    def gamma(self):
        return self._gamma


    @property
    def temperature(self):
        return self._temperature


    @property
    def alpha(self):
        return self._alpha


    @property
    def class_ids(self):
        return self._class_ids


    @property
    def beta(self):
        return self._beta
    

    @property
    def classes_freq(self):
        return self._classes_freq


    @property
    def wcs(self):
        return self._wcs
    

    @property
    def mem_buf_shape(self):
        return self._mem_buf_shape


    @property
    def mem_buf(self):
        return self._mem_buf
    

    @abstractmethod
    def compute_margin_scaler_psi(self) -> tf.Tensor:
        """
        Abstract method to compute the margin scaler for an angular based loss.
        """
        pass


    @abstractmethod
    def has_memory_buffer(self) -> bool:
        """
        Abstract method to check whether this class applies a memory buffer.
        Kappa based losses must have a memory buffer, but in the future new losses
        may appear which do not need it, therefore this function should be kept.
        """
        pass
    

    @abstractmethod
    def compute_concentration_k_hat(self) -> tf.Tensor:
        """
        This function computes equation 5 from KappaFace (with "\hat{r}" instead of "\\bar{r}"), as:
        "\hat{k} = \frac{r * (d - r^2)}{1 - r^2}", where d is the dimension of the extracted feature
        vector and r is "\hat{r}"; also, note that each element in the final tensor is the concentration
        value associated with a specific output class c (eg: output is [k_c1, k_c2, .., k_cn], where n
        is the number of output classes).
        Further, trivially note that the output tensor is ordered as the n_c_all numpy.ndarray (each output
        is associated with one and only one output class).

        Parameters
        ---------

        - z_i: tf.Tensor, memory buffer (see equations 5,6,7,8).
            This tensor must have shape (k,n), where each column represents a feature
            vector, where k depends on the dimension of the hidden extracted vector (512D in KappaFace),
            while n is the number of classes.
        - n_c_all: list of ints, each element is the number of samples in the mini-batch (or all training set) 
            during training for a specific class C.

        Returns
        ------

        Tensor, with shape (n,) where n is the number of classes. Each value in this tensor is a 
        concentration parameter for a specific class as: "\hat{k}_c".
        """
        pass


    # TODO
    # Refactor to setter property
    def assign_mem_buf(self, new_mem_buf: np.ndarray) -> None:
        """
        Assign `new_mem_buf` to the memory buffer.

        Parameters
        ---------

        - new_mem_buf: numpy.ndarray, array with the values to assign to the memory buffer.

        Raises
        -----

        - ValueError: if `new_mem_buf.shape` is different from the memory buffer shape.
        """
        if not isinstance(new_mem_buf, np.ndarray):
            raise ValueError('new_mem_buf must be a numpy array, received: {}'.format(type(new_mem_buf)))
        if self._mem_buf is not None and new_mem_buf.shape != self._mem_buf.numpy().shape:
            raise ValueError('Icompatible new shape, received: {}, mem buf expects: {}'.format(
                new_mem_buf.shape,
                self._mem_buf.numpy().shape
            ))
        self._mem_buf = tf.constant(new_mem_buf)


    def _init_memory_buffer(self) -> None:
        """
        Initialize the memory buffer to a tensor of random values.
        You need to call this method only once.

        Raises
        -----

        - ValueError: if mem_buf_shape doesn't contain ints.
            Also, if mem_buf_shape[-1] is different from the number of output classes.
        """
        if self._mem_buf is None: # already init, do not init again
            for i in self.mem_buf_shape: # assumes it's a tuple
                if not isinstance(i, int):
                    raise ValueError('mem_buf_shape must be all ints; "{}" is not an int'.format(i))
            if self.mem_buf_shape[-1] != len(self._classes_freq):
                    raise ValueError('The memory buffer must have the same \
                    number of columns as the number of output classes. Instead, \
                    memory buffer columns={}, number output classes={}'.format(
                        self.mem_buf_shape[-1], len(self._classes_freq)))
            gen = tf.random.Generator.from_seed(1234, alg='philox') # init generator
            self._mem_buf = gen.normal(shape=self._mem_buf_shape) # generate memory buffer values
    

    def update_memory_buffer(self, z_i: tf.Tensor) -> None:
        """
        Updates the memory buffer as indicated in equation 7 in KappaFace.
        ***This method should only be called in a custom Callback inside the "on_epoch_end" scope***.

        Parameters
        ---------

        - z_i: Tensor, tensor of the extracted hidden features vector.
            This tensor must have shape (k,n), where each column represents a feature
            vector, where k depends on the dimension of the hidden extracted vector (512D in KappaFace),
            while n is the number of classes.

        Raises
        -----

        - ValueError: if z_i has a different shape from self._mem_buf
        """
        if z_i.shape != self.mem_buf.shape: # check shape and ensures init of memory buffer
            raise ValueError('Memory buffer must have the same shape as z_i')
        diff = 1 - self.alpha
        self._mem_buf = (self.alpha * self._mem_buf) + (diff * z_i)
    

    def get_memory_buffer_shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the memory buffer.
        """
        return self.mem_buf.shape
    

    def _compute_r_hat(self) -> tf.Tensor:
       """
       Computes "\hat{r}", which is the approximation of "\\bar{z}" as indicated in 
       equation 8 in KappaFace.
       This function computes the "\hat{r}" for each class c as a Tensor.
       This is the implementation of equation 8 in KappaFace, so this tensor is used to compute
       the approximation of the concentration value k ("\hat{k}", for a vMF distribution) for 
       each output class c.

       Notes
       ----

       - This function assumes the network already takes care of the L2 normalization step.
        This function works at Tensor level, so it assumes to receive all the classes present in the
        mini-batch during training.
        Equation 8: "\hat{r}_c = \\frac{sum_{i|y_i=c}(\\bar{z}_i)}{n_c}".

       - The memory buffer tensor must have shape (k,n), where each column represents a feature
           vector, where k depends on the dimension of the hidden extracted vector (512D in KappaFace),
           while n is the number of classes.

       Examples
       -------

       >>> z_i = tf.constant([[1., 2., 3.], [1., 2., 3.]])
       >>> n_c_all = [100, 101, 200]
       >>> # instatiate object 'kappa' which inherits KappaBaseMarginScaler
       >>> kappa._compute_r_hat(z_i, n_c_all) 
       <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1/100, 2/101, 3/200]), dtype=float32>
       >>> # Here, for clarity the values are shown as 1/100, 2/101, 3/200


       Returns
       ------

       Tensor, "\hat{r}" for each class c, so each element is 
       the sum of the "\\bar{z}_i" (average of the feature vectors) divided by n_c_all[c].
       The returned tensor has the same shape as n_c_all. 
       """
       sum_z_bar = tf.reduce_sum(tf.reduce_mean(self.mem_buf, 0)) 
       z_bar = tf.zeros(len(self._classes_freq), dtype=self.mem_buf.dtype)
       z_bar += sum_z_bar
       return tf.math.divide(z_bar, self._classes_freq) 
 

    def _compute_population_weights_concentration(self, kc_tilde: tf.Tensor) -> tf.Tensor:
        """
        Computes the population weights concentration ("{w_c}^k") as in (EQ. 12).
        The paper version applies a sigmoid function.

        Parameters
        ---------

        - kc_tilde: tf.Tensor, concentration values computed as indicated in (EQ. 11).

        Returns
        ------

        A Tensor with same size as kc_tilde.
        """
        return 1 - self.pop_weights_concentration_fun(tf.math.scalar_mul(self.temperature, kc_tilde))
    

    def _compute_k_tilde(self, khat: tf.Tensor) -> tf.Tensor:
        """
        Computes k tilde, from equation 11 of KappaFace, for each class.
        "k tilde" is the normalized version of the approximized "khat" (concentration of a vMF) value.
        It is computed by subtracting the mean to "khat" and dividing everything by the standard
        deviation.

        Parameters
        ---------

        - khat: tf.Tensor, tensor computed by self.compute_concentration_k_hat(..).

        Notes
        ----

        - equation 11: "{k_c}^~ = \\frac{\hat{k}_c - mu_k}{sigma_k}"
        - equation 10: "sigma_k = \sqrt{\\frac{sum_{c=1}^C{(\hat{k}_c - mu_k)^2}}{C}}" # std dev
        - equation 9: mu_k = "\\frac{sum_{c=1}^C\hat{k}_c}{C}" # mean

        Returns
        ------

        tf.Tensor, tensor with all "{k_c}^~" (k_tilde for each output class c).

        Raises
        -----

        - ValueError: if the std deviation of k has at least one NaN value.
        """
        mean_k = tf.reduce_sum(khat) / len(self._classes_freq)
        std_dev_k = tf.sqrt(tf.reduce_sum(tf.pow(khat - mean_k, 2)) / len(self._classes_freq))
        if std_dev_k is None: # sqrt of neg value => NaN => error or convert to 0. ?
            raise ValueError("Standard deviation of k (concentration parameter of vMF) can't be NaN")
        return (khat - mean_k) / std_dev_k

    
    def kappa_compute_margin_scaler_psi(self, khat: tf.Tensor) -> tf.Tensor:
        """
        Actual computation of the margin scaling factor psi.

        Parameters
        ---------

        - khat: tf.Tensor, tensor of concentration values computed by the subclasses.

        Returns
        ------

        A Tensor which represents the scaler psi.
        """
        return (
            tf.math.scalar_mul(self.gamma, self.wcs)
            ) + (
                tf.math.scalar_mul((1 - self.gamma), 
                self._compute_population_weights_concentration(self._compute_k_tilde(khat)) # wck
                ))


class VoidMarginScaler(MarginScaler):
    """
    Do nothing margin scaler.
    """
    def __init__(self, **kwargs) -> None:
        super(VoidMarginScaler, self).__init__(None, None, None, None, None, None, None, None, **kwargs)
    

    def compute_concentration_k_hat(self) -> tf.Tensor:
        pass
    

    def compute_margin_scaler_psi(self) -> tf.Tensor:
        pass


class KappaSimpleMarginScaler(MarginScaler):
    """
    Simple version to compute the margin scaler psi as indicated in KappaFace.
    """
    def __init__(
        self,
        classes_freq: List[int],
        gamma: float, 
        temperature: float, 
        alpha: float,
        mem_buf_shape_zero: int, # TODO, make it infer from something 
        class_ids: List[str],
        pop_weights_concentration_fun: str = "sigmoid", 
        beta: Union[float, None]=None, 
        mem_buf: tf.Tensor=None,
        **kwargs) -> None:
        """
        Parameters
        ---------

        - classes_freq: list of ints, each value is the total number of samples for a specific class used in training.
        - gamma: float, gamma hyperparameter in [0,1] radians to apply in the scaling factor formula (EQ. 14).
            It balances the contribution of the sample and population concentration.
        - temperature: float, temperature hyperparameter in [0,1] radians.
        - alpha: float, alpha momentum hyperparameter in [0,1] radians needed in the memory buffer update.
        - mem_buf_shape_zero: int, dimension zero of the memory buffer. it is equal to the output shape
            of the layer before L2 normalization.
        - class_ids: list of str, each is the name of the output class associated with the classes_freq list.
        - pop_weights_concentration_fun: str (default="sigmoid"), function to apply to compute the population
            weights concentration. Allowed values are: "sigmoid", "tanh".
            If the value is incorrect it rolls back to the default value ("sigmoid").
        - beta: float (default=None), regulizer hyperparameter in [0,1] radians to mupltiply to n_c in (EQ. 8).
            This is a modification to the original KappaFace function, set it to None if the classic behavior 
            is desired.
        """
        super(KappaSimpleMarginScaler, self).__init__(
            classes_freq, 
            gamma, 
            temperature, 
            alpha,
            mem_buf_shape_zero, 
            class_ids,
            pop_weights_concentration_fun, 
            beta, 
            mem_buf,
            **kwargs)
    

    def compute_margin_scaler_psi(self) -> tf.Tensor:
        """
        Overrides KappaBaseMarginScaler.compute_concentration_k_hat to calculate the simple
        version of the concentration parameters.

        Returns
        ------

        tf.Tensor, each element is the margin scaler psi for a specific class c.
            Each psi is associated with a class c as specified in self._classes_freq, 
            because each number in self._classes_freq is the number of occurrences for a specific
            output class.
        """
        return super().kappa_compute_margin_scaler_psi(
            self.compute_concentration_k_hat()
            )
    

    def has_memory_buffer(self):
        return True

    
    def compute_concentration_k_hat(self) -> tf.Tensor:
        """
        Simple way to compute the concentration value "\hat{k}" for each class c.

        Returns
        ------

        Tensor, with shape (n,) where n is the number of classes. Each value in this tensor is a 
        concentration parameter for a specific class as: "\hat{k}_c".

        Raises
        -----

        - ValueError: if z_i.shape is different from 2.
        - ValueError: if z_i.shape[1] is different from len(n_c_all); different number of 
            feature vectors and output classes (should be equal).
        """
        r_hat = super()._compute_r_hat() 
        d = super().get_memory_buffer_shape()[0]
        r_hat_pow2 = tf.math.multiply(r_hat, r_hat)
        k_hat = r_hat * (d - r_hat_pow2)
        return tf.math.divide(k_hat, (1 - r_hat_pow2))


class KappaNewtonApproxMarginScaler(MarginScaler):
    """
    Different version to compute the margin scaler psi as indicated in KappaFace.
    The difference lies in how to compute the concentration ("\hat{k}") from equation 5
    in KappaFace. Instead, this class computes "\hat{k}" as indicated in (Sra, 2011).
    """
    def __init__(
        self, 
        classes_freq: List[int],
        gamma: float, 
        temperature: float, 
        alpha: float,
        mem_buf_shape_zero: int, # embedding dimension
        class_ids: List[str],
        pop_weights_concentration_fun: str = "sigmoid", 
        beta: Optional[float] = None, 
        tau: float = 0.0001,
        id_bessel_fun: int = 2,
        limit_it_sra_bessel: int = 10000,
        mem_buf: Optional[tf.Tensor] = None,
        **kwargs) -> None:
        """
        Parameters
        ---------

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
       - limit_it_sra_bessel: int (default=10000), max number of iterations allowed.
           If the function takes more than limit_it, then it returns None and raises
           a warning. 
        """
        super(KappaNewtonApproxMarginScaler, self).__init__(
            classes_freq, 
            gamma, 
            temperature, 
            alpha,
            mem_buf_shape_zero, # embedding dimension
            class_ids,
            pop_weights_concentration_fun, 
            beta, 
            mem_buf,
            **kwargs)
        self._tau = tau
        self._embedding_dim = mem_buf_shape_zero
        self.limit_it = limit_it_sra_bessel
        self.order_v = (mem_buf_shape_zero / 2) - 1 # see equation 4 from KappaFace
        if id_bessel_fun is None or not isinstance(id_bessel_fun, int) or id_bessel_fun not in range(4):
            id_bessel_fun = 2
        if id_bessel_fun == 3:
            self._compute_bessel_fn = self._tf_bessel_ive
        elif id_bessel_fun == 2:
            self._compute_bessel_fn = self._bessel_ive
        elif id_bessel_fun == 1:
            self._compute_bessel_fn = self._bessel_iv
        else:
            self._compute_bessel_fn = self._compute_bessel
        self._id_bessel_fun = id_bessel_fun
    

    @property
    def tau(self):
        return self._tau
    

    @property
    def embedding_dim(self):
        return self._embedding_dim
    

    @property
    def id_bessel_fun(self):
        return self._id_bessel_fun
    

    # TODO 
    # debug
    def compute_margin_scaler_psi(self) -> tf.Tensor:
        """
        Truncated Newton method (2 iterations) to compute the concentration value "\hat{k}" for each class c.
        This function is the implementation of (Sra, 2011).
        This function is more accurate than "simple_compute_k_hat(z_i, n_c_all)".

        Parameters
        ---------

        - z_i: Tensor, tensor of the extracted hidden features vector.
            This tensor must have shape (k,n), where each column represents a feature
            vector, where k depends on the dimension of the hidden extracted vector (512D in KappaFace),
            while n is the number of classes.
        - n_c_all: list of ints, each element is the number of samples in the mini-batch (or all training set) 
            during training for a specific class C.
        - p: int (default=64), number of dimension of the vMF unit sphere "S^p".
        - tau: float, hyperparameter which determines the convergence tolerance.
        
        Returns
        ------

        tf.Tensor, each element is the margin scaler psi for a specific class c.
            Each psi is associated with a class c as specified in self._classes_freq, 
            because each number in self._classes_freq is the number of occurrences for a specific
            output class.
        """
        return super().kappa_compute_margin_scaler_psi(
            self.compute_concentration_k_hat()
            )
    

    def has_memory_buffer(self) -> bool:
        return True


    def compute_concentration_k_hat(self) -> tf.Tensor:
        """
        Compute the concentration value "\hat{k}" for each class c with the truncated Newton method
        as indicated in (Sra, 2011).

        Parameters
        ---------

        - z_i: tf.Tensor, memory buffer (see equations 5,6,7,8).
            This tensor must have shape (k,n), where each column represents a feature
            vector, where k depends on the dimension of the hidden extracted vector (512D in KappaFace),
            while n is the number of classes.
        - n_c_all: list of ints, each element is the number of samples in the mini-batch (or all training set) 
            during training for a specific class C.

        Returns
        ------

        Tensor, with shape (n,) where n is the number of classes. Each value in this tensor is a 
        concentration parameter for a specific class as: "\hat{k}_c".

        Raises
        -----

        - ValueError: if z_i.shape is different from 2.
        - ValueError: if z_i.shape[1] is different from len(n_c_all); different number of 
            feature vectors and output classes (should be equal).
        """
        d = super().get_memory_buffer_shape()[0] # dimension of the hypersphere for the data distribution.
        r_hat = super()._compute_r_hat()
        fn = lambda t: self._compute_newton_trunc(t, d)
        return tf.reshape(tf.map_fn(fn, r_hat), -1) # apply fn to each element in r_hat


    def _compute_newton_trunc(
        self, 
        r_bar: Union[float, tf.Tensor], 
        p: Union[int, tf.Tensor]) -> Union[float, tf.Tensor]:
       """
       This method computes the concentration value for a single output class.
       Implementation of equation 3, 6 from (Sra, 2011).

       Parameters
       ---------

       - r_bar: float or tensor with a single float element, EQ 8 from KappaFace paper.
       - p: int or tensor with a single int element, number of dimension of the vMF unit sphere "S^p".

       Returns
       ------

       float or tensor with a single float element, single concentration value approximation 
       for a specific class ("\hat{k}_c").

       References
       ---------

       - (Sra, 2011):
           "A short note on parameter approximation for von Mises-Fisher distributions: 
           and a fast implementation of I s (x)"
       - KappaFace:
           "KappaFace: Adaptive Additive Angular Margin Loss for Deep Face Recognition"

       """
       k = r_bar * (p - (r_bar * r_bar)) # k0 num
       k /= (1 - (r_bar * r_bar)) # init k with EQ 4 from (Sra, 2011)
       k = self._single_step(k, r_bar, p) # first Newton step
       k = self._single_step(k, r_bar, p) # second Newton step
       return k


    def _single_step(
        self, 
        k: Union[float, tf.Tensor], 
        r_bar: Union[float, tf.Tensor], 
        p: Union[int, tf.Tensor]) -> Union[float, tf.Tensor]:
       """
       This method performs a single step in the truncated Newton method to compute
       the concentration value for a single output class.

       Parameters
       ---------

       - k: float or tensor with a single float element, current concentration value.
       - r_bar: float or tensor with a single float element, approximation of "r^bar" as in EQ 8 in KappaFace, s.t: r_bar := ""||sum (z_i) ||_2 / n_c"";
           where z_i is the features vector, n_c is the class number of occurrences in the current mini-batch, ||X||_2
           is the L2 normalization operation.
       - p: int or tensor with a single int element, dimension of the unit sphere (dimension of the feature vector from the last layer in the CNN backbone).

       Returns
       ------

       float or tensor with a single float element, new k  value compute with truncated Newton method.
       """
       apk = self._compute_bessel_fn(k)
       tmp = apk * ((p - 1) / k)
       return k - ((apk - r_bar) / (1 - (apk * apk) - (tmp)))
    

    def _bessel_iv(self, x: float) -> float:
        """
        computes the Bessel function of the first kind with order as:
        "(embedding_dimension/2) - 1".

        Parameters
        ---------

        - x: float, input data to perform the Bessel on.

        Returns
        ------

        float, value of the Bessel function with input x.
        """
        return bessel(x, self.order_v)
    

    def _tf_bessel_ive(self, x: tf.Tensor) -> tf.Tensor:
        """
        computes the exponential (more numerically stable than "_bessel_iv(..)")
        Bessel function of the first kind with order as:
        "(embedding_dimension/2) - 1".
        Wrapper around ``tensorflow_probability.math.bessel_ive``.

        Parameters
        ---------

        - x: tf.Tensor, input data to perform the Bessel on.

        Returns
        ------

        tf.Tensor, value of the Bessel function with input x.
        """
        return bessel_ive(x, self.order_v)
    

    def _bessel_ive(self, x: float) -> float:
        """
        computes the exponential (more numerically stable than "_bessel_iv(..)")
        Bessel function of the first kind with order as:
        "(embedding_dimension/2) - 1".

        Parameters
        ---------

        - x: float, input data to perform the Bessel on.

        Returns
        ------

        float, value of the Bessel function with input x.
        """
        return bessel_exp(x, self.order_v)
    

    # TODO
    # tau needs tuning and understanding of reasonable ranges to apply.
    def _compute_bessel(self, x: Union[tf.Tensor, float]) -> Union[float, tf.Tensor]:
       """
       Computes the Bessel function of the first kind "I_s(x)" via truncated power-series, 
       as presented in algorithm 1 in (Sra, 2011).
       So, this method computes the ratio A_p(k), which is I_s(k) with s=p.
       Parameters
       ---------
       
       - x: float or tensor with a single float element, input data to perform the Bessel on.

       Returns
       ------

       float or tensor with a single float element, Bessel function of the first kind approximation.

       References
       ---------

       - (Sra, 2011):
           "A short note on parameter approximation for von Mises-Fisher distributions: 
           and a fast implementation of I s (x)"
       """
       R = tf.constant([1.0])
       t1 = x * tf.exp(1.)
       s = self.embedding_dim
       t1 /= 2 * s
       t1 = tf.math.pow(t1, s)
       t2 = tf.constant([1 + (1 / (12 * s)) + (1 / (288 * s * s)) - (139 / (51840 * s * s * s))])
       tmp = tf.math.sqrt((s / (2 * math.pi))) / t2
       t1 *= tmp
       M = 1 / s
       it = 1
       converged = False
       while not converged:
           aux_num = 0.25 * x * x
           aux_den = it * (it + s)
           R *= aux_num / aux_den
           M += R
           if (R / M) < self.tau:
               """
               # DEBUG
               self._print_debug_bessel({
               'tau': self.tau,
               'R': R,
               'M': M,
               'R/M': R / M,
               'iteration': it,
               't1': t1,
               't2': t2,
               'limit iterations': self.limit_it
               })
               """
               converged = True
           it += 1
           if it > self.limit_it:
               warnings.warn('algorithm did not converge in {} iterations'.format(self.limit_it))
               return None
       return t1 * M


    def _print_debug_bessel(self, all: Dict) -> None:
        print('Bessel info')
        for k in all.keys():
            print('{}: {}'.format(k, all[k]))
        print('############\n')