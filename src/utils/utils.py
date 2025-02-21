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
#
# ==============================================================================
from typing import Union
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable
from typing import Any

import sys
from os import stat
from os import sep
from itertools import chain
import pandas as pd
import json
import numpy as np
import re
from random import choice
import datetime as dt
from scipy.fftpack import sc_diff

import tensorflow as tf

from src.exceptions.exception_manager import ExceptionManager


def row_count(filename: str) -> int:
    """
    Counts the number of rows in a file.
    """
    gen = (row for row in open(filename, 'r'))
    return sum(1 for row in gen)


def shuffle_train_csv_v1(path_csv: str, new_path_csv: str) -> None:
    """
    Creates a new csv file with the same number of rows 
    as the one stored in path_csv but shuffled.
    Takes way too long to execute but it shouldn't have RAM issues.

    Parameters
    ---------

    - path_csv: str, relative path where the target file is stored.
    - new_path_csv: str, relative path where to save the shuffled file.

    Raises
    -----

    - ValueError if:
        - path_csv is the same as new_path_csv.
        - path_csv does not exist.
        - path_csv or new_path_csv are not strings.
    """
    ExceptionManager.is_csv_file(path_csv)
    if not isinstance(new_path_csv, str):
        raise ValueError('{new_path_csv} is not a string')
    indexes = np.arange(1, row_count(path_csv)) # minus header
    df = pd.read_csv(path_csv, nrows=0)
    df.to_csv(new_path_csv, index=False)
    it = indexes.shape[0]
    index_to_fetch = -1
    indexes_already_shuffled = []
    for i in range(it):
        while index_to_fetch == -1 or index_to_fetch in indexes_already_shuffled:
            index_to_fetch = choice(indexes)
        indexes_already_shuffled.append(index_to_fetch)
        indexes_to_skip = indexes[indexes != index_to_fetch]
        indexes_to_skip = [x + 1 for x in indexes_to_skip]
        df = pd.read_csv(path_csv, nrows=1, skiprows=indexes_to_skip)
        df.to_csv(new_path_csv, header=False, mode='a', index=False)
        index_to_fetch = -1
        progress_bar(i, it, 'Iteration {}/{}'.format(i + 1, it), 'Shufflev1: ')


def shuffle_train_csv_v3(data: Union[pd.DataFrame, str], new_path_csv: str, ignore_file_size: bool=False, flag_write: bool=True):
    """
    Shuffles the pandas.DataFrame data in-place along the rows and optionally saves the data to 
    a csv file.

    Parameters
    ---------

    - data: pandas.DataFrame or str, dataframe to shuffle.
    - new_path_csv: str, path where to save the csv file.
    - ignore_file_size: bool (default=False), flag to indicate whether to ignore the file size and still read the file indicated in data.
    - flag_write: bool (default=True), flag to indicate whether to write to file the shuffled dataframe in new_path_csv.

    Returns
    ------

    - pandas.DataFrame, shuffled data.
    """
    if flag_write and not isinstance(new_path_csv, str):
        raise ValueError('{new_path_csv} is not a string')
    if isinstance(data, str):
        ExceptionManager.is_csv_file(data)
        if ignore_file_size is False and stat(data).st_size > 8000000000:
            raise ValueError('File is too big (> 8GB), use version 1 (far slower though) or set flag "ignore_file_size" to True (may crash RAM)')
        df = pd.read_csv(data, engine='c', na_filter=False)
    else: # data is a pandas.DataFrame
        df = data
    indexes = np.arange(data.shape[0]).tolist()
    shuffled_indexes = np.arange(data.shape[0]).tolist()
    len_shuffled_indexes = len(shuffled_indexes)
    for i in range(len_shuffled_indexes):
        chosen = choice(indexes)
        shuffled_indexes[i] = chosen
        indexes = [i for i in indexes if i != chosen]
        switch_2_df_rows(data, i, chosen)
        progress_bar(i, len_shuffled_indexes, 'Iteration {}/{}'.format(i + 1, len_shuffled_indexes), 'Shufflev3: ')
    if flag_write:
        df.to_csv(new_path_csv, index=False)
    return df


def switch_2_df_rows(df: pd.DataFrame, origin: int, target: int):
    """
    Swtiches two rows in a dataframe.

    Example
    ------
    - Input: {'A': [0,4], 'B'[1,7]}
    - Output: {'A':[4,0], 'B': [7,1]}

    Parameters
    ---------

    - df: pandas.Dataframe in which to switch rows.
    - origin: int, index in "df" to switch with "target".
    - target: int, index in "df" to switch with "origin".
    """
    tmp = df.iloc[origin, :].tolist()
    assign_list_to_row(df, df.iloc[target, :].tolist(), origin)
    assign_list_to_row(df, tmp, target)


# TODO
# shuffle with indexes of the file and then create a new file in append
# mode and add batches of the loaded data instead of the whole file (to save RAM).
def shuffle_train_csv_v2(data: Union[pd.DataFrame, np.ndarray, str], new_path_csv: str, ignore_file_size: bool=False):
    """
    Creates a new csv file with the same number of rows 
    as the one stored in path_csv but shuffled.
    This version is faster but less friendly on RAM.

    Parameters
    ---------

    - data: either pandas.DataFrame of the data to shuffle or a string, which is the relative 
        path where the target file is stored.
    - new_path_csv: string, relative path where to save the shuffled file.
    - ignore_file_size: boolean flag (default=False), if true this function will ignore the size of 
        the input file and try to read it whole anyway. Otherwise, it will not care. 

    Raises
    -----

    - ValueError if:
        - path_csv is the same as new_path_csv.
        - path_csv does not exist.
        - path_csv or new_path_csv are not strings.
        - if ignore_file_size is False and the file stored in path_csv is larger than 8GB.
    """
    if not isinstance(new_path_csv, str):
        raise ValueError('{new_path_csv} is not a string')
    if isinstance(data, str):
        ExceptionManager.is_csv_file(data)
        if ignore_file_size is False and stat(data).st_size > 8000000000:
            raise ValueError('File is too big (> 8GB), use version 1 (far slower though) or set flag "ignore_file_size" to True (may crash RAM)')
        df = pd.read_csv(data, engine='c', na_filter=False)
    else: # data is a pandas.DataFrame
        df = data
    indexes = np.arange(0, df.shape[0]) 
    df_new = pd.DataFrame(df, copy=True)
    for i in range(df.shape[0]):
        index_to_fetch = choice(indexes)
        indexes = indexes[indexes != index_to_fetch]
        assign_list_to_row(df_new, df.iloc[index_to_fetch].tolist(), i)
    df_new.to_csv(new_path_csv, index=False)


def assign_list_to_col(df: pd.DataFrame, col: List, index: int):
    """
    Assigns a list to a specific column in a dataframe.

    Parameters
    ---------

    - df: pandas.DataFrame to which to assign `col`.
    - col: list, with length equal to df.shape[0], to assign to a specific column of `df`.
    - index: int, index of the dataframe to assign `col` to (0 <= index < df.shape[1]).

    Raises
    -----

    - ValueError if:
        - len(col) != df.shape[0].
        - col contains at least one nested list.
        - index is out of bounds.
    """
    if index < 0 or index >= df.shape[1]:
        raise ValueError('Index out of bounds, received: {} (valid range: [0,{}]'.format(index, df.shape[1] - 1))
    if any(isinstance(r, list) for r in col):
        raise ValueError('col is a list of lists')
    if len(col) != df.shape[0]:
        raise ValueError('col shape is different from rows number in df')
    for i in range(df.shape[0]):
        df.iat[i, index] = col[i]


def assign_list_to_row(df: pd.DataFrame, row: List, index: int):
    """
    Assigns a list to a specific row in a dataframe.

    Parameters
    ---------

    - df: pandas.DataFrame to which to assign row.
    - row: list, with length=df.shape[1], to assign to a specific row of df.
    - index: int, index of the dataframe to assign row to (0 <= index < df.shape[0]).

    Raises
    -----

    - ValueError if:
        - len(row) != df.shape[1].
        - row contains at least one nested list.
        - index is out of bounds.
    """
    if index < 0 or index >= df.shape[0]:
        raise ValueError('Index out of bounds')
    if any(isinstance(r, list) for r in row):
        raise ValueError('row is a list of lists')
    if len(row) != df.shape[1]:
        raise ValueError('row shape is different from cols number in df')
    for i in range(df.shape[1]):
        df.iat[index, i] = row[i]


def exp_log(best_hps, log_path: str, file_mode: str='w'):
    """
    Logs to file the best hyperparameters for a tuning experiment.
    It opens the file in overwrite mode by default.

    Parameters
    ---------

    - best_hps: kerastuner HyperParameters object obtained through
        'tuner.get_best_hyperparameters(num_trials=1)[0]'.
    - log_path: str, relative path where to log the best hyperparameters
        obtained from the kerastuner.
    - file_mode: str (default='w'), python file mode.
    """
    with open(log_path, file_mode) as f:
        for b_hp in best_hps.values:
            f.write('{}: {}\n'.format(b_hp, best_hps.values[b_hp]))
        f.write('\n')


def count_unique_csv_labels(path_csv: str) -> int:
    """
    Computes the number of unique labels contained in the dataset stored in "path_csv" 
    with column name equal to "label_col".
    
    Parameters
    ---------

    - path_csv: str, relative path where the files train.csv, test.csv and val.csv are
        stored.
    
    Notes
    ----

    - This function assumes the labels are stored as the first column in each csv file.
        Also, it assumes that the csv dataset in this folder contain the same columns name.
    
    - This is a wrapper function around unique_csv_labels(*), if you also need the unique labels
        in the same scope, then just call unique_csv_labels(*).

    Returns
    ------

    int, number of unique labels in the specified csv dataset.
    """
    return unique_csv_labels(path_csv).shape[0]


def unique_csv_labels(path_csv: str) -> np.ndarray:
    """
    Computes the unique labels contained in the dataset stored in "path_csv" with column name equal to
        "label_col".
    
    Parameters
    ---------

    - path_csv: str, relative path where the files train.csv, test.csv and val.csv are
        stored.
    
    Notes
    ----

    - This function assumes the labels are stored as the first column in the csv file.

    Returns
    ------

    Sorted numpy array (ASC) of the unique labels contained in the specified csv dataset.
    """
    df_cols = pd.read_csv(path_csv, nrows=0)
    label_col = df_cols.columns.tolist()[0]
    df = pd.read_csv(path_csv, usecols=[label_col])
    y_train = df.to_numpy(dtype=np.uint16)
    y_train = np.reshape(y_train, y_train.shape[0])
    return np.sort(pd.unique(y_train))


def flatten_list_any(target: List[List[Any]]) -> List[Any]:
    """
    Flatten a list of lists in a single list. The resulting list keeps the 
    order of comparison of target.
    This function assumes each element in target is a list.

    Examples
    -------

    >>> flatten_list_any([[0], [1,2]])
        [0, 1, 2]

    Parameters
    ---------

    - target: list of lists of objects, target to flatten.

    Returns
    ------

    - list of any object with all the elements from the "target" but 
        flattened.
    """
    return list(chain(*target))


def flatten_list_of_list(main_list: List[List[str]], sep: str) -> List[List[str]]:
    """
    Flattens an input list of lists of strings and joins each inner list
    with "sep". 

    Example 
    ------

    - inputs: main_list=[['a','A'], ['b', 'B']], sep='-' 
    - output: ['a-A', 'b-B']

    Parameters
    ---------

    - main_list: list of lists of strings, target to flatten.
    - sep: string, single character to use as a separator.

    Returns
    ------

    - list of strings with all the elements from the original "main_list" but 
        flattened and joined with "sep".
    """
    res = []
    for l in main_list:
        res.append(sep.join(l))
    return res


def concat_dfs(df_list: List[str]) -> pd.DataFrame:
    """
    Concats a list of pandas.Dataframe from a list of strings (paths to each source file).
    It concats in the order of appeareance in df_list.

    Parameters
    ---------

    - df_list: list of strings, each is the relative path to the source dataset to concat.
    
    Returns
    ------

    pd.DataFrame with the concatted datasets.
    """
    ExceptionManager.is_list_of_n_strings(df_list, len(df_list))
    for i in df_list:
        ExceptionManager.is_csv_file(i)
    df_all = pd.read_csv(df_list[0], engine='c', na_filter=False)
    for i in range(1, len(df_list)):
        tmp_df = pd.read_csv(df_list[i], engine='c', na_filter=False)
        df_all = pd.concat([df_all, tmp_df])
    return df_all


def concat_df_along_cols(path_dfs: List[str]) -> pd.DataFrame:
    """
    Concatenates the datasets along the column axis (1).
    The datasets are specified in the list path_dfs.
    All source datasets must have the same number of rows.

    Parameters
    ---------

    - path_dfs: list of strings, each is the path where the dataset is stored.

    Returns
    ------

    - pd.DataFrame which is the concatenation of all the source datasets specified
        in path_dfs.
    """
    for path in path_dfs:
        ExceptionManager.is_csv_file(path)
    df = pd.read_csv(path_dfs[0], engine='c', na_filter=False)
    for i in range(1, len(path_dfs)):
        df = pd.concat([df, pd.read_csv(path_dfs[i], engine='c')], axis=1)
    return df


def to_string_trim_file_name(target: str) -> str:
    """
    Gets only the path of the file without filename.

    Parameters
    ---------

    - target: str, string representing the file path to trim and return.

    Examples
    -------

    >>> from pathlib import Path
    >>> a = Path('p1', 'p2', 'file.npy')
    >>> to_string_trim_file_path(str(a))
    ... 'p1\\\\p2\\\\' # Windows
    ... 'p1/p2/' # Linux

    Returns
    ------

    - str, string representing only the path of the file obtained from target.
    """
    tmp = target.split(sep)[:-1]
    return ''.join([x + sep for x in tmp])


def to_string_trim_parent_path(target: str) -> str:
    """
    Gets only the name of the file without parent paths.

    Parameters
    ---------

    - target: str, string representing the file path to trim and return.

    Examples
    -------

    >>> from pathlib import Path
    >>> a = Path('p1', 'p2', 'file.npy')
    >>> to_string_trim_parent_path(str(a))
    ... 'file.npy'

    Returns
    ------

    - str, string representing only the name of the file obtained from target.
    """
    return target.split(sep)[-1]


def to_string_trim_path_extension_from_filename(target: str) -> str:
    """
    Gets only the name of the file without parent paths and extension.

    Parameters
    ---------

    - target: str, string representing the file path to trim and return.

    Examples
    -------

    >>> from pathlib import Path
    >>> a = Path('p1', 'p2', 'file.npy')
    >>> to_string_trim_path_extension_from_filename(str(a))
    ... 'file'

    Returns
    ------

    - str, string representing only the name of the file obtained from target.
    """
    return target.split(sep)[-1].split('.')[0]


# TODO
# make it capable of dealing with variable length.
def unpack_data(data, x, e, y, c):
    """
    Unpacks a tuple data with None values if not present.
    Must have a length of 4.

    Parameters
    ---------

    - data: variable to check.
    - x: str, data[0] to print in error message if needed.
    - e: str, data[1] to print in error message if needed.
    - y: str, data[2] to print in error message if needed.
    - c: str, data[3] to print in error message if needed.

    Returns
    ------

    tuple with values set to None if not present in input variable "data".
    """
    if not isinstance(data, tuple):
        return (data, None, None, None)
    elif len(data) == 1:
        return (data[0], None, None, None)
    elif len(data) == 2:
        return (data[0], data[1], None, None)
    elif len(data) == 3:
        return (data[0], data[1], data[2], None)
    elif len(data) == 4:
        return (data[0], data[1], data[2], data[3])
    else:
        raise ValueError("Data is expected to be in format `{x}`, `({x},)`, `({x}, {e})`, " 
            " `({x}, {e}, {y})`, `({x}, {e}, {y}, {c})`, found: {d}".format(x=x, e=e, y=y, c=c, d=data))


def count_x_in_list(target_list: List[Union[int, float, str]], target: Union[int, float, str]) -> int:
    """
    Counts the occurrences of target in target_list.

    Parameters
    ---------

    - target_list: list of values, list where to count the occurrences.
    - target: Any, actual value to check its occurrences in target_list.

    Returns
    ------

    int, number of occurrences of target in target_list.
    """
    return np.count_nonzero([x == target for x in target_list])


def sum_a_b_along_cols(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Sums two tensors "a" and "b", such that "a" is a matrix, while "b" is 
    a vector. The sum is completed along "a" columns, so "a" must have the same 
    number of rows as the number of elements in "b".
    Note that this function first converts "a" and "b" to numpy.ndarray, then
    computes the sum and finally returns a new tf.Tensor with the result.

    Parameters
    ---------

    - a: tf.Tensor, tensor of a matrix to sum along its columns.
    - b: tf.Tensor, tensor to add along the columns of a. This tensor
        must have the same shape as a.shape[0].
    
    Examples
    -------

    >>> a = <tf.Tensor: shape=(3, 4), dtype=int32, numpy= 
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])>
    >>> b = <tf.Tensor: shape=(3,), dtype=int32, numpy=array([10, 10, 10])>
    >>> sum_a_b_along_cols(a, b)
    <tf.Tensor: shape=(3, 4), dtype=int32, numpy=
        array([[10, 11, 12, 13],
               [14, 15, 16, 17],
               [18, 19, 20, 21]])>
    
    Returns
    ------

    tf.Tensor new tensor such that it is the sum of each column in a 
    with the tensor b (eg: a[:,0] = a[:,0] + b).

    Raises
    -----

    - ValueError: if a has a different number of rows from the number of 
        elements in b. Or if a doesn't have only 2 dimensions.
    """
    if len(a.shape) != 2 or len(b.shape) != 1 or b.shape[0] != a.shape[0]:
        raise ValueError('incompatible input shapes, received a: {}, b: {}'.format(a.shape, b.shape))
    nd_a = a.numpy()
    nd_b = b.numpy()
    fn = lambda x: np.add(x, nd_b)
    return tf.constant(np.apply_along_axis(fn, 0, nd_a))
        

def get_mask_tensor_array(size: Tuple[int], index_ones: Union[List[int], List[List[int]]]) -> tf.Tensor:
    """
    Creates a mask array with all elements sets to zero 
    except for the column with input index (basically creates a one-hot encoding
    of an array).
    The output array has type "np.float32".
    This is equivalent to a tf.one_hot(..), this function basically creates a one hot encoding.

    Parameters
    ---------

    - size: tuple of ints, dimension of the array to create.
    - index_ones: list of ints or list of list of ints, indexes of the columns to set to ones.
        Each value inside this list correspond to the j-th column
        of the new rank-1 Tensor. The row order is given by the order of 
        comparison in the list, so the first list element corresponds
        to the first row.
        If you want more than one column to be set to one, then you need to pass
        a list of list of ints, as: index_ones = "[[0,1], [0, 1, 2]]";
        where each inner list corresponds to a row index (as before ordered) and each
        element in each inner list is the j-th column.
        Note that index_ones must either be a 1 dimensional list (eg: [1,2]) or 
        a "2 dimensional" list (eg: [[0], [1,2]]), it can't be mixed (eg: [0, [1,2]] -> error).

    Examples
    -------

    >>> get_mask_tensor_array((2,3), [0])
        tf.Tensor(
            [[1. 0. 0.]
             [0. 0. 0.]], shape=(2, 3), dtype=float32)
    >>> get_mask_tensor_array((2,3), [[0], [1,2]])
        tf.Tensor(
            [[1. 0. 0.]
             [0. 1. 1.]], shape=(2, 3), dtype=float32)

    Returns
    ------

    tf.Tensor with all elements set to zeros except for the index-th
    column (set to ones).

    Raises
    -----

    - ValueError: if size is not a tuple of length 2 with only ints.
        Or if index_ones has length different from size.shape[0] or if
        index_ones contains out of bounds values with respect to an
        array created with shape equal to size (eg: size=(2,3), index_ones[1000]).
    """
    size_ok = isinstance(size, tuple) and len(size) == 2 and \
        isinstance(size[0], int) and isinstance(size[1], int)
    if not size_ok:
        raise ValueError('size is not correct, received: {}'.format(size))
    index_ones_ll_ok = all([isinstance(x, list) for x in index_ones]) # list of list
    index_ones_li_ok = all([isinstance(x, int) for x in index_ones]) # list of ints
    index_ones_shape_ok = len(index_ones) <= size[0] # index_ones shape is valid
    if not index_ones_shape_ok or not (index_ones_ll_ok ^ index_ones_li_ok) or \
        (index_ones_ll_ok and np.max(flatten_list_any(index_ones)) >= size[1]) or \
        (index_ones_li_ok and np.max(index_ones) >= size[1]):
            raise ValueError('index_ones is not correct, received: {}'.format(index_ones))
    res = tf.zeros(size)
    updates_rows_shape = len(index_ones)
    updates = np.zeros((updates_rows_shape, size[-1]))
    indices = [[x] for x in range(len(index_ones))]
    if index_ones_li_ok: # list of ints
        updates[np.arange(updates.shape[0]), index_ones] = 1.
    else: # list of list of ints
        for i in range(len(index_ones)):
            updates[i, index_ones[i]] = 1.
    return tf.tensor_scatter_nd_add(res, indices, updates)
        

def get_mask_nd_array(size: Tuple[int], index_ones: List[int]) -> np.ndarray:
    """
    Creates a mask array with all elements sets to zero 
    except for the column with input index (basically creates a one-hot encoding
    of an array).
    The output array has type "np.float32".

    Parameters
    ---------

    - size: tuple of ints, dimension of the array to create.
    - index_ones: list of ints, indexes of the columns to set to ones.

    Examples
    -------

    >>> get_mask_nd_array((2,2), [0,0])
            array([[1., 0.],
                   [1., 0.]], dtype=float32)

    Returns
    ------

    numpy.ndarray with all elements set to zeros except for the index-th
    column (set to ones).
    """
    res = np.zeros(size, dtype=np.float32)
    res[np.arange(res.shape[0]), index_ones] = 1.
    return res


def progress_bar(
    curr: int, 
    total_its: int, 
    preamble: str='',
    prefix: str='', 
    postfix: str='', 
    bar_size: int=35,
    completed_tick_symbol: str='=',
    todo_tick_symbol: str='.') -> None:
    """
    Plots a progress bar.

    Examples
    ---------

    >>> total = 10
    >>> curr = 0
    >>> for i in range(total):
    ...     # do something
    ...     progress_bar(curr, total, 'Task: ', 'Something', 20)

    Parameters
    ---------

    - curr: int, current iteration in the progress bar.
    - total_its: int, total number of iterations for the progress bar.
    - preamble: str, message to print before the progress bar 
        (this function appends a newline character "\\n" at the end).
    - prefix: str, message to print before the progress bar.
    - postfix: str, message to print after the progress bar.
    - bar_size: int (default=35), total number of ticks to show in output in the progress bar.
    - completed_tick_symbol: str (default="="), symbol used to represent a 
        completed tick in the progress bar. 
    - todo_tick_symbol: str (default='.'), symbol used to represent a todo tick in the progress bar.

    See
    --

    Useful reference (plenty of answers):
        https://stackoverflow.com/questions/3160699/python-progress-bar
    """
    sys.stdout.write('\n\n{}\n\n'.format(preamble)) # a tick is an "="
    curr += 1
    # num_completed_ticks is how much of the total work has been done
    num_completed_ticks = int(round((bar_size * (float(curr) / float(total_its)))))
    completed_ticks = completed_tick_symbol * num_completed_ticks
    todo_ticks = todo_tick_symbol * (bar_size - num_completed_ticks)
    msg = '\r{} [{}] - {}\n'.format(
        prefix, 
        '{}{}'.format(completed_ticks, todo_ticks),
        postfix)
    sys.stdout.write(msg)
    sys.stdout.flush()


def count_elem_in_container(target: Union[np.ndarray, List[Any], Tuple[Any]]) -> int:
    """
    Counts the number of elements in target, which can be either a list or tuple or numpy
    array.

    Parameters
    ---------

    - target: container to count the number of elements.

    Returns
    ------

    int, number of elements in the first dimension of target. This means:
    - if target is an array it returns target.shape[0];
    - if target is a list or tuple it returns len(target).
    """
    if isinstance(target, np.ndarray):
        return target.shape[0]
    return len(target)
        

def return_as_is(target: Any, **kwargs: Dict[str, Any]) -> Any:
    """
    Returns target "as-is".
    This function is useful when you want to apply a wrapper and have a case
    where you don't need the function. Essentially it is used to remove 'ifs'.
    """
    return target


def is_not_in(target: pd.Series, values: List[Any]) -> pd.Series:
    """
    Checks if the elements in `values` are not contained in `target`.

    Parameters
    ---------

    - target: pandas.Series, series to check if it contains elements from values.
    - values: list, list of elements to check if they are contained in target.
    Returns
    ------

    pandas.Series, series of booleans indicating if each element is not in 
    target.

    Raises
    -----
    
    - ValueError: if target is not a pd.Series object or if values is not a list. 
    """
    if not isinstance(target, pd.Series):
        raise ValueError('target is not a pandas.Series; received: {}'.format(type(target)))
    if not isinstance(values, list):
        raise ValueError('values is not a list; received: {}'.format(type(values)))
    res = target.isin(values)
    return res.apply(lambda x: not x)
    

def tslice_index_col(target: tf.Tensor, index: int) -> tf.Tensor:
    """
    Returns the n-th column in the target tensor, where n=index.

    Parameters
    ---------

    - target: tf.Tensor, tensor to select the column from.
    - index: int, index of the column (axis 1) to select.

    Returns
    ------

    rank-1 ("shape=(target.shape[1],)") tensor with the values of the 
    index-th column of target.

    Raises
    -----

    - ValueError: if index is out of bounds.
    """
    if len(target.shape) < 2:
        raise ValueError('Expected tensor of at least rank-2 (matrix)')
    if index < 0 or index > target.shape[1] - 1:
        msg = 'Out of bounds index, expected: [0, {}]; received: {}'
        raise ValueError(msg.format(target.shape[1] - 1, index))
    return tf.gather(target, index, axis=1)


def tslice_cols_with_ith_row(target: tf.Tensor, row_indexes: List[int]) -> tf.Tensor:
    """
    Creates a rank-1 tensor (vector) by selecting the i-th row (from row_indexes)
    for each column.

    Parameters
    ---------

    - target: tf.Tensor, tensor of at least rank-2.
    - row_indexes: list of ints, list of the row indexes to select for each
        column. It must contain exactly target.shape[1] elements.

    Examples
    -------

    >>> a = tf.constant(np.random.randint(0,100,(3,5)))
    >>> <tf.Tensor: shape=(3, 5), dtype=int32, numpy=
    ...     array([
                    [ 5, 70, 97, 59,  2],
                    [30, 76, 21, 83,  6],
                    [60, 90, 48, 39, 67]])>
    >>> labels = tf.constant([0,1,0,2,1])
    ... <tf.Tensor: shape=(5,), dtype=int32, numpy=array([0, 1, 0, 2, 1])>
    >>> tslice_cols_with_ith_row(a, labels)
    ... <tf.Tensor: shape=(5,), dtype=int32, numpy=array([5, 76, 97, 39, 6])>

    Notes
    ----

    - It does not support negative indexing, so negative elements in row_indexes
        are invalid. 

    - This is equivalent of numpy `a[l, np.arange(a.shape[1])]` where a is a numpy.ndarray with 
        shape=(n,m) and l is a list of a.shape[1] elements and l contains the row indexes to 
        select for each column in a.
        Basically, this is equivalent to a[row_indexes, col_indexes], where col_indexes=np.arange(len(row_indexes)).

    Returns
    ------

    rank-1 tensor where each element is the 
    target[row_indexes[i], j] for j in target.shape[1].

    Raises
    -----

    - ValueError: if any value in row_indexes is out of bounds.
        Or if target is not a tensor of rank-2 or higher.
        Or if row_indexes does not have exactly target.shape[1] elements.
    """
    if len(target.shape) < 2:
        raise ValueError('target must be at least a rank-2 tensor')
    if len(row_indexes) != target.shape[1]:
        msg = 'row_indexes must have {} elements; received: {}'
        raise ValueError(msg.format(target.shape[1], len(row_indexes)))
    col_indexes = np.arange(len(row_indexes))
    indexes = np.stack([row_indexes, col_indexes], axis=1)
    return tf.gather_nd(target, indexes)


def is_df_null(target: Union[pd.DataFrame, pd.Series]) -> Union[bool, pd.Series]:
    """
    Checks whether a pandas dataframe or series contains any NaN values.

    Parameters
    ---------

    - target: pandas daframe or series to check if it contains NaNs values.

    Returns
    ------

    - bool, True if the pandas Series contains NaNs values, else false.
    - pd.Series, contains True/False values. Each element indicates if a column contains NaNs;
        true if it has any otherwise false.
    """
    return target.isnull().sum() > 0


def is_json_serializable(target: Any) -> bool:
    """
    Checks if `target` is a JSON serializable object.

    Parameters
    ---------

    - target: python object, object to check if it is JSON serializable.

    Returns
    ------

    True if `target` is JSON serializable, false otherwise.

    References
    ---------

    https://stackoverflow.com/questions/42033142/is-there-an-easy-way-to-check-if-an-object-is-json-serializable-in-python
    """
    try:
        json.dumps(target)
        return True
    except:
        return False


def select_tf_where_isnot_t(target: tf.Tensor, diff_from: Any) -> tf.Tensor:
    """
    Creates a new Tensor by selecting all elements in target, which are different from diff_from.

    Parameters
    ---------

    - target: tf.Tensor, tensor from which to select.
    - diff_from: Any, legal value contained in target to exclude from the selected result.

    Examples
    -------

    >>> a = tf.constant([11.,0.,0.,20.])
    >>> ... <tf.Tensor: shape=(4,), dtype=float32, numpy=array([11.,  0.,  0., 20.], dtype=float32)>
    >>> select_tf_where_isnot_t(a, 0.)
    >>> ... <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
                array([[11.],
                       [20.]], dtype=float32)>

    Returns
    ------

    tf.Tensor, contains all elements in target which are different from diff_from.
    """
    indexes = tf.where(target != diff_from) # [[row_index0, col_index0], [row_index1, col_index1], ..[row_indexN, col_indexN]]
    return tf.gather_nd(target, indexes)


def apply_fun_every_n_elements(
    target: List[Any], 
    func: Callable, 
    batch_size: int) -> List[Any]:
    """
    Applies a function 'func' in batches of size 'batch_size' to 'target'.

    Parameters
    ---------

    - target: list, container to apply 'func' to.
    - func: Callable, function to apply to target. This function does not check whether 'func' may be applied to
        the data stored in target; the caller needs to take care of validity checks.
    - batch_size: int, batch size; how many elements of 'target' to which to apply 'func' multiple times.

    Examples
    -------

    >>> target = np.array([1,1,1,1,1,1,2,1,1,1,1,1,1])
    >>> func1 = lambda x: np.average(x)
    >>> apply_fun_every_n_elements(target, func1, 2)
    ... [1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0]

    Returns
    ------

    list, contains the result of 'func'. This container will have 'len(target) / batch_size' elements.
    """
    res = []
    total_target_size = count_elem_in_container(target)
    for i in range(0, total_target_size, batch_size):
        res.append(func(target[i: i + batch_size]))
    return res


def convert_degrees_in_terms_of_pi(target: List[int]) -> List[str]:
    """
    Converts a list containing angles values in degrees to a list with angles in terms of pi.

    Parameters
    ---------

    - target: list of ints, each value is an angle in degrees.
     
    Examples
    -------
    
    >>> target = [90, 45, 180]
    >>> convert_degrees_in_terms_of_pi(target)
    ... ['$\\pi/4$', '$\\pi/2$', '$\\pi$']

    Returns
    ------

    list of strings, each is the corresponding value in 'target' expressed in terms of pi.
    """
    denominator = 180 # fixed to pi in degrees
    res = []
    for numerator in target:
        gcd = np.gcd(int(numerator), denominator)
        num = numerator // gcd
        if num == 1:
            num = ''
        else:
            num = int(num)
        den = denominator // gcd
        if den == 1:
            den  = ''
        else:
            den = '/{}'.format(int(den))
        tmp = r'${}\pi{}$'.format(num, den)
        res.append(tmp)
    return res


def tf_median(target: tf.Tensor) -> tf.Tensor:
    """
    Computes the median of a given tensor.
    This function's behavior is the same as numpy.median.
    The median of ``target`` is the middle value of a sorted copy of
    ``target``: ``sorted_target[(N - 1) / 2]`` when ``N`` is odd; 
    ``N`` is the number of elements in ``target``. Instead, when ``N`` is even, 
    the median is the average of the two middle values of the sorted copy.
    
    Parameters
    ---------

    - target: tf.Tensor, tensor to compute the median from.

    Returns
    ------

    tf.Tensor, contains the median of target.
    """
    num_elements = np.prod(target.shape.as_list()) # total number of elements in target
    if num_elements % 2 == 0: # even
        middle_index = num_elements // 2
        return tf.reduce_mean(tf.sort(tf.reshape(target, -1))[middle_index:middle_index + 2])
    return target[(num_elements - 1) // 2] # odd


def invert_tf_mask(mask: tf.Tensor) -> tf.Tensor:
    """
    Inverts the values of a target tensor containing only ones and zeros, such that the output
    tensor will have the zeros switched with ones and the ones switched with zeros.

    Parameters
    ---------

    - mask: tf.Tensor, mask of ones and zeros to invert.

    Examples
    -------

    >>> mask = tf.constant([[1.,0.,0.], [0.,0.,1.]])
    ... <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
            array([[1., 0., 0.],
                   [0., 0., 1.]], dtype=float32)>
    >>> invert_tf_mask(mask)
    ... <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
            array([[0., 1., 1.],
                   [1., 1., 0.]], dtype=float32)>

    Returns
    ------

    tf.Tensor, tensor with inverted values.
    """
    return tf.ones_like(mask) - mask


def do_nothing():
    return


def find_object_index_in_list(target: List[Any], target_type: Any) -> int:
    """
    Returns the index of the first occurrence in ``target`` where an object of 
    type ``target_type`` is stored.

    Parameters
    ---------

    - target: list of any, list where to find an object of type ``target_type``.
    - target_type: Any, type of the index of the object to look for.

    Returns
    ------

    int, first occurrence of the index where the object of type ``target_type`` 
        is stored in ``target``.
        Returns 'None' if there is no object of type ``target_type``
        in ``target``. 
    
    Raises
    -----

    - ValueError: if ``target`` is not a list.
    """
    if not isinstance(target, list):
        raise ValueError('target must be a list, received: {}'.format(type(target)))
    for i in range(len(target)):
        if isinstance(target[i], target_type):
            return i
    return None # target_type not in target


def simulated_annealing(
    objective_fun: Callable, 
    bounds: List[int], 
    n_iterations: int, 
    step_size: float, 
    temperature: float):
    """
    Simulated annealing from:
    https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/
    """
    best = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    best_eval = objective_fun(best)
    curr = best
    curr_eval = best_eval
    for i in range(n_iterations):
        candidate = curr + np.random.randn(len(bounds)) * step_size
        candidate_eval = objective_fun(candidate)
        if candidate_eval < best_eval:
            best = candidate
            best_eval = candidate_eval
        diff = candidate_eval - curr_eval
        t = temperature / float(i + 1)
        metropolis = np.exp(-diff / t)
        if diff < 0 or np.random.rand() < metropolis:
            curr = candidate
            curr_eval = candidate_eval
    return [best, best_eval]

