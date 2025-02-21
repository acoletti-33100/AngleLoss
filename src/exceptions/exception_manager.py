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
from typing import Any

import re
from pathlib import Path


class ExceptionManager:
    """
    Stateless class, use the static methods to check if the parameters respect the necessary conditions.
    Use the class because in the future you may need finer control over exceptions, hence better to keep things
    tidy.
    """


    def __init__(self):
        pass

    
    @staticmethod
    def is_list_of_n_strings(target: Any, n: int):
        """
        Checks whether target is a list of n strings.

        Parameters
        ---------

        - target: object to check if it's a list of strings.
        - n: int, number of elements the list must to have.

        Raises
        -----

        ValueError if:
            - target is not a list.
            - target is a list with at least one nested list.
            - target contains at least one element which is not a string.
            - target has length different from n.
        """
        if not isinstance(target, list):
            raise ValueError('target is not a list')
        if any(isinstance(t, list) for t in target):
            raise ValueError('target can not be a list of lists')
        if any(not isinstance(t, str) for t in target):
            raise ValueError('target must be a list of strings only')
        if len(target) != n:
            raise ValueError('target does not contain {} elements'.format(n))

    
    @staticmethod
    def is_csv_file(target: Any):
        """
        Checks whether target is a string and a path to an existing csv file.

        Raises
        -----

        ValueError if:
            - target is None.
            - target is not a string.
            - target is not a file or does not exist.
            - target is not a csv file.
        """
        if target is None:
            raise ValueError('Value can not be None')
        if not isinstance(target, str):
            raise ValueError('{} is not a string'.format(target))
        if not Path(target).is_file():
            raise ValueError('{} is not a file or does not exist'.format(target))
        if not re.search('(.csv)$', target, re.IGNORECASE):
            raise ValueError('{} is not a csv file'.format(target))

    
    @staticmethod
    def is_npy_file(target: Any, none_ok: bool=False, is_existing_file: bool=True):
        """
        Checks whether target is a string or a path to an existing npy file.

        Parameters
        ---------

        - target: variable to check if it is valid.
        - none_ok: bool (default=False), flag to indicate whether to check if target is None.
            If none_ok=True this method does not check if target is None.
        - is_existing_file: bool (default=True), flag to indicate whether this file should already
            exist. If false, it means we are checking for a file to be created and therefore should
            not exist yet.

        Raises
        -----

        ValueError if:
            - target is None (if none_ok=False).
            - target is not a string.
            - target is not a file or does not exist (if is_existing_file=True).
            - target is not an npy file.
        """
        if not none_ok and target is None:
            raise ValueError('Value can not be None')
        if not isinstance(target, str):
            raise ValueError('{} is not a string'.format(target))
        if is_existing_file and not Path(target).is_file():
            raise ValueError('{} is not a file or does not exist'.format(target))
        if not re.search('(.npy)$', target, re.IGNORECASE):
            raise ValueError('{} is not an npy file'.format(target))

    
    @staticmethod
    def is_h5_tf_file(target: Any, none_ok=False):
        """
        Checks whether target is a string or a path to an existing h5 or tf file.

        Parameters
        ---------

        - target: variable to check if it is valid.
        - none_ok: bool (default=False), flag to indicate whether to check if target is None.
            If none_ok=True this method does not check if target is None.

        Raises
        -----

        ValueError if:
            - target is None.
            - target is not a string.
            - target is not a file or does not exist.
            - target is neither a h5 nor tf file.
        """
        if not none_ok and target is None:
            raise ValueError('Value can not be None')
        if not isinstance(target, str):
            raise ValueError('{} is not a string'.format(target))
        if not Path(target).is_file():
            raise ValueError('{} is not a file or does not exist'.format(target))
        if not re.search('(.h5)$|(.h5)$', target, re.IGNORECASE):
            raise ValueError('{} is neither a tf nor a h5 file'.format(target))



    @staticmethod
    def is_list_of_rois(target: Any):
        """
        Checks whether an object is a list of ROIs, so a list of lists of 2 ints each.

        Parameters
        ---------

        - target: object to check if it is a list of lists of ints.

        Raises 
        ------

        ValueError if:

        - target is not a list of lists of ints.
        - target lists have length different from 2.
        - target lists contains non-ints elements.
        """
        if not isinstance(target, list):
            raise ValueError('{t} is not a list'.format(t=target))
        if not (all(isinstance(r, list) for r in target)):
            raise ValueError('not all elements in {t} are lists'.format(t=target))
        rois = [r for r in target]
        if not all([len(r) == 2 for r in rois]):
            raise ValueError('not all elements in {t} have length 2'.format(t=target))
        is_int = []
        for roi in rois:
            is_int += [isinstance(r, int) for r in roi]
        if not all(is_int):
            raise ValueError('not all elements in {t} are ints'.format(t=target))