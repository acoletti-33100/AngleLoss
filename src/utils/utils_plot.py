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
from typing import Tuple
from typing import Union
from typing import List

import io
import matplotlib
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt



def pyplot_to_tf_image(fig):
    """
    Converts a pyplot image to tensorflow image to plot it in tensorboard.
    From:
        - https://www.tensorflow.org/tensorboard/image_summaries
    """
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    img = tf.image.decode_png(buffer.getvalue(), channels=3)
    img = tf.expand_dims(img, 0)
    return img


def get_n_from_unique_colors(colors_num: int, get_color_names=False) -> List[Tuple[float]]:
    """
    Gets a list a n elements, drawn from 40 unique Tuples of 3 floats.
    This function is used to get the class colors as RGB, such that each class
    has a unique identifiable color.

    Parameters
    ---------

    - colors_num: int, number of different RGB tuples (colors) to return.
        If it's bigger than a predifined list of colors, it is set to the redefined
        list length.
    - get_color_names: bool (default=False), flag to indicate whether to return the names of the colors
        too.

    Returns
    ------

    list of tuples, each tuple has 3 elements with at most 2 decimals.
    Each float is in the range [0,1).
    """
    valid_colors = [
    matplotlib.colors.to_rgb('red'),
    matplotlib.colors.to_rgb('blue'),
    matplotlib.colors.to_rgb('lawngreen'),
    matplotlib.colors.to_rgb('black'),
    matplotlib.colors.to_rgb('maroon'),
    matplotlib.colors.to_rgb('fuchsia'),
    matplotlib.colors.to_rgb('yellow'),
    matplotlib.colors.to_rgb('deepskyblue'),
    matplotlib.colors.to_rgb('olivedrab'),
    matplotlib.colors.to_rgb('slategray'),
    matplotlib.colors.to_rgb('violet'),
    matplotlib.colors.to_rgb('sienna'),
    matplotlib.colors.to_rgb('navy'),
    matplotlib.colors.to_rgb('chocolate'),
    matplotlib.colors.to_rgb('tan'),
    matplotlib.colors.to_rgb('mediumorchid'),
    matplotlib.colors.to_rgb('orange'),
    matplotlib.colors.to_rgb('gold'),
    matplotlib.colors.to_rgb('palegoldenrod'),
    matplotlib.colors.to_rgb('yellowgreen'),
    matplotlib.colors.to_rgb('darkgreen'),
    matplotlib.colors.to_rgb('mediumspringgreen'),
    matplotlib.colors.to_rgb('aquamarine'),
    matplotlib.colors.to_rgb('darkslategray'),
    matplotlib.colors.to_rgb('steelblue'),
    matplotlib.colors.to_rgb('royalblue'),
    matplotlib.colors.to_rgb('darkslateblue'),
    matplotlib.colors.to_rgb('blueviolet'),
    matplotlib.colors.to_rgb('purple'),
    matplotlib.colors.to_rgb('hotpink'),
    matplotlib.colors.to_rgb('crimson'),
    matplotlib.colors.to_rgb('pink'),
    matplotlib.colors.to_rgb('olive'),
    matplotlib.colors.to_rgb('orangered'),
    matplotlib.colors.to_rgb('rosybrown')
    ]
    if colors_num > len(valid_colors): 
        colors_num = len(valid_colors)
    if get_color_names:
        color_names = [
    'red',
    'blue',
    'lawngreen',
    'black',
    'maroon',
    'fuchsia',
    'yellow',
    'deepskyblue',
    'olivedrab',
    'slategray',
    'violet',
    'sienna',
    'navy',
    'chocolate',
    'tan',
    'mediumorchid',
    'orange',
    'gold',
    'palegoldenrod',
    'yellowgreen',
    'darkgreen',
    'mediumspringgreen',
    'aquamarine',
    'darkslategray',
    'steelblue',
    'royalblue',
    'darkslateblue',
    'blueviolet',
    'purple',
    'hotpink',
    'crimson',
    'pink',
    'olive',
    'orangered',
    'rosybrown'
        ]
        return valid_colors[:colors_num], color_names[:colors_num]
    return valid_colors[:colors_num]