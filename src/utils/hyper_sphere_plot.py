# General information, notes, disclaimers
#
# author: A. Coletti
# 
# Goals
#
# - I want to plot the extracted features (embeddings) as 3D points on the unit sphere
# during the training of the embeddings extractor.
#
# Steps:
# 
# - Plot a line from the center of the unit sphere to any point on its surface.
#
#
#
# lines starting from the center have 3D coordinates: [0,X], [0,Y], [0,Z]
#
#
# Resources
#
#
# ==============================================================================
from typing import Tuple
from typing import Union
from typing import List

import pandas as pd
import matplotlib
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d 
from matplotlib import ticker
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np

from src.utils.utils_plot import get_n_from_unique_colors
from src.utils.utils import count_elem_in_container
from src.utils.utils import convert_degrees_in_terms_of_pi


def plot_angles(
    deg_median_theta_gt_classes: List[float], 
    deg_avg_theta_gt_classes: List[float], 
    deg_avg_theta_not_gt_classes: List[float]) -> Figure:
    """
    Returns a figure with the plotted angles of the averages of the ground truth and not-corresponding classes 
    and the medians of the ground truth classes for each mini-batch during training.

    Parameters
    ---------

    - deg_median_theta_gt_classes: list of float, median angles in each mini-batch for the corresponding classes
    - deg_avg_theta_gt_classes: list of float, average angles in each mini-batch for the corresponding classes.
    - deg_avg_theta_not_gt_classes: list of float, the average angles in each mini-batch for the non-corresponding 
        classes.

    Returns
    ------

    Figure, matplotlib figure with the x-axis as the number of epochs during training, while 
        the y-axis is the angles values in degrees.
    """
    x_ticks = np.arange(len(deg_avg_theta_gt_classes))
    #fig = plt.figure(figsize=(20, 10))
    fig, ax = plt.subplots(1)
    ax.plot(x_ticks, deg_avg_theta_gt_classes, c='r', label="Average \N{GREEK SMALL LETTER THETA}$_{i,j}$")
    ax.plot(x_ticks, deg_median_theta_gt_classes, c='b', label="Median \N{GREEK SMALL LETTER THETA}$_{i,j}$")
    ax.plot(x_ticks, deg_avg_theta_not_gt_classes, c='g', label="Average \N{GREEK SMALL LETTER THETA}$_{i,j}$, j \u2260 $y_i$")
    plt.xlabel('Num. of iterations')
    plt.ylabel('Degree of theta')
    plt.xticks([])
    y_ticks = convert_degrees_in_terms_of_pi(ax.get_yticks().tolist())
    ax.set_yticklabels(y_ticks) 
    plt.grid(True, linestyle='dotted', which='major', axis='y', color='grey', alpha=0.4)
    plt.legend()
    return fig


def scatter_plot(
    x_coords: Union[np.ndarray, List[float]], 
    y_coords: Union[np.ndarray, List[float]], 
    labels: List[int],
    class_ids: List[str]) -> Figure:
    """

    Parameters
    ---------
 
    - x_coords: np.ndarray, numpy array of the x coordinates; each value is a different 
        point and is associated with y,z coordinates with the same 
        index in y_coords and z_coords respectively.
    - y_coords: np.ndarray, numpy array of the x coordinates; each value is a different 
        point and is associated with x,z coordinates with the same 
        index in x_coords and z_coords respectively.
    - z_coords: np.ndarray, numpy array of the x coordinates; each value is a different 
        point and is associated with x,y coordinates with the same 
        index in x_coords and y_coords respectively.
    - labels: list, contains the labels of the output class associated with each 2D point (must be ints).
        Note that, this label is the true id of the output class, not the predicted class from a model.
    - class_ids: list of str, list with the unique ids associated with the labels.
        This list is ordered as: class with label "0" has id class_ids[0].
        Example: labels[22]=0 and class_ids[0]='E. coli'.
    """
    fig = plt.figure(figsize=(8,8))
    len_class_ids = count_elem_in_container(class_ids)
    colors_ids = get_n_from_unique_colors(len_class_ids)
    colors = [colors_ids[x] for x in labels]
    unique_labels = pd.unique(labels)
    for l in range(len(unique_labels)):
        tmp_x = []
        tmp_y = []
        for i in range(len(labels)):
            if unique_labels[l] == labels[i]:
                tmp_x.append(x_coords[i])
                tmp_y.append(y_coords[i])
        plt.scatter(tmp_x, tmp_y, color=colors[l])
    patches = []
    for i in range(len_class_ids): # each class id gets its own color in the legend
        patches.append(mpatches.Patch(color=colors_ids[i], label=class_ids[i]))
    plt.legend(handles=patches)
    return fig


def sphere_plot(
    x_coords: Union[np.ndarray, List[float]], 
    y_coords: Union[np.ndarray, List[float]], 
    z_coords: Union[np.ndarray, List[float]], 
    labels: Union[np.ndarray, List[int]],
    class_ids: List[str],
    fig_name: str = '',
    grid_thickness: int = 60,
    y_view_point: int = 30,
    x_view_point: int = 45) -> Figure:
    """
    This function plots the intersection between the unit sphere and the feature vectors (must have
    shape (n, 3)).
    To do so, it plots the intersection of the unit sphere with the lines formed by 2 points,
    which are:
    - The centre of the unit sphere.
    - The points given in inputs.
    So, each line if formed by these 2 points and only the intersection between lines and 
    the unit sphere is shown.
    The color of the dot is different for each output class.
    All the lines start from the center of the sphere.
    The idea is to plot the data on the surface of the unit sphere.

    Parameters
    ---------
 
    - x_coords: np.ndarray, numpy array of the x coordinates; each value is a different 
        point and is associated with y,z coordinates with the same 
        index in y_coords and z_coords respectively.
    - y_coords: np.ndarray, numpy array of the x coordinates; each value is a different 
        point and is associated with x,z coordinates with the same 
        index in x_coords and z_coords respectively.
    - z_coords: np.ndarray, numpy array of the x coordinates; each value is a different 
        point and is associated with x,y coordinates with the same 
        index in x_coords and y_coords respectively.
    - labels: list or numpy array, contains the labels of the output class associated with each 3D point (must be ints).
        Note that, this label is the true id of the output class, not the predicted class from a model.
    - class_ids: list of str, list with the unique ids associated with the labels.
        This list is ordered as: class with label "0" has id class_ids[0].
        Example: labels[22]=0 and class_ids[0]='E. coli'.
    - fig_name: str (default=''), name of the output image.
    - grid_thickness: int (default=60, suggested value), controls width and height of each cell in the sphere.
        Internally, it computes theta and phi and if theta is smaller than phi, than the cell will be longer and 
        more stretched horizontally, while the opposite happens with phi smaller than than theta (vertical stretch instead).
    - y_view_point: int (default=30), y view angle of the output image. Rotates over the y-axis. Set it to 90 to view
        the sphere from above, completely (as if you were above the sphere).
    - x_view_point: int (default=45), x view angle of the output image. Rotates over the x-axis (counterclock-wise).

    Notes
    ----

    - set (x_view_point, y_view_point) to:
        - (60, 10), to view the sphere from above.
        - (30, 45), to view from the front almost like a "3/4 portrait" (default).

    Returns
    ------

    matplotlib Figure object of the unit sphere with the data plotted on it.
    """
    t = compute_t(np.array(x_coords), np.array(y_coords), np.array(z_coords))
    intersection_x = x_coords * t
    intersection_y = y_coords * t
    intersection_z = z_coords * t
    x_shape = count_elem_in_container(x_coords)
    len_class_ids = count_elem_in_container(class_ids)
    theta = np.linspace(0, 2 * np.pi, grid_thickness) 
    phi = np.linspace(0, np.pi, int(grid_thickness/3))
    theta_ms, phi_ms = np.meshgrid(theta, phi)
    x = np.sin(phi_ms) * np.cos(theta_ms)
    y = np.sin(phi_ms) * np.sin(theta_ms) 
    z = np.cos(phi_ms)
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(x, y, z, linewidth=1, alpha=0.25, color='gray')
    colors_ids = get_n_from_unique_colors(len_class_ids)
    colors = [colors_ids[x] for x in labels]
    for i in range(x_shape):
        plot_line_in_3d(
            ax, 
            [0, intersection_x[i]], 
            [0, intersection_y[i]], 
            [0, intersection_z[i]], 
            colors[i], 
            class_ids[labels[i]])
    ax.view_init(y_view_point, x_view_point)
    ax.tick_params(axis='both')
    patches = []
    for i in range(len_class_ids): # each class id gets its own color in the legend
        patches.append(mpatches.Patch(color=colors_ids[i], label=class_ids[i]))
    plt.legend(handles=patches)
    plt.title(fig_name)
    return fig


def sphere_4subplot_pca(
    coords: Union[np.ndarray, List[float]], 
    labels: Union[np.ndarray, List[int]],
    class_ids: List[str],
    grid_thickness: int = 60) -> Figure:
    """
    Same as sphere_4subplot(**) but applies a PCA to create only 3 principal
    components when the input 'coords' is an array with shape (n,m) with 'm' > 3.
    """
    pca = PCA(n_components=3)
    pca_res = pca.fit_transform(np.array(coords))
    return sphere_4subplot(
        pca_res[:, 0], 
        pca_res[:, 1], 
        pca_res[:, 2], 
        labels, 
        class_ids, 
        grid_thickness)


def sphere_4subplot(
    x_coords: Union[np.ndarray, List[float]], 
    y_coords: Union[np.ndarray, List[float]], 
    z_coords: Union[np.ndarray, List[float]], 
    labels: Union[np.ndarray, List[int]],
    class_ids: List[str],
    grid_thickness: int = 60) -> Figure:
    """
    This function plots the intersection between the unit sphere and the feature vectors (must have
    shape (n, 3)). It creates 4 subplots of the same data from different angles.
    To do so, it plots the intersection of the unit sphere with the lines formed by 2 points,
    which are:
    - The centre of the unit sphere.
    - The points given in inputs.
    So, each line if formed by these 2 points and only the intersection between lines and 
    the unit sphere is shown.
    The color of the dot is different for each output class.
    All the lines start from the center of the sphere.
    The idea is to plot the data on the surface of the unit sphere.

    Parameters
    ---------
 
    - x_coords: np.ndarray, numpy array of the x coordinates; each value is a different 
        point and is associated with y,z coordinates with the same 
        index in y_coords and z_coords respectively.
    - y_coords: np.ndarray, numpy array of the x coordinates; each value is a different 
        point and is associated with x,z coordinates with the same 
        index in x_coords and z_coords respectively.
    - z_coords: np.ndarray, numpy array of the x coordinates; each value is a different 
        point and is associated with x,y coordinates with the same 
        index in x_coords and y_coords respectively.
    - labels: list or numpy array, contains the labels of the output class associated with each 3D point (must be ints).
        Note that, this label is the true id of the output class, not the predicted class from a model.
    - class_ids: list of str, list with the unique ids associated with the labels.
        This list is ordered as: class with label "0" has id class_ids[0].
        Example: labels[22]=0 and class_ids[0]='E. coli'.
    - grid_thickness: int (default=60, suggested value), controls width and height of each cell in the sphere.
        Internally, it computes theta and phi and if theta is smaller than phi, than the cell will be longer and 
        more stretched horizontally, while the opposite happens with phi smaller than than theta (vertical stretch instead).

    Returns
    ------

    matplotlib Figure object of the unit sphere (from 4 different angles) with the data plotted on it.
    """
    t = compute_t(np.array(x_coords), np.array(y_coords), np.array(z_coords))
    intersection_x = x_coords * t
    intersection_y = y_coords * t
    intersection_z = z_coords * t
    x_shape = count_elem_in_container(x_coords)
    len_class_ids = count_elem_in_container(class_ids)
    theta = np.linspace(0, 2 * np.pi, grid_thickness) 
    phi = np.linspace(0, np.pi, int(grid_thickness/3))
    theta_ms, phi_ms = np.meshgrid(theta, phi)
    x = np.sin(phi_ms) * np.cos(theta_ms)
    y = np.sin(phi_ms) * np.sin(theta_ms) 
    z = np.cos(phi_ms)
    colors_ids = get_n_from_unique_colors(len_class_ids)
    colors = [colors_ids[x] for x in labels]
    y_view_point = [30, 90, 60, 60]
    x_view_point = [45, 45, 270, 320]
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    for i in range(len(x_view_point)):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        plot_single_sphere(
            x, 
            y, 
            z, 
            intersection_x, 
            intersection_y, 
            intersection_z, 
            colors, 
            class_ids, 
            labels, 
            y_view_point[i], 
            x_view_point[i],
            ax, 
            x_shape)
    patches = []
    for i in range(len_class_ids): # each class id gets its own color in the legend
        patches.append(mpatches.Patch(color=colors_ids[i], label=class_ids[i]))
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, handles=patches)
    plt.legend(loc='best', handles=patches)
    return fig


def plot_single_sphere(
    x, 
    y, 
    z, 
    intersection_x, 
    intersection_y, 
    intersection_z, 
    colors,
    class_ids,
    labels,
    y_view_point,
    x_view_point,
    ax, 
    x_shape):
    """
    """
    ax.plot_wireframe(x, y, z, linewidth=1, alpha=0.25, color='gray')
    for i in range(x_shape):
        plot_line_in_3d(
        ax, 
        [0, intersection_x[i]], 
        [0, intersection_y[i]], 
        [0, intersection_z[i]], 
        colors[i], 
        class_ids[labels[i]])
    ax.view_init(y_view_point, x_view_point)
    ax.tick_params(axis='both')


def compute_t(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray]:
    """
    Computes the "t" parameter of the line segment equation intersecting a unit sphere.
    The parametric equation of a line segment is:
        `X = x0 + t*x1`,`Y = y0 + t*y1`,`Z = z0 + t*z1`.
    The implicit equation of the sphere is:
        `X^2 + Y^2 + Z^2 = R^2`.
    Therefore, if we plug the first formula in the second, we obtain that:
        `t = sqrt(1 / x1^2 + y1^2 + z1^2)`.
    Since the unit sphere has radius equal to 1 and `x0 = y0 = z0 = 0`, because we want the line 
    segment to start from the center of the unitt sphere.

    Parameters
    ---------

    - x: numpy array, it contains the x coordinates of a point of the line segment which we want to intersect with the
        unit sphere.
    - y: numpy array, it contains the y coordinates of a point of the line segment which we want to intersect with the
        unit sphere.
    - z: numpy array, it contains the z coordinates of a point of the line segment which we want to intersect with the
        unit sphere.

    Returns
    ------

    numpy.ndarray, array where each value is the t parameter used to compute the intersection between the unit
    sphere and a line segment starting from the sphere's center.

    References
    ---------

    https://stackoverflow.com/questions/32571063/specific-location-of-intersection-between-sphere-and-3d-line-segment
    """
    return np.sqrt(1 / ((x * x) + (y * y) + (z * z)))


def plot_line_in_3d(ax: Figure, x: List[float], y: List[float], z: List[float], color:str, label: str) -> None:
    """
    Plots a straight line in a 3D plane with coordinates x, y, z. The line will be marked at the end
    with a dot; the line has continuos style.

    Parameters
    ---------

    - ax:
    - x:
    - y:
    - z:
    - color:
    - label:

    Notes
    ----

    - ms: controls the size of the mark.
    - to make the dot at the end of each line a full color without borders, apply the same coloro to mfc, mfcalt, mec.
    - c: indicates the color of the body of the line.
    """
    ax.plot(
        x, 
        y,
        z,
        'o', 
        ls='-', 
        c=matplotlib.colors.to_rgb('dimgray'),
        linewidth=1, 
        ms=2, 
        markevery=[-1], 
        mfc=color,
        mfcalt=color,
        mec=color,
        label=label)


def plot_legend_tsne(class_ids):
    len_class_ids = count_elem_in_container(class_ids)#
    colors_ids = get_n_from_unique_colors(len_class_ids)#
    patches = []
    for i in range(len_class_ids): # each class id gets its own color in the legend
        patches.append(mpatches.Patch(color=colors_ids[i], label=class_ids[i]))
    plt.legend(loc='best', prop={'size': 8}, handles=patches)


def plot_tsne_2d(points, class_ids, labels):
    fig_res = plt.figure()
    plt.subplot(2, 2, 1)
    fig = plot_tsne_2d_handler(points, class_ids, labels, 20)
    plt.subplot(2, 2, 2)
    fig1 = plot_tsne_2d_handler(points, class_ids, labels, 30)
    plt.subplot(2, 2, 3)
    fig2 = plot_tsne_2d_handler(points, class_ids, labels, 40)
    plt.subplot(2, 2, 4)
    plot_legend_tsne(class_ids)
    plt.tight_layout()
    return fig_res


def plot_tsne_2d_handler(points, class_ids, labels, perplexity=30):
    n_components = 2
    t_sne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init="random",
        random_state=0,
        )
    X_tsne = t_sne.fit_transform(points)
    return helper_plot_tsne2d(
        X_tsne, 
        class_ids, 
        labels, 
        'p= {}'.format(perplexity))


def helper_plot_tsne2d(points, class_ids, labels, subtitle):
    """
    References:
        - https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
    """
    len_class_ids = count_elem_in_container(class_ids)#
    colors_ids = get_n_from_unique_colors(len_class_ids)#
    colors = [colors_ids[x] for x in labels]#
    x, y = points.T
    fig = plt.scatter(x, y, c=colors, s=5, alpha=0.8)
    plt.title(subtitle)
    return fig


def plot_tsne_3d(points, class_ids, labels):
    #fig_res = plt.figure()
    #plt.subplot(2, 2, 1)
    fig = plot_tsne_3d_handler(points, class_ids, labels, 20)
    #plt.subplot(2, 2, 2)
    #fig1 = plot_tsne_3d_handler(points, class_ids, labels, 30)
    #plt.subplot(2, 2, 3)
    #fig2 = plot_tsne_3d_handler(points, class_ids, labels, 40)
    #plt.subplot(2, 2, 4)
    #plot_legend_tsne(class_ids)
    #plt.tight_layout()
    #return fig_res
    return fig


def plot_tsne_3d_handler(points, class_ids, labels, perplexity):
    #if points.shape[1] > 3:
        #pca = PCA(n_components=3)
        #raw_points = pca.fit_transform(points)
    #else:
        #raw_points = points
    raw_points = points
    t_sne = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="random",
        random_state=0,
        )
    X_tsne = t_sne.fit_transform(raw_points)
    return helper_plot_tsne3d(
        X_tsne, 
        class_ids, 
        labels, 
        'p= {}'.format(perplexity))


def helper_plot_tsne3d(points, class_ids, labels, title):
    """
    References:
        - https://scikit-learn.org/stable/auto_examples/manifold/plot_manifold_sphere.html#sphx-glr-auto-examples-manifold-plot-manifold-sphere-py
    """
    x, y, z = points.T
    len_class_ids = count_elem_in_container(class_ids)#
    colors_ids = get_n_from_unique_colors(len_class_ids)#
    colors = [colors_ids[x] for x in labels]#
    #fig = plt.scatter(x, y, z, c=colors, s=5, alpha=0.8)
    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    ax.scatter(x, y, z, c=colors, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))
    """
    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    ax.scatter(x, y, z, c=colors, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    """
    plt.title(title)
    return fig