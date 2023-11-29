from functools import partial
from itertools import product
from multiprocessing import Pool
import numpy as np
from sklearn import neighbors
from tqdm import tqdm, trange
from .exceptions import UnImpException, ParamError


class Owen(object):
    """A base class for dynamic Owen value computation."""

    def __init__(
        self, x_train, y_train, x_test, y_test, model, init_ov, flags=None, params=None
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model = model
        self.init_ov = init_ov

        self.flags = flags
        self.params = params

    def add_single_point(
        self, add_point_x, add_point_y, flags=None, params=None
    ) -> np.ndarray:
        raise UnImpException("add_single_point")

    def del_single_point(self, del_point_idx, flags=None, params=None) -> np.ndarray:
        raise UnImpException("del_single_point")

    def add_multi_points(
        self, add_points_x, add_points_y, flags=None, params=None
    ) -> np.ndarray:
        raise UnImpException("add_multi_points")

    def del_multi_points(self, del_points_idx, flags=None, params=None) -> np.ndarray:
        raise UnImpException("del_multi_points")


class BaseOwen(Owen):
    """Baseline algorithms for dynamically add point(s)."""

    def __init__(self, x_train, y_train, x_test, y_test, model, init_ov) -> None:
        super().__init__(x_train, y_train, x_test, y_test, model, init_ov)

    def add_single_point(
        self, add_point_x, add_point_y, flags=None, params=None
    ) -> np.ndarray:
        """
        Add a single point and update the Owen value with
        baseline algorithm. (Avg & LOO)

        :param np.ndarray add_point_x:  the features of the adding point
        :param np.ndarray add_point_y:  the label of the adding point
        :param dict flags:              (unused yet)
        :param dict params:             {'method': 'avg' or 'loo'}
        :return: Owen value array `base_so`
        :rtype: numpy.ndarray
        """
        if params is None:
            params = {"method": "avg"}
        return self.add_multi_points(
            np.asarray([add_point_x]), np.asarray([add_point_y]), None, params
        )

    def add_multi_points(
        self, add_points_x, add_points_y, flags=None, params=None
    ) -> np.ndarray:
        """
        Add multiple points and update the Owen value with
        baseline algorithm. (Avg & LOO)

        :param np.ndarray add_points_x:  the features of the adding points
        :param np.ndarray add_points_y:  the labels of the adding points
        :param None flags:               (unused yet)
        :param dict params:              {'method': 'avg' or 'loo'}
        :return: Owen value array `base_sv`
        :rtype: numpy.ndarray
        """
