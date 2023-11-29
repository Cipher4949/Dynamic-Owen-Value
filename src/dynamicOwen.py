from functools import partial
from itertools import product
from multiprocessing import Pool
import numpy as np
from sklearn import neighbors
from tqdm import tqdm, trange
from .exceptions import UnImpException, ParamError
from typing import List
from copy import deepcopy
import math
from .utils import power_set, reduce_1d_list, eval_utility


def exact_owen(x_train, y_train, x_test, y_test, model):
    """
    Calculating the Owen value of data points with exact method
    (Owen value definition)

    :param x_train:  features of train dataset
    :param y_train:  labels of train dataset
    :param x_test:   features of test dataset
    :param y_test:   labels of test dataset
    :param model:    the selected model
    :return: Owen value array `ov`
    :rtype: List[List]
    """

    union_num = len(y_train)
    union_coef = np.zeros(union_num)
    inner_coefs = []
    inner_setss = []
    for i in range(union_num):
        union_coef[i] = 1 / math.comb(union_num - 1, i)
        inner_num = len(y_train[i])
        inner_coef = np.zeros(inner_num)
        for j in range(inner_num):
            inner_coef[j] = 1 / math.comb(inner_num - 1, j)
        inner_coefs.append(inner_coef)
        inner_coalition = np.arange(inner_num)
        inner_sets = list(power_set(inner_coalition))
        inner_setss.append(inner_sets)
    union_coalition = np.arange(union_num)
    union_coalition_set = set(union_coalition)
    union_sets = list(power_set(union_coalition))
    for union_sets_idx in trange(len(union_sets)):
        x_temp = reduce_1d_list(x_train[list(union_sets[union_sets_idx])])
        y_temp = reduce_1d_list(y_train[list(union_sets[union_sets_idx])])
        for inner_idx in union_coalition_set - set(union_sets[union_sets_idx]):
            for inner_set in inner_setss[inner_idx]:
                x_tt = deepcopy(x_temp)
                y_tt = deepcopy(y_temp)
                for inner_set_idx in inner_set:
                    x_tt.append(x_train[inner_idx][inner_set_idx])
                    y_tt.append(y_train[inner_idx][inner_set_idx])
                u = eval_utility(x_tt, y_tt, x_test, y_test, model)


class Owen(object):
    """A base class for dynamic Owen value computation."""

    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        model,
        init_ov,
        union_description,
        flags=None,
        params=None,
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model = model
        self.init_ov = init_ov

        self.union_description = union_description

        self.flags = flags
        self.params = params

    def add_single_point(
        self, add_point_x, add_point_y, add_point_union, flags=None, params=None
    ) -> List[List]:
        raise UnImpException("add_single_point")

    def del_single_point(self, del_point_idx, flags=None, params=None) -> List[List]:
        raise UnImpException("del_single_point")

    def add_multi_points(
        self, add_points_x, add_points_y, add_points_union, flags=None, params=None
    ) -> List[List]:
        raise UnImpException("add_multi_points")

    def del_multi_points(self, del_points_idx, flags=None, params=None) -> List[List]:
        raise UnImpException("del_multi_points")


class BaseOwen(Owen):
    """Baseline algorithms for dynamically add point(s)."""

    def __init__(self, x_train, y_train, x_test, y_test, model, init_ov) -> None:
        super().__init__(x_train, y_train, x_test, y_test, model, init_ov)

    def add_single_point(
        self, add_point_x, add_point_y, add_point_union, flags=None, params=None
    ) -> List[List]:
        """
        Add a single point and update the Owen value with
        baseline algorithm. (GAvg & LAvg)

        :param np.ndarray add_point_x:  the features of the adding point
        :param np.ndarray add_point_y:  the label of the adding point
        :param dict flags:              (unused yet)
        :param dict params:             {'method': 'gavg' or 'lavg'}
        :return: Owen value array `base_so`
        :rtype: List[List]
        """
        if params is None:
            params = {"method": "lavg"}
        return self.add_multi_points(
            np.asarray([add_point_x]),
            np.asarray([add_point_y]),
            np.asarray([add_point_union]),
            None,
            params,
        )

    def add_multi_points(
        self, add_points_x, add_points_y, add_points_union, flags=None, params=None
    ) -> List[List]:
        """
        Add multiple points and update the Owen value with
        baseline algorithm. (GAvg & LAvg)

        :param np.ndarray add_points_x:  the features of the adding points
        :param np.ndarray add_points_y:  the labels of the adding points
        :param None flags:               (unused yet)
        :param dict params:              {'method': 'gavg' or 'lavg'}
        :return: Owen value array `base_sv`
        :rtype: List[List]
        """

        if params is None:
            params = {"method": "lavg"}

        method = params["method"]

        add_num = len(add_points_y)

        if method == "gavg":
            global_ov_sum = 0
            agent_num = 0
            for init_union_ov in self.init_ov:
                global_ov_sum += sum(init_union_ov)
                agent_num += len(init_union_ov)
            global_avg_ov = global_ov_sum / agent_num
            updated_ov = deepcopy(self.init_ov)
            for add_point_union in add_points_union:
                updated_ov[add_point_union].append(global_avg_ov)
            return updated_ov

        elif method == "lavg":
            updated_ov = deepcopy(self.init_ov)
            for add_point_union in add_points_union:
                local_avg_ov = sum(self.init_ov[add_point_union]) / len(
                    self.init_ov[add_point_union]
                )
                updated_ov[add_point_union].append(local_avg_ov)
            return updated_ov

        else:
            raise ParamError
