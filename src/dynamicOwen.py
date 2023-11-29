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
from .utils import power_set, reduce_1d_list, eval_utility, split_permutation_num


def exact_owen(x_train, y_train, x_test, y_test, model, union_description):
    """
    Calculating the Owen value of data points with exact method
    (Owen value definition)

    :param x_train:  features of train dataset
    :param y_train:  labels of train dataset
    :param x_test:   features of test dataset
    :param y_test:   labels of test dataset
    :param model:    the selected model
    :return: Owen value array `ov`
    :rtype: np.ndarray
    """

    exact_ov = np.zeros(len(y_train))
    union_num = len(union_description)
    union_coef = np.zeros(union_num)
    inner_coefs = []
    inner_setss = []
    for i in range(union_num):
        union_coef[i] = 1 / math.comb(union_num - 1, i)
        inner_num = len(union_description[i])
        inner_coef = np.zeros(inner_num)
        for j in range(inner_num):
            inner_coef[j] = 1 / math.comb(inner_num - 1, j)
        inner_coefs.append(inner_coef)
        inner_coalition = union_description[i]
        inner_sets = list(power_set(inner_coalition))
        inner_setss.append(inner_sets)
    union_coalition = np.arange(union_num)
    union_coalition_set = set(union_coalition)
    union_sets = list(power_set(union_coalition))
    for union_sets_idx in trange(len(union_sets)):
        union_set = union_sets[union_sets_idx]
        data_idx = list(reduce_1d_list(union_description[union_set]))
        x_temp = x_train[data_idx]
        y_temp = y_train[data_idx]
        for inner_idx in union_coalition_set - set(union_sets[union_sets_idx]):
            for inner_set in inner_setss[inner_idx]:
                x_tt = deepcopy(x_temp)
                y_tt = deepcopy(y_temp)
                for inner_set_idx in inner_set:
                    x_tt.append(x_train[inner_set_idx])
                    y_tt.append(y_train[inner_set_idx])
                u = eval_utility(x_tt, y_tt, x_test, y_test, model)
                for i in inner_set:
                    exact_ov[i] += (
                        union_coef[len(union_set)]
                        * inner_coefs[inner_idx][len(inner_set) - 1]
                        * u
                        / len(union_description[inner_idx])
                        / union_num
                    )
                for i in set(union_description[inner_idx]) - set(inner_set):
                    exact_ov[i] -= (
                        union_coef[len(union_set)]
                        * inner_coefs[inner_idx][len(inner_set)]
                        * u
                        / len(union_description[inner_idx])
                        / union_num
                    )
    return exact_ov


def mc_owen(
    x_train,
    y_train,
    x_test,
    y_test,
    model,
    union_description,
    m,
    proc_num=1,
    flag_abs=False,
) -> np.ndarray:
    """
    Calculating the Owen value of data points with
    Monte Carlo Method (multi-process supported)

    :param x_train:  features of train dataset
    :param y_train:  labels of train dataset
    :param x_test:   features of test dataset
    :param y_test:   labels of test dataset
    :param model:    the selected model
    :param m:        the permutation number
    :param proc_num: (optional) Assign the proc num with multi-processing
                     support. Defaults to ``1``.
    :param flag_abs: (optional) Whether use the absolution marginal
                     contribution. Defaults to ``False``.
    :return: Owen value array `ov`
    :rtype: numpy.ndarray
    """

    if proc_num < 0:
        raise ValueError("Invalid proc num.")

    # assign the permutation of each process
    args = split_permutation_num(m, proc_num)
    pool = Pool()
    func = partial(
        _mc_owen_sub_task,
        x_train,
        y_train,
        x_test,
        y_test,
        model,
        union_description,
        flag_abs,
    )
    ret = pool.map(func, args)
    pool.close()
    pool.join()
    ret_arr = np.asarray(ret)
    return np.sum(ret_arr, axis=0) / m


def _mc_owen_sub_task(
    x_train, y_train, x_test, y_test, model, union_description, flag_abs, local_m
) -> np.ndarray:
    local_state = np.random.RandomState(None)

    n = len(y_train)
    ov = np.zeros(n)
    union_num = len(union_description)
    union_idxs = np.arange(union_num)
    for _ in trange(local_m):
        local_state.shuffle(union_idxs)
        idxs = []
        for i in union_idxs:
            inner_idxs = np.asarray(union_description[i])
            local_state.shuffle(inner_idxs)
            idxs.append(inner_idxs)
        idxs = list(reduce_1d_list(idxs))
        old_u = 0
        for j in range(1, n + 1):
            temp_x, temp_y = x_train[idxs[:j]], y_train[idxs[:j]]
            temp_u = eval_utility(temp_x, temp_y, x_test, y_test, model)
            contribution = (temp_u - old_u) if not flag_abs else abs(temp_u - old_u)
            ov[idxs[j - 1]] += contribution
            old_u = temp_u
    return ov


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
    ) -> np.ndarray:
        raise UnImpException("add_single_point")

    def del_single_point(self, del_point_idx, flags=None, params=None) -> np.ndarray:
        raise UnImpException("del_single_point")

    def add_multi_points(
        self, add_points_x, add_points_y, add_points_union, flags=None, params=None
    ) -> np.ndarray:
        raise UnImpException("add_multi_points")

    def del_multi_points(self, del_points_idx, flags=None, params=None) -> np.ndarray:
        raise UnImpException("del_multi_points")


class BaseOwen(Owen):
    """Baseline algorithms for dynamically add point(s)."""

    def __init__(self, x_train, y_train, x_test, y_test, model, init_ov) -> None:
        super().__init__(x_train, y_train, x_test, y_test, model, init_ov)

    def add_single_point(
        self, add_point_x, add_point_y, add_point_union, flags=None, params=None
    ) -> np.ndarray:
        """
        Add a single point and update the Owen value with
        baseline algorithm. (GAvg & LAvg)

        :param np.ndarray add_point_x:  the features of the adding point
        :param np.ndarray add_point_y:  the label of the adding point
        :param dict flags:              (unused yet)
        :param dict params:             {'method': 'gavg' or 'lavg'}
        :return: Owen value array `base_so`
        :rtype: np.ndarray
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
    ) -> np.ndarray:
        """
        Add multiple points and update the Owen value with
        baseline algorithm. (GAvg & LAvg)

        :param np.ndarray add_points_x:  the features of the adding points
        :param np.ndarray add_points_y:  the labels of the adding points
        :param None flags:               (unused yet)
        :param dict params:              {'method': 'gavg' or 'lavg'}
        :return: Owen value array `base_sv`
        :rtype: np.ndarray
        """

        if params is None:
            params = {"method": "lavg"}

        method = params["method"]

        add_num = len(add_points_y)

        if method == "gavg":
            global_avg_ov = np.sum(self.init_ov) / len(self.init_ov)
            return np.append(self.init_ov, [global_avg_ov] * add_num)

        elif method == "lavg":
            add_ov = []
            for add_point_union in add_points_union:
                local_avg_ov = np.sum(
                    self.init_ov[self.union_description[add_point_union]]
                ) / len(self.init_ov[self.union_description[add_point_union]])
                add_ov.append(local_avg_ov)
            return np.append(self.init_ov, add_ov)

        else:
            raise ParamError
