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
from .utils import (
    power_set,
    reduce_1d_list,
    eval_utility,
    split_permutation_num,
    split_permutations_t_list,
)


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

    def __init__(
        self, x_train, y_train, x_test, y_test, model, init_ov, union_description
    ) -> None:
        super().__init__(
            x_train, y_train, x_test, y_test, model, init_ov, union_description
        )

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
        :return: Owen value array `base_ov`
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


class PivotOwen(Owen):
    """Pivot-based algorithm"""

    def __init__(
        self, x_train, y_train, x_test, y_test, model, init_ov, union_description
    ) -> None:
        super().__init__(
            x_train, y_train, x_test, y_test, model, init_ov, union_description
        )
        # If no prepare, pass left ov by init ov
        self.lov = init_ov
        self.t_list = None
        self.permutations = None
        self.proc_num = None

    def prepare(self, m, proc_num=1) -> np.ndarray:
        """
        Prepare procedure needed by pivot based dynamic Owen algorithm.
        (Phase Initialization)

        Calculating the left part of the permutations.

        :param proc_num: (optional) Assign the proc num with multi-processing
                         support. Defaults to ``1``.
        :param int m:    The number of the permutations.
        :return: the lov `left_ov` and the t list
        :rtype: np.ndarray
        """

        if proc_num <= 0:
            raise ValueError("Invalid proc num.")

        self.proc_num = proc_num

        n = len(self.y_train)
        self.init_ov = np.zeros(n)

        args = split_permutation_num(m, proc_num)
        pool = Pool()
        func = partial(
            PivotOwen._pivot_prepare_sub_task,
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
            self.model,
            self.union_description,
        )
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        ret_arr = np.asarray(ret, dtype=object)

        self.lov = np.sum([r[0] for r in ret_arr], axis=0) / m
        self.permutations = np.concatenate([r[1] for r in ret_arr], axis=0)
        self.t_list = np.concatenate([r[2] for r in ret_arr], axis=0)
        return self.lov

    @staticmethod
    def _pivot_prepare_sub_task(
        x_train, y_train, x_test, y_test, model, union_description, local_m
    ) -> (np.ndarray, np.ndarray, list):
        local_state = np.random.RandomState(None)

        n = len(y_train)
        lov = np.zeros(n)
        union_num = len(union_description)
        union_idxs = np.arange(union_num)
        t_list = list()
        permutations = list()

        for _ in trange(local_m):
            local_state.shuffle(union_idxs)
            idxs = []
            for i in union_idxs:
                inner_idxs = np.asarray(union_description[i])
                local_state.shuffle(inner_idxs)
                idxs.append(inner_idxs)
            idxs = list(reduce_1d_list(idxs))
            # Draw t from 0 to n
            t = local_state.randint(0, n + 1)

            # Record trim position and the corresponding sequence
            permutations.append(idxs)
            t_list.append(t)

            old_u = 0
            for j in range(1, t + 1):
                temp_x, temp_y = x_train[idxs[:j]], y_train[idxs[:j]]
                temp_u = eval_utility(temp_x, temp_y, x_test, y_test, model)
                lov[idxs[j - 1]] += temp_u - old_u
                old_u = temp_u
        return lov, permutations, t_list

    def add_single_point(
        self,
        add_point_x,
        add_point_y,
        add_point_union,
        m=None,
        proc_num=1,
        flags=None,
        params=None,
    ) -> np.ndarray:
        """
        Add a single point and update the Owen value with pivot based
        algorithm.

        :param np.ndarray add_point_x:  the features of the adding point
        :param np.ndarray add_point_y:  the label of the adding point
        :param int m:                   the num of permutations
        :param int proc_num:            (optional) Assign the proc num with
                                        multi-processing support. Defaults
                                        to ``1``.
        :param dict flags:              {'flag_lov': False} ('flag_lov'
                                        represents that update left ov or not
                                        Defaults to ``False``.)
        :param dict params:             (unused yet)
        :return: Owen value array `pivot_ov`
        :rtype: numpy.ndarray
        """

        if flags is None:
            flags = {"flag_lov": False}

        # Extract flags & params
        flag_lov = flags["flag_lov"]

        new_x_train = np.append(self.x_train, [add_point_x], axis=0)
        new_y_train = np.append(self.y_train, add_point_y)
        new_union_description = deepcopy(self.union_description)
        new_union_description[add_point_union].append(len(new_y_train))

        # Init left part and right part
        lov = np.append(self.lov, 0)

        pool = Pool()
        if m is None:
            m = len(self.t_list)
        args = split_permutation_num(m, proc_num)
        f = PivotOwen._pivot_add_sub_task

        func = partial(
            f,
            new_x_train,
            new_y_train,
            self.x_test,
            self.y_test,
            self.model,
            new_union_description,
        )
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        ret_arr = np.asarray(ret, dtype=object)
        rov = np.sum([r[0] for r in ret_arr], axis=0) / m
        delta_lov = np.sum([r[1] for r in ret_arr], axis=0) / m
        pivot_ov = lov + rov

        if flag_lov:
            self.x_train = new_x_train
            self.y_train = new_y_train
            self.lov = lov * 2 / 3 + delta_lov
        return np.asarray(pivot_ov, dtype=float)

    @staticmethod
    def _pivot_add_sub_task(
        new_x_train, new_y_train, x_test, y_test, model, new_union_description, local_m
    ) -> (np.ndarray, np.ndarray):
        """The Owen value calculation with different permutations."""

        local_state = np.random.RandomState(None)

        n = len(new_y_train) - 1
        rov = np.zeros(n + 1)
        delta_lov = np.zeros(n + 1)
        union_num = len(new_union_description)
        union_idxs = np.arange(union_num)

        for _ in trange(local_m):
            # Draw p from 0 to n + 1 (the size of new_y_train)
            p = local_state.randint(0, n + 2)
            local_state.shuffle(union_idxs)
            idxs = []
            for i in union_idxs:
                inner_idxs = np.asarray(new_union_description[i])
                local_state.shuffle(inner_idxs)
                idxs.append(inner_idxs)
            idxs = list(reduce_1d_list(idxs))
            t = 0
            for t in range(n + 1):
                # Find the new added point's idx
                if idxs[t] == n:
                    break

            # Evaluate utility excluding the new added point
            if t == 0:
                old_u = 0
            else:
                temp_x, temp_y = new_x_train[idxs[:t]], new_y_train[idxs[:t]]
                old_u = eval_utility(temp_x, temp_y, x_test, y_test, model)

            # Evaluate utility including the new added point (from t+1 to n+1)
            for j in range(t + 1, n + 2):
                temp_x, temp_y = new_x_train[idxs[:j]], new_y_train[idxs[:j]]
                temp_u = eval_utility(temp_x, temp_y, x_test, y_test, model)

                if p >= j:
                    delta_lov[idxs[j - 1]] += temp_u - old_u

                rov[idxs[j - 1]] += temp_u - old_u
                old_u = temp_u

        return rov, delta_lov


class DeltaOwen(Owen):
    """Delta based algorithm for dynamically add/delete a single point."""

    def __init__(
        self, x_train, y_train, x_test, y_test, model, init_ov, union_description
    ) -> None:
        super().__init__(
            x_train, y_train, x_test, y_test, model, init_ov, union_description
        )
        self.m = None

    def add_single_point(
        self,
        add_point_x,
        add_point_y,
        add_point_union,
        m,
        proc_num=1,
        flags=None,
        params=None,
    ) -> np.ndarray:
        """
        Add a single point and update the Owen value with delta based
        algorithm.

        :param np.ndarray add_point_x:  the features of the adding point
        :param np.ndarray add_point_y:  the label of the adding point
        :param int m:                   the number of permutations
        :param int proc_num:            the number of proc
        :param dict flags:              (optional) {'flag_update': True or
                                        False} Defaults to ``False``.
        :param dict params:             (unused yet)
        :return: Owen value array `delta_ov`
        :rtype: numpy.ndarray
        """

        self.m = m
        if proc_num <= 0:
            raise ValueError("Invalid proc num.")

        if flags is None:
            flags = {"flag_update": False}

        flag_update = flags["flag_update"]

        new_union_description = deepcopy(self.union_description)
        new_union_description[add_point_union].append(len(self.y_train) + 1)

        # assign the permutation of each process
        args = split_permutation_num(m, proc_num)
        pool = Pool()
        func = partial(
            DeltaOwen._delta_add_sub_task,
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
            self.model,
            add_point_x,
            add_point_y,
            new_union_description,
        )
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        ret_arr = np.asarray(ret)
        delta = np.sum(ret_arr, axis=0) / m
        delta_ov = np.append(self.init_ov, 0) + delta

        if flag_update:
            self.x_train = np.append(self.x_train, [add_point_x], axis=0)
            self.y_train = np.append(self.y_train, add_point_y)
            self.init_ov = delta_ov

        return delta_ov

    @staticmethod
    def _delta_add_sub_task(
        x_train,
        y_train,
        x_test,
        y_test,
        model,
        add_point_x,
        add_point_y,
        new_union_description,
        local_m,
    ) -> np.ndarray:
        local_state = np.random.RandomState(None)

        n = len(y_train)
        union_num = len(new_union_description)
        union_idxs = np.arange(union_num)
        delta = np.zeros(n + 1)

        origin_margin = eval_utility(
            [add_point_x], [add_point_y], x_test, y_test, model
        )

        delta[-1] += origin_margin / (n + 1) * local_m

        for _ in trange(local_m):
            local_state.shuffle(union_idxs)
            idxs = []
            for i in union_idxs:
                inner_idxs = np.asarray(new_union_description[i])
                local_state.shuffle(inner_idxs)
                idxs.append(inner_idxs)
            idxs = list(reduce_1d_list(idxs))
            for i in range(1, n + 1):
                temp_x, temp_y = (x_train[idxs[:i]], y_train[idxs[:i]])

                u_no_np = eval_utility(temp_x, temp_y, x_test, y_test, model)

                u_with_np = eval_utility(
                    np.append(temp_x, [add_point_x], axis=0),
                    np.append(temp_y, add_point_y),
                    x_test,
                    y_test,
                    model,
                )

                current_margin = u_with_np - u_no_np

                delta[idxs[i - 1]] += (current_margin - origin_margin) / (n + 1) * i

                delta[-1] += current_margin / (n + 1)
                origin_margin = current_margin
        return delta

    def del_single_point(
        self, del_point_idx, m, proc_num=1, flags=None, params=None
    ) -> np.ndarray:
        """
        Delete a single point and update the Owen value with
        delta based algorithm. (KNN & KNN+)

        :param int del_point_idx:   the index of the deleting point
        :param m:                   the number of permutations
        :param proc_num:            the number of proc
        :param dict flags:          (optional) {'flag_update': True or False},
                                    Defaults to ``False``.
        :param dict params:         (unused yet)
        :return: Owen value array `delta_ov`
        :rtype: numpy.ndarray
        """

        self.m = m

        if proc_num <= 0:
            raise ValueError("Invalid proc num.")

        if flags is None:
            flags = {"flag_update": False}

        flag_update = flags["flag_update"]

        # assign the permutation of each process
        args = split_permutation_num(m, proc_num)
        pool = Pool()
        func = partial(
            DeltaOwen._delta_del_sub_task,
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
            self.model,
            self.union_description,
            del_point_idx,
        )
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        ret_arr = np.asarray(ret)
        delta = np.sum(ret_arr, axis=0) / m
        delta_ov = np.delete(self.init_ov, del_point_idx) + delta

        if flag_update:
            self.x_train = np.delete(self.x_train, del_point_idx, axis=0)
            self.y_train = np.delete(self.y_train, del_point_idx)
            self.init_ov = delta_ov

        return delta_ov

    @staticmethod
    def _delta_del_sub_task(
        x_train,
        y_train,
        x_test,
        y_test,
        model,
        union_description,
        del_point_idx,
        local_m,
    ) -> np.ndarray:
        local_state = np.random.RandomState(None)

        n = len(y_train)
        deleted_idxs = np.delete(np.arange(n), del_point_idx)
        fixed_idxs = np.copy(deleted_idxs)
        union_num = len(union_description)
        union_idxs = np.arange(union_num)
        delta = np.zeros(n - 1)

        origin_margin = eval_utility(
            [x_train[del_point_idx, :]], [y_train[del_point_idx]], x_test, y_test, model
        )

        for _ in trange(local_m):
            local_state.shuffle(union_idxs)
            idxs = []
            for i in union_idxs:
                inner_idxs = np.asarray(union_description[i])
                local_state.shuffle(inner_idxs)
                idxs.append(inner_idxs)
            idxs = list(reduce_1d_list(idxs))
            deleted_idxs = np.delete(idxs, del_point_idx)
            for j in range(1, n):
                temp_x, temp_y = (x_train[deleted_idxs[:j]], y_train[deleted_idxs[:j]])

                acc_no_op = eval_utility(temp_x, temp_y, x_test, y_test, model)

                temp_x, temp_y = (
                    np.append(temp_x, [x_train[del_point_idx]], axis=0),
                    np.append(temp_y, y_train[del_point_idx]),
                )

                acc_with_op = eval_utility(temp_x, temp_y, x_test, y_test, model)

                current_margin = acc_with_op - acc_no_op

                idx = np.where(fixed_idxs == deleted_idxs[j - 1])[0]
                delta[idx] += (-current_margin + origin_margin) / n * j
                origin_margin = current_margin
        return delta
