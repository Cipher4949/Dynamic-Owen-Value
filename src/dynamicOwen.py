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
    eval_simi,
    eval_ovc,
    get_ele_idxs,
)
from .structures import SimiPreData


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
        to_reduce = []
        for i in union_set:
            to_reduce.append(union_description[i])
        data_idx = list(reduce_1d_list(to_reduce))
        x_temp = x_train[data_idx]
        y_temp = y_train[data_idx]
        for inner_idx in union_coalition_set - set(union_set):
            for inner_set in inner_setss[inner_idx]:
                x_tt = deepcopy(x_temp)
                y_tt = deepcopy(y_temp)
                for inner_set_idx in inner_set:
                    x_tt = np.append(x_tt, x_train[inner_set_idx])
                    y_tt = np.append(y_tt, y_train[inner_set_idx])
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
    x_train,
    y_train,
    x_test,
    y_test,
    model,
    union_description,
    flag_abs,
    local_m,
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
        real_n = len(idxs)
        for j in range(1, real_n + 1):
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
        new_union_description[add_point_union].append(len(new_y_train) - 1)

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
            self.union_description,
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


class YnOwen(Owen):
    """YN-NN algorithm for dynamically delete point(s)."""

    def __init__(
        self, x_train, y_train, x_test, y_test, model, init_ov, union_description
    ) -> None:
        super().__init__(
            x_train, y_train, x_test, y_test, model, init_ov, union_description
        )
        self.MAX_DEL_NUM = 2
        self.del_num = None
        self.yn = None
        self.nn = None

    def prepare(self, del_num, flags=None, params=None) -> [np.ndarray, np.ndarray]:
        """
        The prepare procedure needed by YN-NN algorithm, which needs
        to fill in the multi-dimension array.

        :param del_num:        the number of points which need to be deleted
        :param dict flags:     {'exact': True or False,}
        :param dict params:    (optional) {'mc_type': 0 or 1, 'm': ...} (it is
                               needed when 'exact' is False)
        :return: `yn` and `nn`
        :rtype: tuple([np.ndarray, np.ndarray]
        """
        if flags is None:
            flags = {"exact": True}

        n = len(self.y_train)
        self.del_num = del_num

        if self.del_num > self.MAX_DEL_NUM:
            raise ParamError(
                "the number of delete points cannot > %d" % self.MAX_DEL_NUM
            )

        shape = tuple([n]) * (del_num + 2)
        self.yn = np.zeros(shape=shape)
        self.nn = np.zeros(shape=shape)

        if flags["exact"]:
            union_num = len(self.union_description)
            union_coef = np.zeros(union_num)
            inner_coefs = []
            inner_setss = []
            for i in range(union_num):
                union_coef[i] = 1 / math.comb(union_num - 1, i)
                inner_num = len(self.union_description[i])
                inner_coef = np.zeros(inner_num)
                for j in range(inner_num):
                    inner_coef[j] = 1 / math.comb(inner_num - 1, j)
                inner_coefs.append(inner_coef)
                inner_coalition = self.union_description[i]
                inner_sets = list(power_set(inner_coalition))
                inner_setss.append(inner_sets)
            union_coalition = np.arange(union_num)
            union_coalition_set = set(union_coalition)
            union_sets = list(power_set(union_coalition))
            for union_sets_idx in trange(len(union_sets)):
                union_set = union_sets[union_sets_idx]
                to_reduce = []
                for i in union_set:
                    to_reduce.append(self.union_description[i])
                data_idx = list(reduce_1d_list(to_reduce))
                x_temp = self.x_train[data_idx]
                y_temp = self.y_train[data_idx]
                for inner_idx in union_coalition_set - set(union_set):
                    for inner_set in inner_setss[inner_idx]:
                        x_tt = deepcopy(x_temp)
                        y_tt = deepcopy(y_temp)
                        for inner_set_idx in inner_set:
                            x_tt = np.append(x_tt, self.x_train[inner_set_idx])
                            y_tt = np.append(y_tt, self.y_train[inner_set_idx])
                        u = eval_utility(
                            x_tt, y_tt, self.x_test, self.y_test, self.model
                        )

                        # Assign utility to array
                        l = len(inner_set)
                        Y = list(inner_set)
                        N = list(set(self.union_description[inner_idx]) - set(Y))

                        if self.del_num == 1:
                            for j, k in product(Y, N):
                                self.yn[j, k, l] += (
                                    union_coef[len(union_set)]
                                    * inner_coefs[inner_idx][len(inner_set) - 1]
                                    * u
                                    / len(self.union_description[inner_idx])
                                    / union_num
                                )
                            for j, k in product(N, N):
                                self.nn[j, k, l] += (
                                    union_coef[len(union_set)]
                                    * inner_coefs[inner_idx][len(inner_set)]
                                    * u
                                    / len(self.union_description[inner_idx])
                                    / union_num
                                )
                        else:
                            for j, k, p in product(Y, N, N):
                                self.yn[j, k, p, l] += (
                                    union_coef[len(union_set)]
                                    * inner_coefs[inner_idx][len(inner_set) - 1]
                                    * u
                                    / len(self.union_description[inner_idx])
                                    / union_num
                                )
                            for j, k, p in product(N, N, N):
                                self.nn[j, k, p, l] += (
                                    union_coef[len(union_set)]
                                    * inner_coefs[inner_idx][len(inner_set)]
                                    * u
                                    / len(self.union_description[inner_idx])
                                    / union_num
                                )

        else:
            mc_type = params["mc_type"]
            m = params["m"]

            union_num = len(self.union_description)
            union_idxs = np.arange(union_num)
            for _ in trange(int(n * m)):
                np.random.shuffle(union_idxs)
                idxs = []
                for i in union_idxs:
                    inner_idxs = np.asarray(self.union_description[i])
                    np.random.shuffle(inner_idxs)
                    idxs.append(inner_idxs)
                idxs = list(reduce_1d_list(idxs))
                old_u = 0
                for l in range(1, n + 1):
                    temp_x, temp_y = (self.x_train[idxs[:l]], self.y_train[idxs[:l]])
                    temp_u = eval_utility(
                        temp_x, temp_y, self.x_test, self.y_test, self.model
                    )

                    # Assign utility to array
                    N = idxs[l:]
                    if mc_type == 0:
                        j = idxs[l - 1]
                        if self.del_num == 1:
                            for k in N:
                                self.yn[j, k, l] += temp_u
                                self.nn[j, k, l - 1] += old_u
                        else:
                            for k, p in product(N, N):
                                self.yn[j, k, p, l] += temp_u
                                self.nn[j, k, p, l - 1] += old_u
                    else:
                        Y = idxs[:l]
                        if self.del_num == 1:
                            for j, k in product(Y, N):
                                self.yn[j, k, l] += temp_u
                            for j, k in product(N, N):
                                self.nn[j, k, l] += temp_u
                        else:
                            for j, k, p in product(Y, N, N):
                                self.yn[j, k, p, l] += temp_u
                            for j, k, p in product(N, N, N):
                                self.nn[j, k, p, l] += temp_u

                    old_u = temp_u

            self.yn, self.nn = self.yn / m, self.nn / m

        return self.yn, self.nn

    def del_single_point(self, del_point_idx, flags=None, params=None) -> np.ndarray:
        """
        Delete a single point and update the Owen value with
        YN-NN algorithm.

        :param int del_point_idx:    the index of the deleting point
        :param dict flags:           {'exact': True or False}
        :param dict params:          (unused yet)
        :return: Owen value array `yn_ov`
        :rtype: numpy.ndarray
        """

        if flags is None:
            flags = {"exact": True}

        return self.del_multi_points([del_point_idx], flags, params)

    def del_multi_points(self, del_points_idx, flags=None, params=None) -> np.ndarray:
        """
        Delete multiple points and update the Owen value with
        YN-NN (YNN-NNN) algorithm. (KNN & KNN+)

        :param list del_points_idx:  the index of the deleting points
        :param dict flags:           {'exact': True or False}
        :param dict params:          (unused yet)
        :return: Owen value array `yn_ov`
        :rtype: numpy.ndarray
        """

        if flags is None:
            flags = {"exact": True}

        if len(del_points_idx) > self.del_num:
            raise ParamError("delete too many points")

        n = len(self.y_train)
        yn_ov = np.zeros(n)
        walk_arr = np.delete(np.arange(n), np.asarray(del_points_idx))

        for i, j in product(walk_arr, range(1, 1 + len(walk_arr))):
            t = tuple(del_points_idx[: self.del_num])
            if not flags["exact"] and params["mc_type"] == 0:
                modi_coef_method = 0
            else:
                modi_coef_method = 1
            yn_ov[i] += (
                self.yn[(i,) + t + (j,)] - self.nn[(i,) + t + (j - 1,)]
            ) * YnOwen._modi_coef(n, j, self.del_num, modi_coef_method)
        return np.delete(yn_ov, del_points_idx)

    @staticmethod
    def _modi_coef(n, j, num, method):
        res = 1
        for i in range(num):
            if method == 0:
                res *= n - 1 - i
            else:
                res *= n - i
            res /= n - j - i
        return res


class HeurOwen(Owen):
    """Heuristic dynamic Owen algorithm, including KNN and KNN+ version"""

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
        """
        :param flags: unused yet
        :param params: {'method': 'knn' or 'knn+'}
        """
        super().__init__(
            x_train, y_train, x_test, y_test, model, init_ov, union_description
        )

        if params is None:
            params = {"method": "knn"}

        # Extract param
        self.method = params["method"]

        self.n_neighbors = None
        self.clf = None
        self.simi_type = None
        self.m = None
        self.spd = None

    def prepare(self, flags=None, params=None) -> None:
        """
        The prepare procedure needed by heuristic algorithm, including
        KNN clf training, curve functions generating and etc.

        :param dict flags:  {'exact': True or False,
                             'train': True or False}
        :param dict params: {'n_neighbors': 5,
                             'simi_type': 'ed' or 'cos',
                             'f_shap': 'n*n',
                             'rela': ['poly', 2],
                             'train_idxs': []}
                             (['poly', x] | x in [1, ..., N],
                             in default x is 2)
        :return: None
        :rtype: None
        """

        if flags is None:
            flags = {"exact": False, "train": True}

        # Extract param & flags
        self.n_neighbors = params["n_neighbors"]

        self.clf = neighbors.NearestNeighbors(n_neighbors=self.n_neighbors).fit(
            self.x_train, self.y_train
        )
        if self.method == "knn+":
            # Curve fitting
            flag_train = flags["train"]
            self.simi_type = params["simi_type"]

            n = len(self.y_train)

            if not flag_train:
                self.spd = SimiPreData(params)
            else:
                flag_ext = flags["exact"]
                train_idxs = params["train_idxs"]
                if not flag_ext:
                    self.m = params["m"]

                ovs = np.zeros((len(train_idxs), len(self.y_train) - 1))

                for i, train_idx in enumerate(train_idxs):
                    new_union_d = deepcopy(self.union_description)
                    for _ in new_union_d:
                        if train_idx in _:
                            _.remove(train_idx)
                    if flag_ext:
                        ret = exact_owen(
                            self.x_train,
                            self.y_train,
                            self.x_test,
                            self.y_test,
                            self.model,
                            new_union_d,
                        )
                        ovs[i] = np.delete(ret, train_idx)
                    else:
                        ret = mc_owen(
                            self.x_train,
                            self.y_train,
                            self.x_test,
                            self.y_test,
                            self.model,
                            new_union_d,
                            self.m,
                        )
                        ovs[i] = np.delete(ret, train_idx)
                # Fill in SimiPreData
                self.spd = SimiPreData({"train_idxs": train_idxs, "train_ovs": ovs})

            self._fit_curve(params)

    def add_single_point(
        self, add_point_x, add_point_y, add_point_union, flags=None, params=None
    ) -> np.ndarray:
        """
        Add a single point and update the Owen value with
        heuristic algorithm. (KNN & KNN+)

        :param np.ndarray add_point_x:  the features of the adding point
        :param np.ndarray add_point_y:  the label of the adding point
        :param dict flags:              (unused yet)
        :param dict params:             (unused yet)
        :return: Owen value array `knn_ov` or `knn_plus_ov`
        :rtype: numpy.ndarray
        """
        return self.add_multi_points(
            np.asarray([add_point_x]),
            np.asarray([add_point_y]),
            np.asarray([add_point_union]),
            flags,
            params,
        )

    def add_multi_points(
        self, add_points_x, add_points_y, add_points_union, flags=None, params=None
    ) -> np.ndarray:
        """
        Add multiple points and update the Owen value with
        heuristic algorithm. (KNN & KNN+)

        :param np.ndarray add_points_x:  the features of the adding points
        :param np.ndarray add_points_y:  the labels of the adding points
        :param dict flags:               (unused yet)
        :param dict params:              (unused yet)
        :return: Owen value array `knn_ov` or `knn_plus_ov`
        :rtype: numpy.ndarray
        """
        self.add_points_x = add_points_x
        self.add_points_y = add_points_y
        self.add_points_union = add_points_union

        n = len(self.init_ov)
        add_num = len(add_points_y)
        knn_ov = np.append(self.init_ov, [0] * add_num)

        for i in range(add_num):
            x = add_points_x[i]
            neighbor_list = self.clf.kneighbors([x], self.n_neighbors, False)[0]
            nov = np.sum(self.init_ov[neighbor_list]) / self.n_neighbors
            knn_ov[i + n] = nov
        if self.method == "knn":
            return knn_ov
        else:
            # KNN+
            knn_plus_ov = knn_ov

            simi_type = self.simi_type
            ovc = np.zeros(n)

            for r_idx in trange(add_num):
                x_train = np.append(self.x_train, [add_points_x[r_idx]], axis=0)
                y_train = np.append(self.y_train, add_points_y[r_idx])
                # Always add one point
                simi = eval_simi(x_train, y_train, n, simi_type)
                # Calculate ovc with curve functions
                for i in range(n):
                    # Select the corresponding curve function
                    f = np.poly1d(
                        self.curve_funcs[
                            list(self.f_labels).index(add_points_y[r_idx])
                        ][list(self.all_labels).index(self.y_train[i])]
                    )
                    ovc[i] += -f(simi[i]) if simi[i] != 0 else 0

            added_x_train = np.append(self.x_train, add_points_x, axis=0)
            added_y_train = np.append(self.y_train, add_points_y)

            new_u = eval_utility(
                added_x_train, added_y_train, self.x_test, self.y_test, self.model
            )
            knn_plus_ov[:n] += ovc
            knn_plus_ov *= new_u / np.sum(knn_plus_ov)
            return knn_plus_ov

    def del_single_point(self, del_point_idx, flags=None, params=None) -> np.ndarray:
        """
        Delete a single point and update the Owen value with
        heuristic algorithm. (KNN & KNN+)

        :param int del_point_idx:  the index of the deleting point
        :param dict flags:         (unused yet)
        :param dict params:        (unused yet)
        :return: Owen value array `knn_ov` or `knn_plus_ov`
        :rtype: numpy.ndarray
        """
        return self.del_multi_points([del_point_idx], flags, params)

    def del_multi_points(self, del_points_idx, flags=None, params=None) -> np.ndarray:
        """
        Delete multiple points and update the Owen value with
        heuristic algorithm. (KNN & KNN+)

        :param list del_points_idx:  the index of the deleteing points
        :param dict flags:           (unused yet)
        :param dict params:          (unused yet)
        :return: Owen value array `knn_ov` or `knn_plus_ov`
        :rtype: numpy.ndarray
        """
        self.del_points_idx = del_points_idx
        n = len(self.init_ov)

        knn_ov = np.delete(self.init_ov, del_points_idx)

        idxs = np.arange(n)
        deleted_idxs = np.delete(idxs, del_points_idx)
        deleted_x_train = self.x_train[deleted_idxs]
        deleted_y_train = self.y_train[deleted_idxs]
        # Update clf
        clf = neighbors.NearestNeighbors(n_neighbors=self.n_neighbors).fit(
            deleted_x_train, deleted_y_train
        )
        for i in del_points_idx:
            x = self.x_train[i]
            neighbor_list = clf.kneighbors([x], self.n_neighbors, False)[0]
            for k in neighbor_list:
                idx = deleted_idxs[k]
                knn_ov[k] += self.init_ov[idx] / self.n_neighbors
        if self.method == "knn":
            return knn_ov
        else:
            # KNN+
            knn_plus_ov = knn_ov
            simi_type = self.simi_type

            ovc = np.zeros(n)

            f_labels = list(self.f_labels)
            all_labels = list(self.all_labels)
            for idx in tqdm(del_points_idx):
                simi = eval_simi(self.x_train, self.y_train, idx, simi_type)
                # Calculate ovc with curve functions
                for i in range(n):
                    # Select the corresponding curve function
                    f = np.poly1d(
                        self.curve_funcs[f_labels.index(self.y_train[idx])][
                            all_labels.index(self.y_train[i])
                        ]
                    )
                    ovc[i] += f(simi[i]) if simi[i] != 0 else 0

                new_u = eval_utility(
                    deleted_x_train,
                    deleted_y_train,
                    self.x_test,
                    self.y_test,
                    self.model,
                )
                knn_plus_ov += ovc[deleted_idxs]
                knn_plus_ov *= new_u / np.sum(knn_plus_ov)
            return knn_plus_ov

    def _fit_curve(self, params=None) -> None:
        """
        Generate curve functions which represent the relationship between
        the change of Owen value and the similarity.
        """

        if params is None:
            params = {"f_shap": "n*n", "rela": ["poly", 2], "simi_type": "ed"}

        # Extract params
        f_shap = params["f_shap"]
        rela = params["rela"]
        simi_type = params["simi_type"]

        self.f_labels = set(self.y_train[self.spd.train_idxs])
        self.all_labels = set(self.y_train)

        if rela[0] == "poly":
            curve_funcs = np.zeros(
                (len(self.all_labels), len(self.f_labels), rela[1] + 1)
            )
        else:
            raise ParamError("relationship excepting 'ploy' " "is NOT supported yet")

        if f_shap == "n*n":
            for _, train_idx in product(self.f_labels, self.spd.train_idxs):
                current_label_fidx = list(self.f_labels).index(self.y_train[train_idx])
                for idx, k in enumerate(self.all_labels):
                    label_idxs = get_ele_idxs(k, self.y_train)
                    try:
                        label_idxs.remove(train_idx)
                    except ValueError:
                        pass
                    simi = eval_simi(self.x_train, self.y_train, train_idx, simi_type)
                    # Include the train idx point, del when check same
                    ovs_idx = self.spd.train_idxs.index(train_idx)
                    origin_ov = np.delete(self.init_ov, train_idx)
                    ovc = eval_ovc(self.spd.ovs[ovs_idx], origin_ov)
                    ovc = np.insert(ovc, train_idx, 0)

                    if rela[0] == "poly":
                        x = simi[np.where(simi != 0)]
                        y = ovc[np.where(simi != 0)]
                        z = np.polyfit(x, y, rela[1])
                        p = np.poly1d(z)
                        curve_funcs[current_label_fidx, idx] = p
        self.curve_funcs = curve_funcs
