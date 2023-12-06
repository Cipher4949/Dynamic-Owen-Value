import numpy as np
import pytest
import random
from sklearn import svm

import sys

sys.path.append("..")
from src.dynamicOwen import BaseOwen, DeltaOwen, PivotOwen, HeurOwen, YnOwen


class TestApi(object):
    def setup(self):
        X = [[0.1, 0.2]] * 20
        y = ([0] * 10) + ([1] * 10)
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.union_d = [
            [0, 1, 2, 3],
            [4, 5],
            [6],
            [7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
        ]

        # Perturb
        for i in range(len(self.y)):
            self.X[i][0] += random.randint(0, 10) * 0.01
            self.X[i][1] += random.randint(0, 10) * 0.01

        self.model = svm.SVC(decision_function_shape="ovo")
        self.init_ov = np.zeros(20)

    def test_base_owen(self):
        base_owen = BaseOwen(
            self.X, self.y, self.X, self.y, self.model, self.init_ov, self.union_d
        )
        ov = base_owen.add_single_point(
            self.X[0], self.y[0], 3, params={"method": "gavg"}
        )
        assert len(ov) == len(self.init_ov) + 1
        ov = base_owen.add_single_point(
            self.X[0], self.y[0], 3, params={"method": "lavg"}
        )
        assert len(ov) == len(self.init_ov) + 1
        ov = base_owen.add_multi_points(
            self.X[:2], self.y[:2], [2, 3], params={"method": "gavg"}
        )
        assert len(ov) == len(self.init_ov) + 2
        ov = base_owen.add_multi_points(
            self.X[:2], self.y[:2], [2, 3], params={"method": "lavg"}
        )
        assert len(ov) == len(self.init_ov) + 2

    def test_delta_owen(self):
        delta_owen = DeltaOwen(
            self.X, self.y, self.X, self.y, self.model, self.init_ov, self.union_d
        )
        ov = delta_owen.add_single_point(self.X[0], self.y[0], 3, 10)
        assert len(ov) == len(self.init_ov) + 1

    def test_pivot_owen(self):
        pivot_owen = PivotOwen(
            self.X, self.y, self.X, self.y, self.model, None, self.union_d
        )
        pivot_owen.prepare(100)
        ov = pivot_owen.add_single_point(
            self.X[0],
            self.y[0],
            3,
            proc_num=1,
            params=None,
            flags={"flag_lov": True},
        )
        assert len(ov) == len(self.init_ov) + 1
        assert len(pivot_owen.lov) == len(self.init_ov) + 1

    def test_heur_owen(self):
        heur_owen = HeurOwen(
            self.X,
            self.y,
            self.X,
            self.y,
            self.model,
            self.init_ov,
            self.union_d,
            params={"method": "knn"},
        )
        heur_owen.prepare(params={"n_neighbors": 5})
        ov = heur_owen.add_multi_points(self.X[:2], self.y[:2], [2, 3])
        assert len(ov) == len(self.init_ov) + 2

        heur_owen = HeurOwen(
            self.X,
            self.y,
            self.X,
            self.y,
            self.model,
            self.init_ov,
            self.union_d,
            params={"method": "knn+"},
        )
        heur_owen.prepare(
            flags={"exact": False, "train": True},
            params={
                "n_neighbors": 3,
                "simi_type": "ed",
                "f_shap": "n*n",
                "rela": ["poly", 2],
                "train_idxs": [3, 11],
                "m": 10,
            },
        )
        ov = heur_owen.add_multi_points(self.X[:2], self.y[:2], [2, 3])
        assert len(ov) == len(self.init_ov) + 2

    def test_yn_owen(self):
        yn_owen = YnOwen(
            self.X, self.y, self.X, self.y, self.model, self.init_ov, self.union_d
        )
        yn_owen.prepare(1, flags={"exact": False}, params={"mc_type": 0, "m": 2})
        ov = yn_owen.del_single_point(0)
        assert len(ov) == len(self.init_ov) - 1

        yn_owen = YnOwen(
            self.X[:2],
            self.y[:2],
            self.X,
            self.y,
            self.model,
            self.init_ov[:2],
            [[0], [1]],
        )
        yn_owen.prepare(1, flags={"exact": True})
        ov = yn_owen.del_single_point(0)
        assert len(ov) == len(self.init_ov[:2]) - 1


if __name__ == "__main__":
    pytest.main("-s test_api.py")
