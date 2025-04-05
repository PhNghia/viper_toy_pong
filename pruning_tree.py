import numpy as np
from joblib import load
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from tabulate import tabulate  # Thư viện để định dạng bảng
from tqdm import tqdm
from model.paths import get_viper_pruned_path, get_viper_pruned_path_2
from model.tree_wrapper import TreeWrapper
import csv
import pandas as pd

def prune_tree_by_retraining(original_tree, X_train, y_train, sample_weight, ccp_alpha):
    """
    Huấn luyện lại cây với pruning bằng ccp_alpha và giảm độ phức tạp.
    """
    # max_depth = max(1, int(original_tree.get_depth() * 0.8))
    # max_leaves = max(2, int(original_tree.get_n_leaves() * 0.8))

    max_depth = None
    max_leaves = None

    clf = DecisionTreeClassifier(
        ccp_alpha=ccp_alpha,
        criterion="entropy",
        max_depth=max_depth,
        max_leaf_nodes=max_leaves
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    path = get_viper_pruned_path(ccp_alpha=ccp_alpha, max_depth=max_depth, max_leaves=max_leaves)
    wrapper = TreeWrapper(clf)
    wrapper.print_info()
    wrapper.save(path)

    return clf

# Load cây và dataset để kiểm tra
tree_0_0 = load("log/viper_ToyPong-v0_0.0_all-leaves_all-depth.joblib")
dts_tree_0_0 = load("dataset/viper_ToyPong-v0_0.0_all-leaves_all-depth.joblib")
dts_test = load("dataset/viper_ToyPong-v0_0.0001_all-leaves_all-depth.joblib")  

# Lấy dữ liệu huấn luyện từ dataset
X_train = np.array([traj[0] for traj in dts_tree_0_0])
y_train = np.array([traj[1] for traj in dts_tree_0_0])
sample_weight = np.array([traj[2] for traj in dts_tree_0_0])

# Lấy dữ liệu testtest
X_test = np.array([traj[0] for traj in dts_test])
y_test = np.array([traj[1] for traj in dts_test])

# Huấn luyện lại cây khi tăng ccp_alpha
pruned_tree = prune_tree_by_retraining(tree_0_0, X_train, y_train, sample_weight, ccp_alpha=0.000001)



