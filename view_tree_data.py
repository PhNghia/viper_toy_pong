from joblib import load
import pandas as pd
import numpy as np
# Đường dẫn file
file_path = "log/viper_ToyPong-v0_587_20.joblib"

# # Load cây quyết định
tree_model = load(file_path)

# print(tree_model)  # Kiểm tra model


# # Lấy danh sách các quyết định (rules) trong cây
# n_nodes = tree_model.tree_.node_count
# feature = tree_model.tree_.feature
# threshold = tree_model.tree_.threshold
# value = tree_model.tree_.value

# # Tạo DataFrame
# df = pd.DataFrame({
#     "Node": np.arange(n_nodes),
#     "Feature": feature,
#     "Threshold": threshold,
#     "Value": [v.tolist() for v in value]
# })

# # Lưu ra CSV
# csv_path = "decision_tree.csv"
# df.to_csv(csv_path, index=False)
# print(f"Đã lưu thành CSV: {csv_path}")

from sklearn.tree import export_text

KEY_TO_LABEL = {
    0: 'paddle_x',
    1: 'ball_pos_x',
    2: 'ball_pos_y',
    3: 'ball_vel_x',
    4: 'ball_vel_y'
}

print(export_text(tree_model, feature_names=[KEY_TO_LABEL[i] for i in range(len(KEY_TO_LABEL))]))