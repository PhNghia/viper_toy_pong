import numpy as np
import matplotlib.pyplot as plt
from joblib import load

def plot_feature_importance(tree, feature_names, save_path="feature_importance.png"):
    """
    Vẽ biểu đồ tầm quan trọng của các đặc trưng trong cây quyết định.

    Parameters:
        tree: Mô hình cây quyết định đã huấn luyện.
        feature_names (list): Danh sách tên các đặc trưng.
        save_path (str): Đường dẫn để lưu hình ảnh.
    """
    # Lấy độ quan trọng của các đặc trưng
    importances = tree.feature_importances_

    # Sắp xếp theo mức độ quan trọng giảm dần
    sorted_idx = np.argsort(importances)[::-1]

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color="b")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.title("Feature Importance in Decision Tree")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # In ra giá trị cụ thể
    for i in sorted_idx:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

# Ví dụ sử dụng
feature_names = ["paddle_x", "ball_pos_x", "ball_pos_y", "ball_vel_x", "ball_vel_y"]  # Danh sách đặc trưng

# Load cây quyết định từ file
tree = load("log/viper_ToyPong-v0_0.0_all-leaves_all-depth.joblib")
tree2 = load("log/viper_ToyPong-v0_0.0001_587_20.joblib")

# Gọi hàm vẽ
plot_feature_importance(tree, feature_names, save_path="feature_importance_0.png")
plot_feature_importance(tree2, feature_names, save_path="feature_importance_0.0001.png")
