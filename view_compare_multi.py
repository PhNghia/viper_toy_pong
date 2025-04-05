import csv
import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from joblib import load
from gym_env.toy_pong import ToyPong  # Đảm bảo import đúng môi trường ToyPong
from tabulate import tabulate  # Import thư viện để tạo bảng

# Load chính sách gốc (PPO)
ppo_policy = PPO.load("log/ToyPong-v0_1env_learning_rate-HalfLinearSchedule(initial_value=0.0003)/model.zip")

# Load các cây quyết định (VIPER)
tree_0_0 = load("log/viper_ToyPong-v0_0.0_all-leaves_all-depth.joblib")
tree_0_0001_pruned = load("log/viper_ToyPong-v0_pruned_0.0001_11120_29.joblib")
tree_0_1e05_pruned = load("log/viper_ToyPong-v0_pruned_1e-05_11120_29.joblib")
tree_0_1e06_pruned = load("log/viper_ToyPong-v0_pruned_1e-06_11120_29.joblib")

# Tạo danh sách các cây đã cắt tỉa
pruned_trees = [tree_0_0001_pruned, tree_0_1e05_pruned, tree_0_1e06_pruned]

# Danh sách các chính sách (PPO và VIPER)
models = [
    {"name": "PPO", "model": ppo_policy},
    {"name": "VIPER 0.0", "model": tree_0_0},
    {"name": "VIPER 0.0001 pruned", "model": tree_0_0001_pruned},
    {"name": "VIPER 1e05 pruned", "model": tree_0_1e05_pruned},
    {"name": "VIPER 1e06 pruned", "model": tree_0_1e06_pruned},
    # Thêm chính sách mới ở đây, ví dụ:
    # {"name": "VIPER 0.001", "model": load("log/viper_ToyPong-v0_0.001_all-leaves_all-depth.joblib")},
]

# Class agent sử dụng PPO
class PPOAgent:
    def __init__(self, model):
        self.model = model

    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action  # Trả về hành động từ chính sách gốc (PPO)

# Class agent sử dụng cây quyết định
class DecisionTreeAgent:
    def __init__(self, model):
        self.model = model

    def act(self, observation):
        # Chuyển đổi observation thành np.float32
        observation = np.array(observation).reshape(1, -1).astype(np.float32)

        tree = self.model.tree_
        node = tree.apply(observation)  # Lấy node hiện tại từ cây

        # Duyệt qua các node cho đến khi gặp node lá
        while tree.feature[node] != -2:  # Nếu node không phải là node lá
            # Tiến hành phân nhánh
            if observation[0, tree.feature[node]] <= tree.threshold[node]:
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]

        # Khi gặp node lá, trả về nhãn của node đó
        return np.argmax(tree.value[node])  # Hoặc giá trị hành động từ node

# Tạo môi trường ToyPong
args = type('', (), {})()  # Tạo một object rỗng để truyền đối số
args.render = True  # Hiển thị pygame
args.rand_ball_start = False  # Giữ vị trí ban đầu cố định
env = ToyPong(args)

# Khởi tạo các agent cho mỗi chính sách
agents = {model["name"]: PPOAgent(model["model"]) if model["name"] == "PPO" else DecisionTreeAgent(model["model"]) for model in models}

# Biến để lưu số lần khác biệt
diff_counts = {model["name"]: 0 for model in models}

# Chạy môi trường và so sánh các agent
obs = env.reset()
done = False
step = 0
total_steps = 0

while not done:
    # Lấy hành động từ tất cả các agent
    actions = {name: agent.act(obs) for name, agent in agents.items()}

    # Kiểm tra sự khác biệt giữa các hành động
    for name, action in actions.items():
        if action != actions["PPO"]:
            diff_counts[name] += 1

    # In ra kết quả
    print(f"\n🔹 Step {step}")
    # print(f"   - Observation: {obs}")
    # for name, action in actions.items():
    #     print(f"   - {name} Action: {action}")

    # Chạy môi trường theo chính sách VIPER (cây quyết định)
    obs, reward, done, _ = env.step(actions["VIPER 1e06 pruned"])  # Có thể thay đổi chính sách ở đây

    # Render
    env.render()
    step += 1
    total_steps += 1

env.close()

# Tính tỉ lệ khác biệt
diff_rates = {name: (count / total_steps) * 100 if total_steps > 0 else 0 for name, count in diff_counts.items()}

# Trình bày kết quả dưới dạng bảng
table = []
for name, diff_rate in diff_rates.items():
    if diff_rate == 0:
        conclusion = "✅ Hoàn toàn khớp"
    elif diff_rate < 10:
        conclusion = "✅ Khớp tương đối"
    elif diff_rate < 30:
        conclusion = "⚠️ Có sự khác biệt đáng kể"
    else:
        conclusion = "❌ Khác nhiều, xem lại quá trình huấn luyện"
    
    table.append([name, f"{diff_counts[name]}/{total_steps}", f"{diff_rate:.2f}%", conclusion])

# In ra bảng kết quả
headers = ["Policy", "Differences (PPO)", "Difference Rate (%)", "Conclusion"]
print("\n===================== 📊 KẾT LUẬN 📊 =====================")
print(tabulate(table, headers=headers, tablefmt="grid"))

def compare_multiple_trees(original_tree, pruned_trees, X_train, y_train, X_test, y_test):
    # So sánh các thông số cây
    headers = ["Thuộc tính", "Cây gốc"] + [f"Cây đã cắt tỉa {i+1}" for i in range(len(pruned_trees))]

    # Thông số cây gốc và các cây đã cắt tỉa
    table = [
        ["Độ sâu cây", original_tree.get_depth()] + [tree.get_depth() for tree in pruned_trees],
        ["Số nút lá", original_tree.get_n_leaves()] + [tree.get_n_leaves() for tree in pruned_trees],
        ["Số nút tổng", original_tree.tree_.node_count] + [tree.tree_.node_count for tree in pruned_trees],
        ["Chi phí cắt tỉa (ccp_alpha)", original_tree.ccp_alpha] + [tree.ccp_alpha for tree in pruned_trees],
        ["Độ chính xác trên dữ liệu huấn luyện", original_tree.score(X_train, y_train)] + [tree.score(X_train, y_train) for tree in pruned_trees],
        ["Độ chính xác trên dữ liệu kiếm tra", original_tree.score(X_test, y_test)] + [tree.score(X_test, y_test) for tree in pruned_trees],
    ]
    
    # In bảng
    print("\nSo sánh các thuộc tính cây:\n")
    print(tabulate(table, headers=headers, tablefmt="grid"))
    
    # Lưu bảng so sánh vào file CSV
    with open("result_csv/multiple_tree_comparison.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Ghi tiêu đề cột
        writer.writerow(headers)
        # Ghi nội dung bảng
        writer.writerows(table)

# Load cây và dataset để kiểm tra
dts_tree_0_0 = load("dataset/viper_ToyPong-v0_0.0_all-leaves_all-depth.joblib")
dts_test = load("dataset/viper_ToyPong-v0_0.0001_all-leaves_all-depth.joblib")  

# Lấy dữ liệu huấn luyện từ dataset
X_train = np.array([traj[0] for traj in dts_tree_0_0])
y_train = np.array([traj[1] for traj in dts_tree_0_0])
sample_weight = np.array([traj[2] for traj in dts_tree_0_0])

# Lấy dữ liệu testtest
X_test = np.array([traj[0] for traj in dts_test])
y_test = np.array([traj[1] for traj in dts_test])

# So sánh cây gốc và cây có pruning
compare_multiple_trees(tree_0_0, pruned_trees, X_train, y_train, X_test, y_test)