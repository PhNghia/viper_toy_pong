import gym
import numpy as np
from joblib import load
from gym_env.toy_pong import ToyPong  # Đảm bảo import đúng môi trường ToyPong

# Load cây quyết định từ file
tree = load("log/viper_ToyPong-v0_587_20.joblib")

# Chạy môi trường ToyPong
class DecisionTreeAgent:
    def __init__(self, model):
        self.model = model

    def act(self, observation):
        # Reshape observation để phù hợp với input của cây
        observation = np.array(observation).reshape(1, -1)
        return self.model.predict(observation)[0]  # Dự đoán action từ cây

# Tạo môi trường ToyPong
# args = lambda: None
# args.render = True  # Hiển thị game
# args.rand_ball_start = False  # Không khởi tạo bóng ngẫu nhiên
# env = ToyPong(args)
args = type('', (), {})()  # Tạo một object rỗng để truyền đối số
args.render = True  # Hiển thị pygame
args.rand_ball_start = False  # Giữ vị trí ban đầu cố định (có thể đổi)
env = ToyPong(args)

agent = DecisionTreeAgent(tree)

obs = env.reset()
done = False
step = 0

while not done:
    action = agent.act(obs)  # Chọn action từ cây quyết định
    # 🔹 In thông tin quan trọng
    print(f"\n🔹 Step {step}")
    print(f"   - Observation: {obs}")  # Quan sát của môi trường
    print(f"   - Chọn hành động: {action}")
    obs, reward, done, _ = env.step(action)  # Thực hiện bước đi
    # 🔹 In kết quả sau khi thực hiện action
    print(f"   - Reward: {reward}")
    print(f"   - Done: {done}")
    env.render()
    step += 1

env.close()
