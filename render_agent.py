import gym
import torch
from stable_baselines3 import PPO
from gym_env.toy_pong import ToyPong  # Đảm bảo import đúng môi trường

# Load mô hình PPO đã huấn luyện
model_path = "log/ToyPong-v0_1env_learning_rate-HalfLinearSchedule(initial_value=0.0003)/model.zip"  # Đổi tên nếu cần
model = PPO.load(model_path)

# Khởi tạo môi trường
args = type('', (), {})()  # Tạo một object rỗng để truyền đối số
args.render = True  # Hiển thị pygame
args.rand_ball_start = False  # Giữ vị trí ban đầu cố định (có thể đổi)
env = ToyPong(args)

obs = env.reset()
done = False
step = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)  # Agent chọn hành động
     # 🔹 In thông tin quan trọng
    print(f"\n🔹 Step {step}")
    print(f"   - Observation: {obs}")  # Quan sát của môi trường
    print(f"   - Chọn hành động: {action}")
    obs, reward, done, _ = env.step(action)  # Thực hiện bước đi
    # 🔹 In kết quả sau khi thực hiện action
    print(f"   - Reward: {reward}")
    print(f"   - Done: {done}")
    env.render()  # Hiển thị kết quả
    step += 1

env.close()
