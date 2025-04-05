# Viper

Read the accompanying blog post here (tbd).

**V**erifiability via **I**terative **P**olicy **E**xt**R**action (2019) [paper](https://arxiv.org/abs/1805.08328)]

In this paper the authors distill a Deep Reinforcement Learning such as DeepQN into a decision tree policy which can
then be automatically checked for correctness, robustness, and stability.

This repository implements and tests the viper algorithm on the following environments:

- CartPole
- Atari Pong
- ToyPong (tbd)

## Usage

The entire project can be run using the `main.py` script which can take more options than the ones mentioned below.
To get a full list of options run `python main.py --help`.

### Training the oracle

The commands below reflect configurations that helped achieve a perfect reward averaged over 50 rollouts.

Atari Pong (TODO: only achieves 20.12 +/- 1.66 out of 21):

```
python main.py train-oracle --env-name PongNoFrameskip-v4 --n-env 64 --total-timesteps 15_000_000
```

Toy Pong:

```
python main.py train-oracle --env-name ToyPong-v0 --n-env 1 --total-timesteps 1_000_000
```

Cart pole:

```
python main.py train-oracle --env-name CartPole-v1 --n-env 8 --total-timesteps 100_000
```

You can always resume training a stored model by adding the `--resume` flag to the same command.

### Running viper

Once the oracle policies are trained you can run viper on the same environment:

Cart pole:
```
python main.py train-viper --env-name CartPole-v1 --n-env 1
```

Toy Pong:
```
python main.py train-viper --env-name ToyPong-v0 --n-env 4 --max-leaves 61 --total-timesteps 1_000_000
```

I 
Toy Pong: tạo cây với hiệu suất tốt nhất
python main.py train-viper --env-name ToyPong-v0 --n-env 4 --max-depth 20  --max-leaves 587 --total-timesteps 1_000_000 

🔹 Tổng số bước kiểm tra: 10000
⚠️ Số lần PPO và VIPER khác nhau: 1892
📉 Tỉ lệ khác biệt: 18.92%
⚠️ Cây quyết định CÓ SỰ KHÁC BIỆT đáng kể so với chính sách gốc.


Mỗi đặc trưng đóng góp mức độ khác nhau vào quyết định của cây. Một đặc trưng quan trọng nếu:
Thường xuyên được chọn để phân nhánh sớm trong cây.
Góp phần lớn vào việc giảm độ hỗn loạn (entropy) hoặc giảm Gini impurity.
🔹 Trong sklearn, tầm quan trọng của đặc trưng (feature_importances_) được tính bằng:
Tổng lượng giảm độ hỗn loạn do đặc trưng đó mang lại trên tất cả các lần nó xuất hiện trong cây.

bao nhieu luat
luay yeu co the bo hay ko..

- Luật
+ Số luật: số luật = số node lá, phụ thuộc vào max_depth, max_leaf_nodes và thuật toán phân nhánh entropy
    => đếm số node lá, kiểm tra số luật khi tăng ccp_alpha hoặc giảm max_leaf_nodes
+ Luật yếu: là những nhãnh có ít mẫu dữ liệu (samples) hoặc entropy gần bằng 0
Khi bỏ luật yếu, trạng thái thuộc về luật đó sẽ:    
    Được gán về một nhánh tổng quát hơn (nếu pruning nhẹ)
    Không có nhánh phù hợp (pruning quá mạnh), khi đó agent chọn hành động từ node gần nhất
+ Thực nghiệm:
    So sánh kết quả khi chạy với ccp_alpha = 0.0001 và giá trị lớn hơn
    Kiểm tra có trạng thái nào bị mất quyết định không (so sánh action trước và sau pruning)
+ Cách thực nghiệm
    Chạy agent với cây gốc (không pruning) -> đo hiệu suất
    Tăng dần ccp_alpha để loại bỏ luật yếu -> quan sát thay đổi
    Kiểm tra nếu agent vẫn chơi tốt hoặc hiệu suất giảm
    Nếu hiệu suất giảm mạnh -> giảm pruning để giữ lại những luật quan trọng
+ Mục tiêu thực nghiệm
    Tìm giá trị ccp_alpha tối ưu giúp giảm số luật nhưng vẫn giữ hiệu suất cao
    Xác định mức pruning tối đa mà agent vẫn hoạt động tốt
    Kiểm tra xem agent có gặp trạng thái không xử lý được khi bỏ quá nhiều luật hay không.


1. Không pruning
- `entropy` split criterion
- `ccp_alpha=0`,  # Không pruning
- `max_depth=None`,  # Cho phép cây phát triển tối đa
- `min_samples_split=2`,  # Mỗi node chỉ cần ít nhất 2 mẫu để phân nhánh
- `min_samples_leaf=1`  # Một mẫu duy nhất cũng có thể tạo thành node lá
    
python main.py train-viper --verbose 2 --env-name ToyPong-v0 --n-env 4 --ccp-alpha 0 --total-timesteps 1_000_000 

2. Sử dụng pruning
python main.py train-viper --verbose 2 --env-name ToyPong-v0 --n-env 4 --ccp-alpha 0.0001 --total-timesteps 1_000_000 

