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

1. Không pruning
- `entropy` split criterion
- `ccp_alpha=0`,  # Không pruning
- `max_depth=None`,  # Cho phép cây phát triển tối đa
- `min_samples_split=2`,  # Mỗi node chỉ cần ít nhất 2 mẫu để phân nhánh
- `min_samples_leaf=1`  # Một mẫu duy nhất cũng có thể tạo thành node lá
    
python main.py train-viper --verbose 2 --env-name ToyPong-v0 --n-env 4 --ccp-alpha 0 --total-timesteps 1_000_000 

2. Huấn luyện lại cây và có pruning
- Điều chỉnh các thông số ccp_alpha, max_depth, max_leaf_nodes
python pruning_tree.py

3. So sánh cây gốc và cây cắt tỉa
- Thêm mô hình cây cần so sánh
python view_compare_multi.py

