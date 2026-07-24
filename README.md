# Imitation Learning algorithms and Co-training for Mobile ALOHA


#### Source Project Website: https://mobile-aloha.github.io/

该仓库在 https://mobile-aloha.github.io/ 基础上修改，将操作主体由ViperX机械臂更改为FR3机械臂、任务改为拾取物块并放置到指定位置
You can train and evaluate them in sim or real.
For real, you would also need to install [Mobile ALOHA](https://github.com/MarkFzp/mobile-aloha). 

### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

    # 1. 创建 conda 环境
    conda env create -f conda_env.yaml
    conda activate aloha

    # 2. 安装 Python 依赖
    pip install -r requirements.txt

    # 3. 安装 detr 模块
    cd detr && pip install -e . && cd ..

    # 4. 安装 robomimic (Diffusion Policy 需要)
    git clone https://github.com/ARISE-Initiative/robomimic.git
    cd robomimic && git fetch origin bcbfbb2d4188604557bb42876bc2b1886654e65f && git checkout FETCH_HEAD && pip install -e . && cd ..


### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments (FR3 MuJoCo environments)

以 ``fr3_pick_place_scripted`` 任务为例。
生成 50 条脚本策略演示数据：

    python3 record_sim_episodes.py --task_name fr3_pick_place_scripted --dataset_dir <data save dir> --num_episodes 50

添加 ``--onscreen_render`` 可实时查看渲染画面。
To visualize the simulated episodes after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

Note: to visualize data from the mobile-aloha hardware, use the visualize_episodes.py from https://github.com/MarkFzp/mobile-aloha

### 训练

训练 ACT 策略：

    # FR3 Pick Place 任务
    python3 imitate_episodes.py --task_name fr3_pick_place_scripted --ckpt_dir <ckpt dir> --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_steps 5000 --lr 1e-5 --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

If the policy is jerky or pauses mid-episode, train for more steps (e.g. ``--num_steps 5000``).
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.

### [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.
