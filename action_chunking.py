"""
Day 8 练习：Action Chunking（一次预测 k 步动作）
================================================
运行方式：
    python action_chunking.py

Day 7 的 BC 每步只预测 1 个动作。今天把策略网络升级成一次输出 k=4 步的
"动作块"（action chunk），再对比三种消费这块动作的方式：

    single-step : 每步观测 → 预测 1 步 → 执行 1 步        （Day 7 原始做法，k=1 特例）
    open-loop   : 观测 → 预测 4 步 → 全部执行完才再看世界  （整块开环）
    replan      : 每步观测 → 预测 4 步 → 只执行第 1 步     （逐步重规划）

分工（参考解答 solutions/action_chunking_solution.py 分节注释与 TODO 编号一一对应）：

    TODO-1  build_dataset —— 数据集标签从"1 步动作"改成"k 步动作块"
    TODO-2  ChunkPolicy —— 网络输出维度改成 k*3 并 reshape 成 (B, k, 3)
    TODO-3  run_episode —— 一个回合的闭环主循环，三种 mode 的差别都在这
    TODO-4  evaluate_mode —— 跑 n 个回合统计成功率

规则：
    - "导师提供，禁止修改"部分不许动
    - stub 没填时直接运行会给出中文提示，不会裸崩
"""

# Windows 上 torch 与 numpy 各自带一份 OpenMP 运行库会冲突（见 README 常见问题）。
# --- GBK 控制台防崩（课程统一处理，勿删）---
import sys as _gbk_sys
try:
    if (_gbk_sys.stdout.encoding or "").lower().replace("-", "") in ("gbk", "cp936"):
        _gbk_sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
# --- 防崩块结束 ---
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import random

import torch
import torch.nn as nn

torch.manual_seed(0)

K = 4  # 动作块长度

# ============================================================
# 第一部分：2D 世界 + 专家 + 示范采集 + 特征编码（导师提供，禁止修改）
# 全部与 Day 7 相同——换的只有"标签的形状"和"执行的方式"
# ============================================================

class World:
    """极简 2D 世界：一个夹爪、一支笔、一个笔筒。坐标单位：米。"""

    def __init__(self):
        self.gripper = [0.0, 0.4]
        self.pen = [0.3, 0.0]
        self.cup = [-0.2, 0.0]
        self.gripper_closed = False
        self.pen_held = False
        self.success = False

    def step(self, action):
        dx, dy, gripper_cmd = action
        dx = max(-0.1, min(0.1, float(dx)))
        dy = max(-0.1, min(0.1, float(dy)))
        self.gripper[0] += dx
        self.gripper[1] += dy
        want_closed = float(gripper_cmd) >= 0.5
        if want_closed:
            if not self.pen_held and self._dist(self.gripper, self.pen) < 0.05:
                self.pen_held = True
        else:
            if self.pen_held:
                self.pen_held = False
                if self._dist(self.pen, self.cup) < 0.05:
                    self.success = True
        self.gripper_closed = want_closed
        if self.pen_held:
            self.pen = list(self.gripper)
        return self.success

    @staticmethod
    def _dist(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])


def get_observation(world):
    return {
        "gripper": list(world.gripper),
        "pen": list(world.pen),
        "cup": list(world.cup),
        "gripper_closed": world.gripper_closed,
        "pen_held": world.pen_held,
    }


def rule_policy(observation):
    """专家规则策略（Day 7 同款，只用于采示范和当评测上限对照）。"""
    gx, gy = observation["gripper"]
    px, py = observation["pen"]
    cx, cy = observation["cup"]

    def clip(v):
        return max(-0.1, min(0.1, v))

    if not observation["pen_held"]:
        dx, dy = clip(px - gx), clip(py - gy)
        grip = 1.0 if math.hypot(px - gx, py - gy) < 0.045 else 0.0
    else:
        dx, dy = clip(cx - gx), clip(cy - gy)
        grip = 0.0 if math.hypot(cx - gx, cy - gy) < 0.045 else 1.0
    return [dx, dy, grip]


def collect_demonstrations(n_episodes=200, seed=0, pen_jitter=0.05, max_steps=50):
    """同 Day 7：专家跑 200 回合，返回 episodes=[[(obs, action), ...], ...]。"""
    rng = random.Random(seed)
    episodes = []
    for _ in range(n_episodes):
        world = World()
        world.pen[0] += rng.uniform(-pen_jitter, pen_jitter)
        world.pen[1] += rng.uniform(-pen_jitter, pen_jitter)
        traj = []
        for _step in range(max_steps):
            obs = get_observation(world)
            action = rule_policy(obs)
            traj.append((obs, list(action)))
            if world.step(action):
                break
        assert world.success, "专家策略不应失败"
        episodes.append(traj)
    return episodes


def encode_obs(obs):
    """同 Day 7 的 TODO-1a：8 维特征向量（这次直接给你，把精力留给 chunk）。"""
    return [
        obs["gripper"][0], obs["gripper"][1],
        obs["pen"][0],     obs["pen"][1],
        obs["cup"][0],     obs["cup"][1],
        float(obs["gripper_closed"]), float(obs["pen_held"]),
    ]


# ============================================================
# 第二部分：TODO-1 数据集构建（你来填）——唯一的变化在标签 Y
# ============================================================

def build_dataset(episodes, k=K):
    """TODO-1：把回合列表变成 (obs → k 步动作块) 样本。

    第 i 个样本：X[i] = 第 i 步的观察（encode_obs 编码）；
                Y[i] = 示范轨迹里从第 i 步起的连续 k 个动作。
    问题：轨迹末尾凑不满 k 步怎么办？约定：用零动作 [0,0,0] 补齐
          （原地不动、保持夹爪状态不变）。

    返回：X —— float32，形状 (N, 8)
          Y —— float32，形状 (N, k, 3)
    提示：Y 可以先用 list 收集 [[a_i, a_{i+1}, ..., a_{i+k-1}], ...] 再一把转
          torch.tensor(...).view(-1, k, 3)。注意 k=1 时它就是 Day 7 的 Y。
    """
    xs, ys = [], []
    for traj in episodes:
        for i in range(len(traj)):
            obs, _ = traj[i]
            xs.append(encode_obs(obs))
            actions = [traj[j][1] for j in range(i, min(i + k, len(traj)))]
            while len(actions) < k:
                actions.append([0.0, 0.0, 0.0])  # 补零动作
            ys.append(actions)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)  # 返回 (N, 8) 和 (N, k, 3)
    
    raise NotImplementedError("TODO-1: build_dataset() 还没实现，先看上面的 docstring。")


# ============================================================
# 第三部分：TODO-2 模型（你来填）+ 训练循环（导师提供）
# ============================================================

class ChunkPolicy(nn.Module):
    """TODO-2：MLP 8 → 64 → 64 → k*3，输出 reshape 成 (B, k, 3)。

    关键两点：
    1. 最后一层 nn.Linear(hidden, k * act_dim) —— 一次吐 k 步；
    2. forward 里 out.view(-1, self.k, self.act_dim) 把 (B, k*3) 折成 (B, k, 3)，
       约定第 t 行 = "从现在起第 t 步"的动作。
    """

    def __init__(self, obs_dim=8, act_dim=3, k=K, hidden=64):
        super().__init__()
        self.k = k
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,k*act_dim),
        )
        
       # raise NotImplementedError("TODO-2: ChunkPolicy 的 __init__ 还没实现。")

    def forward(self, x):
        return self.net(x).view(-1, self.k, self.act_dim)  # 对于action块需要重新布局，单独布局成(B, k, 3)符合ACT chunk格式
    
    
        raise NotImplementedError("TODO-2: ChunkPolicy 的 forward 还没实现。")


def train_supervised(model, X, Y, epochs=40, batch_size=64, lr=1e-3, verbose=True):
    """通用监督循环（导师提供，和 Day 7 的 TODO-3 等价——只是模型从外面传进来）。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = X.shape[0]                          #老几样，快快去复习，尽可能默写出来，这是基本功问题
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), Y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  loss={total / n:.5f}")
    return model


def make_chunk_step_fn(model):
    """推理包装（导师提供）：obs 字典 → 动作块 list，形状 (k, 3)。"""
    model.eval()

    def chunk_fn(obs):
        x = torch.tensor(encode_obs(obs), dtype=torch.float32).unsqueeze(0)  # (1, 8)
        with torch.no_grad():
            out = model(x)[0]              # (k, 3)
        return out.tolist()

    return chunk_fn


# ============================================================
# 第四部分：TODO-3 三种执行方式（你来填）——今天的核心概念全在这
# ============================================================

def run_episode(chunk_fn, mode, pen_jitter=0.05, rng=None, max_steps=50,
                verbose=False, tag="", pen0=None):
    """TODO-3：跑一个回合。chunk_fn(obs) 返回 (k, 3) 动作块。

    三种 mode 唯一的区别是"什么时候允许看一眼世界"：
    - "single-step"：每一步都 观测 → 预测 → 执行块里的第 1 个动作
                     （k=1 模型上这就是 Day 7；k=4 模型上等价于 replan）
    - "open-loop"  ：观测 → 预测整块 → 把 k 个动作全部执行完，中间绝不重新观测
                     （执行中途 success 也要立刻停下）
    - "replan"     ：观测 → 预测整块 → 只执行第 1 个动作，下一步重新观测重新预测

    提示：open-loop 需要一个"剩余动作清单"（比如一个 list，观测时装满、
    执行时逐个弹出）；另两种每次都用新鲜预测的第一个动作。
    参数 pen_jitter/rng/pen0/verbose 的处理照抄 docstring 下方骨架——已给你写好
    建 World 和初始笔位置的部分，你只写主循环。
    返回：(是否成功 ok, 用了多少步 step)
    """

    world = World()
    if pen0 is not None:
        world.pen = [float(pen0[0]), float(pen0[1])]
    elif pen_jitter > 0 and rng is not None:
        world.pen[0] += rng.uniform(-pen_jitter, pen_jitter)
        world.pen[1] += rng.uniform(-pen_jitter, pen_jitter)
    
    action_queue = []  # 动作缓存（open-loop 才用到）
    ok = False 
    
    for step in range(max_steps):
        #1,什么时候需要看世界？
        #对于single-step和replan，每一步都要看世界，所以在循环里每次都要获取obs
        #对于open-loop，只在第一次看世界，之后就不再看世界，所以在循环里第一次获取obs后，之后就不再获取obs，而是使用剩余动作清单
        if mode == "open-loop":
            if not action_queue:  #第一次获取obs后，在执行完动作前不再获取obs，而是使用剩余动作清单
                action_queue = chunk_fn(get_observation(world)).copy() 
        else:
            action_queue = [chunk_fn(get_observation(world))[0]]  #每次都要看世界，所以在循环里每次都要获取obs，并且只取第一个动作
        #2,从缓存中取出将要执行的动作
            
        if action_queue:
            action = action_queue.pop(0)    #pop的意思就是执行完一个扔掉一个，所以列表的长度会逐渐变短，直到为空
        else:
            break
            
        #3，执行动作，世界推进
        ok = world.step(action)
        if verbose:
            print(f"step {step:2d}  gripper={world.gripper}  pen={world.pen}  "
                f"gripper_closed={world.gripper_closed}  pen_held={world.pen_held}  "
                    f"success={world.success}")
        #4，检查是否成功
        if ok:
            break
    
    return ok, step+1  # 返回是否成功和用了多少步    

    raise NotImplementedError("TODO-3: run_episode() 的主循环还没实现，先看上面的 docstring。")


# ============================================================
# 第五部分：TODO-4 评测（你来填）
# ============================================================

def evaluate_mode(chunk_fn, mode, n_episodes=40, seed=123, pen_jitter=0.05,
                  max_steps=50):
    """TODO-4：用 run_episode 跑 n_episodes 个回合，返回成功率（0~1 的浮点数）。

    套路同 Day 7：rng = random.Random(seed) 在函数开头建一次，每回合把
    (chunk_fn, mode, pen_jitter, rng, max_steps) 交给 run_episode。
    """
    rng = random.Random(seed)
    successes = 0
    for _ in range(n_episodes):
        ok, _ = run_episode(chunk_fn, mode, pen_jitter=pen_jitter, rng=rng, max_steps = max_steps)
        if ok:
            successes += 1
    return successes / n_episodes  # 返回成功率（0~1 的浮点数）
    
    
    raise NotImplementedError("TODO-4: evaluate_mode() 还没实现，先看上面的 docstring。")


# ============================================================
# 主流程（导师提供，禁止修改）
# ============================================================

def main():
    print("=" * 64)
    print(f"Day 8 练习：Action Chunking（k={K}）三种执行方式对比")
    print("=" * 64)
    print(f"torch 版本：{torch.__version__}")

    print("\n[1/4] 采集示范（同 Day 7：200 回合，笔 ±0.05 m）")
    episodes = collect_demonstrations(n_episodes=200, seed=0)
    n_pairs = sum(len(t) for t in episodes)
    print(f"      {n_pairs} 个观测时刻 → {n_pairs} 个 (obs, {K}步动作块) 样本")

    try:
        print("\n[2/4] 训练两个模型（同数据）")
        X1, Y1 = build_dataset(episodes, k=1)          # Day 7 的单步 BC
        m_single = ChunkPolicy(k=1)
        print(f"      单步模型 (8→64→64→1×3, k=1):")
        train_supervised(m_single, X1, Y1)
        Xk, Yk = build_dataset(episodes, k=K)          # Day 8 的 chunk BC
        m_chunk = ChunkPolicy(k=K)
        print(f"      整块模型 (8→64→64→{K}×3, k={K}):")
        train_supervised(m_chunk, Xk, Yk)

        f_single = make_chunk_step_fn(m_single)
        f_chunk = make_chunk_step_fn(m_chunk)

        print("\n[3/4] 成功率对比（每格 40 回合，seed=123）")
        grid = [
            ("单步 BC（Day 7）",      f_single, "single-step"),
            ("整块开环（预测4步执行4步）", f_chunk,  "open-loop"),
            ("整块重规划（预测4步用1步）", f_chunk,  "replan"),
        ]
        for jitter in (0.05, 0.10):
            head = "示范内分布" if jitter == 0.05 else "超出示范分布（±0.10 随机化）"
            print(f"\n  ▶ 评测随机化 ±{jitter:.2f} m（{head}）")
            sr_e = evaluate_mode(lambda obs: [rule_policy(obs)], "single-step", pen_jitter=jitter)
            print(f"     专家规则（对照上限）        {sr_e * 100:5.0f}%")
            for name, fn, mode in grid:
                sr = evaluate_mode(fn, mode, pen_jitter=jitter)
                print(f"     {name:26s} {sr * 100:5.0f}%")

        print("\n[4/4] 失败分析：open-loop 在超出分布时怎么'盲执行'出错")
        rng = random.Random(123)
        pens = [(0.3 + rng.uniform(-0.10, 0.10), 0.0 + rng.uniform(-0.10, 0.10))
                for _ in range(40)]
        for ep, pen in enumerate(pens):
            ok, _ = run_episode(f_chunk, "open-loop", 0, None, 50, pen0=pen)
            if not ok:
                print(f"  --- open-loop 失败回合 {ep}（笔初始位置 {[round(v, 3) for v in pen]}）---")
                run_episode(f_chunk, "open-loop", 0, None, 50,
                            verbose=True, pen0=pen)
                break

    except NotImplementedError as e:
        print(f"\n⏸  {e}")
        print("\n你还没有填完全部 TODO。全部填完后，参考解答在 solutions/action_chunking_solution.py。")


if __name__ == "__main__":
    main()
