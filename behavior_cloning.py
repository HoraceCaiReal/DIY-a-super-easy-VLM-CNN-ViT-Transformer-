"""
Day 7 练习：用行为克隆（BC）让 MLP 学会"笔 → 笔筒"任务
======================================================
运行方式：
    python behavior_cloning.py

流程：采集示范（导师给的规则专家）→ 编码特征 → 训练 MLP → 放回 World 闭环评测。

分工（参考解答 solutions/behavior_cloning_solution.py 按同样的分节注释，
TODO 编号一一对应，做完自己的再去看）：

    TODO-1  encode_obs + build_dataset —— 观察字典 → 8 维向量；轨迹 → 训练张量
    TODO-2  BCPolicy —— 神经网络本体（8 → 64 → 64 → 3 的 MLP）
    TODO-3  train_policy —— 监督训练循环（batch + MSE loss + Adam）
    TODO-4  evaluate —— 闭环评测：跑 n 个回合，返回成功率

规则：
    - 标注"导师提供，禁止修改"的部分不许动（World 是"物理定律"，
      专家是"考卷答案的数据来源"）
    - stub 没填时直接运行会给出中文提示，不会崩得难看
"""

# Windows 上 torch 与 numpy 各自带一份 OpenMP 运行库（libiomp5md.dll）会冲突，
# 这是官方认可的环境变量开关（见 README"常见问题"）。必须在 import torch 之前设置。
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

# ============================================================
# 第一部分：2D 世界（复制自 day01/policy_interface.py，导师提供，禁止修改）
# ============================================================

class World:
    """极简 2D 世界：一个夹爪、一支笔、一个笔筒。坐标单位：米。"""

    def __init__(self):
        self.gripper = [0.0, 0.4]    # 夹爪位置 (x, y)
        self.pen = [0.3, 0.0]        # 笔的位置
        self.cup = [-0.2, 0.0]       # 笔筒位置
        self.gripper_closed = False  # 夹爪是否闭合
        self.pen_held = False        # 笔是否被拿着
        self.success = False         # 任务是否成功

    def step(self, action):
        """执行一步动作 [dx, dy, gripper_cmd]，返回是否成功。规则见 day01。"""
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
    """把世界状态打包成观察字典（复制自 day01）。"""
    return {
        "gripper": list(world.gripper),
        "pen": list(world.pen),
        "cup": list(world.cup),
        "gripper_closed": world.gripper_closed,
        "pen_held": world.pen_held,
    }


# ============================================================
# 第二部分：专家策略（Day 1 练习 B 参考解的同类，导师提供，禁止修改）
# ============================================================

def rule_policy(observation):
    """规则策略——就是让你 Day 1 手写的那类策略。

    阶段逻辑：没拿笔 → 朝笔移动，够近就闭合；拿着笔 → 朝笔筒移动，够近就张开。
    今天的任务：不写这个函数，而是让神经网络从它的示范数据里把它"学出来"。
    """
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


# ============================================================
# 第三部分：示范采集（导师提供，禁止修改）——读懂它，它是"数据来源"
# ============================================================

def collect_demonstrations(n_episodes=200, seed=0, pen_jitter=0.05, max_steps=50):
    """运行专家策略，采集示范数据集 D = {(o, a)}。

    每个回合开始时把笔的位置随机扰动 ±pen_jitter 米——这样示范数据才能
    覆盖"笔不在标准位置"的状态（否则 200 个回合会是同一条轨迹）。

    返回：episodes = [trajectory, ...]，
          trajectory = [(obs字典, action列表), ...]（按时间顺序）
    """
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
            traj.append((obs, list(action)))   # 记录 (观察, 动作) 对
            if world.step(action):
                break
        assert world.success, "专家策略不应失败（失败说明 World 或专家被改坏了）"
        episodes.append(traj)
    return episodes


# ============================================================
# 第四部分：TODO-1 特征编码 + 数据集构建（你来填）
# ============================================================

def encode_obs(obs):
    """TODO-1a：把观察字典编码成长度 8 的浮点数列表。

    要求（顺序必须固定，参考解答与评测都按这个约定）：
        [gripper_x, gripper_y, pen_x, pen_y, cup_x, cup_y,
         gripper_closed, pen_held]
    提示：最后两维是布尔值，网络只吃浮点数——用 float(...) 转成 0.0/1.0。
    """
    return [obs["gripper"][0], obs["gripper"][1],
            obs["pen"][0], obs["pen"][1],
            obs["cup"][0], obs["cup"][1],    #分别手机各个向量的x，y信息
            float(obs["gripper_closed"]), float(obs["pen_held"])]
            
            
    raise NotImplementedError("TODO-1a: encode_obs() 还没实现，先看上面的 docstring。")


def build_dataset(episodes):
    """TODO-1b：把回合列表摊平成训练张量。

    输入：collect_demonstrations 的返回值（回合列表，每回合是 (obs, action) 序列）
    输出：X —— torch.float32 张量，形状 (N, 8)；Y —— torch.float32 张量，形状 (N, 3)
    提示：N = 所有回合的步数之和。先收集 Python list，最后 torch.tensor(...) 一把转。
    """
    xs, ys = [], []
    for traj in episodes:
        for obs, action in traj:   
            xs.append(encode_obs(obs))  
            ys.append(action)      
    return (torch.tensor(xs, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.float32))
        
    raise NotImplementedError("TODO-1b: build_dataset() 还没实现，先看上面的 docstring。")


# ============================================================
# 第五部分：TODO-2 模型 + TODO-3 训练循环（你来填）
# ============================================================

class BCPolicy(nn.Module):
    """TODO-2：策略网络——3 层全连接 MLP：obs_dim → hidden → hidden → act_dim，
    中间用 ReLU。结构提示：Day 1 讲义 §6 的"MLP 头"你已经写过无数遍了。
    """

    def __init__(self, obs_dim=8, act_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        #raise NotImplementedError("TODO-2: BCPolicy 的 __init__ 还没实现。")

    def forward(self, x):
        return self.net(x)
    
    
        raise NotImplementedError("TODO-2: BCPolicy 的 forward 还没实现。")


def train_policy(X, Y, epochs=40, batch_size=64, lr=1e-3, verbose=True):
    """TODO-3：标准监督训练循环，返回训练好的模型。

    需要的零件（就是你 Day 4-6 天天用的五件套）：
        model = BCPolicy(obs_dim=X.shape[1], act_dim=Y.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
    每个 epoch：torch.randperm(N) 打乱 → 按 batch_size 切片 →
        opt.zero_grad(); loss = loss_fn(model(X[idx]), Y[idx]);
        loss.backward(); opt.step()
    收尾：verbose 时打印 epoch 号和平均 loss（第 1 个 epoch 和每 10 个）。
    """
    model = BCPolicy(obs_dim = X.shape[1], act_dim = Y.shape[1])
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn = nn.MSELoss()
    N = X.shape[0]           #总样本数，今天我们要手动分batch
    for ep in range (1,epochs+1):
        perm = torch.randperm(N)  #torch.randperm用来打乱batch顺序
        # 如果不打乱，每个 epoch 都按"轨迹 0 的第 1 步、第 2 步…，轨迹 1…"的顺序喂，模型会学到"相邻样本相关"的假规律（比如连续几步动作很像），泛化变差。打乱让每个 batch 里的样本独立随机，梯度估计更无偏。
        tot, nb = 0.0, 0
        for s in range(0, N, batch_size):  #s是起始索引，range(0,N,batch_size)会生成0,64,128,...直到N
            idx = perm[s:s+batch_size]   #取64个随机索引
            loss = loss_fn(model(X[idx]),Y[idx])  #Y[idx]是专家动作，model(X[idx])是模型预测动作，loss_fn是MSELoss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()   #.item()返回一个python数值，loss是tensor类型
            nb += 1  #batch数量
        if verbose and (ep == 1 or ep % 10 == 0):
            print(f"epoch {ep:3d}/{epochs}  MSE = {tot/nb:.6f}")        
    return model  
    
    raise NotImplementedError("TODO-3: train_policy() 还没实现，先看上面的 docstring。")


def make_policy_fn(model):
    """把 torch 模型包装成 policy_fn(obs字典) -> [dx, dy, grip]。（导师提供）"""
    model.eval()

    def policy_fn(obs):
        x = torch.tensor(encode_obs(obs), dtype=torch.float32)
        with torch.no_grad():
            action = model(x)
        return action.tolist()

    return policy_fn


# ============================================================
# 第六部分：TODO-4 闭环评测（你来填）
# ============================================================

def evaluate(policy_fn, n_episodes=40, seed=123, pen_jitter=0.05,
             max_steps=50, verbose_episodes=0):
    """TODO-4：把策略放回 World 里闭环跑 n_episodes 个回合，返回成功率。

    每个回合：
        1. world = World()；若 pen_jitter > 0，把 world.pen 两个坐标各加
           rng.uniform(-pen_jitter, pen_jitter)（rng = random.Random(seed)，
           整个函数开头建一次即可——这样同学之间结果可比）
        2. 循环 max_steps 次：obs = get_observation(world);
           action = policy_fn(obs); ok = world.step(action); if ok: break
        3. 统计成功回合数，返回 成功数 / n_episodes
    选做：verbose_episodes > 0 时，逐步打印前这么多个回合的
        [step 动作 夹爪位置 笔位置 held] 日志（失败分析用，格式随意）。
    """
    rng = random.Random(seed)
    successes = 0
    for ep in range(n_episodes):  #外层：40个回合
        world = World()           #每个回合新世界
        if pen_jitter > 0: 
            world.pen[0] += rng.uniform(-pen_jitter, pen_jitter)  
            world.pen[1] += rng.uniform(-pen_jitter, pen_jitter)  
        ok = False   #每次随机扰动笔的位置，ok意味着成功放入笔筒，因此开始时一定不成功
        for step in range(max_steps):    #单次测试时最多给50步
            obs = get_observation(world)  #看世界，接受世界返回信息
            action = policy_fn(obs)  #模型开始决策
            ok = world.step(action)   #执行，世界变化
            if ep < verbose_episodes:    
                 print(f"    ep{ep} step{step:2d} a={[round(v,3) for v in action]}"
                      f" gripper=({world.gripper[0]:.2f},{world.gripper[1]:.2f})"
                      f" held={world.pen_held}")
            if ok:   #成功就停
                break
        if ok:  #成功就计数
            successes += 1
    return successes / n_episodes
                
    
    
    raise NotImplementedError("TODO-4: evaluate() 还没实现，先看上面的 docstring。")


# ============================================================
# 主流程（导师提供，禁止修改）——TODO 没填完时它会给你中文提示
# ============================================================

def main():
    print("=" * 60)
    print("Day 7 练习：行为克隆（BC）")
    print("=" * 60)

    print(f"torch 版本：{torch.__version__}")

    print("\n[1/4] 采集示范：200 回合（笔初始位置随机 ±0.05 m）")
    episodes = collect_demonstrations(n_episodes=200, seed=0)
    n_pairs = sum(len(t) for t in episodes)
    steps = [len(t) for t in episodes]
    print(f"      共 {n_pairs} 个 (o, a) 样本对；每回合步数 {min(steps)}–{max(steps)}")

    try:
        print("\n[2/4] 构建数据集")
        X, Y = build_dataset(episodes)
        print(f"      X 形状 {tuple(X.shape)}（N, obs_dim=8）  Y 形状 {tuple(Y.shape)}（N, act_dim=3）")

        print("\n[3/4] 训练 MLP（8→64→64→3，MSE + Adam，40 epochs）")
        model = train_policy(X, Y)

        print("\n[4/4] 评测（每格 40 回合）")
        policy_fn = make_policy_fn(model)
        sr_fixed = evaluate(policy_fn, n_episodes=40, seed=123, pen_jitter=0.0)
        sr_rand = evaluate(policy_fn, n_episodes=40, seed=123, pen_jitter=0.05)
        print(f"      标准初始状态（笔在 (0.3, 0)）      成功率：{sr_fixed * 100:.0f}%")
        print(f"      随机初始状态（笔 ±0.05 m）          成功率：{sr_rand * 100:.0f}%")

        print("\n对照：专家规则策略在随机初始状态下的成功率")
        sr_expert = evaluate(rule_policy, n_episodes=40, seed=123, pen_jitter=0.05)
        print(f"      专家                                成功率：{sr_expert * 100:.0f}%")

        print("\n失败分析：打印 2 个回合的逐步日志（观察 BC 在什么状态附近出错）")
        evaluate(policy_fn, n_episodes=40, seed=123, pen_jitter=0.05, verbose_episodes=2)

    except NotImplementedError as e:
        print(f"\n⏸  {e}")
        print("\n你还没有填完全部 TODO——上面是第 2/4 步开始的流程，"
              "回到本文件搜索 TODO 逐个实现即可。")
        print("全部填完后，参考解答在 solutions/behavior_cloning_solution.py，"
              "先有自己的版本再对照。")


if __name__ == "__main__":
    main()

    """一句话总结
十来步就断 = 成功了提前 break，不是失败、不是 max_steps 截断。跑满 50 步才停的才是失败。

日志里的"瑕疵"：ep0 step 4–7，grip 在 0.5 边界抖动（0.476→1.045→0.659→0.649），且朝笔走过头 0.06m。这是 BC 回归连续动作的经典弱点，但没导致失败。

想看真失败回合：改 verbose_episodes 逻辑，只在 ok=False 时打印；或者把评测回合数调大，看成功率。

你现在看到的日志，其实是成功案例的"不优雅版本"——模型学会了任务，只是在边界状态上不够利落。这恰恰是 BC 的典型表现：模仿到了"大致对"，没模仿到"精确"。
    """