"""
Day 9 练习：多峰示范 —— MSE 学出"平均动作"会怎样？
===================================================
运行方式：
    python multimodal_actions.py

任务（横墙世界）：点机器人从 (0, -0.3) 到目标 (0, +0.4)，正前方一堵水平墙
（|y| < 0.05 且 |x| ≤ 0.52），只能从墙左端外或右端外绕——示范两族各占一半，
绕哪边都对。

实验 1（脚手架已给，直接能跑）：Day 7 的 MSE-MLP 一字不改地学两族混合示范，
看"平均动作"在闭环里发生什么。
实验 2（你来填）：2 分量 MDN（混合密度网络）——让策略表示 π(a|o) 这个分布，
并从中采样，而不是输出均值。
附加（选做 TODO-4）：动作离散成 17 bin + 分类头 + softmax 采样（Day 6 的老朋友）。

分工（参考解答 solutions/multimodal_actions_solution.py 分节注释与 TODO 编号一一对应）：

    TODO-1  MDNPolicy —— 输出头：K 权重 logit + K×2 均值 + K×2 σ原始参数
    TODO-2  mdn_nll —— 混合高斯负对数似然
    TODO-3  mdn_sample —— 先抽分量、再抽高斯样本
    TODO-4（选做） discrete_pipeline —— 离散化 + 交叉熵 + 采样

规则：
    - "导师提供，禁止修改"部分不许动（World、专家、采集、评测循环）
    - 实验 1 不依赖你的任何代码，stub 全空也能跑出撞墙结果
"""

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

MAX_STEPS = 100
GOAL_R = 0.15         # 距目标 < 它 = 成功（刻意留大：量化动作头有 ±0.1 级抖动）
WALL_HALF_L = 0.52    # 水平墙：|x| ≤ 0.52 是实心段；|x| > 0.52 是左右绕行通道
WALL_HALF_T = 0.05    # 厚度：|y| < 0.05
START = (0.0, -0.3)   # 距墙下缘 0.25 m ≈ 7 步
GOAL = (0.0, 0.4)
K_COMP = 2            # MDN 分量数

# ============================================================
# 第一部分：WallWorld（导师提供，禁止修改）
# ============================================================

class WallWorld:
    """点机器人 2D 世界：obs=(位置, 目标)，action=(dx,dy) 每步限幅 ±0.1。

    终止规则（step 返回码）：
        1 = 成功（距目标 < GOAL_R）
        0 = 还活着
       -1 = 失败：撞墙（新位置进入 |x|<0.52 且 |y|<0.05；撞墙不移动，直接判负）
    世界边界 |x|≤1.0, |y|≤0.9，越界裁剪、不算失败。
    """

    def __init__(self, start=START, goal=GOAL):
        self.pos = [float(start[0]), float(start[1])]
        self.goal = [float(goal[0]), float(goal[1])]
        self.done_code = 0
        self.traj = [list(self.pos)]

    def step(self, action):
        if self.done_code != 0:
            return self.done_code
        dx = max(-0.1, min(0.1, float(action[0])))
        dy = max(-0.1, min(0.1, float(action[1])))
        nx = max(-1.0, min(1.0, self.pos[0] + dx))
        ny = max(-0.9, min(0.9, self.pos[1] + dy))
        if abs(nx) < WALL_HALF_L and abs(ny) < WALL_HALF_T:
            self.done_code = -1
            return self.done_code
        self.pos = [nx, ny]
        self.traj.append(list(self.pos))
        if math.hypot(nx - self.goal[0], ny - self.goal[1]) < GOAL_R:
            self.done_code = 1
        return self.done_code

    def observation(self):
        return {"pos": list(self.pos), "goal": list(self.goal)}


def encode_obs(obs):
    return [obs["pos"][0], obs["pos"][1], obs["goal"][0], obs["goal"][1]]


def jittered_starts(n, seed=0, start_jitter=0.01):
    """生成一串起点（两族示范/评测共用同一串 → 配对抖动，见讲义 §4.4）。"""
    rng = random.Random(seed)
    return [(START[0] + rng.uniform(-start_jitter, start_jitter),
             START[1] + rng.uniform(-start_jitter, start_jitter))
            for _ in range(n)]


def ascii_map(world, title):
    """把轨迹画成 ASCII 地图（撞墙实验的证据，不依赖任何绘图库）。"""
    cols, rows = 19, 19          # x: -0.9..0.9，y: 0.9..-0.9，步长 0.1
    grid = [[" "] * cols for _ in range(rows)]
    for c in range(cols):
        x = -0.9 + c * 0.1
        if abs(x) < WALL_HALF_L:
            grid[9][c] = "#"                    # 水平墙画在 y=0.0 那一行
    def put(p, ch):
        c = round((p[0] + 0.9) / 0.1)
        r = round((0.9 - p[1]) / 0.1)
        if 0 <= r < rows and 0 <= c < cols and grid[r][c] == " ":
            grid[r][c] = ch
    for p in world.traj[:-1]:
        put(p, ".")
    put(world.traj[-1], "o")
    put(world.pos, "C" if world.done_code == -1 else "o")
    put(world.goal, "G")
    print(f"\n{title}")
    print("  " + "-" * (cols + 2))
    for r, row in enumerate(grid):
        y = 0.9 - r * 0.1
        print(f"{y:+.1f} |" + "".join(row) + "|")
    print(f"     {'x=-0.9':>8}{'x=0':^3}{'x=+0.9':>7}")


def make_plot(demo_trajs, eval_worlds, path):
    """matplotlib 存图（导师提供）：两族示范 + 评测轨迹。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 6))
    for traj, col, lab in ((demo_trajs[0], "tab:blue", "demo: detour right"),
                           (demo_trajs[1], "tab:green", "demo: detour left")):
        for i, w in enumerate(traj[:12]):
            xs = [p[0] for p in w.traj]; ys = [p[1] for p in w.traj]
            ax.plot(xs, ys, color=col, alpha=0.35, lw=1,
                    label=lab if i == 0 else None)
    ax.add_patch(Rectangle((-WALL_HALF_L, -WALL_HALF_T), 2 * WALL_HALF_L,
                           2 * WALL_HALF_T, color="k", label="wall"))
    for j, w in enumerate(eval_worlds):
        xs = [p[0] for p in w.traj]; ys = [p[1] for p in w.traj]
        col = "tab:red" if w.done_code == -1 else "tab:purple"
        ax.plot(xs, ys, color=col, lw=1.5)
        ax.plot(xs[-1], ys[-1], marker="X" if w.done_code == -1 else "o",
                color=col, ms=8)
    ax.plot(START[0], START[1], marker="s", color="gray", label="start")
    ax.plot(GOAL[0], GOAL[1], marker="*", color="gold", ms=14, label="goal")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.95, 0.95)
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("bimodal demos (right=blue/left=green) vs MSE-BC(red) vs MDN(purple)")
    fig.savefig(path, dpi=110); plt.close(fig)
    print(f"      图片已保存：{path}")


# ============================================================
# 第二部分：两族示范专家 + 采集 + 数据集（导师提供，禁止修改）
# ============================================================

WAYPOINTS = {
    "right": [(0.92, 0.05), (0.95, 0.35), (0.0, 0.4)],
    "left":  [(-0.92, 0.05), (-0.95, 0.35), (0.0, 0.4)],
}


def make_expert(family):
    """脚本专家：斜奔到墙端外，纵穿，再到目标。带单调推进的内部航点索引，
    每个回合要用一个新实例。"""
    wps = WAYPOINTS[family]
    state = {"i": 0}

    def expert(obs):
        x, y = obs["pos"]
        while state["i"] < len(wps) and math.hypot(wps[state["i"]][0] - x,
                                                   wps[state["i"]][1] - y) < 0.15:
            state["i"] += 1
        tx, ty = wps[state["i"]] if state["i"] < len(wps) else obs["goal"]
        dx, dy = tx - x, ty - y
        d = math.hypot(dx, dy)
        if d < 1e-9:
            return [0.0, 0.0]
        s = min(0.1, d) / d
        return [dx * s, dy * s]
    return expert


def run_episode(world, act_fn, max_steps=MAX_STEPS):
    """闭环跑一回合：act_fn(obs) -> [dx, dy]。返回世界对象（含轨迹/结局码）。"""
    code = 0
    for _ in range(max_steps):
        code = world.step(act_fn(world.observation()))
        if code != 0:
            break
    return world


def collect_multimodal(n_per_family=100, seed=0, start_jitter=0.01):
    """两族各采 n_per_family 个回合（共用同一串起点抖动）。返回 (ep_right, ep_left)。"""
    starts = jittered_starts(n_per_family, seed=seed, start_jitter=start_jitter)
    out = {"right": [], "left": []}
    for fam in ("right", "left"):
        for start in starts:
            w = run_episode(WallWorld(start=start), make_expert(fam))
            assert w.done_code == 1, "脚本专家不应失败"
            out[fam].append(w)
    return out["right"], out["left"]


def build_dataset(ep_right, ep_left):
    """摊平成 X:(N,4) Y:(N,2)。同一起点上两族动作 dx 相反、dy 相同——多峰就在这一行里。"""
    xs, ys = [], []
    for traj in (ep_right + ep_left):
        obs = {"pos": traj.traj[0], "goal": traj.goal}
        for i in range(1, len(traj.traj)):
            xs.append(encode_obs(obs))
            ys.append([traj.traj[i][0] - obs["pos"][0], traj.traj[i][1] - obs["pos"][1]])
            obs = {"pos": traj.traj[i], "goal": traj.goal}
    return (torch.tensor(xs, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.float32))


# ============================================================
# 第三部分：实验 1 —— MSE-MLP（脚手架全部给出，直接运行看结果）
# ============================================================

class MLPPolicy(nn.Module):
    """Day 7 的网络原样平移（act_dim=2）。它只会输出"一个动作"。"""
    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, x):
        return self.net(x)


def train_supervised(model, X, Y, loss_fn, epochs=40, batch_size=64, lr=1e-3,
                     verbose="epoch10"):
    """通用监督循环（导师提供）：注意 loss_fn 是自己传的——这就是实验 2 的接口。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
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
            print(f"    epoch {epoch:3d}  loss={total / n:.5f}")
    return model


def evaluate(act_fn, n_episodes=50, seed=7, start_jitter=0.01):
    """统计：成功率 / 撞墙率 / 超时率 / 成功者从哪边绕（导师提供）。"""
    starts = jittered_starts(n_episodes, seed=seed, start_jitter=start_jitter)
    succ = collide = timeout = 0
    right = left = 0
    worlds = []
    for start in starts:
        w = run_episode(WallWorld(start=start), act_fn)
        worlds.append(w)
        if w.done_code == 1:
            succ += 1
            cross = [p for p in w.traj if abs(p[0]) >= WALL_HALF_L]
            if cross and cross[0][0] >= 0:
                right += 1
            else:
                left += 1
        elif w.done_code == -1:
            collide += 1
        else:
            timeout += 1
    return {"succ": succ / n_episodes, "collide": collide / n_episodes,
            "timeout": timeout / n_episodes, "right": right, "left": left,
            "worlds": worlds}


def mse_policy_fn(model):
    model.eval()
    def act(obs):
        x = torch.tensor(encode_obs(obs), dtype=torch.float32)
        with torch.no_grad():
            return model(x).tolist()
    return act


# ============================================================
# 第四部分：实验 2 —— MDN（TODO-1/2/3，你来填）
# ============================================================

class MDNPolicy(nn.Module):
    """TODO-1：混合密度网络头。网络本体同 MLP，只换最后一层和输出解释。

    最后一层输出 k + 2*k*act_dim = 2 + 4 + 4 = 10 个数（K=2, 动作 2 维），切分约定：
        out[:, 0:2]                → 权重 logit（K 个，softmax 后是 π_k）
        out[:, 2:6]  .view(-1,k,2) → 均值 μ（每个分量 2 维）
        out[:, 6:10] .view(-1,k,2) → σ 的原始参数 raw
    σ 用 sigmoid 参数化（别用 clamp！clamp 截断区梯度恒 0）：
        stds = 0.002 + 0.048 * torch.sigmoid(raw)
    提示：forward 返回 (logits, means, stds) 三元组即可，训练和采样都吃这三样。
    """
    def __init__(self, obs_dim=4, act_dim=2, k=K_COMP, hidden=64):
        super().__init__()
        self.act_dim = act_dim
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, k + 2 * k * act_dim),    #最后一层输出k+2*k*act_dim个数契合题目
        )
        
        
        #raise NotImplementedError("TODO-1: MDNPolicy 的 __init__ 还没实现。")

    def forward(self, x):
        out = self.net(x)
        logits = out[:, :self.k]  #抽取logits
        means = out[:, self.k:self.k + self.k * self.act_dim].view(-1, self.k, self.act_dim)  #抽取means
        s = out[:, self.k + self.k * self.act_dim:].view(-1, self.k, self.act_dim)  #抽取stds的原始参数
        stds = 0.002 + 0.048 * torch.sigmoid(s)  # 参数化标准差
        return logits, means, stds
    


def mdn_nll(logits, means, stds, target):
    """TODO-2：混合高斯负对数似然（对 MSE 的替换品）。

    对每个样本 i：
        logN_k = Σ_dims [ -0.5*((a-μ_k)/σ_k)² - log σ_k - 0.5*ln(2π) ]
        loss_i = -log Σ_k π_k exp(logN_k) = -logsumexp_k( log π_k + logN_k )
    返回 batch 平均 loss（越小越好）。
    零件：torch.log_softmax / logsumexp；target 形状 (B,2)，用
          target.unsqueeze(1) 变 (B,1,2) 再和 (B,K,2) 相减（广播）。
    """
    # π 的对数：log_softmax 一步到位（比先 softmax 再 log 数值稳）
    log_pi = torch.log_softmax(logits, dim=-1)  # 计算log π_k,(B,K)
    
    t = target.unsqueeze(1)  #(B,1,2)
    z = (t-means)/stds
    log_nmormal = (
        - 0.5 * (z ** 2).sum(-1)                              # 二次项，对动作维求和
        - stds.log().sum(-1)                                 # -log σ₁ - log σ₂
        - means.shape[-1] * 0.5 * math.log(2 * math.pi)
    )
    return -torch.logsumexp(log_pi + log_nmormal, dim=-1).mean()  # -log Σ_k π_k exp(logN_k) 对 batch 平均
    
    #raise NotImplementedError("TODO-2: mdn_nll() 还没实现。")


def mdn_sample(logits, means, stds, generator=None):
    """TODO-3：从混合分布采一个动作——"从 π(a|o) 里抽"的第一步正式实现。

    两步抽样：
        1) 按权重 π=softmax(logits) 抽分量号 k（torch.multinomial）
        2) 从该分量 N(μ_k, σ_k²) 抽：a = μ_k + σ_k * ε, ε~N(0,1)
    返回形状 (B,2) 的动作张量。提示：b = torch.arange(B); mu = means[b, idx]。
    """
    B = logits.shape[0]
    pi = torch.softmax(logits, dim=-1)  # 计算 π_k
    
    idx = torch.multinomial(pi, num_samples=1, replacement=True, generator=generator).squeeze(-1)  # 按 π 抽分量号 k
    b = torch.arange(B)
    mu = means[b, idx]  # 取出对应分量的均值
    sigma = stds[b, idx]  # 取出对应分量的标准差
    eps = torch.randn_like(mu, generator=generator)  # ε~N(0,1)
    return mu + sigma*eps
    

def mdn_policy_fn(model, seed=0):
    """推理包装（导师提供）：每步做一次 TODO-3 的抽样。"""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    def act(obs):
        x = torch.tensor(encode_obs(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, means, stds = model(x)
        return mdn_sample(logits, means, stds, generator=gen)[0].tolist()
    return act


# ============================================================
# 第五部分：附加 TODO-4（选做）—— 动作离散化 + 分类头
# ============================================================

# 17 个动作 bin：16 个方向（步长 0.1，22.5° 一档）+ 静止
ACTION_BINS = [[0.0, 0.0]] + [
    [round(math.cos(a) * 0.1, 6), round(math.sin(a) * 0.1, 6)]
    for a in [i * math.pi / 8 for i in range(16)]
]


def discrete_pipeline(X, Y, epochs=80):
    """TODO-4（选做）：离散化动作头一条龙，返回 act_fn(obs)->[dx,dy]。

    步骤提示：
    1) 标签：每个示范动作归到最近的 bin（参考解答里的 nearest_bin）
    2) 模型：同 MLP，最后一层输出 17 个 logits（分类头，Day 6 的老朋友）
    3) 损失：nn.CrossEntropyLoss()（注意标签是 bin 索引，不是动作值！）
    4) 推理：probs=softmax(logits) → torch.multinomial 抽一个 bin → 返回其动作
    跑分标准：成功率 ≥ 90%（参考解答实测 96%，见 solutions.md）。
    """
    bins = torch.tensor(ACTION_BINS, dtype=torch.float32)            # (17, 2)
    d2 = ((Y.unsqueeze(1) - bins.unsqueeze(0)) ** 2).sum(-1)         # (N, 17)
    labels = d2.argmin(dim=1)                                        # 每个动作归最近 bin

    model = nn.Sequential(
        nn.Linear(4, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, len(ACTION_BINS)),     # 17 个 logits，分类头
    )
    train_supervised(model, X, labels, nn.CrossEntropyLoss(), epochs=epochs, lr=1e-3)
    model.eval()

    def act(obs):
        x = torch.tensor(encode_obs(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=-1)
            idx = torch.multinomial(probs, 1).item()
        return ACTION_BINS[idx]              # 每步随机抽一个 bin —— 同样会多峰分化
    return act
    
    #raise NotImplementedError("TODO-4: discrete_pipeline() 还没实现（选做，可留空）。")


# ============================================================
# 主流程（导师提供，禁止修改）
# ============================================================

def main():
    print("=" * 62)
    print("Day 9 练习：多峰示范 —— MSE 平均撞墙 vs MDN 采样绕行")
    print("=" * 62)
    print(f"torch 版本：{torch.__version__}")

    print("\n[1/5] 采集两族示范：右绕/左绕各 100 回合（共用同一串起点抖动）")
    ep_right, ep_left = collect_multimodal(n_per_family=100, seed=0)
    X, Y = build_dataset(ep_right, ep_left)
    print(f"      X {tuple(X.shape)}  Y {tuple(Y.shape)}（同一起点上两族动作 dx 相反、dy 相同）")

    print("\n[2/5] 实验 1（脚手架已给）：MSE-MLP 学两族示范")
    mse = train_supervised(MLPPolicy(), X, Y, nn.MSELoss())
    res_mse = evaluate(mse_policy_fn(mse))
    w_bad = next((w for w in res_mse["worlds"] if w.done_code == -1), res_mse["worlds"][0])
    ascii_map(w_bad, "MSE-MLP 的一条轨迹（C=撞墙处）：")
    print(f"      MSE   成功 {res_mse['succ']:.0%} | 撞墙 {res_mse['collide']:.0%} | "
          f"超时 {res_mse['timeout']:.0%}")

    try:
        print("\n[3/5] 实验 2（你来填）：2 分量 MDN + NLL + 采样")
        mdn = train_supervised(MDNPolicy(), X, Y,
                               lambda o, t: mdn_nll(o[0], o[1], o[2], t),
                               epochs=150, lr=3e-3)
        res_mdn = evaluate(mdn_policy_fn(mdn))
        w_good = next((w for w in res_mdn["worlds"] if w.done_code == 1), None)
        print(f"      MDN   成功 {res_mdn['succ']:.0%} | 撞墙 {res_mdn['collide']:.0%} | "
              f"超时 {res_mdn['timeout']:.0%}"
              f"（成功者：右绕 {res_mdn['right']} / 左绕 {res_mdn['left']}）")

    except NotImplementedError as e:
        print(f"\n⏸  {e}")
        print("\n实验 1 已经跑完了——看到了吧：MSE 学出的'平均动作'在闭环里撞墙。")
        print("回到本文件搜索 TODO-1/2/3 把 MDN 填上，再来看 [4/5] 之后的输出。")
        print("参考解答在 solutions/multimodal_actions_solution.py，先有自己的版本再对照。")
        return

    print("\n[4/5] 附加（选做）：动作离散成 17 bin + 分类头 + 采样")
    try:
        disc_act = discrete_pipeline(X, Y)
        res_disc = evaluate(disc_act)
        print(f"      离散  成功 {res_disc['succ']:.0%} | 撞墙 {res_disc['collide']:.0%} | "
              f"超时 {res_disc['timeout']:.0%}"
              f"（成功者：右绕 {res_disc['right']} / 左绕 {res_disc['left']}）")
    except NotImplementedError as e:
        print(f"      （{e} 选做题没填也正常，先确保 TODO-1/2/3 完成）")

    print("\n[5/5] 存图")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fig_multimodal.png")
    make_plot((ep_right, ep_left), [w_bad] + ([w_good] if w_good else []), path)


if __name__ == "__main__":
    main()
