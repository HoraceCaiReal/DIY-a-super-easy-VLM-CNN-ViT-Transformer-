"""
Day 6 练习：组装并训练一个能"听话指物"的 mini-VLA
====================================================
配套讲义：lectures/day06_组装MiniVLA.md（先读 §1 的数据流图再动手）

运行方式：
    python mini_vla.py
（若终端中文乱码，先执行  $env:PYTHONIOENCODING="utf-8"  再运行；训练在 CPU 上约 1–2 分钟）

任务：合成图像（32×32 白底 + 红蓝两个方块）+ 指令 "point to the red/blue square"
      → 模型输出目标方块中心的归一化坐标 (x, y)，各维 ∈ (0, 1)。

分工：
- 数据生成器、Tokenizer、2 层 Transformer 主干、训练循环、评测与计时：导师已写好，禁改
- TODO-1：PatchEmbedding.forward()  —— 图像 → 视觉 token（Day 4 你写过）
- TODO-2：MiniVLA.forward() 文本分支 —— ids → 文本 token（Day 5 你写过）
- TODO-3：MiniVLA.forward() 拼接    —— [视觉 | sep | 文本] → (B, 22, d)（Day 5 你写过）
- TODO-4：MiniVLA.forward() 读出+头 —— 池化 → MLP → (B, 2)

通关标准：训练后测试集上 "指对方块" 准确率 > 90% 且 平均坐标误差 < 0.05。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
# GBK 控制台防崩：emoji 无法编码时降级替换（同 day13/day16 的处理）
try:
    if (sys.stdout.encoding or "").lower().replace("-", "") in ("gbk", "cp936"):
        sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import time

import torch
import torch.nn as nn

# ----------------------------- 全局设定 -----------------------------
D = 64                 # token 维度
IMG, C, P = 32, 3, 8   # 图像边长 / 通道 / patch 边长
N = (IMG // P) ** 2    # 16 个视觉 token
VOCAB = ["<pad>", "<sep>", "point", "to", "the", "red", "blue", "square"]
V = len(VOCAB)
L = 5                  # 模板句固定 5 词，如 "point to the red square"
SEQ = N + 1 + L        # 22 = 视觉16 + sep1 + 文本5
TRAIN_N, TEST_N, BATCH = 768, 256, 64
EPOCHS, LR = 150, 1e-3
HINT = 0.1             # "指向正确方块"的坐标误差判据（归一化单位，≈3 像素）

# ------------------- 导师提供：分词器（Day 5 已练过，这里直接用） -------------------
class Tokenizer:
    def __init__(self, vocab):
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for i, w in enumerate(vocab)}

    def encode(self, text):
        return [self.word2id.get(w, self.word2id["<pad>"]) for w in text.lower().split()]

    def texts_to_ids(self, texts):
        """list[str] → (B, L) int64 张量。"""
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)


TOKENIZER = Tokenizer(VOCAB)

# ------------------- 导师提供：数据生成器 -------------------
def make_dataset(n, seed):
    """合成数据集。返回 dict：
    images  (n,3,32,32) 0–1  float   白底 + 红蓝两个 7×7 方块
    texts   list[str]                "point to the red square" / "...blue..." 各半
    actions (n,2) float              目标方块中心归一化坐标 (x, y) = ((cx+0.5)/32, (cy+0.5)/32)
    red_xy / blue_xy (n,2) float     两方块中心的归一化坐标（评测"是否指对方块"用）
    """
    g = torch.Generator().manual_seed(seed)
    half = 3
    images = torch.ones(n, C, IMG, IMG)
    actions = torch.zeros(n, 2)
    red_xy = torch.zeros(n, 2)
    blue_xy = torch.zeros(n, 2)
    texts = []
    for i in range(n):
        while True:
            cr = [int(torch.randint(4, 28, (1,), generator=g)),
                  int(torch.randint(4, 28, (1,), generator=g))]  # 红块中心 (列x, 行y)
            cb = [int(torch.randint(4, 28, (1,), generator=g)),
                  int(torch.randint(4, 28, (1,), generator=g))]  # 蓝块中心
            if abs(cr[0] - cb[0]) >= 2 * half + 1 or abs(cr[1] - cb[1]) >= 2 * half + 1:
                break  # 两方块不重叠
        red_xy[i] = torch.tensor([(cr[0] + 0.5) / IMG, (cr[1] + 0.5) / IMG])
        blue_xy[i] = torch.tensor([(cb[0] + 0.5) / IMG, (cb[1] + 0.5) / IMG])
        for center, col in ((cr, [1.0, 0.0, 0.0]), (cb, [0.0, 0.0, 1.0])):
            cx, cy = center
            r, c0 = slice(cy - half, cy + half + 1), slice(cx - half, cx + half + 1)
            images[i, :, r, c0] = torch.tensor(col).view(3, 1, 1)
        target = "red" if int(torch.randint(2, (1,), generator=g)) == 0 else "blue"
        texts.append(f"point to the {target} square")
        actions[i] = red_xy[i] if target == "red" else blue_xy[i]
    return {"images": images, "texts": texts, "actions": actions,
            "red_xy": red_xy, "blue_xy": blue_xy}


# ------------------- 导师提供：Transformer 主干（2 层，你手写过的结构） -------------------
def build_encoder():
    layer = nn.TransformerEncoderLayer(
        d_model=D, nhead=4, dim_feedforward=128, activation="gelu",
        batch_first=True, dropout=0.0)
    return nn.TransformerEncoder(layer, num_layers=2)


# ============================ TODO-1（必做） ============================
class PatchEmbedding(nn.Module):
    """Day 4 内容回收：图像 → (B, 16, d) 视觉 token。__init__ 已给，写 forward。"""

    def __init__(self, d=D, p=P):
        super().__init__()
        self.proj = nn.Linear(p * p * C, d)
        self.pos = nn.Parameter(torch.zeros(1, N, d))

    def forward(self, images):
        """(B, 3, 32, 32) → (B, 16, d)。三步：手工 patchify（reshape+permute+reshape）、
        self.proj、加 self.pos。不许改成 Conv2d 作弊（那是 Day 5 的"已知等价"，这里练 Day 4）。
        """
        b,c,h,w = images.shape
        n = h//P
        p = P
        x = images.reshape(b, c, n, p, n, p) #经典reshape到（B， C， H//p（patch高度）, p, W//p（patch宽度） ,p）
        x = x.permute(0,2,4,3,5,1)    #经典交换顺序重组，让patch能到前面
        x = x.reshape(b, n*n, p*p*C)    #经典最后重塑成（B, patch数量， 每个patch维度）
        return self.proj(x)+self.pos    
        
        raise NotImplementedError(
            "TODO-1 未填写：PatchEmbedding.forward()。先 reshape 成 (B,C,4,8,4,8)，"
            "permute 成 (B,4,4,8,8,C)，reshape 成 (B,16,192)，再过 proj 和 pos。见 day04 §5。")


# ============================ TODO-2/3/4（必做） ============================
class MiniVLA(nn.Module):
    """完整数据流：图像+指令 → 视觉token+文本token → 拼接 → Transformer → 动作头 → (B, 2)。"""

    def __init__(self):
        super().__init__()
        self.vision = PatchEmbedding()                 # TODO-1 在这里被调用
        self.sep = nn.Parameter(torch.zeros(1, 1, D))  # <sep> 的嵌入，可学习
        self.text_emb = nn.Embedding(V, D)
        self.text_pos = nn.Parameter(torch.zeros(1, L, D))
        self.encoder = build_encoder()
        self.head = nn.Sequential(                     # 动作头：回归式 MLP（讲义 §3）
            nn.Linear(D, 128), nn.GELU(), nn.Linear(128, 2)
            )

    def forward(self, images, texts):
        """images: (B,3,32,32) float；texts: list[str]，长度 B。返回 (B, 2) 预测坐标。

        你要写的四步（本函数的主体）：
        TODO-1 已在上面的类里做（本函数第 1 行调用它）
        TODO-2: ids = TOKENIZER.texts_to_ids(texts)；txt = self.text_emb(ids) + self.text_pos
        TODO-3: fused = torch.cat([vis, sep展开到batch, txt], dim=1)，应得 (B, 22, d)
        TODO-4: h = self.encoder(fused)；读出=按序列维平均池化；动作=self.head(读出)
        """
        vis = self.vision(images)   #1
        ids = TOKENIZER.texts_to_ids(texts)
        txt = self.text_emb(ids) + self.text_pos        #2
        fused = torch.cat([vis, self.sep.expand(vis.shape[0],-1,-1),txt], dim=1)  #3
        h = self.encoder(fused)   #4
        pooled = h.mean(dim = 1)  #最后的池化
        return self.head(pooled)
        
        
        raise NotImplementedError(
            "TODO-2/3/4 未填写：实现 MiniVLA.forward()。依次：视觉 token → 文本 token "
            "(查表+位置) → torch.cat 拼成 (B,22,d) → self.encoder → mean(dim=1) → self.head。")


# ==================== 导师提供：训练循环、评测与计时，禁改 ====================
def sanity_check(data, k=16):
    """自检数据：被指方块中心的像素颜色必须真是那个颜色（防生成器 bug 把任务变不可能）。"""
    for i in range(k):
        cx, cy = (data["red_xy"][i] * IMG).long().tolist()
        bx, by = (data["blue_xy"][i] * IMG).long().tolist()
        im = data["images"][i]
        assert im[0, cy, cx] > 0.5 and im[2, cy, cx] < 0.5, f"第{i}张红块中心不是红色"
        assert im[2, by, bx] > 0.5 and im[0, by, bx] < 0.5, f"第{i}张蓝块中心不是蓝色"
        want = data["texts"][i].split()[3]  # "red"/"blue"
        ref = data["red_xy"][i] if want == "red" else data["blue_xy"][i]
        assert torch.allclose(ref, data["actions"][i]), f"第{i}张指令与动作不一致"
    print(f"  数据自检通过（前 {k} 个样本：颜色/中心/动作/指令一致）")


def train(model, data):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    imgs, acts = data["images"], data["actions"]
    texts = data["texts"]
    log_every = max(1, EPOCHS // 10)
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(TRAIN_N)
        tot, nb = 0.0, 0
        for s in range(0, TRAIN_N, BATCH):
            idx = perm[s:s + BATCH]
            batch_texts = [texts[i] for i in idx.tolist()]
            pred = model(imgs[idx], batch_texts)
            loss = crit(pred, acts[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if ep % log_every == 0 or ep == 1:
            print(f"  epoch {ep:3d}/{EPOCHS}  train MSE = {tot / nb:.6f}")
    return model


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    pred = model(data["images"], data["texts"])
    acts = data["actions"]
    mse = nn.functional.mse_loss(pred, acts).item()
    err = (pred - acts).norm(dim=1)                    # 每个样本到"被指方块中心"的距离
    # "指对方块"：预测点离被指方块中心比离另一个方块中心更近。
    # d_red/d_blue 中较大者必然不是 acts 那个，即"另一个方块"的距离。
    d_other = torch.maximum((pred - data["red_xy"]).norm(dim=1),
                            (pred - data["blue_xy"]).norm(dim=1))
    point_acc = (err < d_other).float().mean().item()  # 指令遵守率
    hit_rate = (err < HINT).float().mean().item()      # 误差 < 0.1 的比例
    print(f"  测试 MSE = {mse:.6f} | 平均坐标误差 = {err.mean().item():.4f}"
          f" | 误差<0.1 占比 = {hit_rate:.1%} | 指对方块 = {point_acc:.1%}")
    return mse, err.mean().item(), hit_rate, point_acc


@torch.no_grad()
def time_inference(model, data, iters=200):
    model.eval()
    img1 = data["images"][:1]
    txt1 = [data["texts"][0]]
    t0 = time.perf_counter()
    for _ in range(iters):
        model(img1, txt1)
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  单步推理耗时（CPU、batch=1）≈ {ms:.2f} ms → 理论上限 {1000 / ms:.0f} Hz")


def main():
    torch.manual_seed(0)
    model = MiniVLA()
    n_params = sum(p.numel() for p in model.parameters())
    train_data = make_dataset(TRAIN_N, seed=11)
    test_data = make_dataset(TEST_N, seed=77)
    print(f"参数量 ≈ {n_params / 1e6:.2f} M | 训练 {TRAIN_N} 样本 | 测试 {TEST_N} 样本"
          f" | 序列长 {SEQ} = 16 视觉 + 1 sep + 5 文本")
    sanity_check(train_data)
    print("训练（MSE loss，Adam）：")
    train(model, train_data)
    print("评测：")
    _, mean_err, hit_rate, point_acc = evaluate(model, test_data)
    time_inference(model, test_data)
    ok = point_acc > 0.9 and mean_err < 0.05
    print(("全部通关 ✅  ") if ok else "尚未达标 ⚠️（多训几个 epoch 或找导师）",
          "这就是 Day 1 预告的 mini-VLA：图像+指令进，坐标动作出，端到端训练。")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print("\n" + "=" * 62)
        print(f"⚠️  练习未完成：{e}")
        print("=" * 62)
        print("提示：先读讲义 lectures/day06_组装MiniVLA.md §1 数据流图，")
        print("      Day 4/5 的参考解答思路可复用。填完 4 个 TODO 后重跑。")
