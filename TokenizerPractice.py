"""
Day 5 练习：分词 + Embedding + 视觉/文本 token 拼接
====================================================
配套讲义：lectures/day05_语言Token与多模态融合.md（先读 §4 再动手）

运行方式：
    python token_fusion.py
（若终端中文乱码，先执行  $env:PYTHONIOENCODING="utf-8"  再运行）

分工：
- VOCAB、Tokenizer 类骨架、视觉侧 ConvPatchEmbed（= Day 4 答案的卷积版）、
  Transformer 演示层、主流程断言：导师已写好，禁改
- TODO-1：Tokenizer.encode()          —— 字符串 → token id 列表
- TODO-2：TextEmbedding.forward()      —— id → (B, L, d) 文本 token
- TODO-3：fuse()                        —— 视觉 token + <sep> + 文本 token → (B, 22, d)

通关标准：最后打印 "全部断言通过 ✅"。
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

import torch
import torch.nn as nn

# ------------------------- 导师提供：设定与词表 -------------------------
B, H, W, C = 2, 32, 32, 3
P, N, D = 8, 16, 64              # patch 边长 / patch 数 / token 维度
MAX_LEN = 5                      # 练习里的模板句都是 5 个词

# 玩具词表：21 个词，够说"指哪个颜色方块/圆圈"这类指令（讲义 §2）
# 这是一个人工定义的列表
VOCAB = ["<pad>", "<unk>", "<sep>", "a", "apple", "blue", "circle", "cup",
         "green", "in", "on", "pen", "plate", "point", "put", "red", "small",
         "square", "the", "to", "yellow"]
V = len(VOCAB)
SEP_ID = VOCAB.index("<sep>")

# ------------------------- 导师提供：Day 4 的视觉侧 -------------------------
def make_toy_images(batch=B, seed=3):
    """32×32 白底 + 红蓝两方块（与 day04 相同的玩具图像）。"""
    g = torch.Generator().manual_seed(seed)
    imgs = torch.ones(batch, C, H, W)
    half = 3
    palette = {"red": (1.0, 0.0, 0.0), "blue": (0.0, 0.0, 1.0)}
    for b in range(batch):
        while True:
            cr = [int(torch.randint(4, 28, (1,), generator=g)),
                  int(torch.randint(4, 28, (1,), generator=g))]
            cb = [int(torch.randint(4, 28, (1,), generator=g)),
                  int(torch.randint(4, 28, (1,), generator=g))]
            if abs(cr[0] - cb[0]) >= 2 * half + 1 or abs(cr[1] - cb[1]) >= 2 * half + 1:
                break
        for center, name in ((cr, "red"), (cb, "blue")):
            cx, cy = center
            r, c0 = slice(cy - half, cy + half + 1), slice(cx - half, cx + half + 1)
            imgs[b, :, r, c0] = torch.tensor(palette[name]).view(3, 1, 1)
    return imgs


class ConvPatchEmbed(nn.Module):
    """Day 4 练习的卷积版答案（讲义 §5.3 的等价性），导师替你写好，读懂即可。"""

    def __init__(self, d=D, p=P):
        super().__init__()
        self.conv = nn.Conv2d(C, d, kernel_size=p, stride=p)
        self.pos = nn.Parameter(torch.randn(1, N, d) * 0.02)

    def forward(self, images):                      # (B,3,32,32) → (B,16,d)
        return self.conv(images).flatten(2).transpose(1, 2) + self.pos


# ============================ TODO-1（必做） ============================
class Tokenizer:
    """词级分词器：空格切词 + 查词表。真实 LLM 用子词（BPE），套路相同。"""

    def __init__(self, vocab):
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for i, w in enumerate(vocab)}

    def encode(self, text):
        """字符串 → token id 列表（list[int]）。

        规则（讲义 §2）：
        1. 转小写、按空格切词
        2. 每个词查 self.word2id；词表里没有的词 → 用 <unk> 的 id
        例："point to the red square" → [13, 19, 18, 15, 17]
        """
        words = text.lower().split()       #lower是全转小写，split是分开
        unk = self.word2id["<unk>"]        #unk是unknown，专用未知id
        return [self.word2id.get(w,unk) for w in words]   #dict里的get用法：找到就返回w，没找到就返回unk
    
        raise NotImplementedError(
            "TODO-1 未填写：实现 Tokenizer.encode()。lower().split() 然后逐个查 "
            "self.word2id，查不到用 self.word2id['<unk>']。见讲义 day05 §2。")

    def decode(self, ids):
        """参考实现（id 列表 → 词列表），读懂 encode/decode 的互逆关系。"""
        return [self.id2word[i] for i in ids]


# ============================ TODO-2（必做） ============================
class TextEmbedding(nn.Module):
    """文本侧：id → embedding 查表 → 加位置编码（对照 ConvPatchEmbed 的三步）。"""

    def __init__(self, vocab_size=V, d=D, max_len=MAX_LEN):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)          # E ∈ R^(V×d)，讲义 §4.1
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)

    def forward(self, ids):
        """ids: (B, L) 的 int64 张量 → 返回 (B, L, d)。

        两步：self.emb(ids) 查表，然后加 self.pos（广播到 batch 维）。
        """
        return self.emb(ids)+self.pos
        
        raise NotImplementedError(
            "TODO-2 未填写：实现 TextEmbedding.forward()。两行：查表 self.emb(ids)，"
            "再加 self.pos。见讲义 day05 §4.1 与 §4.5。")


# ============================ TODO-3（必做） ============================
def fuse(visual, text, sep):
    """模态拼接：视觉 token + <sep> + 文本 token → 一个多模态序列。

    visual: (B, 16, d)   text: (B, L, d)   sep: (1, 1, d) 的 <sep> 嵌入（nn.Parameter）
    返回:   (B, 16+1+L, d)，顺序必须是 [视觉 | sep | 文本]（讲义 §4.5）
    提示：sep 的 batch 维是 1，需要 expand 到 B（或用 torch.cat 的广播限制想清楚）。
    """
    return torch.cat([visual, sep.expand(B, -1, -1),text],dim=1)  
    #忘记 expand——torch.cat 不做广播，(1,1,d) 不展开会报形状不匹配。这正是"拼接是多模态的全部魔法，但每个 tensor 都得自己对齐规格"的微观体现。
    
    raise NotImplementedError(
        "TODO-3 未填写：实现 fuse()。一行 torch.cat([visual, sep.expand(...), text], "
        "dim=1)，dim=1 是序列维。见讲义 day05 §4.3。")


# ======================= 以下为导师提供的验证流程，禁改 =======================
def main():
    torch.manual_seed(0)

    tokenizer = Tokenizer(VOCAB)
    texts = ["point to the red square", "point to the blue square"]

    # ---- 分词：字符串 → ids ----
    ids_list = [tokenizer.encode(t) for t in texts]
    assert ids_list[0] == [13, 19, 18, 15, 17], f"分词结果错：{ids_list[0]}"
    assert [tokenizer.id2word[i] for i in ids_list[1]] == texts[1].split()
    print(f"① 分词: {texts[0]!r} → ids {ids_list[0]}  (L={len(ids_list[0])})")

    ids = torch.tensor(ids_list, dtype=torch.long)
    assert ids.shape == (B, MAX_LEN), f"ids 形状错：{tuple(ids.shape)}"
    print(f"② batch 化: {tuple(ids.shape)}  int64")

    # ---- 视觉 token（Day 4 内容，导师代劳） ----
    images = make_toy_images()
    vision = ConvPatchEmbed()
    vis = vision(images)
    assert vis.shape == (B, N, D), f"视觉 token 形状错：{tuple(vis.shape)}"
    print(f"③ 视觉 token: {tuple(vis.shape)}  (B, 16, d)")

    # ---- 文本 token ----
    text_emb = TextEmbedding()
    txt = text_emb(ids)
    assert txt.shape == (B, MAX_LEN, D), f"文本 token 形状错：{tuple(txt.shape)}"
    print(f"④ 文本 token: {tuple(txt.shape)}  (B, L, d)")

    # ---- 拼接 ----
    sep = nn.Parameter(torch.randn(1, 1, D) * 0.02)
    fused = fuse(vis, txt, sep)
    L_total = N + 1 + MAX_LEN
    assert fused.shape == (B, L_total, D), f"拼接后形状错：{tuple(fused.shape)}"
    print(f"⑤ 拼接: [视觉16 | sep1 | 文本5] → {tuple(fused.shape)}  (B, {L_total}, d)")

    # ---- 过一层真正的 Transformer：它眼里只有 (B, T, d) ----
    layer = nn.TransformerEncoderLayer(
        d_model=D, nhead=4, dim_feedforward=128, activation="gelu", batch_first=True)
    out = layer(fused)
    assert out.shape == fused.shape, "Transformer 层不应改变形状"
    print(f"⑥ TransformerEncoderLayer: {tuple(fused.shape)} → {tuple(out.shape)}"
          f"  ← 两种来源的 token 已无法从'形状'上区分")

    print("\n全部断言通过 ✅  两种模态已住进同一个序列——Day 6 就把动作头接到它后面。")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print("\n" + "=" * 62)
        print(f"⚠️  练习未完成：{e}")
        print("=" * 62)
        print("提示：先读讲义 lectures/day05_语言Token与多模态融合.md §2/§4，")
        print("      填完三个 TODO 后重新运行。参考解答在 solutions/，先别偷看。")
