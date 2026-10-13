import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(0)
random.seed(0)


# === Hyperparameters ===
D_MODEL = 32
N_HEADS = 4
D_FF = 64
N_LAYERS = 2
NUM_COLORS = 4          # 红绿蓝黄
IMG_SIZE = 32
NUM_IMG_TOKENS = 16     # CNN 输出的 token 数
BATCH = 16
LR = 1e-3
STEPS = 2000

COLOR_RGB = [
    [1.0, 0.0, 0.0],    # 红
    [0.0, 1.0, 0.0],    # 绿
    [0.0, 0.0, 1.0],    # 蓝
    [1.0, 1.0, 0.0],    # 黄
]


# === 位置嵌入 ===
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.pos = nn.Embedding(max_len, D_MODEL)

    def forward(self, x):
        B, T, C = x.shape
        pos = torch.arange(T, device=x.device)
        return x + self.pos(pos)


# === 图像编码器 ===
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),    # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),   # 8 -> 4
            nn.ReLU(),
        )
        self.proj = nn.Linear(32, D_MODEL)

    def forward(self, img):
        # img: (B, 3, 32, 32)
        feat = self.cnn(img)              # (B, 32, 4, 4)
        feat = feat.flatten(2)            # (B, 32, 16)
        feat = feat.transpose(1, 2)       # (B, 16, 32)
        return self.proj(feat)            # (B, 16, D_MODEL)


# === 多头注意力 ===
class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# === 前馈网络 ===
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.ReLU(),
            nn.Linear(D_FF, D_MODEL),
        )

    def forward(self, x):
        return self.net(x)


# === Transformer 块 ===
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MHA()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# === Mini VLM ===
class MiniVLM(nn.Module):
    def __init__(self):
        super().__init__()
        # CLS token 可学习
        self.cls_token = nn.Parameter(torch.randn(1, 1, D_MODEL))
        # 文本嵌入：每个颜色一个向量
        self.text_emb = nn.Embedding(NUM_COLORS, D_MODEL)
        # 类型嵌入：0=CLS, 1=图像, 2=文本
        self.type_emb = nn.Embedding(3, D_MODEL)

        self.img_encoder = ImageEncoder()

        max_len = 1 + NUM_IMG_TOKENS + 1   # CLS + 图像 + 文本
        self.pos_emb = PositionalEmbedding(max_len)

        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 2)   # 二分类

    def forward(self, img, text):
        B = img.shape[0]
        device = img.device

        # 1. CLS token
        cls = self.cls_token.expand(B, -1, -1)
        cls = cls + self.type_emb(torch.zeros(B, 1, dtype=torch.long, device=device))

        # 2. 图像 token
        img_tok = self.img_encoder(img)                              # (B, 16, D)
        img_tok = img_tok + self.type_emb(torch.ones(B, NUM_IMG_TOKENS, dtype=torch.long, device=device))

        # 3. 文本 token
        txt_tok = self.text_emb(text).unsqueeze(1)                   # (B, 1, D)
        txt_tok = txt_tok + self.type_emb(2 * torch.ones(B, 1, dtype=torch.long, device=device))

        # 4. 拼接: [CLS, 图像×16, 文本×1]
        x = torch.cat([cls, img_tok, txt_tok], dim=1)                # (B, 18, D)

        # 5. 位置嵌入
        x = self.pos_emb(x)

        # 6. Transformer
        x = self.blocks(x)
        x = self.ln_f(x)

        # 7. 取 CLS 位置的输出分类
        cls_out = x[:, 0, :]
        return self.head(cls_out)


# === 数据生成 ===
def make_image(color_idx, size=IMG_SIZE):
    rgb = torch.tensor(COLOR_RGB[color_idx]).view(3, 1, 1)
    img = rgb.repeat(1, size, size)
    img = img + 0.1 * torch.randn(3, size, size)   # 加噪声
    return img.clamp(0, 1)


def get_batch(batch_size=BATCH):
    imgs, texts, labels = [], [], []
    for _ in range(batch_size):
        c1 = random.randint(0, NUM_COLORS - 1)   # 图像颜色
        c2 = random.randint(0, NUM_COLORS - 1)   # 文本颜色
        label = 1 if c1 == c2 else 0
        imgs.append(make_image(c1))
        texts.append(c2)
        labels.append(label)
    imgs = torch.stack(imgs)
    texts = torch.tensor(texts, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, texts, labels


# === 训练 ===
model = MiniVLM()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for step in range(STEPS):
    imgs, texts, labels = get_batch()
    logits = model(imgs, texts)
    loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        print(f"step {step}, loss {loss.item():.4f}, acc {acc:.2%}")


# === 测试 ===
model.eval()
print("\n=== 测试 ===")
with torch.no_grad():
    tests = [
        (0, 0, "图像红 + 文本红"),
        (0, 2, "图像红 + 文本蓝"),
        (1, 1, "图像绿 + 文本绿"),
        (3, 1, "图像黄 + 文本绿"),
        (2, 2, "图像蓝 + 文本蓝"),
    ]
    for c_img, c_txt, name in tests:
        img = make_image(c_img).unsqueeze(0)
        txt = torch.tensor([c_txt])
        logits = model(img, txt)
        pred = logits.argmax(dim=-1).item()
        print(f"{name} → {'匹配' if pred == 1 else '不匹配'}")