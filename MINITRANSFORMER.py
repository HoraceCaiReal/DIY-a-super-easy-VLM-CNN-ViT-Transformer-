import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # 解决 Windows 上的 OpenMP 冲突

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# ========== 超参数 ==========
VOCAB = 10
D_MODEL = 16
N_HEADS = 2
D_FF = 32
N_LAYERS = 1
MAX_LEN = 8
BATCH = 4
LR = 1e-3
STEPS = 2000


# ========== 嵌入层 ==========
class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        return self.tok(x) + self.pos(pos)


# ========== 多头自注意力 ==========
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# ========== 前馈网络 ==========
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


# ========== Transformer 块 ==========
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ========== 完整模型 ==========
class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = Embedding()
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, x):
        x = self.emb(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


# ========== 数据生成 ==========
def get_batch():
    starts = torch.randint(0, VOCAB, (BATCH,))
    x = (starts.unsqueeze(1) + torch.arange(MAX_LEN)) % VOCAB
    y = (starts.unsqueeze(1) + torch.arange(1, MAX_LEN + 1)) % VOCAB
    return x, y


# ========== 训练 ==========
model = MiniTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for step in range(STEPS):
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step}, loss {loss.item():.4f}")


# ========== 测试 ==========
model.eval()
x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
with torch.no_grad():
    pred = model(x).argmax(dim=-1)

print("输入:", x.tolist())
print("预测:", pred.tolist())