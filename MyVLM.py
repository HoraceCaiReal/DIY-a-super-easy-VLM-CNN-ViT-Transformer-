import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # 解决 Windows 上的 OpenMP 冲突
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
torch.manual_seed(0)


"""HYPERPARAMETERS"""
#相比最初的识数字transformer，VLM的输入变成了输入图像token
NUM_COLORS = 4
IMG_SIZE=32
NUM_IMG_TOKENS = 16 #CNN最终卷积后输出数
COLOR_RGB=[[1.0,0.0,0.0],
           [0.0,1.0,0.0],
           [0.0,0.0,1.0],
           [1.0,1.0,0.0]] #红蓝绿黄

D_MODEL = 32
N_HEADS = 4
D_FF = 64
N_LAYERS = 2
BATCH = 4
LR = 0.005
STEPS = 2000
#在这里，MAX_LEN变成了一个组合输入token，所以它不是超参数了



class PositionalEmbedding(nn.Module):
    def __init__(self,max_len):
        super().__init__()
        self.pos = nn.Embedding(max_len, D_MODEL)
    
    def forward(self,x):
        B,T,C = x.shape    #在VLM里，鉴于输入的是图像，输入就变成了B,T,C三维
        pos = torch.arange(T,device=x.device)
        return x + self.pos(pos)

class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, D_MODEL*3)
        self.proj = nn.Linear(D_MODEL ,D_MODEL)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3,dim = -1)  # Split into Q,K,V
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1,2)   #TRANSPOSE Here is to make the shape of Q,K,V to be [B, N_HEADS, T, head_dim] for next calculation
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1,2) 
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1,2) 
        
        scores = q @ k.transpose(-2,-1) / (self.head_dim**0.5)  #Original Formula of Attention: QK^T/sqrt(d_k)
        
        #mask = torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()  #triu Means Upper Triangular Matrix, diagonal=1 means the first diagonal above the main diagonal
        #scores = scores.masked_fill(mask, float('-inf'))  # Masking future tokens
        
        attn = F.softmax(scores,dim=-1)  #Get attention weights
        out = attn @ v    #Get attenttion weights multiplied by V to get the output of ATTENTION LAYER
        
        out = out.transpose(1,2).contiguous().view(B,T,C)  #transpose back to original shape [B,T,C] and use CONTIGUOUS to make sure the memory is continuous for nex layer
        return self.proj(out)  #Mix information from differenet heads and project back to D_MODEL dimension

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1,stride=2),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,padding=1,stride=2),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3,padding=1,stride=2),
            nn.ReLU(),              
        )
        
        self.proj= nn.Linear(32,D_MODEL)
        
    
    def forward(self,img):
        feat = self.cnn(img)
        feat = feat.flatten(2)
        feat = feat.transpose(1,2)
        return self.proj(feat)


class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        self.qkv = nn.Linear(D_MODEL, D_MODEL*3)
        self.proj = nn.Linear(D_MODEL ,D_MODEL)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3,dim = -1)  # Split into Q,K,V
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1,2)   #TRANSPOSE Here is to make the shape of Q,K,V to be [B, N_HEADS, T, head_dim] for next calculation
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1,2) 
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1,2) 
        
        scores = q @ k.transpose(-2,-1) / (self.head_dim**0.5)  #Original Formula of Attention: QK^T/sqrt(d_k)
        
        #在VLM中，鉴于引入掩码会导致后面的学习混乱，图像无法自己看自己，因此不要掩码
        #mask = torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()  #triu Means Upper Triangular Matrix, diagonal=1 means the first diagonal above the main diagonal
        #scores = scores.masked_fill(mask, float('-inf'))  # Masking future tokens
        
        attn = F.softmax(scores,dim=-1)  #Get attention weights
        out = attn @ v    #Get attenttion weights multiplied by V to get the output of ATTENTION LAYER
        
        out = out.transpose(1,2).contiguous().view(B,T,C)  #transpose back to original shape [B,T,C] and use CONTIGUOUS to make sure the memory is continuous for nex layer
        return self.proj(out)  #Mix information from differenet heads and project back to D_MODEL dimension

#==FeedForward==#
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.ReLU(),
            nn.Linear(D_FF,D_MODEL)
        )
    def forward(self,x):
        return self.net(x)
    
#===Transformer Block  Basic Unit of Transformer, finish forward propagation inside instead of complete class===
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MHA()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
    
    
    #x += y 在 Python 里等价于 x = x.__iadd__(y)，也就是在原张量的内存上直接改。但 PyTorch 做反向传播时，需要用到前向传播时的原始 x 来算梯度。你把 x 原地改了，autograd 记录的是旧版本，再回头算梯度时发现张量已经变了，就对不上了，于是报错
    def forward(self,x):   #All LayerNorm are inside Blocks
        x = x+ self.attn(self.ln1(x))   #Before doing attn DO Normalization First!    
        x = x+ self.ff(self.ln2(x))     #Before doing ff DO normlization Again! 
        return x   #无论如何一定记得要return！！！不然会直接报错！！！     
    
class MyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        #可学习的聚类token “Class Token”，不纳入输入token的维度
        self.cls_token = nn.Parameter(torch.randn(1,1,D_MODEL)*0.02)
        #文本嵌入：每个颜色一个向量
        self.text_emb = nn.Embedding(NUM_COLORS, D_MODEL)
        #类型嵌入：0 CLS, 1 图像，2 文本
        self.type_emb = nn.Embedding(3, D_MODEL)
        self.img_encoder = MyCNN()
       
        #输入token拼接
        max_len = 1 + NUM_IMG_TOKENS + 1  # CLS + 图像token + 文本
        self.pos_emb = PositionalEmbedding(max_len)
        
        self.block = nn.Sequential(*[Block()for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL,2) #二分类头
        
    def forward(self,img,text):
        B = img.shape[0]
        device = img.device
        
        #1. CLS token
        cls = self.cls_token.expand(B,-1,-1)
        cls = cls + self.type_emb(torch.zeros(B,1 , dtype=torch.long,device=device))
        
        #图像token
        img_tok = self.img_encoder(img)
        img_tok = img_tok + self.type_emb(torch.ones(B,NUM_IMG_TOKENS, dtype=torch.long, device=device))
        
        #文本token
        txt_tok = self.text_emb(text).unsqueeze(1)
        txt_tok = txt_tok + self.type_emb(2*torch.ones(B,1,dtype=torch.long,device=device))
        
        
        #拼接
        x = torch.cat([cls,img_tok,txt_tok],dim=1)
        
        #嵌入位置
        x = self.pos_emb(x)
        
        #塞入Transformer
        x = self.block(x)
        x = self.ln(x)
        
        #取CLS位置输出分类
        cls_out = x[:,0,:]
        return self.head(cls_out)
    
 #生成图像数据   
def make_image(color_idx, size=IMG_SIZE):
    rgb = torch.tensor(COLOR_RGB[color_idx]).view(3,1,1)
    img = rgb.repeat(1,size,size)
    img = img+0.1*torch.randn(3,size,size) #加噪声
    return img.clamp(0,1)


def get_batch(batch_size=BATCH):
    imgs, texts ,labels = [], [], []
    for _ in range(batch_size):
        c1 = random.randint(0, NUM_COLORS-1) #图像颜色
        c2 = random.randint(0, NUM_COLORS-1) #文本颜色
        label = 1 if c1 == c2 else 0
        imgs.append(make_image(c1))
        texts.append(c2)
        labels.append(label)
    imgs = torch.stack(imgs)
    texts = torch.tensor(texts, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs,texts,labels
        
# === 训练 ===
model = MyVLM()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for step in range(STEPS):
    imgs, texts, labels = get_batch()
    logits = model(imgs, texts)
    class_weights = torch.tensor([1.0,3.0])
    loss = F.cross_entropy(logits, labels,weight=class_weights)

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
        
    
    
    
        
        
        
  
    
         
    
           