import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # 解决 Windows 上的 OpenMP 冲突
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)



#===Hyperparameters===#
VOCAB_NUM = 10
D_MODEL = 16
N_HEADS = 4
D_FF = 32
MAX_LEN = 8
N_LAYERS = 2
BATCH = 4
LR = 0.001
STEPS = 1000

#====Embedding Layer
# Derive token and position embeddings by encoding====#
class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_NUM,D_MODEL)
        self.pos = nn.Embedding(MAX_LEN,D_MODEL)
    
    def forward(self,x):
        B,T = x.shape
        pos = torch.arange(T,device=x.device)  #device = x.device是为了让tok编码和pos编码处于同一内存，不然跨内存会报错
        return self.tok(x) + self.pos(pos)

#====MultiHeadAttention====

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
        
        mask = torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()  #triu Means Upper Triangular Matrix, diagonal=1 means the first diagonal above the main diagonal
        scores = scores.masked_fill(mask, float('-inf'))  # Masking future tokens
        
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
        
#==Complete Homemade Transformer
class HMT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = EmbeddingLayer()
        self.block= nn.Sequential(*[Block() for _ in range(N_LAYERS)])    #Determines how many blocks in a single transformer
        self.ln_f = nn.LayerNorm(D_MODEL)    #LayerNorm Outside Block
        self.head = nn.Linear(D_MODEL,VOCAB_NUM) #
        
    def forward(self,x):  #Input starts to go forward
        x = self.emb(x)
        x = self.block(x)
        x = self.ln_f(x)
        return self.head(x)
    
 #===Genearate data===
def get_batch():
       starts = torch.randint(0,VOCAB_NUM, (BATCH,))   #生成BATCH个0~VOCAB_NUM的随机数（如[3,4,0,9]）
       x = (starts.unsqueeze(1)+torch.arange(MAX_LEN)) % VOCAB_NUM     # 1,unsqueeze是为了给第一维加一个维度(如[[3],[4],[0],[9]])，方便和后面的arange（MAX_LEN）广播相加变成4条MAX_LEN维的输入样本
       #2，%VOCAB_NUM是为了取余避免超出9超载
       
       y = (starts.unsqueeze(1)+torch.arange(1, MAX_LEN+2)) % VOCAB_NUM  #与x无差异，只是整体往后挪了一格，作为该任务“预测下一个数”的样本
       return x, y
      
     

#==开始训练==
model = HMT()
optimizer = torch.optim.Adam(model.parameters(),LR) #这就是之前调用nn.Module的好处，内存都存进了model.parameters,优化器可以直接调用

for step in range(STEPS):
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1,VOCAB_NUM),y.view(-1))
    
    #老三样
    optimizer.zero_grad() #清零上次梯度
    loss.backward()   #预反向
    optimizer.step()  #执行
    
    if step %200 == 0:
        print(f"第{step}轮，损失为{loss}")
        
#==开测==
model.eval()
x =torch.tensor([[0,1,3,5,7]])
with torch.no_grad():
    y_pred = model(x).argmax(dim=-1)
print("输入", x.tolist())
print("预测", y_pred.tolist())
        
        
        
        
