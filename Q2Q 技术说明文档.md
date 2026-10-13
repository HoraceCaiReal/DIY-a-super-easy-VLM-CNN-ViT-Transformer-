Q2Q 技术说明文档
Qualitative-to-Quantitative: A Neuro-Symbolic Reflex Architecture for Edge-Deployable VLA

版本 1.0 | 2026-10

目录
概述

问题背景与动机

整体架构

Q2QBlock：神经反射弧

Q2QHarness：编排层

符号后端：凸优化与路径规划

训练方法

可解释性

与现有工作的对比

应用场景

路线图

1. 概述
Q2Q（Qualitative-to-Quantitative）是一种面向边缘设备的神经符号控制架构。其核心思想是：

将控制问题分解为“定性决策”和“定量执行”两个层次。神经网络负责选择策略（定性），凸优化/路径规划负责生成动作（定量）。两者通过可学习的卡尔曼门控残差融合，形成闭环反射弧。

Q2Q 由三部分组成：

组件	职责	可微性
Q2QBlock	策略选择 + 参数预测	可微
Q2QHarness	调度、执行、监控、恢复	不可微
符号后端	OSQP / A* 求解	不可微
2. 问题背景与动机
2.1 现有 VLA 的局限
问题	端到端 VLA	纯符号方法
可解释性	❌ 黑盒	✅ 可检查
安全性	❌ 无保证	✅ 可验证
算力需求	❌ 高（GPU）	✅ 低（CPU）
可学习性	✅ 从数据学	❌ 人工设计
泛化性	✅ 强	❌ 弱
2.2 Q2Q 的定位
Q2Q 在端到端和纯符号之间找到一条中间路线：

text
可微前端（神经网络） + 不可微后端（凸优化）
= 可学习 + 可验证 + 可解释 + 低算力
3. 整体架构
text
┌─────────────────────────────────────────────────────────┐
│  LLM (System 2)                                         │
│  职责：长时序规划、子目标生成、监工                      │
│  部署：云端 / 本地大模型                                 │
└──────────────────────┬──────────────────────────────────┘
                       │ 子目标
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Q2QHarness                                             │
│  职责：调度、执行、监控、恢复                            │
│  部署：ROS2 节点，CPU                                   │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ 状态管理  │ │ 数据采集  │ │ 错误恢复  │ │ 日志记录  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │ 子目标解析│ │ 策略路由  │ │ 求解器封装│               │
│  └──────────┘ └──────────┘ └──────────┘               │
└──────────────────────┬──────────────────────────────────┘
                       │ 传感态 + 非传感态
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Q2QBlock                                               │
│  职责：策略选择 + 参数预测                               │
│  部署：边缘设备，几百万参数                              │
│                                                         │
│  传感态 token ──→ 卡尔曼门控残差 ──→ 锚点直通           │
│  非传感态 token ──→ 共享 Transformer                    │
│  迭代反馈：CLS + PARAM 替换回输入                        │
│  输出：case (分类) + params (回归)                       │
└──────────────────────┬──────────────────────────────────┘
                       │ case + params
                       ▼
┌─────────────────────────────────────────────────────────┐
│  符号后端求解器                                          │
│  OSQP（机械臂操作） / A* / Hybrid A*（移动导航）         │
│  部署：CPU，毫秒级                                       │
└──────────────────────┬──────────────────────────────────┘
                       │ 轨迹
                       ▼
┌─────────────────────────────────────────────────────────┐
│  ROS2 + Gazebo 执行层                                    │
└─────────────────────────────────────────────────────────┘
3.1 算力分层
text
高算力: LLM（云端）
中算力: Q2QHarness（CPU）
低算力: Q2QBlock（边缘设备）+ OSQP（CPU）
4. Q2QBlock：神经反射弧
4.1 输入定义
类型	来源	示例	可信度
传感态	直接测量	关节角、末端位姿、力、图像	高（仿真真值）
非传感态	推断/给定	子目标、进度、历史、prompt	低（有误差）
关键区分：传感态是直接测量的，非传感态是推断或给定的。两者的可信度不同，需要区别对待。

4.2 序列结构
text
输入序列：
  [CLS] + [传感态×16] + [非传感态×16] + [PARAM×K]

  位置:  0     1~16        17~32        33~33+K

  CLS:         用于 case 分类
  传感态:       锚点直通
  非传感态:     网络处理
  PARAM:       用于参数回归，K 由参数组数决定
4.3 卡尔曼门控残差
核心公式：

text
x_sensed_new = α · x_sensed_original + (1-α) · x_sensed_after
其中：

x_sensed_original：原始传感态编码（锚点，跨层不变）

x_sensed_after：经过 Transformer 后的传感态

α：可学习门控，对应卡尔曼增益

α 的计算：

python
alpha = sigmoid(gate([x_sensed_original, x_sensed_after]))
α 的语义：

text
α → 1: 信任传感器
α → 0: 信任网络推断
α = 0.5: 两者等权
4.4 位置选择性门控
门控只作用于传感态位置，其他位置用普通残差：

python
# 普通残差（所有位置）
x = x + attn(ln(x))
x = x + ff(ln(x))

# 门控残差（只对传感态位置）
x_sensed_after = x[:, 1:17, :]
alpha = gate([x_sensed_original, x_sensed_after])
x_sensed_new = alpha * x_sensed_original + (1-alpha) * x_sensed_after
x[:, 1:17, :] = x_sensed_new
维度天然对齐：(B, 16, 32) → (B, 16, 32)。

4.5 迭代反馈
text
第 1 轮: [CLS₀, 传感态, 非传感态, PARAM₀] → [CLS₁, ..., PARAM₁]
第 2 轮: [CLS₁, 传感态, 非传感态, PARAM₁] → [CLS₂, ..., PARAM₂]
...
第 N 轮: [CLSₙ₋₁, 传感态, 非传感态, PARAMₙ₋₁] → [CLSₙ, ..., PARAMₙ]
每轮监督：

python
loss = Σ_k [loss_case_k + λ · loss_param_k] / N
4.6 输出
输出	来源	损失
case	CLS 位置 → case_head	CrossEntropy
params	PARAM 位置 → param_heads	MSE（掩码）
4.7 参数分组
python
PARAM_GROUPS = {
    "weights":  ["pos_weight", "vel_weight", "acc_weight"],   # 3 个
    "limits":   ["v_max", "a_max", "jerk_max"],               # 3 个
    "safety":   ["safety_margin", "collision_dist"],          # 2 个
    "horizon":  ["horizon_length"],                           # 1 个
}
每个组一个 PARAM token，一个预测头。

4.8 完整伪代码
python
class Q2QBlock(nn.Module):
    def __init__(self):
        self.cls_token = Parameter(1, 1, D_MODEL)
        self.param_tokens = Parameter(1, K, D_MODEL)
        self.type_emb = Embedding(4, D_MODEL)
        self.sensed_proj = Linear(SENSED_DIM, D_MODEL)
        self.nonsensed_proj = Linear(NONSENSED_DIM, D_MODEL)
        self.blocks = [Block() for _ in range(N_LAYERS)]
        self.case_head = Linear(D_MODEL, NUM_CASES)
        self.param_heads = [Linear(D_MODEL, dim) for dim in PARAM_DIMS]

    def forward(self, sensed, nonsensed, n_iters=3):
        x_sensed = self.sensed_proj(sensed)
        x_sensed_original = x_sensed
        x_nonsensed = self.nonsensed_proj(nonsensed)
        cls = self.cls_token.expand(B, -1, -1)
        param = self.param_tokens.expand(B, -1, -1)

        x = cat([cls, x_sensed, x_nonsensed, param], dim=1)

        for i in range(n_iters):
            for block in self.blocks:
                x = x + block.attn(block.ln1(x))
                x = x + block.ff(block.ln2(x))
                x_sensed_after = x[:, 1:17, :]
                alpha = block.gate(cat([x_sensed_original, x_sensed_after]))
                x[:, 1:17, :] = alpha * x_sensed_original + (1-alpha) * x_sensed_after

            cls_new = x[:, 0:1, :]
            param_new = x[:, -K:, :]
            x = cat([cls_new, x[:, 1:-K, :], param_new], dim=1)

        case_logits = self.case_head(x[:, 0, :])
        params = [head(x[:, -K+i, :]) for i, head in enumerate(self.param_heads)]
        return case_logits, params
5. Q2QHarness：编排层
5.1 组件
text
1. 子目标解析器    LLM 输出 → 结构化子目标
2. 状态管理        WorldState，历史，进度
3. 数据采集        ROS2 订阅，组装传感态/非传感态
4. 模型调用        Q2QBlock 推理
5. 策略路由        case → 求解器
6. 求解器封装      OSQP / A* / Hybrid A*
7. 执行监控        ROS2 发布，状态监控
8. 错误恢复        重试 / 换 case / 重规划
9. 日志记录        全链路可追溯
10. 可视化         α 热力图，case 分布
5.2 状态管理
python
@dataclass
class WorldState:
    # 传感态
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    ee_pose: np.ndarray
    gripper_state: float
    image: np.ndarray
    force_torque: np.ndarray

    # 非传感态
    current_subgoal: Subgoal
    task_progress: float
    history: list
    last_case: int
    step_count: int
5.3 策略库
python
CASE_POOL = {
    # 机械臂操作（OSQP）
    0: {"name": "快速接近", "solver": "osqp",
        "weights": {"pos": 10.0, "vel": 1.0},
        "constraints": {"joint": True, "collision": False},
        "horizon": 10},
    1: {"name": "精细对齐", "solver": "osqp",
        "weights": {"pos": 100.0, "vel": 0.1},
        "constraints": {"joint": True, "collision": True},
        "horizon": 20},
    2: {"name": "避障绕行", "solver": "osqp",
        "weights": {"pos": 1.0, "vel": 1.0},
        "constraints": {"collision": True, "safety_margin": 0.1},
        "horizon": 30},

    # 移动底盘导航（A*）
    10: {"name": "最短路径", "solver": "astar",
         "heuristic": "euclidean", "grid_resolution": 0.05},
    11: {"name": "安全绕行", "solver": "astar",
         "heuristic": "euclidean", "safety_margin": 0.3},
    12: {"name": "狭窄通道", "solver": "hybrid_astar",
         "heuristic": "dubin", "turning_radius": 0.3},
}
5.4 主循环
python
def run(self, task):
    subgoals = self.planner.plan(task, self.state)

    for subgoal in subgoals:
        self.state.current_subgoal = subgoal
        success = self.execute_subgoal(subgoal)

        if not success:
            action = self.recovery.handle_failure(subgoal, self.state)
            if action == "retry":
                success = self.execute_subgoal(subgoal)
            elif action[0] == "change_case":
                self.state.forced_case = action[1]
                success = self.execute_subgoal(subgoal)
            else:
                subgoals = self.planner.replan(task, self.state)
                break

    return self.state.task_progress >= 1.0

def execute_subgoal(self, subgoal):
    for step in range(MAX_STEPS):
        sensed = self.ros.get_sensed()
        nonsensed = self.ros.get_nonsensed()

        case, params = self.model.predict(sensed, nonsensed, subgoal)
        trajectory = self.solver.solve(case, params, sensed, subgoal)

        if trajectory is None:
            return False

        self.executor.execute(trajectory)
        self.state.history.append(StepRecord(...))

        if self.monitor.check_subgoal(subgoal, self.state):
            return True

    return False
6. 符号后端：凸优化与路径规划
6.1 OSQP（机械臂操作）
python
def solve_osqp(case_params, state, subgoal):
    H = case_params["horizon"]
    n = 2 * (H + 1) * DOF

    P = build_P(case_params["weights"], H)
    q = build_q(state, subgoal, case_params["weights"], H)
    A, l, u = build_constraints(case_params["constraints"], state, H)

    prob = osqp.OSQP()
    prob.setup(P=sp.csc_matrix(P), q=q,
               A=sp.csc_matrix(A), l=l, u=u)
    result = prob.solve()
    return result.x.reshape(-1, DOF)
保证：最优性 + 约束满足。

6.2 A*（移动导航）
python
def solve_astar(case_params, state, subgoal):
    grid = build_grid(state.map, case_params["grid_resolution"])
    path = astar(grid, state.position, subgoal.target,
                 heuristic=case_params["heuristic"])
    return path
保证：路径最优性（给定启发式）。

6.3 solver-agnostic
python
class StrategySolver:
    def solve(self, case_id, state, subgoal):
        case_params = self.case_pool[case_id]
        solver = case_params["solver"]

        if solver == "osqp":
            return self.solve_osqp(...)
        elif solver == "astar":
            return self.solve_astar(...)
        elif solver == "hybrid_astar":
            return self.solve_hybrid_astar(...)
Q2QBlock 不关心后面接什么求解器。

7. 训练方法
7.1 损失函数
python
loss_case = F.cross_entropy(case_logits, case_label)

case_correct = (case_logits.argmax(dim=-1) == case_label).float()
loss_param = F.mse_loss(param_pred, param_label, reduction='none')
loss_param = (loss_param * case_correct).mean()

loss = loss_case + λ * loss_param
关键：case 错时，param loss 被掩码。

7.2 迭代监督
python
loss = 0
for i in range(n_iters):
    case_logits, param_pred = model(x)
    loss_case_i = F.cross_entropy(case_logits, case_label)
    loss_param_i = masked_mse(param_pred, param_label, case_label)
    loss += loss_case_i + λ * loss_param_i
loss = loss / n_iters
7.3 数据采集
随机环境 + 兜底触发：

python
def collect_episode(env, model, harness):
    config = sample_environment()
    env.reset(config)

    records = []
    for step in range(MAX_STEPS):
        sensed = env.get_sensed()
        nonsensed = env.get_nonsensed()
        case_logits, params = model(sensed, nonsensed)

        fallback_type = harness.check_fallback(env.state, case_logits)

        if fallback_type:
            records.append({"type": "boundary", ...})
            harness.execute_fallback(fallback_type)
            records.append({"type": "recovery", ...})
        else:
            env.execute(case_logits.argmax(), params)
            records.append({"type": "normal", ...})

        if env.task_done():
            break

    return records
四类样本：

类型	触发条件	用途
normal	正常运行	主任务训练
boundary	兜底触发	鲁棒性训练
recovery	兜底执行	恢复能力
failure	兜底也失败	失败预测
7.4 后训练（RLHF / DPO）
text
SFT 阶段:  case CE + param MSE
RLHF 阶段: case + param 用 reward 更新
DPO 阶段:  case + param 用偏好对更新
8. 可解释性
8.1 α 可视化
图表	横轴	纵轴	颜色
层次热力图	传感维度	网络层	α
时空热力图	时间步	传感维度	α
统计分布	α 值	频次	-
case 相关性	case id	α 均值	-
能看出：

浅层信任传感器，深层信任网络

某些维度 α 低（传感器不可靠）

α 突然变化（传感器故障）

不同 case 的 α 模式不同

8.2 case 混淆矩阵
python
cm = confusion_matrix(case_labels, case_preds)
8.3 参数预测 vs 真值
python
plt.scatter(param_labels[:, i], param_preds[:, i])
8.4 注意力图
python
attn = model.blocks[0].attn.last_attn
sns.heatmap(attn.mean(dim=(0,1)))
9. 与现有工作的对比
维度	标准 Transformer	MoE	Action Expert	Concept Bottleneck	Q2Q
输入分离	❌	❌	❌	❌	✅ 传感/非传感
门控残差	❌	❌	❌	❌	✅ 卡尔曼
传感态锚点	❌	❌	❌	❌	✅
迭代反馈	❌	❌	❌	❌	✅
多任务 token	❌	❌	❌	❌	✅ CLS+PARAM
凸优化输出	❌	❌	❌	❌	✅ OSQP
可解释	❌	❌	❌	✅	✅
安全保证	❌	❌	❌	❌	✅
低算力	❌	❌	❌	❌	✅
Q2Q 的核心区别：

不是 MoE：选的是凸优化问题，不是神经网络

不是 Action Expert：动作由求解器生成，不是网络生成

不是 Concept Bottleneck：case 是策略选择，不是概念预测

是神经符号优化：神经网络选择优化问题形式，凸优化求解

10. 应用场景
10.1 工厂边缘设备
text
场景: 机械臂分拣、AGV 调度、质检、装配
需求: 实时、安全、可解释、低功耗、可扩展
Q2Q: 策略库按产线配置，case 可人工干预，OSQP 保证安全
10.2 机械假肢
text
场景: 上肢/下肢假肢、外骨骼
需求: 超低功耗、超低延迟、安全、个性化、可解释
Q2Q: MCU 部署、反射弧、case 可患者自定义、α 可适应个体差异
10.3 具身智能
text
场景: 移动机械臂、人形机器人
需求: 低算力反射弧 + 云端规划
Q2Q: Q2QBlock 在边缘，LLM 在云端，分层解耦
11. 路线图
短期（毕设期间）
text
□ 实现 Q2QBlock
□ 实现 Q2QHarness
□ 搭 PyBullet / Gazebo 环境
□ 随机环境数据采集
□ 训练 Q2Q
□ α 可视化
□ 扰动测试
□ 写毕设论文
中期（毕设后 6~12 个月）
text
□ 补实验，投 CoRL / ICRA / IROS
□ 开源代码
□ 扩展移动机械臂场景
□ 加 RLHF 后训练
长期（3~5 年）
text
□ 工业落地
□ 人道主义假肢项目
□ 读博 / 工作 / 创业
附录：符号表
符号	含义
B	batch size
T	序列长度
C / D_MODEL	特征维度
H	注意力头数
D	head_dim
K	PARAM token 数
α	卡尔曼增益
λ	损失权重
N	迭代轮数
文档结束

Q2Q 的核心叙事：

不是所有智能都需要云端大模型。边缘反射弧 + 云端规划 = 完整具身智能。定性选策略，定量求解器，两者互为阴阳。