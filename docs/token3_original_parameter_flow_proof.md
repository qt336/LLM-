# 去掉 WV 后的原始参数流：可训练 embedding 的二 gap kernel 证明

> **范围说明**
>
> 本文把 $v_o,v_s$ 当作独立于 embedding 的固定 value。它不适用于
> $o_t=\sum_i a_{ti}e_{X_i}$、即 attention 直接乘可训练 embedding 的模型；
> 后一种模型还有 direct value-to-embedding gradient，必须另加残差项分析。

本文研究一个比 QKV+embedding 模型更干净的版本：

$$
\Phi=(E,W_Q,W_K)
$$

全部参与训练，而 value 向量

$$
v_o,\ v_s
$$

是固定、独立于 embedding 的模型常量。也就是说，本文删掉的是整条可训练
value 参数路径，而不是简单地令

$$
v_a=e_a.
$$

如果令 $v_a=e_a$，value loss 会直接进入 embedding gradient，原来
value-to-score 的干扰仍然存在；那并没有真正解决问题。

在固定 value 的解释下，可以完全避开六向量预条件器的全谱分析。loss 对
$E,W_Q,W_K$ 的 score 依赖只经过两个标量

$$
\boldsymbol\Delta
:=
\begin{pmatrix}
\Delta_o\\
\Delta_s
\end{pmatrix},
$$

因此原始参数梯度流精确诱导出一个 $2\times2$ Gram-kernel ODE：

$$
\boxed{
\dot{\boldsymbol\Delta}
=
\rho_v K(\Phi)
\begin{pmatrix}
\Lambda_o\\
\Lambda_s
\end{pmatrix}.
}
$$

这里 $K$ 是两个 gap 在原始参数空间中的 gradient Gram 矩阵。它自动正半定，
但非对角元可以为负。

由这个 ODE 可以把“无条件能证明什么”和“要两个 gap 同时增长还缺什么”
分得非常清楚：

| 设置 | 附加条件 | 可以严格推出的结论 |
|---|---|---|
| 完整 NTP score loss | 无额外几何条件 | loss 不增；两个 gap 不可能同时下降；至少一个 gap 的瞬时速度为正 |
| 完整 NTP score loss | $K_{os}\ge0$ | $\Delta_o,\Delta_s$ 同时不减，非退化时同时严格增加 |
| 只取严格 post-3 项 $J<t$ | 无 kernel 符号条件 | $\Delta_o$ 严格增加 |
| query 在结构上共享，只有一个 gap | 无 kernel 符号条件 | 完整 score loss 下唯一 gap 严格增加 |

这条路线不要求

$$
W_K(0)=0,
$$

不冻结 embedding，也不要求原始六特征 Jacobian 接近各向同性。

---

## 1. 数据、固定 value 模型与 ordinary 对称性

序列长度为

$$
L=20.
$$

special token 记为

$$
s:=3,
$$

ordinary token 集合为

$$
O:=\{201,\ldots,501\},
\qquad
n:=|O|=301.
$$

每条序列恰好包含一个 $s$，其位置

$$
J\sim\operatorname{Unif}\{1,\ldots,L\};
$$

其余位置独立均匀采自 $O$。理论 vocabulary 为 $O\cup\{s\}$。

所有 ordinary token 在对称子空间上共享 embedding

$$
e_o\in\mathbb R^m,
$$

special embedding 为

$$
e_s\in\mathbb R^m.
$$

定义

$$
\delta e:=e_s-e_o.
$$

query 和 key 为

$$
q_o=W_Qe_o,
\qquad
q_s=W_Qe_s,
$$

$$
k_o=W_Ke_o,
\qquad
k_s=W_Ke_s,
$$

并记

$$
\kappa:=k_s-k_o=W_K\delta e.
$$

value 不由 embedding 生成，而是两个固定向量

$$
v_o,\ v_s\in\mathbb R^p,
\qquad
\delta v:=v_s-v_o.
$$

attention output 和固定 readout 为

$$
o_t=\sum_{i\le t}a_{ti}v_{X_i},
\qquad
z_t=Uo_t,
\qquad
p_t=\operatorname{softmax}(z_t).
$$

population NTP loss 是

$$
\mathcal L(\Phi)
=
\sum_{t=1}^{L-1}\mathbb E[-\log p_t(X_{t+1})].
$$

训练参数只有

$$
\Phi=(E,W_Q,W_K),
$$

并使用原始欧氏 gradient flow。

### 1.1 ordinary 对称子空间

假设：

1. 所有 ordinary embedding 在初值相等；
2. $U$ 的所有 ordinary 输出行相同；
3. 所有 ordinary token 使用同一个固定 value $v_o$。

数据分布和模型都对 ordinary token 置换不变。因此 ordinary-symmetric
子空间对 population gradient flow 不变，所有 ordinary embedding 在训练中
继续相等。

需要注意：原始 embedding table 中有 $n$ 个不同的 ordinary 行。若
$\mathcal L_{\rm red}(e_o,e_s,W_Q,W_K)$ 是 loss 在对称子空间上的限制，
则单个 ordinary embedding 行的速度是

$$
\dot e_o
=
-\frac1n\nabla_{e_o}\mathcal L_{\rm red},
$$

而 special embedding 的速度为

$$
\dot e_s
=
-\nabla_{e_s}\mathcal L_{\rm red}.
$$

记

$$
\beta:=\frac1n.
$$

在 reduced coordinates

$$
\phi=(e_o,e_s,W_Q,W_K)
$$

上，原始参数度量对应的 inverse metric 为

$$
P
:=
\operatorname{diag}
\left(
\beta I_m,\ I_m,\ I_{dm},\ I_{dm}
\right),
$$

所以

$$
\dot\phi
=
-P\nabla_\phi\mathcal L_{\rm red}.
$$

下文使用加权内积与范数

$$
\langle x,y\rangle_P:=x^\top Py,
\qquad
\|x\|_P^2:=x^\top Px.
$$

---

## 2. 两个 attention gap 和数据给出的负 score gradient

定义两个 special-key score gap：

$$
\Delta_o
:=
\frac{\langle q_o,\kappa\rangle}{\sqrt d},
$$

$$
\Delta_s
:=
\frac{\langle q_s,\kappa\rangle}{\sqrt d}.
$$

$\Delta_o$ 用于 special 已经出现在更早位置、当前 query 为 ordinary 的事件

$$
J<t;
$$

$\Delta_s$ 用于当前 query 本身就是 special 的事件

$$
J=t.
$$

当 prefix 长度为 $t$ 且存在一个 special key 时，special attention mass 为

$$
\alpha_{t,x}
=
\frac{\exp(\Delta_x)}
{t-1+\exp(\Delta_x)},
\qquad
x\in\{o,s\}.
$$

只要 $t>1$，

$$
\frac{\partial\alpha_{t,x}}{\partial\Delta_x}
=
\alpha_{t,x}(1-\alpha_{t,x})>0.
$$

定义

$$
A_t
:=
\mathbf 1\{J<t\}\alpha_{t,o}
+\mathbf 1\{J=t\}\alpha_{t,s}.
$$

因为 values 固定，

$$
o_t=v_o+A_t\delta v.
$$

若 $J>t$，则 $A_t=0$，此时 $o_t=v_o$，Q/K score 不影响输出。

### 2.1 readout detector

令

$$
\bar{\mathbf e}_O
:=
\frac1n\sum_{a\in O}\mathbf e_a,
$$

$$
d_*
:=
U^\top(\bar{\mathbf e}_O-\mathbf e_s),
$$

其中 $\mathbf e_a$ 是 vocabulary one-hot。定义固定 value 的 detector
orientation

$$
\rho_v
:=
\langle d_*,\delta v\rangle.
$$

本文只需要唯一一个语义方向条件：

$$
\boxed{\rho_v>0.}
\tag{V}
$$

这个条件与 attention 更新无关；它只表示固定 special value 相对 ordinary
value 更偏向 readout 的 ordinary-vs-special 方向。也可以直接在模型构造中令

$$
\delta v=d_*,
$$

此时

$$
\rho_v=\|d_*\|^2>0
$$

自动成立。

### Lemma 2.1（数据结构给出两个负 score gradient）

定义

$$
G_o:=\frac{\partial\mathcal L}{\partial\Delta_o},
\qquad
G_s:=\frac{\partial\mathcal L}{\partial\Delta_s}.
$$

在 ordinary-symmetric readout 和条件 (V) 下，

$$
\boxed{
G_o=-\rho_v\Lambda_o<0,
\qquad
G_s=-\rho_v\Lambda_s<0,
}
$$

其中

$$
\Lambda_o
:=
\frac1L\sum_{t=1}^{L-1}
(t-1)\pi_{t,o}
\alpha_{t,o}(1-\alpha_{t,o})>0,
$$

$$
\Lambda_s
:=
\frac1L\sum_{t=1}^{L-1}
\pi_{t,s}
\alpha_{t,s}(1-\alpha_{t,s})>0,
$$

而 $\pi_{t,x}$ 是相关输出点预测 special token 的概率。

#### 证明

进入 $G_o$ 的事件满足 $J<t$，进入 $G_s$ 的事件满足 $J=t$。由于每条
序列只有一个 special token，这两类事件都推出

$$
X_{t+1}\in O.
$$

在 ordinary-symmetric readout 下，

$$
p_{t,x}
=
\pi_{t,x}\mathbf e_s
+(1-\pi_{t,x})\bar{\mathbf e}_O.
$$

因此对 ordinary label，

$$
\zeta_{t,x}
:=
\nabla_{o_t}\ell_t
=
U^\top(p_{t,x}-\bar{\mathbf e}_O)
=
-\pi_{t,x}d_*.
$$

又因为

$$
\frac{\partial o_t}{\partial\Delta_x}
=
\alpha_{t,x}(1-\alpha_{t,x})\delta v,
$$

所以每个相关 loss 项对 $\Delta_x$ 的导数是

$$
\left\langle
-\pi_{t,x}d_*,
\alpha_{t,x}(1-\alpha_{t,x})\delta v
\right\rangle
=
-\rho_v\pi_{t,x}
\alpha_{t,x}(1-\alpha_{t,x}).
$$

最后按 $J$ 均匀分布计数：$J<t$ 有 $t-1$ 个可能位置，$J=t$ 有一个
可能位置，得到上述 $\Lambda_o,\Lambda_s$。softmax 概率严格为正，
且至少存在 $t>1$，故两者严格为正。证毕。

---

## 3. 两个 gap 在原始参数空间中的精确梯度

本节直接对

$$
\phi=(e_o,e_s,W_Q,W_K)
$$

求导，不经过六特征预条件器。

为简化记号，定义 embedding-space 向量

$$
b:=W_Q^\top\kappa,
$$

$$
a_o:=W_K^\top q_o,
\qquad
a_s:=W_K^\top q_s.
$$

### Lemma 3.1（原始参数 gap gradients）

$\Delta_o$ 的四个 reduced-coordinate gradient 为

$$
\nabla_{e_o}\Delta_o
=
\frac1{\sqrt d}(b-a_o),
$$

$$
\nabla_{e_s}\Delta_o
=
\frac1{\sqrt d}a_o,
$$

$$
\nabla_{W_Q}\Delta_o
=
\frac1{\sqrt d}\kappa e_o^\top,
$$

$$
\nabla_{W_K}\Delta_o
=
\frac1{\sqrt d}q_o\delta e^\top.
$$

$\Delta_s$ 的四个 gradient 为

$$
\nabla_{e_o}\Delta_s
=
-\frac1{\sqrt d}a_s,
$$

$$
\nabla_{e_s}\Delta_s
=
\frac1{\sqrt d}(b+a_s),
$$

$$
\nabla_{W_Q}\Delta_s
=
\frac1{\sqrt d}\kappa e_s^\top,
$$

$$
\nabla_{W_K}\Delta_s
=
\frac1{\sqrt d}q_s\delta e^\top.
$$

#### 证明

由

$$
\Delta_x
=
\frac1{\sqrt d}
e_x^\top W_Q^\top W_K(e_s-e_o)
$$

分别对 $e_o,e_s,W_Q,W_K$ 做一阶变分即可。例如

$$
\delta\Delta_o
=
\frac1{\sqrt d}
\left[
\langle W_Q\delta e_o,\kappa\rangle
+\langle q_o,W_K(\delta e_s-\delta e_o)\rangle
\right],
$$

因此

$$
\nabla_{e_o}\Delta_o
=
\frac1{\sqrt d}
(W_Q^\top\kappa-W_K^\top q_o),
$$

$$
\nabla_{e_s}\Delta_o
=
\frac1{\sqrt d}W_K^\top q_o.
$$

其他块同理。证毕。

---

## 4. 精确的二 gap Gram-kernel ODE

定义 $2\times2$ 矩阵

$$
K(\phi)
:=
\begin{pmatrix}
K_{oo}&K_{os}\\
K_{os}&K_{ss}
\end{pmatrix},
$$

其中

$$
K_{xy}
:=
(\nabla_\phi\Delta_x)^\top
P
(\nabla_\phi\Delta_y).
$$

等价地，如果把原始 embedding table 的所有 ordinary 行都写出来，
$K_{xy}$ 就是两个 gap 的完整原始参数 gradient 内积。

因此

$$
K\succeq0.
$$

### Lemma 4.1（kernel 的显式公式）

对角元为

$$
\boxed{
dK_{oo}
=
\|\kappa\|^2\|e_o\|^2
+\|q_o\|^2\|\delta e\|^2
+\beta\|b-a_o\|^2
+\|a_o\|^2,
}
$$

$$
\boxed{
dK_{ss}
=
\|\kappa\|^2\|e_s\|^2
+\|q_s\|^2\|\delta e\|^2
+\beta\|a_s\|^2
+\|b+a_s\|^2.
}
$$

非对角元为

$$
\boxed{
\begin{aligned}
dK_{os}
={}&
\|\kappa\|^2\langle e_o,e_s\rangle
+\|\delta e\|^2\langle q_o,q_s\rangle\\
&+\langle a_o,b\rangle
-\beta\langle b,a_s\rangle
+(1+\beta)\langle a_o,a_s\rangle.
\end{aligned}
}
$$

#### 证明

将 Lemma 3.1 中的四个 gradient block 代入

$$
K_{xy}
=
\beta
\langle\nabla_{e_o}\Delta_x,\nabla_{e_o}\Delta_y\rangle
+\langle\nabla_{e_s}\Delta_x,\nabla_{e_s}\Delta_y\rangle
$$

$$
\qquad
+\langle\nabla_{W_Q}\Delta_x,\nabla_{W_Q}\Delta_y\rangle_F
+\langle\nabla_{W_K}\Delta_x,\nabla_{W_K}\Delta_y\rangle_F.
$$

矩阵外积满足

$$
\langle ux^\top,vy^\top\rangle_F
=
\langle u,v\rangle\langle x,y\rangle.
$$

逐项展开即得。证毕。

### Theorem 4.2（原始参数流的 exact 二 gap ODE）

在固定 value 模型中，

$$
\boxed{
\dot{\boldsymbol\Delta}
=
\rho_v K(\phi)
\begin{pmatrix}
\Lambda_o\\
\Lambda_s
\end{pmatrix}.
}
$$

即

$$
\boxed{
\dot\Delta_o
=
\rho_v(K_{oo}\Lambda_o+K_{os}\Lambda_s),
}
$$

$$
\boxed{
\dot\Delta_s
=
\rho_v(K_{os}\Lambda_o+K_{ss}\Lambda_s).
}
$$

#### 证明

固定 value 后，所有 $J>t$ 的项与 Q/K score 无关。其余 loss 对
$E,W_Q,W_K$ 的依赖只通过 $(\Delta_o,\Delta_s)$，所以

$$
\nabla_\phi\mathcal L_{\rm red}
=
G_o\nabla_\phi\Delta_o
+G_s\nabla_\phi\Delta_s.
$$

原始约化梯度流为

$$
\dot\phi=-P\nabla_\phi\mathcal L_{\rm red}.
$$

因此

$$
\dot\Delta_x
=
(\nabla_\phi\Delta_x)^\top\dot\phi
=
-\sum_{y\in\{o,s\}}
K_{xy}G_y.
$$

代入

$$
G_y=-\rho_v\Lambda_y
$$

即得。证毕。

这个 ODE 对任意 $W_K$ 成立，embedding 也完整参与训练。

---

## 5. 不加 kernel 符号假设时，数学上最多能推出什么

正半定性不能保证 $K$ 把正向量逐坐标映成正向量，但它能给出一个严格的
aggregate 结论。

### Theorem 5.1（无额外几何假设的 aggregate 增长）

在条件 (V) 下，

$$
\boxed{
\Lambda_o\dot\Delta_o
+\Lambda_s\dot\Delta_s
=
\rho_v
\begin{pmatrix}
\Lambda_o&\Lambda_s
\end{pmatrix}
K
\begin{pmatrix}
\Lambda_o\\
\Lambda_s
\end{pmatrix}
\ge0.
}
$$

更精确地，

$$
\boxed{
\Lambda_o\dot\Delta_o
+\Lambda_s\dot\Delta_s
=
\rho_v
\left\|
\Lambda_o\nabla_\phi\Delta_o
+\Lambda_s\nabla_\phi\Delta_s
\right\|_P^2.
}
$$

因此只要当前 score gradient 非零，就有严格不等式

$$
\Lambda_o\dot\Delta_o
+\Lambda_s\dot\Delta_s>0.
$$

特别地，$\dot\Delta_o,\dot\Delta_s$ 不可能同时为负；非退化时至少有一个
gap 严格增长。

#### 证明

由 Theorem 4.2，

$$
\boldsymbol\Lambda^\top\dot{\boldsymbol\Delta}
=
\rho_v\boldsymbol\Lambda^\top K\boldsymbol\Lambda.
$$

$K$ 是两个原始 gap gradient 的 Gram 矩阵，所以

$$
\boldsymbol\Lambda^\top K\boldsymbol\Lambda
=
\left\|
\Lambda_o\nabla_\phi\Delta_o
+\Lambda_s\nabla_\phi\Delta_s
\right\|_P^2
\ge0.
$$

由于 $\Lambda_o,\Lambda_s>0$，若两个 gap 速度都为负，左侧必为负，
与上式矛盾。证毕。

### Corollary 5.2（loss 的精确耗散恒等式）

沿原始参数流，

$$
\boxed{
\dot{\mathcal L}
=
-\rho_v^2
\boldsymbol\Lambda^\top K\boldsymbol\Lambda
=
-\rho_v
\left(
\Lambda_o\dot\Delta_o
+\Lambda_s\dot\Delta_s
\right)
\le0.
}
$$

因此对任意存在区间 $[0,T]$，

$$
\mathcal L(T)
+\rho_v^2
\int_0^T
\boldsymbol\Lambda(u)^\top
K(u)
\boldsymbol\Lambda(u)\,du
=
\mathcal L(0).
$$

这里没有冻结 embedding，也没有使用 $W_K$ 初始化条件。

---

## 6. 两个 gap 同时增长：一个简单的“任务不冲突”条件

### Theorem 6.1（kernel cooperativity）

若在某个状态

$$
\boxed{K_{os}\ge0,}
\tag{K+}
$$

则

$$
\dot\Delta_o\ge0,
\qquad
\dot\Delta_s\ge0.
$$

若再有

$$
K_{oo}>0,
\qquad
K_{ss}>0,
$$

则

$$
\dot\Delta_o>0,
\qquad
\dot\Delta_s>0.
$$

如果 (K+) 在一个时间区间上保持，则两个 gap 以及所有存在竞争 key 的
对应 special attention mass 在该区间上同时不减；非退化时严格增加。

#### 证明

由 Theorem 4.2，

$$
\dot\Delta_o
=
\rho_v(K_{oo}\Lambda_o+K_{os}\Lambda_s).
$$

其中

$$
\rho_v>0,\quad
\Lambda_o,\Lambda_s>0,\quad
K_{oo}\ge0.
$$

若 $K_{os}\ge0$，右侧非负。$\Delta_s$ 同理。若两个对角元严格为正，
则各自的 diagonal term 已经严格为正。attention mass 对 gap 严格单调，
故结论成立。证毕。

条件 (K+) 的含义不是“假设 attention 增长”，而是两个任务

$$
\text{提高 }\Delta_o
\qquad\text{与}\qquad
\text{提高 }\Delta_s
$$

在原始参数空间中的梯度夹角不钝：

$$
K_{os}
=
\langle\nabla_\Phi\Delta_o,\nabla_\Phi\Delta_s\rangle
\ge0.
$$

它只是一项 $2\times2$ kernel 几何，而不是 $6d\times6d$ Jacobian 的
全谱假设。

### 6.1 $K_{os}<0$ 时的精确条件

即使 $K_{os}<0$，两个 gap 仍可能同时增长。Theorem 4.2 给出的充要条件是

$$
K_{oo}\Lambda_o+K_{os}\Lambda_s>0,
$$

$$
K_{os}\Lambda_o+K_{ss}\Lambda_s>0.
$$

等价地，

$$
|K_{os}|
<
\min\left\{
\frac{K_{oo}\Lambda_o}{\Lambda_s},
\frac{K_{ss}\Lambda_s}{\Lambda_o}
\right\}.
$$

这个版本比 (K+) 弱，但它已经使用了当前 loss 权重；如果希望假设尽量
上游、与结论距离更远，优先使用 (K+)。

---

## 7. 为什么不能无条件证明两个 gap 都增长

Theorem 5.1 已经是仅由 $K\succeq0$ 能得到的最强逐点偏序结论之一：
正权 aggregate 增长，但单个坐标可能下降。

一个抽象的 $2\times2$ 例子是

$$
K=
\begin{pmatrix}
1&-a\\
-a&1
\end{pmatrix},
\qquad
0<a<1.
$$

它正定。若

$$
\boldsymbol\Lambda
=
\begin{pmatrix}
\varepsilon\\
1
\end{pmatrix},
\qquad
0<\varepsilon<a,
$$

则

$$
(K\boldsymbol\Lambda)_o
=
\varepsilon-a<0,
$$

而

$$
(K\boldsymbol\Lambda)_s
=
1-a\varepsilon>0.
$$

所以一个 gap 可以下降，尽管

$$
\boldsymbol\Lambda^\top K\boldsymbol\Lambda>0
$$

且总 loss 正在下降。

这不只是抽象线性代数现象。在当前原始参数化中，取标量模型

$$
d=m=1,
$$

并令

$$
e_o=1,
\qquad
e_s=-1,
\qquad
W_Q=W_K=1.
$$

则

$$
\delta e=-2,
\qquad
q_o=1,
\qquad
q_s=-1,
\qquad
\kappa=-2.
$$

由 Lemma 4.1，

$$
dK_{os}=-11-3\beta<0.
$$

同时

$$
dK_{oo}=9+9\beta>0,
$$

$$
dK_{ss}=17+\beta>0.
$$

当 $\Lambda_s/\Lambda_o$ 足够大时，

$$
K_{oo}\Lambda_o+K_{os}\Lambda_s<0,
$$

于是 $\Delta_o$ 的速度为负。

因此，在允许任意 $E,W_Q,W_K$ 且要求完整 loss 同时控制两个 gap 的模型中，
不加任何“两个 gap 参数梯度不冲突”的条件，就不可能证明二者总是同时增长。

---

## 8. 两条完全不需要 $K_{os}$ 条件的标量路线

如果研究目标允许只保留一个真正相关的 gap，原始参数流会变成标量 kernel
flow，此时正半定性已经足够，不需要任何 cross-gap 假设。

### 8.1 只分析严格 post-3 ordinary-query objective

定义

$$
\mathcal L_{<}
:=
\sum_{t=1}^{L-1}
\mathbb E[
\mathbf1\{J<t\}\ell_t
].
$$

这个 objective 只包含 special 已经在更早位置出现的 loss 项，因此其 score
依赖只经过 $\Delta_o$：

$$
\nabla_\phi\mathcal L_{<}
=
G_o\nabla_\phi\Delta_o
=
-\rho_v\Lambda_o\nabla_\phi\Delta_o.
$$

### Theorem 8.1（post-3 单 gap 全局单调）

沿 $\mathcal L_{<}$ 的原始参数梯度流，

$$
\boxed{
\dot\Delta_o
=
\rho_v\Lambda_o K_{oo}
=
\rho_v\Lambda_o
\|\nabla_\phi\Delta_o\|_P^2
\ge0.
}
$$

只要 gap gradient 非零，

$$
\dot\Delta_o>0.
$$

因此所有严格位于 token 3 之后的 ordinary query 对 token 3 的 attention
在整个非退化训练区间严格增加。

这里 $W_K$ 任意初始化，embedding 完整训练。

#### 证明

这是 Theorem 4.2 的一维版本：

$$
\dot\Delta_o
=
-\langle\nabla_\phi\Delta_o,
P\nabla_\phi\mathcal L_{<}\rangle
=
-G_o\|\nabla_\phi\Delta_o\|_P^2.
$$

再代入 $G_o=-\rho_v\Lambda_o<0$。证毕。

这个 theorem 证明的是 conditional post-3 objective，不等于完整 NTP objective；
完整 objective 还包含 $J=t$ 的 special-query 梯度，后者通过 $K_{os}$ 影响
$\Delta_o$。

### 8.2 结构上共享 query

如果模型在结构上让 ordinary 和 special 使用同一个 query，使

$$
\Delta_o=\Delta_s=:\Delta,
$$

那么完整 score loss 也只依赖一个 gap，并且

$$
\frac{\partial\mathcal L}{\partial\Delta}
=
G_o+G_s
=
-\rho_v(\Lambda_o+\Lambda_s)<0.
$$

因此对任意包含可训练 embedding 的原始参数化，

$$
\boxed{
\dot\Delta
=
\rho_v(\Lambda_o+\Lambda_s)
\|\nabla_\phi\Delta\|_P^2
>0
}
$$

只要参数化非退化。

这说明两个 gap 的困难并不是 $W_K$ 是否为零，也不是 embedding 是否训练；
真正的困难是完整模型同时存在两个不同的 score coordinate，而它们的原始
参数梯度可能互相冲突。

---

## 9. 区间单调与趋于 full attention 的条件

### Theorem 9.1（完整二 gap 模型的区间单调）

设在时间区间 $I$ 上：

$$
\rho_v>0,
\qquad
K_{os}(u)\ge0,
\qquad
K_{oo}(u),K_{ss}(u)>0.
$$

则

$$
\Delta_o(u),\Delta_s(u)
$$

在 $I$ 上严格增加，对应 attention mass 也严格增加。

这是 Theorem 6.1 的逐时应用。只在初值检查 $K_{os}\ge0$ 只能由连续性给出
短时间结论，不能自动推出任意长区间，因为 kernel 会随 $E,W_Q,W_K$ 改变。

### Corollary 9.2（一个趋于 attention 1 的充分条件）

进一步假设训练全局存在，并且对所有足够大的 $u$，

$$
K_{os}(u)\ge0,
$$

$$
K_{oo}(u)\ge k_o>0,
\qquad
K_{ss}(u)\ge k_s>0.
$$

则

$$
\Delta_o(u)\to+\infty,
\qquad
\Delta_s(u)\to+\infty,
$$

从而对每个固定 prefix 长度 $t>1$，

$$
\alpha_{t,o}(u)\to1,
\qquad
\alpha_{t,s}(u)\to1.
$$

#### 证明

固定 values 意味着所有可能的 attention output 都位于紧线段

$$
\{v_o+a\delta v:0\le a\le1\}.
$$

softmax special 概率在这条紧线段上连续且严格为正，所以存在

$$
\underline\pi>0
$$

使所有相关 $\pi_{t,x}\ge\underline\pi$。

反设 $\Delta_o$ 有有限上界。因为它单调，$\alpha_{t,o}(1-\alpha_{t,o})$
在至少一个 $t>1$ 的相关有界 gap 区间上有严格正下界，因此

$$
\Lambda_o\ge\underline\Lambda_o>0.
$$

于是

$$
\dot\Delta_o
\ge
\rho_v k_o\underline\Lambda_o>0,
$$

与 $\Delta_o$ 有上界矛盾。$\Delta_s$ 同理。最后由 attention 的 logistic
公式得到极限 $1$。证毕。

---

## 10. 实验验证：只需要一个 $2\times2$ kernel

这条证明不需要构造

$$
6d\times6d
$$

预条件器，也不需要计算其极端特征值。

### 10.1 检查固定 value orientation

计算

$$
d_*=U^\top(\bar{\mathbf e}_O-\mathbf e_s),
$$

$$
\rho_v=\langle d_*,v_s-v_o\rangle.
$$

检查

$$
\rho_v>0.
$$

如果 values 直接设置为

$$
v_s-v_o=d_*,
$$

则该条件自动成立。

### 10.2 计算 kernel

有两种方式。

第一种是使用 Lemma 4.1 的显式公式，只需当前

$$
e_o,e_s,W_Q,W_K.
$$

第二种是分别反传两个 gap：

$$
g_o^\Phi:=\nabla_\Phi\Delta_o,
\qquad
g_s^\Phi:=\nabla_\Phi\Delta_s.
$$

然后计算

$$
K_{oo}=\|g_o^\Phi\|^2,
$$

$$
K_{ss}=\|g_s^\Phi\|^2,
$$

$$
K_{os}=\langle g_o^\Phi,g_s^\Phi\rangle.
$$

对 untied ordinary embedding table，必须对全部 $n$ 个 ordinary 行计算
原始 gradient 内积；如果使用 reduced formula，则 ordinary embedding block
要乘 $\beta=1/n$。

建议同时报告 normalized conflict cosine

$$
\chi
:=
\frac{K_{os}}{\sqrt{K_{oo}K_{ss}}}
\in[-1,1].
$$

$\chi\ge0$ 对应 Theorem 6.1 的简单协同条件。

### 10.3 验证 exact ODE

从 forward 计算 $\Lambda_o,\Lambda_s$，然后比较 autograd 实测 gap 速度与

$$
\widehat{\dot\Delta}_o
=
\rho_v(K_{oo}\Lambda_o+K_{os}\Lambda_s),
$$

$$
\widehat{\dot\Delta}_s
=
\rho_v(K_{os}\Lambda_o+K_{ss}\Lambda_s).
$$

两边应在 population/minibatch 近似误差内一致。

实验报告应把三件事分开：

1. **无条件结论**：
   $$
   \Lambda_o\dot\Delta_o+\Lambda_s\dot\Delta_s\ge0.
   $$
2. **是否协同**：
   $$
   K_{os}\ge0.
   $$
3. **实际两个速度**：
   $$
   K_{oo}\Lambda_o+K_{os}\Lambda_s,
   \qquad
   K_{os}\Lambda_o+K_{ss}\Lambda_s.
   $$

这样不会把最终速度符号伪装成模型假设。

---

## 11. 假设层次与最终结论

去掉 $W_V$ 后，必须先明确 value 的含义：

- 若 $v_o,v_s$ 固定且独立于 embedding，本文的二 gap kernel 证明严格成立；
- 若令 $v_a=e_a$，value loss 仍直接更新 embedding，不能使用本文的
  score-only factorization；
- 若 $v_a=W_Ve_a$ 且 $W_V$ 可训练，则回到 QKV+embedding 的共享参数问题。

在固定 value、embedding 正常训练、$W_K$ 任意的模型中，最强的无条件
原始参数结论是

$$
\boxed{
\Lambda_o\dot\Delta_o
+\Lambda_s\dot\Delta_s
=
\rho_v\boldsymbol\Lambda^\top K\boldsymbol\Lambda
\ge0.
}
$$

所以两个 gap 不会同时下降，非退化时至少一个增长，且 loss 严格耗散。

如果希望完整 NTP loss 下两个 gap 同时增长，最少还需要控制它们在原始
参数空间中的冲突。最简单的 task-independent 条件是

$$
\boxed{K_{os}\ge0.}
$$

如果只研究真正的“后续 ordinary token 回看 token 3”，即条件事件 $J<t$，
则问题只有一个 gap，此时无需任何 cross-gap 条件：

$$
\boxed{
\dot\Delta_o
=
\rho_v\Lambda_o
\|\nabla_\Phi\Delta_o\|^2
>0.
}
$$

因此，去掉 $W_V$ 后真正剩下的数学障碍不是 key 初始化，也不是 embedding
训练，而是完整 loss 中 ordinary-query gap 与 special-query gap 是否在共享
参数上发生 gradient conflict。这个障碍被一个可直接计算的 $2\times2$
Gram kernel 完整刻画。
