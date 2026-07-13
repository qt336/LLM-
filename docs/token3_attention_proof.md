# token `3` attention 局部增长：方向任意的完整六向量耦合 flow

本文讨论 ordinary/special 两类型六向量参数化

$$
\theta=(q_o,q_s,k_o,k_s,v_o,v_s),
\qquad
q_o,q_s,k_o,k_s,v_o,v_s\in\mathbb R^d.
$$

和旧的同向写法不同，本文不要求

$$
q_o,q_s,k_o,k_s,v_o,v_s
$$

共用同一个方向，也不要求 query、key、value 三类方向相同。主证明只使用完整六向量耦合 gradient flow。结论分两层：

1. 对任意六向量，attention gap 的速度有 exact ODE。
2. 在 ordinary-symmetric readout 下，数据结构“token `3` 后面的 label 不再是 `3`”只推出 score-gradient 标量 \(G_o^0,G_s^0\) 的可计算符号形式；真正的 attention 增长还需要一个最小几何条件。

---

## 1. 数据、模型与记号

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
O:=\{201,202,\dots,501\}.
$$

每条序列中，\(s\) 恰好出现一次，位置记为

$$
J\in\{1,\dots,L\}.
$$

假设 \(J\) 均匀，其余位置独立均匀采自 \(O\)。训练目标是 population next-token prediction loss：

$$
\mathcal L(\theta)
=
\sum_{t=1}^{L-1}\mathbb E[\ell_t(\theta)],
\qquad
\ell_t=-\log p_t(X_{t+1}).
$$

六向量两类型模型为：ordinary token 使用

$$
(q_o,k_o,v_o),
$$

special token 使用

$$
(q_s,k_s,v_s).
$$

定义

$$
\kappa:=k_s-k_o,
\qquad
\delta v:=v_s-v_o.
$$

当 prefix 中已经出现唯一 token `3` 时，special key 相对 ordinary key 的 score gap 为

$$
\Delta_o:=\frac{\langle q_o,\kappa\rangle}{\sqrt d}
\qquad\text{ordinary query},
$$

$$
\Delta_s:=\frac{\langle q_s,\kappa\rangle}{\sqrt d}
\qquad\text{special query}.
$$

若 prefix 长度为 \(t\)，且其中有一个 special key 和 \(t-1\) 个 ordinary key，则 special attention mass 为

$$
\alpha_t^{(o)}
=
\frac{\exp(\Delta_o)}{t-1+\exp(\Delta_o)}
\qquad(J<t),
$$

$$
\alpha_t^{(s)}
=
\frac{\exp(\Delta_s)}{t-1+\exp(\Delta_s)}
\qquad(J=t).
$$

并且

$$
\frac{d\alpha}{d\Delta}=\alpha(1-\alpha).
$$

统一定义位置 \(t\) 对 special value 的 attention mass：

$$
A_t
:=
\mathbf 1\{J<t\}\alpha_t^{(o)}
+
\mathbf 1\{J=t\}\alpha_t^{(s)}.
$$

于是 attention output 为

$$
o_t=(1-A_t)v_o+A_tv_s
=v_o+A_t\delta v.
$$

若 \(J>t\)，prefix 尚未出现 token `3`，则 \(A_t=0\) 且 \(o_t=v_o\)。

读出为

$$
z_t=Uo_t,
\qquad
p_t=\operatorname{softmax}(z_t),
$$

其中 \(U\) 固定。上游梯度为

$$
\zeta_t:=\nabla_{o_t}\ell_t
=U^\top(p_t-e_{X_{t+1}}^{\rm vocab}).
$$

---

## 2. 完整六向量耦合梯度流

定义

$$
g_t:=\langle \zeta_t,\delta v\rangle.
$$

再定义两个 score-gradient 标量：

$$
G_o
:=
\sum_{t=1}^{L-1}
\mathbb E\left[
\mathbf 1\{J<t\}
 g_t\alpha_t^{(o)}(1-\alpha_t^{(o)})
\right],
$$

$$
G_s
:=
\sum_{t=1}^{L-1}
\mathbb E\left[
\mathbf 1\{J=t\}
 g_t\alpha_t^{(s)}(1-\alpha_t^{(s)})
\right].
$$

定义 value-path 梯度

$$
V_o:=\sum_{t=1}^{L-1}\mathbb E[(1-A_t)\zeta_t],
\qquad
V_s:=\sum_{t=1}^{L-1}\mathbb E[A_t\zeta_t].
$$

直接对六个独立特征向量求梯度，得到

$$
\nabla_{v_o}\mathcal L=V_o,
\qquad
\nabla_{v_s}\mathcal L=V_s,
$$

$$
\nabla_{q_o}\mathcal L=\frac1{\sqrt d}G_o\kappa,
\qquad
\nabla_{q_s}\mathcal L=\frac1{\sqrt d}G_s\kappa,
$$

$$
\nabla_{k_s}\mathcal L
=
\frac1{\sqrt d}(G_oq_o+G_sq_s),
\qquad
\nabla_{k_o}\mathcal L
=-\frac1{\sqrt d}(G_oq_o+G_sq_s).
$$

因此完整欧氏六向量 gradient flow 为

$$
\dot v_o=-V_o,
\qquad
\dot v_s=-V_s,
$$

$$
\dot q_o=-\frac1{\sqrt d}G_o\kappa,
\qquad
\dot q_s=-\frac1{\sqrt d}G_s\kappa,
$$

$$
\dot k_s=-\frac1{\sqrt d}(G_oq_o+G_sq_s),
\qquad
\dot k_o=\frac1{\sqrt d}(G_oq_o+G_sq_s).
$$

特别地，

$$
\dot\kappa
=
\dot k_s-\dot k_o
=-\frac2{\sqrt d}(G_oq_o+G_sq_s).
$$

这是本文唯一使用的动力学。这里没有冻结任何变量；\(A_t\)、\(\alpha_t^{(o)}\)、\(\alpha_t^{(s)}\)、\(G_o\)、\(G_s\)、\(V_o\)、\(V_s\) 都是当前六向量状态诱导出的函数。

---

## 3. exact attention-gap ODE

由于

$$
\Delta_x=\frac{\langle q_x,\kappa\rangle}{\sqrt d},
\qquad x\in\{o,s\},
$$

完整耦合 flow 给出闭式 ODE。

### Lemma 3.1（gap ODE）

沿完整六向量 gradient flow，逐点有

$$
\dot\Delta_o
=
-
\frac1d
\left[
G_o\|\kappa\|^2
+2G_o\|q_o\|^2
+2G_s\langle q_o,q_s\rangle
\right],
$$

$$
\dot\Delta_s
=
-
\frac1d
\left[
G_s\|\kappa\|^2
+2G_o\langle q_s,q_o\rangle
+2G_s\|q_s\|^2
\right].
$$

#### 证明

对

$$
\Delta_x=\frac{\langle q_x,\kappa\rangle}{\sqrt d}
$$

求导：

$$
\dot\Delta_x
=
\frac1{\sqrt d}
\left(
\langle \dot q_x,\kappa\rangle
+
\langle q_x,\dot\kappa\rangle
\right).
$$

对 \(x=o\)，代入

$$
\dot q_o=-\frac1{\sqrt d}G_o\kappa,
\qquad
\dot\kappa=-\frac2{\sqrt d}(G_oq_o+G_sq_s),
$$

得到

$$
\dot\Delta_o
=
-
\frac1d
\left[
G_o\|\kappa\|^2
+2G_o\|q_o\|^2
+2G_s\langle q_o,q_s\rangle
\right].
$$

\(x=s\) 同理。证毕。

### Corollary 3.2（无 readout 假设的最少局部增长条件）

定义

$$
H_o
:=
G_o\|\kappa\|^2
+2G_o\|q_o\|^2
+2G_s\langle q_o,q_s\rangle,
$$

$$
H_s
:=
G_s\|\kappa\|^2
+2G_o\langle q_s,q_o\rangle
+2G_s\|q_s\|^2.
$$

在任意初始六向量 \(\theta^0\) 上，若

$$
H_o^0<0,
\qquad
H_s^0<0,
$$

则

$$
\dot\Delta_o(0)>0,
\qquad
\dot\Delta_s(0)>0.
$$

因此存在 \(\epsilon>0\)，使得对所有 \(0<u\le\epsilon\)，

$$
\Delta_o(u)>\Delta_o(0),
\qquad
\Delta_s(u)>\Delta_s(0).
$$

从而所有有竞争 key 的对应 token `3` attention mass 相比初值严格增大。

更精确地说，因为

$$
\dot\Delta_x(0)=-\frac1dH_x^0,
$$

所以 \(H_x^0<0\) 与 \(\dot\Delta_x(0)>0\) 等价。这是完全不依赖 readout symmetry、也不依赖方向同向性的 exact 一阶条件。

#### 证明

由 Lemma 3.1，\(\dot\Delta_x(0)=-H_x^0/d\)。梯度流右端是有限维 smooth 函数与有限 expectation 的组合，解局部 \(C^1\)，因此正一阶导数推出右侧局部严格增长。attention mass 对 \(\Delta_x\) 单调，因为 \(d\alpha/d\Delta=\alpha(1-\alpha)>0\)。证毕。

---

## 4. label 结构推出的 score-gradient 符号形式

本节只把数据直觉写进 \(G_o^0,G_s^0\)。这里仍然不要求 \(q,k,v\) 方向相同。

令右上角 \(0\) 表示初始值。定义初始输出路径上的点

$$
o_{t,o}^0:=v_o^0+\alpha_{t,o}^0\delta v^0,
\qquad
o_{t,s}^0:=v_o^0+\alpha_{t,s}^0\delta v^0,
$$

其中

$$
\alpha_{t,o}^0
=
\frac{\exp(\Delta_o^0)}{t-1+\exp(\Delta_o^0)},
\qquad
\alpha_{t,s}^0
=
\frac{\exp(\Delta_s^0)}{t-1+\exp(\Delta_s^0)}.
$$

### Assumption 4.1（初始 attention-output 路径上的 ordinary-symmetric readout）

对所有出现在 \(o_{t,o}^0,o_{t,s}^0\) 中的有限多个点，存在

$$
\pi_{t,o},\pi_{t,s}\in(0,1),
$$

使得

$$
\operatorname{softmax}(Uo_{t,o}^0)
=
\pi_{t,o}e_s^{\rm vocab}+(1-\pi_{t,o})\bar e_O,
$$

$$
\operatorname{softmax}(Uo_{t,s}^0)
=
\pi_{t,s}e_s^{\rm vocab}+(1-\pi_{t,s})\bar e_O,
$$

其中

$$
\bar e_O:=\frac1{|O|}\sum_{a\in O}e_a^{\rm vocab}.
$$

定义 detector 方向

$$
d_*:=U^\top(\bar e_O-e_s^{\rm vocab}),
$$

以及 value detector 投影

$$
\rho_v^0:=\langle d_*,\delta v^0\rangle.
$$

这个假设只要求 readout 在初始 attention output 会访问的有限多个点上对 ordinary token 对称；它不要求 \(v_o^0,v_s^0\) 与某个全局方向同向。

### Lemma 4.2（post-`3` label 给出 \(G_o^0,G_s^0\) 的符号形式）

在 Assumption 4.1 下，

$$
G_o^0=-\rho_v^0\Lambda_o^0,
\qquad
G_s^0=-\rho_v^0\Lambda_s^0,
$$

其中

$$
\Lambda_o^0
:=
\frac1L\sum_{t=1}^{L-1}
(t-1)\pi_{t,o}\alpha_{t,o}^0(1-\alpha_{t,o}^0)
\ge0,
$$

$$
\Lambda_s^0
:=
\frac1L\sum_{t=1}^{L-1}
\pi_{t,s}\alpha_{t,s}^0(1-\alpha_{t,s}^0)
\ge0.
$$

由于 \(L=20\)、\(\pi_{t,o},\pi_{t,s}\in(0,1)\)，且 softmax 给出 \(0<\alpha<1\)，事实上

$$
\Lambda_o^0>0,
\qquad
\Lambda_s^0>0.
$$

因此若

$$
\rho_v^0>0,
$$

则

$$
G_o^0<0,
\qquad
G_s^0<0.
$$

#### 证明

出现在 \(G_o\) 中的位置满足 \(J<t\)，出现在 \(G_s\) 中的位置满足 \(J=t\)。由于每条序列只有一个 token `3`，所以对 \(t\le L-1\)，

$$
J<t\quad\Longrightarrow\quad X_{t+1}\in O,
$$

以及

$$
J=t\quad\Longrightarrow\quad X_{t+1}\in O.
$$

也就是说，所有进入 \(G_o,G_s\) 的 post-`3` score-gradient loss 项，其 next-token label 都是 ordinary。

若当前输出为某个初始路径点 \(o\)，且 label 对 ordinary token 条件取均值，则由 Assumption 4.1，

$$
\mathbb E[\zeta_t^0\mid o_t^0=o,
X_{t+1}\in O]
=
U^\top(\operatorname{softmax}(Uo)-\bar e_O)
=-\pi(o)d_*.
$$

于是

$$
g_t^0
=
\langle \zeta_t^0,\delta v^0\rangle
=-\pi(o_t^0)\langle d_*,\delta v^0\rangle
=-\pi(o_t^0)\rho_v^0.
$$

代入 \(G_o,G_s\) 的定义，并按 \(J\) 均匀计数：对固定 \(t\)，事件 \(J<t\) 有 \(t-1\) 个可能位置，事件 \(J=t\) 有一个可能位置。因此得到上式。证毕。

### Remark 4.3（这里真正由数据结构消去的假设）

Lemma 4.2 消去的是如下方向符号假设：post-`3` loss 项的上游梯度在 detector 方向上是 suppressor，即

$$
\mathbb E[\zeta_t^0\mid J<t]
\text{ 和 }
\mathbb E[\zeta_t^0\mid J=t]
$$

沿着 \(-d_*\) 方向。这个符号不是额外假设，而是由

$$
J<t\text{ 或 }J=t\quad\Longrightarrow\quad X_{t+1}\in O
$$

和 ordinary-symmetric readout 推出。

但是这还没有直接证明 attention gap 增长。attention gap 的符号还要经过 Lemma 3.1 中的 \(q_o,q_s,\kappa\) 几何项。

---

## 5. 不要求 \(q,k,v\) 同向的局部增长定理

把 Lemma 4.2 代入 Lemma 3.1，可以得到完全显式的初始速度。

定义

$$
B_o^0
:=
\Lambda_o^0(\|\kappa^0\|^2+2\|q_o^0\|^2)
+2\Lambda_s^0\langle q_o^0,q_s^0\rangle,
$$

$$
B_s^0
:=
\Lambda_s^0(\|\kappa^0\|^2+2\|q_s^0\|^2)
+2\Lambda_o^0\langle q_o^0,q_s^0\rangle.
$$

### Theorem 5.1（ordinary-symmetric readout 下的 exact 初始速度）

在 Assumption 4.1 下，完整六向量耦合 flow 满足

$$
\dot\Delta_o(0)=\frac{\rho_v^0}{d}B_o^0,
\qquad
\dot\Delta_s(0)=\frac{\rho_v^0}{d}B_s^0.
$$

因此 token `3` attention 在初值后局部严格增大的充要一阶条件是

$$
\rho_v^0B_o^0>0,
\qquad
\rho_v^0B_s^0>0.
$$

这里没有使用 \(q,k,v\) 同向假设。

#### 证明

由 Lemma 4.2，

$$
G_o^0=-\rho_v^0\Lambda_o^0,
\qquad
G_s^0=-\rho_v^0\Lambda_s^0.
$$

代入 Lemma 3.1：

$$
\dot\Delta_o(0)
=
-\frac1d
\left[
G_o^0\|\kappa^0\|^2
+2G_o^0\|q_o^0\|^2
+2G_s^0\langle q_o^0,q_s^0\rangle
\right]
=
\frac{\rho_v^0}{d}B_o^0.
$$

\(\dot\Delta_s(0)\) 同理。由于 attention mass 对 \(\Delta_x\) 单调，且梯度流解局部 \(C^1\)，\(\rho_v^0B_o^0>0\)、\(\rho_v^0B_s^0>0\) 等价于两个 gap 的正一阶增长，从而推出 token `3` attention 局部严格增大。证毕。

### Corollary 5.2（一个不要求同向的简单充分条件）

因为 \(\Lambda_o^0,\Lambda_s^0>0\)，若

$$
\rho_v^0>0
$$

且

$$
\langle q_o^0,q_s^0\rangle
>
-\min\left\{
\frac{\Lambda_o^0(\|\kappa^0\|^2+2\|q_o^0\|^2)}{2\Lambda_s^0},
\frac{\Lambda_s^0(\|\kappa^0\|^2+2\|q_s^0\|^2)}{2\Lambda_o^0}
\right\},
$$

则

$$
B_o^0>0,
\qquad
B_s^0>0,
$$

从而

$$
\dot\Delta_o(0)>0,
\qquad
\dot\Delta_s(0)>0.
$$

一个更容易检查的充分条件是

$$
\rho_v^0>0,
\qquad
\langle q_o^0,q_s^0\rangle\ge0,
$$

并且排除零梯度退化：

$$
\|\kappa^0\|^2+2\|q_o^0\|^2>0,
\qquad
\|\kappa^0\|^2+2\|q_s^0\|^2>0.
$$

在这个条件下同样有 \(B_o^0>0,B_s^0>0\)。例如只要 \(q_o^0,q_s^0\) 都非零，就满足这个非退化条件。

#### 证明

由 \(B_o^0,B_s^0\) 的定义直接得到。若 \(\langle q_o^0,q_s^0\rangle\ge0\)，两个 cross 项非负；再由非退化条件和 \(\Lambda_o^0,\Lambda_s^0>0\)，两个主项分别严格为正。证毕。

### Corollary 5.3（常用的分块方向凝聚特例）

如果初始态只满足分块方向凝聚

$$
q_o^0=a_oh_q,
\qquad
q_s^0=a_sh_q,
$$

$$
k_o^0=b_oh_k,
\qquad
k_s^0=b_sh_k,
$$

其中

$$
\|h_q\|=\|h_k\|=1,
\qquad
a_o,a_s,b_o,b_s>0,
$$

但不要求

$$
h_q=h_k,
$$

也不要求 value 方向与 \(h_q,h_k\) 相同，则

$$
\langle q_o^0,q_s^0\rangle=a_oa_s>0.
$$

因此在 Assumption 4.1 和

$$
\rho_v^0=\langle d_*,v_s^0-v_o^0\rangle>0
$$

下，有

$$
\dot\Delta_o(0)>0,
\qquad
\dot\Delta_s(0)>0.
$$

这里 \(h_q\)、\(h_k\) 可以不同；\(v_s^0-v_o^0\) 也可以是任意方向，只需要它在 detector 方向上的投影为正。

---

## 6. 条件性区间不减与更简单充分条件

上面的结论是局部增长。若要证明某个时间区间 \(I\) 上 attention 不降低，exact 判据仍然是 Lemma 3.1 的 driver。

定义任意时刻的

$$
H_o(u)
:=
G_o\|\kappa\|^2
+2G_o\|q_o\|^2
+2G_s\langle q_o,q_s\rangle,
$$

$$
H_s(u)
:=
G_s\|\kappa\|^2
+2G_o\langle q_s,q_o\rangle
+2G_s\|q_s\|^2.
$$

### Theorem 6.1（exact 条件性区间不减）

若对所有 \(u\in I\)，都有

$$
H_o(u)\le0,
\qquad
H_s(u)\le0,
$$

则

$$
\Delta_o(u),\Delta_s(u)
$$

在 \(I\) 上不减，因而对应 token `3` attention mass 在 \(I\) 上不减。

若两个不等式在某个子区间上严格，且 attention 未饱和，则 attention mass 在该子区间严格增加。

#### 证明

由 Lemma 3.1，

$$
\dot\Delta_o=-\frac1dH_o,
\qquad
\dot\Delta_s=-\frac1dH_s.
$$

因此 \(H_o,H_s\le0\) 推出 \(\dot\Delta_o,\dot\Delta_s\ge0\)。attention mass 对 gap 单调，故结论成立。证毕。

### Corollary 6.2（更简单的几何充分条件）

若对所有 \(u\in I\)，都有

$$
G_o(u)\le0,
\qquad
G_s(u)\le0,
$$

以及

$$
\langle q_o(u),q_s(u)\rangle\ge0,
$$

则

$$
H_o(u)\le0,
\qquad
H_s(u)\le0,
$$

从而 token `3` attention mass 在 \(I\) 上不减。

#### 证明

由定义，

$$
H_o
=
G_o(\|\kappa\|^2+2\|q_o\|^2)
+2G_s\langle q_o,q_s\rangle.
$$

其中

$$
\|\kappa\|^2+2\|q_o\|^2\ge0.
$$

若 \(G_o\le0\)，第一项非正；若同时 \(G_s\le0\) 且 \(\langle q_o,q_s\rangle\ge0\)，第二项也非正。因此 \(H_o\le0\)。同理，

$$
H_s
=
G_s(\|\kappa\|^2+2\|q_s\|^2)
+2G_o\langle q_o,q_s\rangle
\le0.
$$

再用 Theorem 6.1 得结论。证毕。

### Corollary 6.3（用 detector 投影写出的数据型充分条件）

若在区间 \(I\) 上，ordinary-symmetric readout 沿实际 post-`3` attention-output 轨道成立，使得对所有相关 \(u,t,x\)，都有

$$
\mathbb E[\zeta_t(u)\mid J<t\text{ 或 }J=t]
=-\pi_{t,x}(u)d_*,
\qquad
\pi_{t,x}(u)\ge0,
$$

并且对所有 \(u\in I\)，都有

$$
\rho_v(u):=\langle d_*,\delta v(u)\rangle\ge0,
$$

以及

$$
\langle q_o(u),q_s(u)\rangle\ge0,
$$

则 token `3` attention mass 在 \(I\) 上不减。

#### 证明

由数据结构，进入 \(G_o,G_s\) 的位置满足 \(J<t\) 或 \(J=t\)，其 next-token label 都是 ordinary。由区间版 ordinary-symmetric readout，

$$
g_t(u)=\langle\zeta_t(u),\delta v(u)\rangle
=-\pi_{t,x}(u)\rho_v(u)\le0.
$$

又因为 \(\alpha(1-\alpha)\ge0\)，所以

$$
G_o(u)\le0,
\qquad
G_s(u)\le0.
$$

再由 Corollary 6.2 得结论。证毕。

### Corollary 6.4（初始简单条件推出短时间版本）

若在 \(u=0\) 有严格 margin

$$
\rho_v^0=\langle d_*,\delta v^0\rangle>0,
\qquad
\langle q_o^0,q_s^0\rangle>0,
$$

并且 ordinary-symmetric readout 在一段实际轨道邻域内成立，则由连续性，存在 \(\epsilon>0\)，使得对所有 \(0\le u\le\epsilon\)，

$$
\rho_v(u)>0,
\qquad
\langle q_o(u),q_s(u)\rangle>0.
$$

因此在 \([0,\epsilon]\) 上满足 Corollary 6.3 的条件，token `3` attention mass 在这段短时间内不减。若同时排除零梯度退化，则初始处严格增长。

这个 corollary 解释了为什么初始条件

$$
\langle d_*,\delta v^0\rangle>0,
\qquad
\langle q_o^0,q_s^0\rangle>0
$$

可以替代更复杂的 \(H_o^0,H_s^0<0\) 检查来给出短时间结论。对任意预先指定的长区间，仅有初始正号还不够；需要沿区间保持 \(\rho_v(u)\ge0\) 和 \(\langle q_o(u),q_s(u)\rangle\ge0\)，或者直接检查 exact driver。

---

## 7. 假设层次与结论

本文的最少 exact 结论是 Corollary 3.2：对任意六向量初值，不加 readout symmetry、不加方向同向性，局部 attention 增长等价于

$$
H_o^0<0,
\qquad
H_s^0<0.
$$

若加入 ordinary-symmetric readout，则数据结构“token `3` 后面的 label 不再是 `3`”推出

$$
G_o^0=-\rho_v^0\Lambda_o^0,
\qquad
G_s^0=-\rho_v^0\Lambda_s^0.
$$

这一步只消去了 post-`3` score-gradient 的方向符号假设。它没有单独消去 attention gap 增长所需的 \(q\)-几何条件。真正的 exact 初始增长公式是

$$
\dot\Delta_o(0)=\frac{\rho_v^0}{d}B_o^0,
\qquad
\dot\Delta_s(0)=\frac{\rho_v^0}{d}B_s^0.
$$

因此，最少的 data-driven 局部增长条件是

$$
\rho_v^0B_o^0>0,
\qquad
\rho_v^0B_s^0>0.
$$

一个自然且容易检查的局部充分条件是

$$
\rho_v^0>0,
\qquad
\langle q_o^0,q_s^0\rangle\ge0,
$$

再加上非退化条件

$$
\|\kappa^0\|^2+2\|q_o^0\|^2>0,
\qquad
\|\kappa^0\|^2+2\|q_s^0\|^2>0.
$$

如果要得到区间不减，则可以使用更简单的沿轨道充分条件：

$$
G_o(u),G_s(u)\le0,
\qquad
\langle q_o(u),q_s(u)\rangle\ge0.
$$

在区间版 ordinary-symmetric readout 下，前两个不等式又可由

$$
\langle d_*,\delta v(u)\rangle\ge0
$$

推出。只用初始的 \(\langle d_*,\delta v^0\rangle>0\) 和 \(\langle q_o^0,q_s^0\rangle>0\) 可以通过连续性给出短时间版本，但不能单独给出任意长区间的不减。

这允许 \(q\)、\(k\)、\(v\) 三类方向完全不同；甚至允许 \(v_o^0,v_s^0\) 不在同一条过原点射线上。所有 attention 变化都由完整六向量耦合 gradient flow 诱导，没有冻结任何变量。
