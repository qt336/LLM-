# RoPE 六向量梯度流：marker 后五个位置的 attention 能否一直增加

本文研究下面的命题：

> 序列中的 marker token 后面五个 token 都不是 marker。模型使用 RoPE。
> 沿完整六向量耦合 gradient flow，后面五个 token 对 marker 的 attention
> 是否都会一直增加，直到不再变化？

结论是：

$$
\boxed{
\text{仅凭“后五个 token 不是 marker”，不能推出五个 attention 同时单调增加。}
}
$$

失败原因不是 value 路径，也不是冻结变量，而是 RoPE 使不同相对位移共享
同一组 query/key 参数时产生带符号的 gradient interference。即使五个位置
各自的 loss 都希望提高 marker attention，它们的合成更新仍可能降低其中
某一个位置的 attention。

严格可证明的版本是：

1. 五个 log-attention 的速度满足一个 exact Gram-kernel ODE；
2. 该 kernel 自动正半定，所以 loss-gradient 加权的总 log-attention 速度
   非负，至少一个位置会增长；
3. 若 kernel 逐项非负，并且五个位置的 suppressor 符号沿轨道保持，则五个
   attention 才会同时单调增加；
4. attention 有界，所以单调轨道必有极限；在额外的有界光滑条件下，速度
   趋于零，即渐近进入平台，而不一定在有限时间突然不变。

---

## 1. 六向量模型与 RoPE

marker 类型记为

$$
m,
$$

ordinary 类型记为

$$
o.
$$

完整六向量状态为

$$
\theta=(q_o,q_m,k_o,k_m,v_o,v_m),
\qquad
q_o,q_m,k_o,k_m,v_o,v_m\in\mathbb R^d.
$$

本文使用独立六向量欧氏 gradient flow：

$$
\dot\theta=-\nabla_\theta\mathcal L(\theta).
$$

RoPE 位置矩阵记为

$$
R_p
=
\bigoplus_{j=1}^{d/2}
\operatorname{Rot}(p\omega_j),
$$

并满足

$$
R_p^\top R_q=R_{q-p}.
$$

位置 $t$ 的 query 与位置 $i$ 的 key 的 score 为

$$
s_{t,i}
=
\frac1{\sqrt d}
\langle R_tq_{X_t},R_ik_{X_i}\rangle
=
\frac1{\sqrt d}
q_{X_t}^\top R_{i-t}k_{X_i}.
$$

设 marker 位于位置 $J$。对 marker 后第 $r$ 个位置，

$$
t_r:=J+r,
\qquad
r\in\{1,2,3,4,5\}.
$$

因为这五个 query token 都不是 marker，它们都使用 $q_o$。marker key 的
score 为

$$
s_r^{(m)}
:=
s_{J+r,J}
=
\frac1{\sqrt d}
q_o^\top R_{-r}k_m.
$$

其他 ordinary key 的 score 依赖各自的相对位移：

$$
s_{r,i}^{(o)}
=
\frac1{\sqrt d}
q_o^\top R_{i-(J+r)}k_o.
$$

因此 RoPE 下 ordinary keys 不再具有同一个 score。无位置编码证明中的单一
logistic gap

$$
\frac{\langle q_o,k_m-k_o\rangle}{\sqrt d}
$$

不能直接复用。

marker attention 为

$$
A_r
:=
\frac{\exp(s_r^{(m)})}
{\exp(s_r^{(m)})
+\sum_{i\le J+r,\ i\ne J}\exp(s_{r,i}^{(o)})}.
$$

---

## 2. 一个必须先修正的标签条件

NTP 在 query 位置 $J+r$ 使用的 label 是

$$
X_{J+r+1}.
$$

所以要让 $r=1,\ldots,5$ 五个 query 的 label 全部不是 marker，需要

$$
X_{J+2},X_{J+3},X_{J+4},X_{J+5},X_{J+6}\ne m.
$$

仅知道

$$
X_{J+1},\ldots,X_{J+5}\ne m
$$

并不能控制第五个 query 的 label $X_{J+6}$。因此原命题还存在一个
off-by-one：

$$
\boxed{
\text{若 marker 可能再次出现，则还必须要求第六个后继 token 也不是 marker。}
}
$$

若每条序列中 marker 全局只出现一次，则这个附加要求自动成立。

---

## 3. log-attention gradient 是正确的五维坐标

定义

$$
\ell_r:=-\log p_{J+r}(X_{J+r+1})
$$

为第 $r$ 个 post-marker query 的 NTP loss，并定义上游梯度

$$
\zeta_r:=\nabla_{o_{J+r}}\ell_r.
$$

attention output 为

$$
o_{J+r}
=
v_o+A_r(v_m-v_o)
$$

时，定义

$$
g_r
:=
\langle\zeta_r,v_m-v_o\rangle.
$$

如果 label 不是 marker、readout 对 ordinary token 对称，并且

$$
\rho_v
:=
\langle d_*,v_m-v_o\rangle\ge0,
$$

则和无位置编码证明相同，

$$
g_r\le0.
$$

对 q/k 参数而言，

$$
\nabla_{q,k}\ell_r
=
g_r\nabla_{q,k}A_r
=
g_rA_r\nabla_{q,k}\log A_r.
$$

定义

$$
\lambda_r:=-g_rA_r\ge0
$$

以及 log-attention gradient

$$
h_r
:=
\nabla_{q_o,k_o,k_m}\log A_r.
$$

于是第 $r$ 个 loss 对 q/k 的 gradient 为

$$
\nabla_{q_o,k_o,k_m}\ell_r
=
-\lambda_rh_r.
$$

这里 value 向量也按完整六向量 flow 更新；但 $A_r$ 本身只依赖 q/k，所以
它的瞬时速度只需要 q/k gradient。value 更新会通过后续时刻的 $g_r$ 和
$\lambda_r$ 间接影响 attention。

---

## 4. 五个 attention 的 exact Gram-kernel ODE

先固定一条满足窗口条件的序列，并考虑只由这五个 post-marker loss 构成的
条件 objective：

$$
\mathcal L_5
:=
\sum_{r=1}^5\ell_r.
$$

对 population 或 minibatch objective，需要把“样本编号”也并入 Gram
kernel 的坐标索引；不同样本之间同样可能产生 gradient interference。

定义 $5\times5$ Gram 矩阵

$$
H_{rs}
:=
\langle h_r,h_s\rangle,
\qquad
r,s\in\{1,\ldots,5\},
$$

其中内积包含共享的 $q_o,k_o,k_m$ 三个向量块。显然

$$
H\succeq0.
$$

### Theorem 4.1（五维 exact log-attention ODE）

沿 $\mathcal L_5$ 的完整六向量欧氏 gradient flow，

$$
\boxed{
\frac d{du}\log A_r
=
\sum_{s=1}^5H_{rs}\lambda_s,
\qquad
r=1,\ldots,5.
}
$$

向量形式为

$$
\boxed{
\frac d{du}
\begin{pmatrix}
\log A_1\\
\vdots\\
\log A_5
\end{pmatrix}
=
H
\begin{pmatrix}
\lambda_1\\
\vdots\\
\lambda_5
\end{pmatrix}.
}
$$

#### 证明

q/k gradient 为

$$
\nabla_{q,k}\mathcal L_5
=
-\sum_{s=1}^5\lambda_sh_s.
$$

因此 gradient flow 给出

$$
\dot\theta_{q,k}
=
\sum_{s=1}^5\lambda_sh_s.
$$

对 $\log A_r$ 使用链式法则：

$$
\frac d{du}\log A_r
=
\langle h_r,\dot\theta_{q,k}\rangle
=
\sum_{s=1}^5
\lambda_s\langle h_r,h_s\rangle.
$$

证毕。

### Corollary 4.2（无额外几何假设的最强 aggregate 结论）

因为 $H\succeq0$，

$$
\boxed{
\sum_{r=1}^5
\lambda_r\frac d{du}\log A_r
=
\lambda^\top H\lambda
=
\left\|
\sum_{r=1}^5\lambda_rh_r
\right\|^2
\ge0.
}
$$

所以当 q/k score gradient 非零时，至少一个 $A_r$ 增长；五个 attention
不可能同时严格下降。

但是正半定性不能推出

$$
H\lambda\ge0
$$

逐坐标成立。非对角元 $H_{rs}$ 可以为负，而 RoPE 正是产生负
cross-offset 内积的直接来源。

---

## 5. RoPE 为什么产生负的 offset interference

先看一个比完整 softmax 更简单、但已经足以暴露问题的 type-gap：

$$
\Delta_r
:=
\frac1{\sqrt d}
q_o^\top R_{-r}(k_m-k_o).
$$

记

$$
\kappa:=k_m-k_o.
$$

其六向量 gradient 为

$$
\nabla_{q_o}\Delta_r
=
\frac1{\sqrt d}R_{-r}\kappa,
$$

$$
\nabla_{k_m}\Delta_r
=
\frac1{\sqrt d}R_rq_o,
$$

$$
\nabla_{k_o}\Delta_r
=
-\frac1{\sqrt d}R_rq_o.
$$

因此五个 type-gap 的 Gram kernel 为

$$
\boxed{
K_{rs}
=
\frac1d
\left[
\kappa^\top R_{r-s}\kappa
+2q_o^\top R_{s-r}q_o
\right].
}
$$

把 $q_o,\kappa$ 分解到各个二维 RoPE frequency plane：

$$
q_o=(q_{o,1},\ldots,q_{o,d/2}),
$$

$$
\kappa=(\kappa_1,\ldots,\kappa_{d/2}),
$$

则

$$
\boxed{
K_{rs}
=
\frac1d
\sum_{j=1}^{d/2}
\left(
\|\kappa_j\|^2+2\|q_{o,j}\|^2
\right)
\cos((r-s)\omega_j).
}
$$

所以只要有效能量落在某些满足

$$
\cos((r-s)\omega_j)<0
$$

的 frequency plane 上，offset 间 kernel 就会为负。

---

## 6. 标准 RoPE 频率给出的显式反例

标准 RoPE 的最高 frequency 通常为

$$
\omega_1=1.
$$

只在这个二维 frequency plane 上取非零 $q_o,\kappa$。若五个 gap loss
对各 offset 给出相同的正驱动力

$$
\lambda_1=\cdots=\lambda_5=\lambda>0,
$$

则第一个 offset 的 type-gap 速度正比于

$$
\sum_{s=1}^5\cos((1-s)\omega_1)
=
1+\cos1+\cos2+\cos3+\cos4.
$$

数值上

$$
1+\cos1+\cos2+\cos3+\cos4
\approx
-0.51948<0.
$$

因此

$$
\dot\Delta_1<0
$$

尽管五个 loss 对各自 gap 的直接导数全都为负，即每个 loss 单独都希望提高
自己的 marker gap。

这已经否定了“post-marker label 全为 ordinary 就自动推出五个 offset 同时
增长”的普遍命题。

### 6.1 对真实 causal attention 的反例结构

上面的 type-gap 省略了不同 ordinary key 的 RoPE score。真实 attention 中
同样存在负 interference。

取 marker 位于位置 $1$，并在某个状态令

$$
k_o=0.
$$

对 offset $r$，ordinary keys 的 score 此时全为零。定义局部 marker-vs-
ordinary-average contrast

$$
C_r
:=
s_r^{(m)}
-\frac1r\sum_{h=0}^{r-1}
\frac1{\sqrt d}q_o^\top R_{-h}k_o.
$$

在 $k_o=0$ 的状态，$\nabla\log A_r$ 与 $\nabla C_r$ 只差正标量
$1-A_r$，所以 gradient 内积符号相同。

令

$$
u_r
:=
\frac1r\sum_{h=0}^{r-1}R_hq_o.
$$

当

$$
q_o=k_m\ne0
$$

且二者都只位于 frequency $\omega=1$ 的同一个二维平面时，
contrast kernel 的符号由

$$
\widetilde H_{rs}
=
2\cos((r-s)\omega)
+\frac{\langle u_r,u_s\rangle}
{\|q_o\|^2}
$$

控制。特别地，

$$
\widetilde H_{1,5}
=
2\cos4
+\frac15\sum_{h=0}^4\cos h
\approx
-1.41118<0.
$$

因此只要第五个位置的驱动力相对足够大，第五个 loss 的更新就会使第一个
位置的 marker attention 瞬时下降。数据条件只决定 $\lambda_r$ 的符号，
并不限制五个 $\lambda_r$ 的比例，所以不能排除这种情况。

---

## 7. 五个 attention 同时单调的严格充分条件

### Theorem 7.1（RoPE kernel cooperativity）

设在时间区间 $I$ 上：

1. 五个 query 的 label 都不是 marker；
2. value/readout suppressor 保持
   $$
   g_r(u)=\langle\zeta_r(u),v_m(u)-v_o(u)\rangle\le0,
   \qquad
   r=1,\ldots,5;
   $$
3. 五个 log-attention gradient Gram 矩阵逐项非负：
   $$
   H_{rs}(u)\ge0,
   \qquad
   r,s=1,\ldots,5;
   $$
4. 训练 objective 只包含这五个位置，或其他位置的 q/k gradient 与每个
   $h_r$ 的内积也非负。

则对每个 $r=1,\ldots,5$，

$$
\frac d{du}\log A_r(u)\ge0,
$$

从而

$$
\dot A_r(u)\ge0.
$$

若第 $r$ 行至少有一个严格正的 $H_{rs}\lambda_s$，则

$$
\dot A_r(u)>0.
$$

#### 证明

条件 1 和 2 给出

$$
\lambda_s=-g_sA_s\ge0.
$$

由 Theorem 4.1，

$$
\frac d{du}\log A_r
=
\sum_{s=1}^5H_{rs}\lambda_s.
$$

条件 3 使右侧每项非负。若存在严格正项，则和严格为正。因为

$$
\dot A_r
=
A_r\frac d{du}\log A_r
$$

且 $A_r>0$，得到结论。其他位置的 gradient 在条件 4 下只增加非负项。
证毕。

### Remark 7.2（为什么这不是数据条件能自动推出的）

“后五个 token 不是 marker”只控制

$$
\lambda_r\ge0.
$$

它完全不控制

$$
H_{rs}
=
\langle\nabla\log A_r,\nabla\log A_s\rangle.
$$

$H$ 由 RoPE frequencies、当前 q/k 能量分布、prefix 中各 ordinary key 的
attention 权重共同决定。标准 RoPE 的高频 plane 允许 $H_{rs}<0$。

---

## 8. 一个更具体但较强的 frequency 条件

对第 5 节的简化 type-gap kernel，若所有承载 $q_o,\kappa$ 能量的 RoPE
frequency 都满足

$$
\omega_j\le\frac{\pi}{8},
$$

则因为

$$
|r-s|\le4,
$$

有

$$
|(r-s)\omega_j|\le\frac\pi2
$$

以及

$$
\cos((r-s)\omega_j)\ge0.
$$

因此

$$
K_{rs}\ge0
\qquad(r,s=1,\ldots,5).
$$

这给出 type-gap 五维系统的一个明确低频充分条件。

但它不能直接替代真实 attention 的 $H_{rs}\ge0$ 检查，因为真实 causal
denominator 还包含多个不同 relative offset 的 ordinary keys。对真实模型，
最可靠的实验量仍然是直接计算

$$
H_{rs}
=
\langle\nabla_{q,k}\log A_r,
\nabla_{q,k}\log A_s\rangle.
$$

---

## 9. “一直增大，直至不变”的严格含义

在 Theorem 7.1 的条件沿整个训练区间成立时，

$$
0<A_r(u)\le1
$$

且 $A_r(u)$ 单调不减。因此每个 attention 都存在极限

$$
A_r^\infty
:=
\lim_{u\to\infty}A_r(u)
\in(0,1].
$$

所以可以严格说：

$$
\boxed{
\text{五个 attention 单调增加并渐近趋于某个平台。}
}
$$

仅由单调有界不能断言它们在某个有限训练时间后完全恒定。若再假设：

1. 六向量轨道有界；
2. gradient-flow 向量场及其一阶导数在该轨道上有界；

则 $\dot A_r$ 一致连续。又因为

$$
\int_0^\infty\dot A_r(u)\,du
=
A_r^\infty-A_r(0)<\infty,
$$

由 Barbalat lemma，

$$
\dot A_r(u)\to0.
$$

平台可能来自：

- $A_r\to1$，softmax 导数饱和；
- suppressor 驱动力 $\lambda_r\to0$；
- 相关 q/k gradient 退化；
- kernel 行与当前驱动力正交。

因此“直至不变”一般应理解为速度渐近趋零，而不是有限时间停止。

---

## 10. 完整训练 objective 的额外问题

如果训练使用完整序列 loss，而不只使用 marker 后五个位置，则 q/k gradient
还包含

$$
g_{\rm rest}
:=
\nabla_{q,k}\mathcal L_{\rm rest}.
$$

这时 exact ODE 变成

$$
\frac d{du}\log A_r
=
\sum_{s=1}^5H_{rs}\lambda_s
-\langle h_r,g_{\rm rest}\rangle.
$$

即使 $H_{rs}\ge0$，其他位置也可能通过共享的 $q_o,k_o,k_m$ 降低目标
attention。要证明完整训练下的全程单调，还必须控制

$$
\langle h_r,g_{\rm rest}\rangle
\le
\sum_{s=1}^5H_{rs}\lambda_s
$$

或使用更强但更简单的条件

$$
\langle h_r,g_{\rm rest}\rangle\le0.
$$

所以“局部五位置 conditional objective”和“完整 NTP objective”必须明确
区分。

---

## 11. 实验上应直接测什么

对每个满足 marker 后窗口条件的样本：

1. 取
   $$
   A_r
   =
   \operatorname{attn}[J+r,J],
   \qquad
   r=1,\ldots,5.
   $$
2. 分别反传
   $$
   h_r
   =
   \nabla_{q_o,k_o,k_m}\log A_r.
   $$
3. 构造
   $$
   H_{rs}
   =
   \langle h_r,h_s\rangle.
   $$
4. 记录 suppressor 标量
   $$
   g_r
   =
   \langle\zeta_r,v_m-v_o\rangle
   $$
   和
   $$
   \lambda_r=-g_rA_r.
   $$
5. 比较实测速度与理论预测
   $$
   \frac d{du}\log A_r
   \stackrel{?}{=}
   \sum_sH_{rs}\lambda_s
   $$
  ；若使用完整 loss，再单独记录
   $$
   -\langle h_r,g_{\rm rest}\rangle.
   $$

建议报告：

$$
\min_{r,s}H_{rs},
$$

$$
\min_r(H\lambda)_r,
$$

$$
\lambda^\top H\lambda,
$$

以及五个 attention 的 finite-step 变化。

若只观察到

$$
\lambda^\top H\lambda>0,
$$

只能支持“加权 aggregate 改善”，不能声称五个 offset 全部单调。

---

## 12. 最终结论

仅由

$$
\text{marker 后五个 token 都不是 marker}
$$

最多能帮助推出相关 post-marker loss 的直接驱动力

$$
\lambda_r\ge0.
$$

它不能控制 RoPE 造成的 cross-offset kernel 符号。标准 frequency

$$
\omega=1
$$

已经能产生负的 offset interference，因此无条件命题

$$
\boxed{
A_1,\ldots,A_5
\text{ 沿六向量 gradient flow 一直同时增大}
}
$$

是假的。

严格可成立的条件化版本是：

$$
\boxed{
\lambda_r(u)\ge0
\quad\text{且}\quad
H_{rs}(u)\ge0
\quad
\forall r,s\in\{1,\ldots,5\}
}
$$

并且完整 objective 的其他 gradient 不产生负干扰。在这些条件沿轨道保持
时，五个 attention 单调不减；若严格非退化则严格增加，并因有界而趋于
极限。在有界光滑轨道条件下，其增长速度趋于零。
