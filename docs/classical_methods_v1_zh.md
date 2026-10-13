# 三维经典算法：来源、离散化与求解登记

本文件定义独立 `reconstruction/classical.py` 的实际方法；原冻结 projectors.py 不改动。所有输出是 endpoint-aligned XYZ 64³ CH4 质量分数 C∈[0,1]，域边长0.6m，节点间距 h=0.6/63m。摩尔质量采用原配置 CH4=16.043、air=28.965（同单位比值）。初始化为零场，无 GT/network warm start、logit/sigmoid 参数化。

## 有界非负三维 SART

来源：Andersen & Kak (1984), *Simultaneous algebraic reconstruction technique (SART): a superior implementation of the ART algorithm*, Ultrasonic Imaging 6:81–94，[DOI](https://doi.org/10.1016/0161-7346(84)90008-7)。本次查阅了 [scikit-image 官方 iradon_sart 文档](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.iradon_sart) 关于线性采样、零初始化、裁剪、relaxation 与迭代噪声，以及 [TIGRE 官方 SART 源码](https://github.com/CERN/TIGRE/blob/master/Python/tigre/algorithms/art_family_algorithms.py) 和 [块更新实现](https://github.com/CERN/TIGRE/blob/master/Python/tigre/algorithms/iterative_recon_alg.py)。DOI 页面访问得到验证页面，未声称已读原文全文。本实现依据 SART 行列归一化更新，独立构建真实三维射线算子；不调用二维 Radon，也不复制 TIGRE 的 CT 几何或近似反投影。

未知量为摩尔分数 chi，`Y=A chi=10^6 sum_k w_rkj chi_j ds_r`。每条实际 valid ray 在 near+length×(k+0.5)/96 处取样，使用 XYZ 三线性权重、border clamp 与 ds=length/96；矩阵保存 CSR，重复 ray/voxel 系数求和，仅构造稀疏矩阵及其真正的转置 CSR。valid 的 Y=0 保留为方程，invalid 射线完全排除；padding 不进入独立 station-ID 清单。

每个 station 的64×64扫描为一个 block b。令 R_b=diag(A_b 1)，D_b=diag(A_b^T 1)，更新为：

```
chi <- clip_[0,1](chi + relaxation D_b^† A_b^T R_b^† (Y_b - A_b chi))
```

未触达列的更新为零；按冻结输入 station 顺序遍历全部4 block 一次为一个 sweep。保存相对平方残差目标 `0.5 ||A chi-Y||²/||Y||²` 与 relative L2，它是监控目标，SART 不是普通梯度下降，也不承诺每个 station 更新都使全局目标单调下降。

精确转换：`C=chi M_CH4/[chi M_CH4+(1-chi)M_air]`。不能把 ppm·m 当质量分数积分，不能对全浓度假设低浓度线性关系。

## 有界非线性 Tikhonov 三维平滑重建

Tikhonov 形式参照 Hansen, *Regularization Tools v4 manual*，[官方手册](https://www.imm.dtu.dk/~pcha/Regutools/RTv4manual.pdf)，章节 General-form Tikhonov regularization：数据拟合平方项与 `||Lx||²`。下载手册并保存来源 hash。这里保留广义 Tikhonov 正则形式，但数据项为原 P_diff(C)，故名称明确为**非线性有界变分重建**，不是闭式线性 Tikhonov 解。

```
min_0<=C<=1 0.5 mean_valid[(P_diff(C)-Y)/s]^2
                + lambda/2 mean_voxel[(Dx C)^2+(Dy C)^2+(Dz C)^2]
```

s=冻结 train p99.5=32284.894580078173 ppm·m。三轴正向差分除物理 h，末端差分设零（Neumann）；正则项采用均匀节点平均，等价于体积归一的离散平滑 proxy，未冒称精确连续积分求积。

## 有界平滑三维 isotropic TV 重建

来源：Rudin–Osher–Fatemi (1992), *Nonlinear total variation based noise removal algorithms*, [DOI](https://doi.org/10.1016/0167-2789(92)90242-F)；查阅 [scikit-image 官方 nD TV 文档](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle)，包括三维示例和 isotropic gradient norm。本实现是平滑 TV 变分目标，**不调用其图像去噪器，也不声称使用 Chambolle proximal 或精确非平滑 TV 解**。

```
min_0<=C<=1 0.5 mean_valid[(P_diff(C)-Y)/s]^2
       + lambda mean_voxel[sqrt((Dx C)^2+(Dy C)^2+(Dz C)^2+epsilon^2)-epsilon]
```

epsilon=1e-4 m^-1，使用真正三维 gradient norm，非二维切片堆叠。

两种变分方法使用投影 Barzilai–Borwein 梯度及单调 Armijo backtracking：`Cnew=clip(C-alpha grad J)`；接受条件 `Jnew<=J+1e-4 gradJ·(Cnew-C)+1e-12`；拒绝则 alpha/2，最多20次。BB next alpha=`||deltaC||²/(deltaC·deltaGrad)`，正曲率时使用并限制[1e-5,1e6]，否则保留接受的 alpha；initial alpha=1000。梯度通过原 frozen differentiable_project 的 autograd 计算。保存 objective/data/regularization、step、backtrack、projected gradient infinity proxy，不宣称找到全局最优解。

## 两种前向离散化的差异

SART 在线性算子中插值 chi；原 P_diff 先插值 C，再逐采样点执行非线性 mass_to_mole。二者不交换，虽然路径、K96、坐标与单位一致，仍是不同离散化。每个 SART 输出单独比较 `A(mass_to_mole(C))` 与 `P_diff(C)`，以原 P_diff 的有效射线 norm 为分母保存 relative L1/L2；SART 自身 fit 残差和统一重投影误差分别保存，不能将离散化差异全部归因于算法恢复能力。

## 有限校准与停止

`configs/classical_v4_v1.yaml` 登记候选：SART relaxation {0.25,0.5,0.8}；Tikhonov lambda {1e-5,1e-4,1e-3}；TV lambda {1e-4,1e-3,1e-2}。每种3个全局配置，不按 Val 扩展。12 train×2布局×3方法×3候选=216实际重建，SART至多150 sweep、变分至多200迭代。

只按输入目标相对变化1e-5、连续8次停止，SART最少30 sweep、变分最少40步；状态包括 objective_stable、budget_exhausted、line_search_failed/nonfinite。H/GT只能选择全局校准配置，不选择单case迭代。最终参数及源码 hash 在 `selected_configs.json/.sha256` 冻结，随后才运行 Val。完整日志保留预算耗尽/失败，不按 Val 调参重试。
