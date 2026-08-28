# 问题一最终方法说明

## 1. 目标、输入与主张范围

问题一要求在给定塔位和1745面定日镜坐标下，计算题目规定的60个太阳状态中各镜的光学效率和镜场输出热功率，并形成月平均与年平均结果。模型输入包括日期、当地时刻、纬度、海拔、镜心坐标、镜面尺寸与安装高度、塔和圆柱集热器几何以及题面给出的DNI、大气透射率、反射率和功率公式；输出包括余弦效率、阴影遮挡效率、大气透射率、截断效率、综合光学效率、单镜功率、镜场功率及其月年聚合。

建模者通过 `q1_method_choice` 选择M1耦合低差异光线追迹作为主方法，通过 `q1_sunshape_weighting_choice` 确认半角4.65 mrad的均匀pillbox太阳盘和60状态等权口径，并通过 `q1_tower_shadow_convention`、`q1_tower_height_revision` 和 `q1_receiver_vertical_interval_revision` 确认塔身与集热器的最终几何口径。

人的选择原话和理由保存在决策账本。本说明不根据结果优劣重新编造选择原因，而只解释已经选定的方法如何完整求解问题。

## 2. 题面事实、必要口径与简化假设

题面直接给出东—北—天顶坐标系、塔位$(0,0)$、直径7 m和高度8 m的集热器、1745面$6\times6$ m镜面、镜心高度4 m以及每月21日五个计算时点。按建模者修订口径，塔身是半径3.5 m、$0\le z<72$ m的实体圆柱，集热器有效受光面是半径3.5 m、$72\leq z\leq80$ m的圆柱侧面，其中心瞄准点为$(0,0,76)$。阴影包括入射光被邻镜或塔身截断；集热器不作为入射阴影体，只承担反射光命中判定。

简化假设包括：镜面为无面形、斜率和跟踪误差的理想平面镜；太阳锥采用半角$4.65\times10^{-3}$ rad的均匀pillbox；五个时点等权形成月平均，十二个月等权形成题面离散年平均；不增加题面没有提供的接收器吸收率。这些假设决定模型适用边界，不能当作题面事实。

## 3. 太阳位置与DNI

定义状态集合

$$
\mathcal{T}=\{(m,h):m=1,\ldots,12;\ h\in\{9,10.5,12,13.5,15\}\}.
$$

对$t=(m,h)$，由每月21日相对春分的天数$D_m$计算赤纬角$delta_m$，由时刻计算时角$omega_h$：

$$
\sin\delta_m=\sin\left(\frac{2\pi D_m}{365}\right)\sin 23.45^\circ,
\qquad
\omega_h=\frac{\pi}{12}(h-12).
$$

高度角满足

$$
\sin\alpha_s=\cos\delta_m\cos\varphi\cos\omega_h+\sin\delta_m\sin\varphi.
$$

题面只给出方位角余弦，故先计算

$$
C_\gamma=\frac{\sin\delta_m-\sin\alpha_s\sin\varphi}{\cos\alpha_s\cos\varphi},
$$

再按

$$
\gamma_s=\begin{cases}
\arccos C_\gamma,&\omega_h\leq0,\\
2\pi-\arccos C_\gamma,&\omega_h>0
\end{cases}
$$

补足象限。镜场指向太阳中心的单位向量为

$$
\boldsymbol{s}=\left(\cos\alpha_s\sin\gamma_s,\ \cos\alpha_s\cos\gamma_s,\ \sin\alpha_s\right)^{\mathsf T}.
$$

海拔$H=3$ km时，题面DNI公式化为

$$
\mathrm{DNI}=1.366\left[0.34981+0.5783875\exp\left(-\frac{0.275745}{\sin\alpha_s}\right)\right]\ \mathrm{kW/m^2}.
$$

$\boldsymbol{s}$负责几何方向，DNI负责能量尺度；DNI已经是垂直太阳光线平面的辐照度，不能再重复投影到水平面。

## 4. 镜面姿态与余弦效率

第$i$面镜的镜心为$\boldsymbol{c}_i=(x_i,y_i,4)$，接收器中心为$\boldsymbol{c}_R=(0,0,76)$。令

$$
\boldsymbol{t}_i=\frac{\boldsymbol{c}_R-\boldsymbol{c}_i}{\|\boldsymbol{c}_R-\boldsymbol{c}_i\|}.
$$

太阳中心光线的传播方向为$-\boldsymbol{s}$。理想镜面反射算子为

$$
\mathcal{R}(\boldsymbol{d},\boldsymbol{n})=\boldsymbol{d}-2(\boldsymbol{d}\cdot\boldsymbol{n})\boldsymbol{n}.
$$

控制系统要求$\mathcal{R}(-\boldsymbol{s},\boldsymbol{n}_i)=\boldsymbol{t}_i$，故镜面朝阳法向唯一确定为

$$
\boldsymbol{n}_i=\frac{\boldsymbol{s}+\boldsymbol{t}_i}{\|\boldsymbol{s}+\boldsymbol{t}_i\|}.
$$

该式不是只计算一次的静态安装角。对固定镜位，$\boldsymbol{t}_i$在一天内不变，而$\boldsymbol{s}(t)$随太阳运动改变，因此每个规定时点都重新计算$\boldsymbol{n}_i(t)$。将法向写成控制系统可读的目标姿态：

$$
\beta_i(t)=\operatorname{mod}\!\left[
\operatorname{atan2}(n_{x,i}(t),n_{y,i}(t)),2\pi\right],
\qquad
\tau_i(t)=\arccos n_{z,i}(t),
$$

其中$\beta_i$是从正北顺时针量取的法向方位角，$\tau_i$是镜面相对水平面的倾角。二者确定目标镜面平面；具体电机轴角依赖题面未给出的支架零位，不在模型中虚构。该姿态使太阳中心光线经镜心反射后指向接收器中心，是题面规定的中心瞄准，并非另行求解接收器瞄准点优化。

余弦效率由此解析得到

$$
\eta_{\cos,i}=\boldsymbol{n}_i\cdot\boldsymbol{s}=\sqrt{\frac{1+\boldsymbol{s}\cdot\boldsymbol{t}_i}{2}}.
$$

为表示真实有限镜面，令$\boldsymbol{k}=(0,0,1)$，构造

$$
\boldsymbol{e}_{w,i}=\frac{\boldsymbol{k}\times\boldsymbol{n}_i}{\|\boldsymbol{k}\times\boldsymbol{n}_i\|},
\qquad
\boldsymbol{e}_{h,i}=\boldsymbol{n}_i\times\boldsymbol{e}_{w,i}.
$$

镜面点集为

$$
\mathcal{M}_i(t)=\{\boldsymbol{c}_i+u\boldsymbol{e}_{w,i}+v\boldsymbol{e}_{h,i}:u,v\in[-3,3]\}.
$$

该有限边界同时用于入射阴影和出射遮挡相交判定。

## 5. 太阳盘方向测度与逐方向反射

在$\boldsymbol{s}$的正交平面构造单位基$\boldsymbol{a},\boldsymbol{b}$，太阳盘方向写为

$$
\boldsymbol{s}'=\boldsymbol{s}\cos\theta+(\boldsymbol{a}\cos\psi+\boldsymbol{b}\sin\psi)\sin\theta,
$$

其中$0\leq\theta\leq\theta_\odot$、$0\leq\psi<2\pi$、$\theta_\odot=4.65\times10^{-3}$ rad。均匀pillbox表示单位立体角内能量相同，故归一化测度为

$$
\mathrm{d}\mu_s=\frac{\sin\theta\,\mathrm{d}\theta\,\mathrm{d}\psi}{2\pi(1-\cos\theta_\odot)}.
$$

数值采样必须使$\cos\theta$均匀，不能使$\theta$均匀。镜面根据太阳中心方向跟踪，但太阳盘中的每个方向都按

$$
\boldsymbol{r}'=-\boldsymbol{s}'+2(\boldsymbol{s}'\cdot\boldsymbol{n}_i)\boldsymbol{n}_i
$$

单独反射。因此，中心光线指向接收器中心并不意味着所有太阳盘方向都能命中有限接收器。

## 6. 统一光线样本空间和三个事件

对镜$i$和时点$t$定义联合样本空间

$$
\Omega_i(t)=\mathcal{M}_i(t)\times\mathcal{S}(t),
\qquad
\mathrm{d}\mu_i=\frac{\mathrm{d}A}{36}\,\mathrm{d}\mu_s.
$$

一个样本$\xi=(\boldsymbol{p},\boldsymbol{s}')$代表太阳盘中一小份能量射向镜面点$\boldsymbol{p}$。

入射阴影从镜面点沿太阳方向回溯：

$$
\boldsymbol{q}_{\mathrm{in}}(\lambda)=\boldsymbol{p}+\lambda\boldsymbol{s}',\quad\lambda>0.
$$

令$J_i(\xi)$表示入射射线不与任何其他有限镜面相交，$T_i(\xi)$表示入射射线不与塔身实体圆柱

$$
\mathcal{C}_T=\{(x,y,z):x^2+y^2\le3.5^2,\ 0\le z<72\}
$$

相交，则联合入射可见指示量为$I_i(\xi)=J_i(\xi)T_i(\xi)$。集热器高度区间不参与入射阴影判定。逐方向反射后，从

$$
\boldsymbol{q}_{\mathrm{out}}(\lambda)=\boldsymbol{p}+\lambda\boldsymbol{r}',\quad\lambda>0
$$

追踪出射光；若它在进入接收器区域前不与邻镜相交，令$O_i(\xi)=1$，否则为0。

接收器侧面为

$$
\mathcal{C}=\{(x,y,z):x^2+y^2=3.5^2,\ 72\leq z\leq80\}.
$$

将射线的$x,y$分量代入圆柱方程得到

$$
(r_x'^2+r_y'^2)\lambda^2+2(p_xr_x'+p_yr_y')\lambda+(p_x^2+p_y^2-3.5^2)=0.
$$

取最小正根并检查交点高度。命中圆柱侧面时$H_i(\xi)=1$，否则为0；顶面和底面不计入接收面。

## 7. 由联合事件定义效率

令$V_i=I_iO_i$。阴影遮挡效率定义为

$$
\eta_{\mathrm{sb},i}=\int_{\Omega_i}V_i(\xi)\,\mathrm{d}\mu_i(\xi),
$$

截断效率必须按未被阻挡的反射光条件化：

$$
\eta_{\mathrm{trunc},i}=
\frac{\int_{\Omega_i}V_i(\xi)H_i(\xi)\,\mathrm{d}\mu_i(\xi)}
{\int_{\Omega_i}V_i(\xi)\,\mathrm{d}\mu_i(\xi)}.
$$

于是严格满足

$$
\eta_{\mathrm{sb},i}\eta_{\mathrm{trunc},i}
=\int_{\Omega_i}V_i(\xi)H_i(\xi)\,\mathrm{d}\mu_i(\xi).
$$

该恒等式是防止重复扣损的核心：一条光线无论与多少邻镜相交，只记录一次布尔损失；截断效率也不能使用全部样本作分母。

## 8. 大气透射、综合效率与功率

令$d_{\mathrm{HR},i}=\|\boldsymbol{c}_R-\boldsymbol{c}_i\|$。附件中全部距离小于1000 m，按题面公式

$$
\eta_{\mathrm{at},i}=0.99321-0.0001176d_{\mathrm{HR},i}+1.97\times10^{-8}d_{\mathrm{HR},i}^2.
$$

镜面反射率为$\eta_{\mathrm{ref}}=0.92$，故

$$
\eta_i=\eta_{\mathrm{sb},i}\eta_{\cos,i}\eta_{\mathrm{at},i}\eta_{\mathrm{trunc},i}\eta_{\mathrm{ref}},
$$

$$
E_i(t)=\mathrm{DNI}(t)\,36\,\eta_i(t),
\qquad
E_{\mathrm{field}}(t)=\mathrm{DNI}(t)\sum_{i=1}^{1745}36\eta_i(t).
$$

由于题面功率公式没有接收器吸收率，本问不再加入题面外参数。

## 9. 月与年的聚合

Q1镜面等面积，故时点平均效率是逐镜算术平均。月平均是当月五个时点的算术平均，题面离散年平均是十二个月的算术平均，即60个状态等权。功率必须先逐时点计算后再平均：

$$
\bar E_{\mathrm{year}}=\frac{1}{60}\sum_{t\in\mathcal{T}}\mathrm{DNI}(t)\sum_i36\eta_i(t).
$$

单位镜面面积年平均功率为

$$
P_A=\frac{\bar E_{\mathrm{year}}}{1745\times36}.
$$

不能用平均DNI与平均效率的乘积替代上述平均，因为二者随时点共同变化。

## 10. M1数值求解器

连续积分包含有限矩形可见性和有限圆柱命中，没有简洁闭式解。M1在镜面局部面积和太阳盘立体角上分别生成随机扰动Sobol低差异点，计算两组样本的笛卡尔积，并在每条联合光线上依次判定$I_i,O_i,H_i$。正式细分辨率为64个镜面点乘32个太阳方向，即每个“镜面—时点—种子”2048条联合光线；种子固定为2023、2024、2025和2026，正式结果取四个种子的算术平均。

为验证离散误差，同时计算32乘16粗分辨率。相同种子的粗样本是细Sobol序列前缀。GPU仅分块并行射线相交，不改变样本、事件或效率定义。60 m邻镜候选半径用于加速，并与全1745镜的相交判定进行分层复核。

## 11. 可用基线与回退规则

B1对镜面采用$5\times5$中点网格，利用太阳中心方向计算阴影遮挡，并以16个太阳盘方向估计条件截断。它共享太阳角、镜面姿态、有限圆柱和功率聚合公式，能够生成完整的题目表格，因此是可用基线而非仅供诊断的参考。

B1的低分辨率和因子化近似不能替代M1。预定回退规则是：若M1未达到收敛阈值，则提高同一方法到128乘64并返回实验判断点；不得把B1自动升级为最终方法。本轮所有阈值通过，回退未触发。

## 12. 验证、稳健性与适用范围

验证覆盖太阳向量单位化、方位象限、反射恒等式、有限圆柱根选择、TensorFlow与NumPy相交计数、邻镜剪枝、效率范围、联合事件分解、功率单位和聚合。嵌套分辨率使年均综合效率变化$6.0874\times10^{-5}$，年均功率相对变化0.0104%，月度单位面积功率最大相对变化0.0599%，均低于预设阈值。四种子年均功率极差为0.008771 MW。

太阳盘半角$\pm5\%$使年均功率改变0.3219%和0.3363%；按月天数加权相对于十二个月等权改变0.0146%，均低于1%返回决策阈值。以上证据说明数值求解对已定义模型稳定，但不证明理想镜面或pillbox太阳盘等同于真实电站。实际工程外推仍需镜面误差、污染、逐日气象和接收器热损失数据。
