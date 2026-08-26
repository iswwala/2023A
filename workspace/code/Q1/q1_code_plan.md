# Q1 Python代码与实验契约

> 模型边界：本文件只规定如何数值求解 `workspace/analysis/Q1/modeling_logic.md` 中已经定义的连续物理模型。M1、B1、Sobol采样和GPU分块均不是物理假设，不能反向改变阴影、遮挡、截断或功率的定义。

## 1. 实现边界

- 目标语言：Python 3.12，统一环境 `/home/iswwala/venvs/tf/bin/python`。
- 正式轮次：`round1`。
- 批准决定：`q1_method_choice`；主方法为 `M1` 耦合低差异光线追迹。
- 可用基线：`B1` 因子化规则网格光线追迹。
- 假设口径：`q1_sunshape_weighting_choice` 的2A，以及暂行的 `q1_optics_convention_choice` 3A。
- 不设置或实现独立回退方法。若M1未达到收敛，只提高M1分辨率并重新运行；B1不得自动升级为正式方法。

## 2. 文件与数据契约

### 2.1 输入

唯一数据输入为 `workspace/data_clean/Q1/heliostat_positions.csv`，必须包含：

| 字段 | 类型 | 单位 | 约束 |
|---|---|---|---|
| `heliostat_id` | 整数 | 1 | 唯一、连续，共1745个 |
| `x_m` | 浮点 | m | 东向坐标 |
| `y_m` | 浮点 | m | 北向坐标 |
| `z_m` | 浮点 | m | Q1全部等于4 |

固定参数为纬度$39.4^\circ$、镜宽和镜高6 m、集热器中心$(0,0,80)$ m、圆柱半径3.5 m、侧面高度$z\in[76,84]$ m、反射率0.92、太阳盘半角4.65 mrad。月份日期偏移与五个时点严格使用题面口径。

### 2.2 可执行脚本

使用一个共享实现 `scripts/Q1/q1_optical_model.py`，通过命令行选择 `validate`、`baseline`、`main`、`sensitivity` 或 `all`。共享太阳位置、镜面姿态、射线相交、聚合和输出函数，禁止为M1与B1复制两套物理公式。

### 2.3 round1输出

```text
results/Q1/experiments/round1/
├── figures/
├── tables/
│   ├── q1_main_monthly.csv
│   ├── q1_main_annual.csv
│   ├── q1_baseline_monthly.csv
│   ├── q1_baseline_annual.csv
│   └── q1_main_per_heliostat_state.csv.gz
├── metrics/
│   ├── q1_validation.json
│   ├── q1_convergence.json
│   ├── q1_sensitivity.json
│   └── q1_method_comparison.json
└── run_summary.json
```

表格和JSON是正式输出；成功运行不保留完整控制台日志。失败、警告或复现异常时才建立 `logs/`。

## 3. 共享物理计算

1. 生成12个月乘5时点共60个太阳状态，依次计算$\delta,\omega,\alpha_s,\gamma_s,\boldsymbol{s}$与DNI；方位角上午取反余弦分支、下午取$2\pi-\arccos$分支。
2. 对每个时点和每面镜计算镜心到接收器中心方向$\boldsymbol{t}_i$、法向$\boldsymbol{n}_i$、水平宽度轴$\boldsymbol{e}_{w,i}$和镜面高度轴$\boldsymbol{e}_{h,i}$。
3. 镜面面积点写为$\boldsymbol{p}=\boldsymbol{c}_i+u\boldsymbol{e}_{w,i}+v\boldsymbol{e}_{h,i}$。所有矩形相交均同时检查正向射线参数与$u,v\in[-3,3]$。
4. 入射阶段从$\boldsymbol{p}$沿指向太阳的方向回溯；先与邻镜相交即记阴影。出射阶段先按$\boldsymbol{r}'=-\boldsymbol{s}'+2(\boldsymbol{s}'\cdot\boldsymbol{n}_i)\boldsymbol{n}_i$反射，再检查到接收器之前是否与邻镜相交。
5. 接收器命中只计有限圆柱侧面：求$x^2+y^2=3.5^2$的正根，并要求交点$z\in[76,84]$。不计顶面、底面，也不额外乘接收器吸收率。
6. 60 m邻域剪枝用于加速，但必须通过分层样本与全1745镜相交结果对照；阴影遮挡效率差不得超过$10^{-4}$。
7. 每条光线仅保存`incoming_clear`、`outgoing_clear`和`receiver_hit`布尔量。令$V$为前两者之交、$H$为圆柱命中，则
   $\eta_{sb}=P(V)$，$\eta_{trunc}=P(H\mid V)$，并检查$\eta_{sb}\eta_{trunc}=P(V\cap H)$。
8. 余弦效率和大气透射率不通过抽样估计：$\eta_{cos}=\boldsymbol{n}_i\cdot\boldsymbol{s}$；$\eta_{at}$严格使用题面给定的$d_{HR}$二次式。
9. 单镜综合效率为$\eta_i=\eta_{sb}\eta_{cos}\eta_{at}\eta_{trunc}\times0.92$；时点功率为$\mathrm{DNI}\sum_i36\eta_i$ kW。

## 4. M1主方法

对每面镜、每个时点、每个固定种子，分别生成二维scrambled Sobol镜面点和二维均匀立体角pillbox方向，并计算两组样本的笛卡尔积。效率在同一批联合光线上统计，禁止先独立平均阴影与截断后相乘。

- 固定种子：`2023, 2024, 2025, 2026`。
- 粗分辨率：32个镜面点乘16个太阳方向，共512条联合光线/镜时种子。
- 细分辨率：64个镜面点乘32个太阳方向，共2048条联合光线/镜时种子。
- 正式M1值：四个细分辨率种子估计的算术平均；种子标准差单独保存。
- 嵌套要求：相同种子的粗级样本必须是细级Sobol序列的前缀。
- 执行：按太阳时点、目标镜和射线块分批；优先使用TensorFlow GPU，批大小由8 GB显存实测确定，不得一次构造全场全部射线张量。
- 条件加密：若聚合收敛阈值不通过，增加到128镜面点乘64太阳方向，并只重跑未收敛的正式M1链条。

## 5. B1可用基线

镜面使用$5\times5$中点网格。阴影和遮挡只用太阳中心方向计算；在已通过中心方向可见性检查的网格点上，再用16个固定低差异太阳盘方向估算截断。B1仍需输出与M1完全相同的单镜效率字段、月表、年表和功率单位，但在结果中明确标记为 `usable_baseline`。

## 6. 聚合与直接比较

- 月平均：当月五个题面时点算术平均。
- 年平均：12个月算术平均，即60个时点各权重$1/60$。
- 镜场平均效率：同尺寸Q1中对1745面镜等权；同时由单镜效率重算总功率核对能量恒等式。
- 表1字段：月份、平均光学效率、平均余弦效率、平均阴影遮挡效率、平均截断效率、单位面积镜面平均输出热功率（kW/m$^2$）。
- 表2字段：年平均光学效率、余弦效率、阴影遮挡效率、截断效率、年平均输出热功率（MW）、单位面积镜面年平均输出热功率（kW/m$^2$）。
- M1与B1比较：逐月和全年报告五项效率及功率的绝对差；另报告单镜时点综合效率绝对差的均值、95%分位数和最大值。

## 7. 必须监控的风险和验证

1. 60个太阳高度角与DNI全部为正，太阳向量范数误差不超过$10^{-12}$。
2. 中心光线反射方向与$\boldsymbol{t}_i$最大残差不超过$10^{-12}$，中心光线圆柱命中率为1。
3. 所有效率位于$[0,1]$，输出无NaN或无穷值；联合能量分解最大残差不超过$10^{-12}$。
4. 60 m剪枝在分层镜时样本上的$\eta_{sb}$差不超过$10^{-4}$。
5. 粗细级年平均光学效率绝对差不超过0.002；年平均功率相对差不超过0.3%；任一月单位面积功率相对差不超过0.5%。
6. 保存四种子年平均结果的均值、标准差和范围，不以单种子结果冻结。
7. 敏感性必须包含太阳盘半角$\pm5\%$及按月天数加权。任一情景使年均功率变化超过1%时，把核心结论标为条件成立并回到人类假设判断点。
8. 记录实际墙钟时间、峰值GPU显存或无法取得该指标的原因、设备名和依赖版本。

## 8. run_summary与回退字段

`run_summary.json`必须记录问题、轮次、实现目标、四个种子、批准决定ID、M1与B1的角色、脚本、状态、运行时间、输入输出、关键指标、警告和错误。`fallback_trigger`固定为：

```json
{
  "fallback_id": null,
  "condition": "M1未达到收敛时提高同一方法分辨率并返回实验判断点；B1不得自动替代M1",
  "observed": false,
  "evidence": "results/Q1/experiments/round1/metrics/q1_convergence.json"
}
```

## 9. 下游代码审查命名检查

Python审查文件应放入 `workspace/code/Q1/reviews/q1_python_review.json`，并逐项给出：

- `syntax`
- `input_contract`
- `method_alignment`
- `reproducibility`
- `output_contract`

审查还必须核对公式方向、圆柱有限高度、遮挡截止距离、单位换算和M1/B1角色未发生互换。
