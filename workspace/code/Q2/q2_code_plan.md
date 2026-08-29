# Q2 Python代码与实验契约

> 模型边界：本文件只实现已经由建模者批准的M2主方法和B2可用基线。风险探针数值用于确定搜索邻域，不作为正式结果；正式数值必须由本契约规定的耦合低差异光线追迹重新计算。

> Round2修订：`q2_method_choice_revision_size_layout_search` 已将主方法更新为M2R。round1及其数值只作为历史候选；round2必须先完成全允许区间规格粗筛、多个候选精筛和分区初始布局搜索，才可形成新的正式候选。

## 1. 实现边界

- 目标语言：Python 3.12，环境 `/home/iswwala/venvs/tf/bin/python`。
- 正式轮次：`round1`。
- 批准决定：`q2_method_choice`；主方法为M2，可用基线为B2。
- 可执行代码按仓库规则放在 `scripts/Q2/`；本计划属于中间材料，放在 `workspace/code/Q2/`。
- 不实现独立回退方法。若M2不能通过正式功率、收敛或稳定性检查，返回人类判断点；B2不得自动升级为主方法。

## 2. 输入与固定口径

Q2没有需要清洗的外部数值数据。输入由题面固定参数、Q1已核验太阳状态和Q2批准设计域组成：

| 输入 | 单位 | 正式取值或约束 |
|---|---:|---|
| 建设场地半径 | m | 350 |
| 塔周禁布半径 | m | 100，按镜心判断 |
| 吸收塔高度 | m | 80 |
| 集热器 | m | 半径3.5，圆柱侧面高度76至84 |
| 镜面规格 | m | 统一宽高，$2\le h\le w\le8$ |
| 安装高度 | m | 统一且$2\le z_H\le6$，同时$z_H-h/2>0$ |
| 镜间距 | m | 镜心距离不小于$w+5$ |
| 时间状态 | 1 | 12个月乘5时点，共60状态且等权 |
| 太阳盘 | rad | 半角$4.65\times10^{-3}$的均匀pillbox |
| 固定种子 | 1 | 2023、2024、2025、2026 |

所有设计先生成绝对场地坐标，再平移到以塔为原点的光学坐标计算镜面法向、塔镜距离和接收器相交；场界仍在绝对坐标中检查。

## 3. B2可用基线

- 塔位$(0,0)$ m；统一镜面$7\times7$ m；安装高度4 m。
- 采用最小中心距12 m的确定性六角交错点阵，以塔为中心生成，再按建设场界和100 m禁布区裁剪。
- 预期镜数2832。代码必须重新生成并验证，不读取探针中的镜位列表。
- 使用与M2完全相同的正式耦合评价器、种子和收敛判据。

## 4. M2主方法

### 4.1 参数化合法布局

正式round1使用风险筛查选出的局部最优结构作为搜索入口：塔位$(0,-40)$ m，统一$7\times7$ m镜面，安装高度4 m，点阵间距系数1.01，即实际中心距12.12 m。点阵生成同时支持塔位、尺寸、方向、间距和横纵相位参数，所有非法候选在光学评价前拒绝。

### 4.2 多保真筛选和删镜修复

1. 对完整参数化镜场用耦合低差异快速层计算60状态逐镜年平均贡献。
2. 按逐镜年平均功率从高到低排序，保留快速层估计功率达到61.2 MW的最小前缀；61.2 MW沿用已批准风险筛查的修复目标，不是最终可行性阈值或固定误差校正。
3. 删除后重新计算全部剩余镜面的遮挡和截断，禁止把原贡献简单求和作为最终功率。
4. 若正式结果不能达到60 MW或功率裕量不足，则记录触发状态并返回判断点；不得以快速层结果替代正式结果。

## 5. 正式耦合光学评价

对每面镜、每个时点和每个种子，分别生成scrambled Sobol镜面点与均匀pillbox太阳方向并取笛卡尔积。在同一联合样本上统计入射阴影、出射遮挡和有限圆柱命中：

- 粗分辨率：16个镜面点乘8个太阳方向；
- 细分辨率：32个镜面点乘16个太阳方向；
- 四个种子分别计算，正式值为细分辨率联合事件计数的种子池化结果；
- 同种子粗细Sobol序列嵌套；
- 镜面边界使用动态半宽$w/2$和半高$h/2$，禁止沿用Q1中只适用于6 m方镜的3 m常数；
- M2和B2均按同一方式计算$\eta_{sb}$、$\eta_{cos}$、$\eta_{at}$、$\eta_{trunc}$、$\eta_{total}$、场功率和单位面积功率。

风险探针运行后确认当前WSL进程不存在 `/dev/dxg`，`nvidia-smi` 报告GPU访问被操作系统阻断，因此RTX 4060无法用于本轮。上述分辨率是在保留双层、四种子和全部60状态的前提下按1至2小时预算缩减；是否足够由同一收敛阈值判断，不能因设备不可见而放宽阈值。

## 6. round1输出

```text
results/Q2/experiments/round1/
├── figures/
│   ├── q2_layout_comparison_diagnostic.png
│   └── q2_monthly_comparison_diagnostic.png
├── tables/
│   ├── q2_main_layout.csv
│   ├── q2_baseline_layout.csv
│   ├── q2_main_monthly.csv
│   ├── q2_main_annual.csv
│   ├── q2_baseline_monthly.csv
│   └── q2_baseline_annual.csv
├── metrics/
│   ├── q2_validation.json
│   ├── q2_design_selection.json
│   ├── q2_convergence.json
│   └── q2_method_comparison.json
└── run_summary.json
```

正式round1不直接写 `result2.xlsx`。Excel、论文表1至表3和冻结数值必须在结果稳定性经人工判断后，由同一解决方案包生成。

## 7. 验证与风险条件

1. 太阳向量、中心反射方向、有限圆柱中心光命中和TensorFlow/NumPy小样本计数通过Q1同等级回归检查。
2. 动态$7\times7$ m边界必须通过专门相交测试，并验证$6\times6$ m退化结果与Q1内核一致。
3. 所有镜心位于350 m建设圆内、距塔至少100 m；最近镜心距离不小于$w+5$；最低旋转包络离地严格为正。
4. 所有效率有限且在$[0,1]$；检查$\eta_{sb}\eta_{trunc}=P(V\cap H)$。
5. 粗细年平均综合效率绝对差不超过0.002，场功率相对差不超过0.3%，任一月单位面积功率相对差不超过0.5%。
6. 报告四种子细分辨率功率的标准差和范围。
7. M2正式年平均功率必须不低于60 MW；同时记录相对60 MW的裕量。
8. 保存单位面积指标差值及相对改进；只有其超过数值离散和后续扰动最不利变化时，才允许形成改进结论。
9. 快速层与完整层秩相关0.486的探针风险保持生效：快速层只淘汰，正式结论只引用完整60状态细分辨率值。
10. 记录TensorFlow是否实际识别RTX 4060；GPU不可见只作为运行警告，不改变物理计算。

## 8. run_summary与回退字段

`run_summary.json`记录批准决定ID、M2与B2角色、脚本、运行状态、输入输出、种子、环境、关键指标、退化检查、警告和错误。回退字段固定为：

```json
{
  "fallback_id": null,
  "condition": "M2正式功率低于60 MW、粗细收敛失败或正式单位面积优势不超过数值与扰动不确定性时，返回人类判断点；B2不得自动替代M2",
  "observed": false,
  "evidence": "results/Q2/experiments/round1/metrics/q2_convergence.json"
}
```

## 9. 下游Python审查

审查文件放在 `workspace/reviews/Q2/q2_python_review.json`，必须逐项检查：

- `syntax`
- `input_contract`
- `method_alignment`
- `reproducibility`
- `output_contract`

附加检查包括动态镜面边界、塔位坐标平移、删镜后重新评价、60 MW硬约束、单位换算以及M2/B2评价口径一致性。

## 10. Round2 M2R分层搜索契约

### 10.1 粗筛

- 镜宽覆盖$[2,8]$ m，镜高覆盖$[2,w]$ m，粗筛默认步长1 m。
- 对每个$(w,h)$至少检查最低安全安装高度、一个中间高度和6 m上界中的非重复合法值。
- 先以镜数、面积、平均DNI和效率上界进行容量预筛；上界不足60 MW的候选可不进入光线追迹，但必须记录淘汰原因。
- 光学粗筛使用四季、上午/正午/下午共12个代表状态、固定公共种子和同一“密排交错母池＋径向/方位分区”初始布局。
- 入围集合同时按单位面积指标、容量边界距离和规格多样性生成，不能只保留一个第一名。
- 入围时按$w<6$、$6\le w<7$和$w\ge7$三个尺度层强制保留容量代表，使6 m以下已评价规格不会仅因快速功率较低而全部退出精筛。

### 10.2 精筛和完整复核

- 对多个粗筛入围规格使用小于1 m的非整数步长生成合法邻域。
- 精筛仍只负责压缩候选数；至少保留5个互异候选进入完整60状态快速复核。
- 完整复核按60状态重新排序，不使用快速层固定偏差修正。
- 正式高精度round2只复算完整复核中的少量前列候选，并继续采用16乘8、32乘16和四固定种子。

### 10.3 初始位置策略

- 新生成器先构造满足$d_{ij}\ge w+5$的密排交错母池，再按三径向区和三方位类别执行确定性稀疏化；所有分区数值是待筛参数，不是文献常数。
- 生成后必须逐项验证固定350 m场界、塔周100 m禁布区、$d_{ij}\ge w+5$及严格正离地。
- 分区布局与旧统一六角交错布局至少保留一组同规格对照，以判断位置先验是否真正改善指标。
- 后续位置细化只搜索塔位、分区边界、径向/方位保留比例、点阵方向和相位；不得在本轮直接开放数千个独立镜位变量。

### 10.4 Round2输出

```text
results/Q2/experiments/round2/
├── tables/
│   ├── q2_coarse_size_candidates.csv
│   ├── q2_fine_size_candidates.csv
│   ├── q2_full_recheck_candidates.csv
│   ├── q2_targeted_refine_candidates.csv
│   ├── q2_targeted_refine_full_rechecks.csv
│   ├── q2_formal_candidate_comparison.csv
│   ├── q2_formal_candidate_1_layout.csv
│   ├── q2_formal_candidate_2_layout.csv
│   ├── q2_formal_candidate_3_layout.csv
│   └── q2_shortlisted_layout_*.csv
├── metrics/
│   ├── q2_staged_search_summary.json
│   ├── q2_targeted_refine_summary.json
│   ├── q2_formal_shortlist.json
│   ├── q2_formal_convergence.json
│   ├── q2_formal_candidate_comparison.json
│   └── q2_layout_strategy_checks.json
└── run_summary.json
```

Round2不得覆盖round1文件。新的正式结果、稳健性和人工结果判定完成前，`freeze`和`paper_writing`保持禁止。
