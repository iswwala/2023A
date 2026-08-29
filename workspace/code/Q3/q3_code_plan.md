# Q3 Python代码与首轮实验契约

## 1. 目标与批准范围

- 目标语言：Python 3.12，环境`/home/iswwala/venvs/tf`。
- 正式问题：Q3；首轮输出目录：`results/Q3/experiments/round1/`。
- 人工决定：`q3_method_choice`。
- 主方法：M3G，从空场独立生成布局的全局粒子群--逐镜连续细化。
- 可用基线：B3，Q2 round17最终方案通过Q3异质接口复算。
- 条件回退：F3；本轮不实现，除非后续出现人工`fallback_activation`记录。

## 2. 输入合同

| 输入 | 单位 | 来源 |
|---|---:|---|
| Q2正式指标、收敛和种子结果 | 多单位 | `results/Q2/experiments/round17_zoned_formal/metrics/q2_zoned_formal.json` |
| 60个太阳状态及光学常数 | 多单位 | `scripts/Q1/q1_optical_model.py` |
| 塔身、接收器及共享追迹逻辑 | 多单位 | `scripts/Q2/q2_optical_model.py` |

主方法不读取Q2镜位。每个粒子从场界、塔位及布局参数生成$(x_i,y_i,z_i,w_i,h_i)$；Q2指标只用于最终对照。

## 3. 异质接口

1. 以数组形式保存$w_i,h_i,z_i$，镜面采样点按各镜自己的宽高缩放。
2. 候选遮挡镜的相交边界使用对应的$w_j/2,h_j/2$，不能使用统一标量。
3. 镜心三维坐标使用各镜$z_i$；法向、镜面局部轴、大气透射率和接收器相交均随其变化。
4. 场功率逐镜计算：
   $$P(t)=\mathrm{DNI}(t)\sum_i w_ih_i\eta_i(t).$$
5. 各效率按面积加权：
   $$\bar\eta_k(t)=\frac{\sum_iw_ih_i\eta_{k,i}(t)}{\sum_iw_ih_i}.$$
6. 异尺寸间距按$d_{ij}\ge(w_i+w_j)/2+5$全量检查。

## 4. 必须先通过的回归

把Q2布局的所有$w_i,h_i,z_i$设为round17统一值，用Q3接口在相同状态、分辨率与种子下复算。年平均效率、功率、逐月指标和四种子结果必须在浮点/事件计数容差内等于Q2。回归不通过时停止M3搜索，保存失败证据并触发返回判断点。

## 5. M3G全局求解

第一层用PSO联合搜索塔位、分区边界、蜂窝/同心环相位与密度、规格空间场和安全离地余量。每个粒子从空场独立生成镜位，不读取或扰动Q2坐标。

1. 快速层使用覆盖四季与低/中/高太阳高度的代表状态和公共种子评价群体。
2. 生成器先建立分区母点阵，再为每面镜按半径和南北位置赋连续$w_i,h_i,z_i$，并按$d_{ij}\ge(w_i+w_j)/2+5$接受或修复候选。
3. 以容量优先的约束排序比较粒子：未达到安全功率时优先提高功率，达到后最大化$P_A$。
4. 多个非重复前列候选进入60状态中精度复核；仅对排序稳定且有容量裕量的候选执行逐镜残差细化。
5. 最终候选使用两级分辨率和四固定种子正式复算；任一种子低于60 MW即判不可行。

首轮用于验证逐镜异质规格是否产生可用信号；宽度、镜位和塔位的联合块更新需在人类查看首轮结果后继续，不在本轮静默扩大范围。

## 6. B3基线

通过同一Q3接口复算Q2最终设计。B3必须保持2983面、塔位$(0,-33.092418)$ m、统一$6.708297\times6.708297$ m、安装高度3.892642 m，并重现round17正式结果。B3与M3使用相同状态、种子、光线采样及面积加权聚合。

## 7. 首轮输出

```text
results/Q3/experiments/round1/
├── figures/
├── tables/
│   ├── q3_baseline_layout.csv
│   ├── q3_main_layout.csv
│   ├── q3_baseline_monthly.csv
│   └── q3_main_monthly.csv
├── metrics/
│   ├── q3_equal_spec_regression.json
│   ├── q3_geometry.json
│   ├── q3_search_trace.json
│   ├── q3_method_comparison.json
│   └── q3_output_degeneracy.json
└── run_summary.json
```

首轮成功运行不等于最终冻结；正式两级四种子复算、宽度/镜位联合细化与人类结果判断仍是后续门禁。

## 8. 风险与回退字段

- 记录逐镜宽、高、安装高度的唯一值、标准差、边界质量和面积集中度。
- 记录快速层与60状态层排序是否反转。
- 记录几何最小松弛、功率裕量、运行时间和预计完整预算。
- F3触发条件沿用方法卡；只记录`observed`，不实现F3。

## 9. 下游审查

Python审查必须包含`syntax`、`input_contract`、`method_alignment`、`reproducibility`和`output_contract`，并额外检查异质边界数组、面积加权恒等式、等规格回归、间距半和公式、逐镜离地以及B3/M3可比性。
