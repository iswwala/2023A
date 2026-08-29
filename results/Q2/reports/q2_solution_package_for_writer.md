# Q2写作解决方案包

## 写作授权与范围

- 最终方案：`M2ZT_symmetric_zoned_formal`；决定ID：`q2_round17_final_result_acceptance`。
- 数值包签署：`q2_package_signoff`。
- 基线范围：`q2_baseline_scope_waiver`；不写基线，不报告相对提升。
- 允许主张：方案在既定物理与数值口径下通过全部几何检查、两级收敛检查和四种子60 MW检查。
- 禁止主张：所有逐镜布局中的全局最优、充分工程裕量、相对可用基线提升。

## 顶线数值

| 主张 | 数值 | 冻结来源 | 稳健性与限制 |
|---|---:|---|---|
| 塔位 | $(0,-33.092418)$ m | `q2_tower_x_m`, `q2_tower_y_m` | $x_T=0$且严格东西对称 |
| 镜面规格 | $6.708297\times6.708297$ m | `q2_width_m`, `q2_height_m` | 统一规格；矩形自由度最终退化为方形 |
| 安装高度 | 3.892642 m | `q2_installation_height_m` | 地面净空0.538493 m |
| 镜数与面积 | 2983面，134238.711280 m$^2$ | `q2_mirror_count`, `q2_total_area_m2` | Excel明细行数等于2983 |
| 年平均功率 | 60.025692 MW | `q2_field_power_mw` | 池化裕量0.025692 MW；最小种子60.003732 MW |
| 单位面积功率 | 0.447156 kW/m$^2$ | `q2_unit_area_power_kw_m2` | 当前参数化布局族内接受的边界最优方案 |
| 年平均光学效率 | 0.458734 | `q2_eta_total` | 32×16四种子池化值 |

全部精确值与逐月结果读取`results/Q2/reports/frozen_numbers.json`，论文显示值只能由其统一四舍五入。

## 方法叙述顺序

1. 建立带60 MW硬约束的单位面积功率最大化模型；
2. 用严格东西对称三角密排母池降低镜位维数；
3. 将塔位、统一镜宽高、安装高度、径向分区边界、相位和过渡带纳入粒子群；
4. 先用代表状态筛除，再用60状态复排；
5. 对最终候选进行16×8、32×16和四固定种子正式复算；
6. 说明分区相位与矩形自由度最终退化，不把未采用结构写成结果特征；
7. 报告边界裕量与适用范围。

## 最终文件

- 题目表1至表3：`results/Q2/reports/tables/`
- 逐镜答案：`results/Q2/reports/result2.xlsx`
- 冻结数值：`results/Q2/reports/frozen_numbers.json`
- 最终方法解释：`workspace/methods/Q2/q2_final_method_explanation.md`
- 最终结果分析：`results/Q2/reports/q2_final_result_analysis.md`
- 稳健性：`results/Q2/reports/q2_robustness_report_round17.md`

