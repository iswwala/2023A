# Q3写作解决方案包

## 写作授权与范围

- 最终方案：round36容量修复后的2939面异质镜场。
- 人工签署：`q3_round36_paper_signoff`。
- 允许主张：方案在记录搜索预算、既定物理模型和正式数值口径下通过全部几何、两级收敛及四种子60 MW检查；是当前最佳已验证可行解。
- 禁止主张：所有连续镜位和逐镜规格组合中的全局最优；对施工、控制和光学误差具有充分工程裕量。

## 顶线数值

| 主张 | 数值 | 冻结ID | 稳健性与限制 |
|---|---:|---|---|
| 塔位 | $(0,-37.426544)$ m | `q3_tower_x_m`, `q3_tower_y_m` | 外层GA连续搜索结果 |
| 镜数与面积 | 2939面，132952.120200 m$^2$ | `q3_mirror_count`, `q3_total_area_m2` | Excel明细2939行 |
| 宽度范围 | $[4.755444,6.8]$ m | `q3_width_range_m` | 搜索边界为$[2,8]$ m，实际范围是结果 |
| 高度范围 | $[4.748198,6.8]$ m | `q3_height_range_m` | 逐镜满足$h_i\le w_i$ |
| 安装高度范围 | $[3.409111,4.822522]$ m | `q3_installation_height_range_m` | 最小离地净空0.05 m |
| 年平均功率 | 60.111866 MW | `q3_field_power_mw` | 最低种子60.080454 MW |
| 单位面积功率 | 0.452132 kW/m$^2$ | `q3_unit_area_power_kw_m2` | 相对Q2基线提高1.1127% |
| 年平均光学效率 | 0.463851 | `q3_eta_total` | $32\times16$四种子池化值 |

## 方法叙述顺序

1. 定义逐镜面积加权功率链和带60 MW硬约束的单位面积目标；
2. 写出场界、禁布区、异尺寸间距、尺寸顺序、安装高度及离地约束；
3. 说明两区三角密排GA的所有连续搜索范围以及镜数如何自然变化；
4. 说明入围布局上的逐镜连续宽、高、安装高度细化；
5. 用round34失败解释正式复核和容量裕量修复的必要性；
6. 报告round36三张正式表、四种子稳定性和结论边界。

## 最终文件

- 冻结数字：`results/Q3/reports/frozen_numbers.json`
- 表1至表3：`results/Q3/reports/tables/`
- 逐镜答案：`results/Q3/reports/result3.xlsx`
- 方法解释：`workspace/methods/Q3/q3_final_method_explanation.md`
- 结果分析：`results/Q3/reports/q3_final_result_analysis.md`
- 稳健性报告：`results/Q3/reports/q3_robustness_report.md`

