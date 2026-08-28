# Q1冻结结果变更记录

## 2026-08-28：加入吸收塔塔身阴影

- 变更原因：建模者要求把塔身建模为高72 m、与集热器同半径的圆柱，并把集热器高度区间修订为72--80 m（中心高度76 m），同时明确集热器本体不计入入射阴影。
- 影响类别：FROZEN。
- 解冻范围：Q1阴影遮挡效率、综合光学效率、镜场功率、月度和年度结果，以及由这些结果生成的图表和论文表述。
- canonical source：`scripts/Q1/q1_optical_model.py`。
- 人工决策：`q1_tower_shadow_convention`、`q1_tower_height_revision`、`q1_receiver_vertical_interval_revision`。
- 后续动作：修改光线追迹、重跑Q1、重新审查、重冻结并执行Q1范围一致性检查。
