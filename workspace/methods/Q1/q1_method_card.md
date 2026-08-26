# Q1 Method Card

## Goal and success criteria

在题面给定的1745面、$6\times6$ m、安装高度4 m的镜场上，计算60个规定时点的五项光学效率、热功率、月平均和年平均结果，并生成表1、表2及可复核的几何解释。成功标准为：所有效率合法；太阳角、反射和能量恒等式通过；阴影遮挡不重复计损；截断结果达到明确的数值收敛阈值；正式结果与论文完全同源。

## Human constraints

- Output form: `workspace/analysis/Q1/`保存详细推导、验证和图表证据；论文Q1正文完整展示建模过程、公式、结果和限制。
- Priority: `1A`，物理精度优先。
- Unacceptable failure: `2A`，物理或几何口径错误不可接受。
- Experiment budget: `3D`，RTX 4060 Laptop GPU 8 GB可用；用户未指定墙钟时间上限。
- Decision evidence: `workspace/methods/Q1/q1_decisions.jsonl#q1_pre_method_preferences`。

## Shortlist

| ID | Role | Mathematical idea | Why eligible | Main risk | Implementation cost |
|---|---|---|---|---|---|
| M1 | main_candidate | 耦合低差异光线追迹：对镜面面积与太阳盘方向联合采样，逐条做有向矩形阴影/遮挡相交和有限圆柱侧面命中，按联合概率分解$\eta_{sb}$与$\eta_{trunc}$ | 直接对应题面锥光、反射和截断定义；可用布尔光线掩码消除重复计损；可给出分辨率收敛证据 | 高分辨率CPU代价高；太阳锥能量分布和采样分辨率会影响局部截断 | 高；需GPU分块、低差异采样、收敛与多精度复算 |
| B1 | usable_baseline | 因子化规则网格光线追迹：镜面$5\times5$中点网格计算中心太阳方向的阴影遮挡，再用16条太阳盘光线估算可见网格点的圆柱截断 | 能完成全部Q1输出；与M1使用相同太阳角、镜面姿态、接收器和功率公式，结果可直接比较；CPU全量估算约3.2分钟 | 因子化和低分辨率造成局部损失边界量化粗糙；探针中单镜时点总效率与M1最大差约0.099 | 低；适合作为可用基线和回归测试 |

未设置独立条件回退方案。若M1不能达到收敛或GPU规模要求，应回到人类决策点调整精度/方法，不能静默把B1升级为最终方法。

## Baseline validity

- Real task completed: 是。B1能给出每镜、每时点的$\eta_{cos},\eta_{sb},\eta_{at},\eta_{trunc},\eta$与功率，并按题面聚合为表1和表2。
- Comparable output/metric: 是。B1与M1仅在阴影/截断数值积分口径和分辨率上不同，其他输入、物理公式、单位与聚合完全相同。
- Limitation: B1只用于基线比较、回归和异常定位；在当前“物理几何错误不可接受”的约束下，不能在未通过高精度复算时支撑最终主张。

## Risk-probe summary

| ID | Executability | Data/assumptions | Degeneracy | Sensitivity | Scale | Verdict |
|---|---|---|---|---|---|---|
| M1 | PASS：216个分层镜时组合，8.54 s | PASS/CONDITIONAL：反射残差$8.20\times10^{-16}$，剪枝与全场一致；sunshape待确认 | PASS：总效率216个六位小数唯一值，标准差0.134 | CONDITIONAL：粗细分辨率平均差0.0060、最大0.0197；锥角$\pm5\%$最大跨度0.00839 | CONDITIONAL：探针分辨率CPU全量估算4139.95 s；高分辨率需GPU分块 | CONDITIONAL |
| B1 | PASS：216个分层镜时组合，0.397 s | CONDITIONAL：低分辨率因子化近似 | PASS：总效率213个六位小数唯一值，标准差0.137 | CONDITIONAL：与M1平均绝对差0.0137、最大0.0989 | PASS：CPU全量估算192.65 s | CONDITIONAL（可用基线） |

完整证据见 `workspace/methods/Q1/probes/risk_probe_summary.json` 和 `risk_probe_raw_metrics.json`。

## Required final checks if M1 is chosen

1. 60个太阳时点全部满足$\alpha_s>0$、DNI为正、太阳向量单位化。
2. 中心光线反射方向与镜心到接收器中心方向最大残差不超过$10^{-12}$。
3. 对分层镜时样本，用全1745镜检查复核60 m候选剪枝，$\eta_{sb}$差不超过$10^{-4}$。
4. 使用至少两个嵌套分辨率；年平均光学效率绝对变化不超过0.002，年平均功率相对变化不超过0.3%，任一月单位面积功率相对变化不超过0.5%。
5. 使用至少4个固定scramble种子；报告年均结果的种子离散度，不以单次随机结果冻结。
6. pillbox半角$\pm5\%$和按月天数加权作为敏感性；若年均功率变化超过1%，核心结论标为条件成立并返回假设决策点。
7. B1与M1的差异按月份和空间位置解释，不只报告一个全场均值。

## Approved method and assumption state

- 正式主方法：`1A`，即M1耦合低差异光线追迹；B1保持可用基线角色。
- 太阳锥与聚合口径：`2A`，即A5采用半角4.65 mrad均匀pillbox太阳盘，A7采用五时点与十二个月等权；两者按方法卡执行敏感性检查。
- 太阳位置实现：A1、A2随M1按题面公式和已定义坐标系实现，并由对称性与向量恒等式机械核验。
- 其余光学口径：A3、A4、A6、A8按 `q1_optics_convention_final` 采用3A正式口径。

## Model-solver boundary

Q1的物理模型以 `workspace/analysis/Q1/modeling_logic.md` 中定义的连续联合样本空间、光路事件和能量关系为准。M1与B1均是该模型的数值求解器：M1用耦合低差异光线追迹估计连续积分，B1用低分辨率因子化近似提供对照。采样数、GPU分块和运行时间不参与物理效率的定义，也不能用于倒推或修改模型口径。

## Compact history

- 2026-08-26：建立M1高精度候选与B1可用基线；依据 `q1_pre_method_preferences` 把物理几何恒等式设为硬检查。
- 2026-08-26：建模者对第3项光学口径“暂时先按A”；记录为 `q1_optics_convention_choice`，方法与sunshape/年均口径在当时仍待决定。
- 2026-08-26：建模者选择1A与2A；`q1_method_choice` 确认M1为正式主方法，`q1_sunshape_weighting_choice` 确认A5与A7，Q1达到G2.5。
- 2026-08-26：建模者要求开始撰写Q1论文，正文须完整、严谨、逐步说明建模理由，所有正式图表使用中文；记录为 `q1_writer_handoff_request`。该请求不替代尚缺的结果冻结决定。
- 2026-08-26：建模者选择结果卡A，接受M1 round1与当前稳定性证据，将3A转为Q1正式口径，并把主张限制在60个代表时点、理想平面镜和4.65 mrad均匀太阳盘范围内；同时授权绘制示意图。
