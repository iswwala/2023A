# 2023 A题论文框架

本目录依据 `workspace/raf_paper/` 的 CUMCM LaTeX 工程构筑，但不复用其中的 2024A 正文、数值、图片或程序。

## 编译

在本目录运行：

```bash
latexmk -xelatex -interaction=nonstopmode main.tex
```

## 填充边界

- 方法段落来自 `workspace/methods/Qx/qx_final_method_explanation.md`。
- 数值只来自 `results/Qx/reports/frozen_numbers.json`。
- 结果解释、贡献和置信范围来自人类决策记录。
- 图片只使用通过渲染验证的 Type 2--4 图。
- 文中的“待证据完成”框是框架接口，提交前必须全部消除。
