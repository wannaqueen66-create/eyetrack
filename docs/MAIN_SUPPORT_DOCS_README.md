# Main Support Docs README

这份说明页用于解释 `--build_main_support_docs` 自动生成的辅助文件分别做什么。

## 1. `main_branch_results_manifest.json`
结果包的统一元数据入口。
适合被脚本读取，也适合人工快速确认：
- 当前结果根目录
- 主/补充 outcome 列表
- 全样本 / QC后 两条轨道的关键入口路径

## 2. `MAIN_BRANCH_FIGURE_CAPTIONS.md`
主线 figure caption 草稿（中英双语）。
适合直接拿去做论文图注初稿，再按实际结果微调。

## 3. `main_branch_figure_captions.json`
与图注草稿对应的结构化 JSON。
适合后续自动拼装 figure index / caption packet。

## 4. `MAIN_BRANCH_PACKET_SUMMARY.md`
给作者/合作者快速回忆：
- 主显著性 packet 在哪里
- 哪些 outcome 是 primary
- 哪些 outcome 是 supplement/exploratory
- 每个 outcome 建议按什么顺序读

## 5. `MAIN_BRANCH_WRITING_GUIDE.md`
主线结果写作提纲。
适合在开始写 Results 章节前快速定段落结构和引用顺序。

## 6. `figure_pack_main_branch/`
从当前结果包里复制出的轻量图包。
适合：
- 整理候选正文图
- 快速发给合作者预览
- 后续继续扩展 caption / usage map

## 推荐使用顺序
1. 先读 `README_研究输出说明.txt`
2. 再看 `MAIN_BRANCH_PACKET_SUMMARY.md`
3. 然后看 `MAIN_BRANCH_WRITING_GUIDE.md`
4. 接着看 `figure_pack_main_branch/FIGURE_PACK_INDEX.md`
5. 最后用 `MAIN_BRANCH_FIGURE_CAPTIONS.md` 写图注或汇报材料
