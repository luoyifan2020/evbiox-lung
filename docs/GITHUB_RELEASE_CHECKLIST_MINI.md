# GITHUB RELEASE CHECKLIST MINI

## 1) 是否还存在绝对路径
- **结论：存在，需处理。**
- 在 `evbiox-lung/nsclc_swinunetr/configs/` 下大量 YAML 命中绝对路径模式（如 `D:/...`）。
- 在 `evbiox-lung/nsclc_swinunetr/scripts/` 下也有部分脚本命中绝对路径模式。
- 发布建议：将对外保留的核心 config 改为相对路径或占位变量（如 `${PROJECT_ROOT}`）。

## 2) 是否还存在敏感信息
- **结论：未发现明显密钥类高危文本，但仍需人工复核。**
- 轻量关键词扫描仅在 `evbiox-lung/nsclc_swinunetr/scripts/08_extract_radiomics.py` 命中 `token/secret` 相关字样（需人工确认语义是否仅普通变量名）。
- 未在本轮扫描中发现明显私钥块（如 `BEGIN PRIVATE KEY`）或常见 API key 片段。

## 3) README 是否足够让陌生人看懂
- **结论：基础可读，建议再补一层“可执行环境说明”。**
- 现有 `evbiox-lung/README.md` 已说明：项目目标、主流程、公开范围、隐私边界。
- 当前短板：缺少明确依赖清单与最小命令示例（可在下一步补 `requirements.txt` 或 `environment.yml`）。

## 4) 是否有推荐放到 assets/ 的 demo 图片
- **结论：当前未快速检出可直接复用的 PNG 图片文件；建议手工导出 3-5 张。**
- 推荐放入 `demo/assets/images/` 的截图类型：
  - 交互页面总览截图（来自 `interactive_report_v1.0.html`）
  - 3D STL 视图截图（tumor gt/pred 对比）
  - 生存曲线图截图（由 `km_curves_q3.csv` 驱动）
  - 风险分组图截图（由 `risk_groups.csv` 驱动）
  - 单病例指标卡截图（由 `R01-056_tumor_metrics.csv` 驱动）

## 5) 是否需要补 requirements.txt / environment.yml
- **结论：需要，优先级高。**
- 目前 `evbiox-lung/` 下未检测到 `requirements*.txt` 或 `environment*.yml`。
- 建议至少补一个：
  - `requirements.txt`（轻量公开）或
  - `environment.yml`（对研究复现更友好）

## 6) 哪些文件还不适合公开
- 高优先级不公开：
  - `evbiox-lung/outputs/`
  - `evbiox-lung/data/raw/`
  - `evbiox-lung/data/processed/`
  - `evbiox-lung/data/raw_data/`
- 高优先级复核后再公开：
  - `evbiox-lung/nsclc_swinunetr/configs/*.yaml`（绝对路径清理）
  - `evbiox-lung/nsclc_swinunetr/scripts/` 中含绝对路径或本地化硬编码的脚本
- 中优先级：
  - `evbiox-lung/data/roi/`（可能包含可逆定位信息，建议仅保留极小脱敏样例）

---

## GitHub 仓库一句话介绍
A showcase-ready NSCLC pipeline for CT tumor segmentation and multimodal risk fusion, with curated demo artifacts and privacy-first release boundaries.

## README 首页 5 条亮点
- End-to-end method path: extraction -> preprocessing -> datalist -> segmentation -> fusion report.
- Public-showcase structure with minimal, readable documentation.
- Privacy-first policy: no raw medical data or internal paths in release assets.
- Curated demo asset guidance for interactive and 3D visualization.
- Clear separation between core workflow and historical heavy artifacts.

## 建议公开发布前手工再检查的 5 个点
- 逐个检查拟公开的 `configs/*.yaml` 是否仍包含本机绝对路径。
- 抽查核心脚本头部参数，确认无内部用户名/盘符/私网路径。
- 对 demo CSV/STL 做一次去标识复核（字段名、内容、文件名）。
- 在全新目录做一次 README 引导下的“冷启动阅读测试”（陌生人 5 分钟可理解）。
- 用 `git status` + staged diff 做发布前白名单确认，确保无 `data/raw`、`outputs`、大文件误入。
