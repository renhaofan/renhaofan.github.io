# 添加一篇论文阅读笔记

网站使用 GitHub Pages 原生支持的 Jekyll collection。每篇论文对应 `_papers` 目录中的一个 Markdown 文件；推送到 GitHub 后，论文列表和独立网页会自动生成。

## 推荐流程

1. 把论文 PDF、官方 HTML 或 arXiv 链接交给 Codex。
2. 使用下面的请求：

   ```text
   请使用 paper-reading-notes skill 精读这篇论文，并参考
   _templates/paper-note-template.md 的 front matter 和正文结构，
   将完整中文笔记保存为 _papers/YYYY-MM-DD-short-title.md。
   重要结论需要注明对应表格、图、章节或页码；作者观点和个人分析要明确区分。
   ```

3. 检查 Markdown 文件开头的元数据，尤其是 `title`、`date`、`paper_url`、`tags` 和 `summary`。
4. 提交并推送：

   ```bash
   git add _papers/你的文件名.md
   git commit -m "Add paper note: 论文简称"
   git push
   ```

5. GitHub Pages 构建完成后：

   - 列表页：`https://renhaofan.github.io/papers/`
   - 单篇页：`https://renhaofan.github.io/papers/完整文件名去掉扩展名/`

## 文件命名

推荐使用 `YYYY-MM-DD-short-title.md`，例如：

```text
_papers/2026-08-29-comvs-gs.md
```

Jekyll 会把集合文件的完整文件名作为默认页面地址的一部分。若希望地址始终简短稳定，可在 front matter 中额外设置：

```yaml
permalink: /papers/comvs-gs/
```

## 图片和公式

- 图片放入 `images/papers/<short-title>/`，Markdown 中用站点根路径引用，例如 `/images/papers/comvs-gs/pipeline.png`。
- 行内公式使用 `$...$`，独立公式使用 `$$...$$`；单篇论文页面已接入 MathJax 渲染 LaTeX。行内公式中的星号请写成 `\ast` 或 `\star`，不要直接写 `*`，以免被 Markdown 解析成斜体。
- 不要提交没有授权公开传播的论文 PDF 或图片；优先链接官方论文地址，并使用自己绘制或获得授权的示意图。
