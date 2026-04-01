# NeurIPS Paper Starter

This folder is a paper workspace for the current project.

What is included:
- `main.tex`: paper entry point with a NeurIPS-style preamble
- `sections/`: section stubs for the main paper
- `references.bib`: starter bibliography for the current literature set
- `notes/related_papers.md`: curated follow-up reading list
- `notes/methodology_outline.md`: methodology outline derived from the repository

Notes:
- The TeX toolchain is not installed in this environment, so the project was not compiled here.
- `main.tex` is set up to use the official `neurips_2026.sty` if you place that file in this folder.
- If the style file is not present yet, the document falls back to a simple article layout so the text remains editable.

Suggested next steps:
- Add the official `neurips_2026.sty` from the NeurIPS/Overleaf template.
- Move the methodology outline into `sections/method.tex` as the implementation stabilizes.
- Expand `references.bib` with any papers you decide to cite directly.
