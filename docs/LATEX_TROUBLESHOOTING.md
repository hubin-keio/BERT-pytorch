# LaTeX Compilation Fix Guide

## Error: "LaTeX Error: File `utf-8.def' not found"

This error occurs when your LaTeX installation doesn't have UTF-8 encoding support.

---

## **Solution 1: Use the Simplified LaTeX File (RECOMMENDED)**

I've created `BERT_Architecture_Simple.tex` - a version with minimal dependencies.

```bash
cd docs
pdflatex BERT_Architecture_Simple.tex
```

**Why this works:**
- Removes the `inputenc` UTF-8 requirement
- Uses only standard LaTeX packages
- Should compile on any TeX distribution

---

## **Solution 2: Install Missing LaTeX Packages**

If you want to compile the original file, install the required packages:

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y \
  texlive-latex-base \
  texlive-fonts-recommended \
  texlive-latex-extra \
  texlive-lang-english
```

### macOS (with Homebrew)
```bash
brew install basictex
sudo tlmgr update --self
sudo tlmgr install collection-basic collection-fontsrecommended
```

### Windows
Download and install one of:
- [MiKTeX](https://miktex.org/) (automatic package installation)
- [TeX Live](https://www.tug.org/texlive/)

---

## **Solution 3: Use Online Overleaf Editor**

No installation needed!

1. Go to https://www.overleaf.com/
2. Create new project → "Upload Project"
3. Upload `docs/BERT_Architecture.tex`
4. Overleaf automatically compiles to PDF
5. Download the PDF from the menu

---

## **Solution 4: Fix the UTF-8 Declaration**

If you have LaTeX installed but missing UTF-8 support, modify line 2:

**Change from:**
```latex
\usepackage[utf-8]{inputenc}
```

**Change to:**
```latex
\usepackage[utf8]{inputenc}
```

Note: `utf8` (no hyphen) instead of `utf-8`.

---

## **Solution 5: Remove inputenc Package Entirely**

For maximum compatibility, remove UTF-8 declaration:

**Find this line:**
```latex
\usepackage[utf-8]{inputenc}
```

**Delete it entirely**, and add this instead:
```latex
% UTF-8 encoding assumed by modern LaTeX
```

---

## **Quick Reference**

| Option | Setup Required | Result |
|--------|-----------------|--------|
| **Simplified .tex** | None | PDF via pdflatex |
| **Install packages** | Yes (15-30 min) | Full PDF support |
| **Overleaf online** | None (browser) | PDF via web |
| **Fix UTF-8 decl** | Minimal | Works if LaTeX exists |

---

## **Recommended Steps** (in order)

### Step 1: Try Simplified Version
```bash
cd /home/user/BERT-pytorch/docs
pdflatex BERT_Architecture_Simple.tex
```

If this works → Done! You have PDF

### Step 2: If Above Fails, Use Overleaf
- Browser-based, no installation
- Automatic PDF generation
- Most reliable method

### Step 3: If You Want Local Installation
```bash
# Linux
sudo apt-get install texlive-latex-base texlive-latex-extra

# macOS
brew install basictex
```

Then compile:
```bash
pdflatex BERT_Architecture.tex
```

---

## **Verify Successful Compilation**

After running pdflatex, you should see:

```
This is pdfTeX, Version 3.14159...
...
Output written on BERT_Architecture.pdf...
```

And the file `BERT_Architecture.pdf` will be created.

---

## **Files Available**

| File | UTF-8? | Complexity | Notes |
|------|--------|-----------|-------|
| `BERT_Architecture.tex` | Yes | Full features | Original comprehensive version |
| `BERT_Architecture_Simple.tex` | No | Minimal | Easier to compile, still complete |
| `BERT_ARCHITECTURE.md` | Yes | N/A | Markdown (no compilation needed) |

---

## **If Nothing Works**

**Use the Markdown version instead:**
```bash
cat docs/BERT_ARCHITECTURE.md
# or open in any text editor
```

The markdown contains ALL the same mathematical content, just in readable text format instead of PDF.

---

## **Troubleshooting Checklist**

- [ ] Try Simplified LaTeX version first
- [ ] Check if `pdflatex` is installed: `which pdflatex`
- [ ] Check LaTeX version: `pdflatex --version`
- [ ] Use Overleaf if local setup fails
- [ ] Use markdown if you don't need PDF format
- [ ] Read markdown directly (no compilation needed!)

---

## **Getting Help**

If you still have issues, the Markdown version (`BERT_ARCHITECTURE.md`) has all the same content and needs no compilation:

```bash
cat docs/BERT_ARCHITECTURE.md  # View in terminal
# or open in VS Code, GitHub, or any text editor
```
