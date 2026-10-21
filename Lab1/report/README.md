# Lab 1 report

- PDF: `Md_Zunayedul_Islam_Model_Pruning.pdf`
- Editable LaTeX source: `main.tex`
- Generated tables and measurement macros: `data/`
- Original and regenerated figures: `figures/`
- Raw experiment data: `../artifacts/`

The report contains a cover, contents, background, exercise-by-exercise
discussion, final evaluation, conclusion and references. It ends immediately
after the references. All architecture descriptions, measurements and plots
concern this lab's VGG-style model and saved experiments.
The bonus is identified as implemented/tested on synthetic tensors, without a
full CIFAR100 experiment.

## Build the PDF

Install XeLaTeX with the usual TeX Live packages (`fontspec`, `unicode-math`,
`booktabs`, `listings`, `placeins`, `needspace`, `microtype`, and `hyperref`).
With Times New Roman installed on your system, run from the assignment root:

```bash
bash Lab1/report/build.sh
```

The build runs XeLaTeX three times to resolve the contents, citations and figure
references. Its logs and temporary files go into `report/build/`. The finished
PDF is copied into this directory.

If the font is not installed, supply your local Times New Roman font directory:

```bash
TIMES_FONT_DIR=/path/to/times-fonts bash Lab1/report/build.sh
```

This directory should contain `Times.TTF`, `Timesbd.TTF`, `Timesi.TTF` and
`Timesbi.TTF` (regular, bold, italic and bold italic). Capitalisation matters on
Linux. A system font installation avoids the filename requirement. Font files
are not bundled. The completed PDF embeds Times New Roman subsets, so readers
do not need to install the font to view it. Times-style mathematical symbols
and monospaced code are used where appropriate. Four original experiment PNGs
retain their original chart labels; regenerated chart labels use Times New Roman.

## Regenerate figures and tables

This step uses the existing training environment, including Matplotlib,
NumPy and PyTorch; it does not rerun training or pruning:

```bash
# With Times New Roman installed:
bash Lab1/report/generate_figures.sh

# Or provide the font directory explicitly:
TIMES_FONT_DIR=/path/to/times-fonts bash Lab1/report/generate_figures.sh
```

The script selects `../../.venv/bin/python` relative to the report directory if
available, otherwise `python3`. Override it with `PYTHON_BIN` when needed.
It contains inline Python in a shell script; no new Python files are created.
It regenerates seven vector PDF figures and their PNG previews, copies the four
original pruning PNGs unchanged, and derives table rows from the raw data.
`data/provenance.json` records input SHA256 hashes. If raw experiments change,
regenerate the figures/tables and review the narrative before rebuilding.

## Recorded result and provenance

The selected model has 60% zero parameters, 72.77% validation accuracy in the
original GPU experiment, and 72.97% final test accuracy. The baseline test
accuracy is 73.23%, so the test difference is 0.26 percentage points. A separate
final evaluation reported 72.75% validation accuracy; the report preserves that
measurement separately and does not assign an unrecorded cause to the difference.

The report documents checkpoint selection over fine-tuning epochs 0--5. At low
sparsity, the pre-fine-tuning state was retained when all updates were worse.
No speedup, reduced dense file size, matched-budget iterative comparison or
unmeasured bonus accuracy is claimed.
