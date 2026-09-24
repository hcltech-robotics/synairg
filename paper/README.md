# SynAirG manuscript

**Geometry-First Generative Worlds as Simulation Substrates in Bronchoscopy**
Chris von Csefalvay, Pranav Doma and Tamas Foldi · HCLTech
IROS 2026 SurgTwin Workshop

[PDF](../docs/synairg-paper.pdf) · [BibTeX](../docs/synairg-paper.bib)

## Build

Install a TeX distribution with `latexmk`, pdfLaTeX, BibTeX, PGF/TikZ and the
standard LaTeX packages used in `main.tex`. From this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The output is `build/main.pdf`. The published PDF is `../docs/synairg-paper.pdf`.
The verified build has four pages. `SynAirGIEEEtran.bst` is an explicitly named
LPPL derivative of the supplied `IEEEtran.bst`; it adds DOI hyperlinks to reference
titles while keeping the compact reference layout. The original style and class
are retained unchanged with their original licence notices.

## Provenance and scope

Imported from the author-supplied Overleaf source export
`Geometry_First_Generative_Worlds_forBronchoscopy_Digital_Twins.zip`.
Its SHA-256 is
`3cff75be0cf4d21a0dcb02837e7e1652a5ff024ffdcec859764ba61052a3d86d`.

The public revision restores the three-author byline, HCLTech affiliation,
corresponding-author address, dataset DOI and project link. It corrects the code
and dataset licence statement, a typographical error, and bibliography metadata.
See [the identifier audit](REFERENCE-AUDIT.md) for 27 reference records.

The abstract, scientific claims, measurements, plots and table values are retained
from the supplied source. No experiments or figure-generation scripts were rerun.
`SOURCE-SHA256SUMS` records the original exported files; only `main.tex` and
`references.bib` are intentionally changed among those files.

`spark_stats.py` and `make_3d_and_montage.py` are the original analysis scripts,
preserved as source provenance. They are outside the maintained package and its
lint gate; their historical runtime dependencies and data must be supplied before
attempting a scientific reproduction.

## Identifiers and licences

The [1000lungs dataset DOI](https://doi.org/10.57967/hf/9209) is
`10.57967/hf/9209`. It is not a manuscript DOI. The code licence is Apache-2.0;
the dataset card specifies OpenMDW 1.1. The included IEEE class and bibliography
styles retain their own licence notices. These statements do not assign a new
licence to the manuscript itself.
