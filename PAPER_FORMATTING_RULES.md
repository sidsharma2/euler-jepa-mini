# Paper formatting and structure rules

These rules apply to the Euler-JEPA paper and future conference manuscripts
derived from this example.

## Paper structure

Use a professional conference-paper narrative in this order:

1. Abstract: problem, method, main quantitative result, and claim boundary.
2. Introduction: engineering motivation, research gap, research question, and
   explicit contributions.
3. Related work: only literature that motivates the chosen model or comparison.
4. Physical problem: governing equations, definitions, assumptions, units,
   characteristic scales, and limiting cases.
5. Learning framework: information set, architecture, objective, optimization,
   and justification for every physics or ML component.
6. Experimental method: data generation/acquisition, splits, baselines,
   metrics, uncertainty, and reproducibility controls.
7. Results: quantitative evidence first, followed by interpretation.
8. Discussion: mechanism, failure modes, comparison with prior work, and scope.
9. Conclusions: findings, limitations, and a concrete next-step research path.

The manuscript must tell one coherent story. Audit history may be retained in a
separate report, but the main paper should explain only the audit facts needed
to motivate the corrected experiment.

## Page and typography rule

Use the target venue's two-column template. The complete manuscript, including
references, must remain within the stated page limit. For this project the hard
limit is six pages. Keep the title professional and use the author name as the
only author-line subtitle. Mark the corresponding author with a dagger and put
the matching corresponding-author footnote on page one.

## Figure rule

Figures are vector-first. Prefer native TikZ/PGFPlots. Imported plots or
illustrations must use PDF, EPS, or SVG when possible; do not use raster images
for quantitative figures unless no vector source exists.

Figures with a wide aspect ratio, multi-panel content, dense labels, or a
mechanism/architecture diagram must use a full-width `figure*` environment and
be sized relative to `\textwidth`, not `\columnwidth`. Single-column figures
are reserved for genuinely compact plots that remain legible at column width.
Never make a figure unreadable merely to keep it in one column.

Every figure needs a caption that states what is shown, the data or model
condition, and the interpretation needed by the reader. Every plot must label
quantities, units, and normalization. Keep editable figure source beside the
paper and compile it with the manuscript.

## Evidence rule

Separate code verification, solution verification, calibration, sensitivity,
uncertainty, and experimental validation. A low training or latent loss is not
evidence of physical correctness. Report negative results and do not claim
transfer to real machinery without real validation.
