# agent.md

## Scope
These notes apply to work in `/Users/xiyin/ResearchIdeas/String-Monte-Carlo/lightcone formalism`.

## Primary files
- Main working note: `tex/lightcone_discrete_sigma_note.tex`
- Running multi-agent review log: `agents_chat.md`

## Accuracy standard
- Do not leave undefined symbols in equations. Define symbols at first use whenever reasonably possible.
- If a symbol was introduced much earlier and is easy to forget, briefly remind the reader locally instead of forcing backtracking.
- Do not use vague words such as "schematic" unless the note explicitly explains what is being suppressed and why.
- If a formula is imported from continuum lightcone string theory rather than derived in the note, say so explicitly.
- If a quantity is not fully fixed because of missing conventions or an unresolved calculation, say that plainly.
- Distinguish carefully between:
  - lattice-defined quantities and continuum target quantities,
  - local vertex data and global sewn-diagram data,
  - vertex labels and transverse/spinor indices,
  - exact formulas, convention-dependent formulas, and deferred formulas.

## Reader burden
- Optimize for low cognitive load.
- Every displayed equation should be understandable from nearby text.
- Prefer explicit sentences like "Here X means ..." or "This is not yet ..." over relying on expert inference.
- When there is a nontrivial basis choice, state the basis, the physical subspace, and what is actually inverted or solved.
- When the note makes an approximation, interpolation, refinement, or continuum-limit claim, state the precise status and limitations.

## Figures
- Add figures for geometric constructions, bookkeeping maps, and local interaction-region data when they materially help.
- Keep figures visually clean. Do not place long equations inside drawings if they cause overlap or clutter.
- Use captions to explain the mathematical role of the figure, not just its visual content.

## Review workflow
- Read relevant entries in `agents_chat.md` before making another accuracy pass.
- When addressing an audit, treat it as a checklist and make sure each item is resolved by one of:
  - a local definition,
  - a notation cleanup,
  - an explicit cross-reference,
  - an honest statement that the quantity is not derived here and what extra input is required.
- If an issue cannot be fully solved in the note, record the obstacle clearly in the prose instead of hiding it.

## TeX workflow
- Edit source only, never generated artifacts.
- Rebuild locally with `latexmk -pdf -bibtex lightcone_discrete_sigma_note.tex` from `lightcone formalism/tex` when bibliography-sensitive changes are made.
- A successful build with only overfull/underfull box warnings is acceptable; undefined references, math errors, or missing symbol definitions are not.

## House style for this note
- Prefer precision over brevity.
- State conventions explicitly.
- Use consistent notation across sections once a symbol is chosen.
- If a notation change is made in one section, check downstream sections for consistency.
- When the note claims a formula is "exact," ensure the exact domain of validity is stated.
