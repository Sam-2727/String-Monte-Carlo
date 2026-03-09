# Overarching goal
- The ultimate goal is to construct arbitrary genus string amplitudes using kontsevich's ribbon graph formulation
- This is done by using a parameterization of the ribbon graph formulation that essentially involves sewing together edges of the disc

# Current status
- Right now, we can successfully reproduce the bosonic torus amplitude for arbitrary modular parameter, using the recipes outlined in string MC_modified.tex. This is done by coming up with a numerical map between strebel length parameteris \ell_i and torus modular parameter \tau, as explained in string MC_modified.tex and implemented in the various code in the repository.
- We are struggling to match the free fermion and bc ghost determinant. Various parts of the .ipynb files test 


# Repository outline
- All code and notes live under `covariant formalism/`
- `covariant formalism/tex/string MC_modifed.tex` is the human text explanation of the various procedures that we hope to perform. Not all sections are implemented in code currently, but the core code methods are explained there. It is important to refer to this file when confused about the purpose of something, before trying to reason through yourself.
- `covariant formalism/python/partition_function.py` constructs the numerical formulation of the matter and bc ghost partition functions at genus one, which is what we ultimately want to do for arbitrary genus
- `covariant formalism/python/ell_to_tau.py` constructs analytic formula that allow comparison to the
- `covariant formalism/notebooks/` contains .ipynb files to test the various formula in partition_function.py and ell_to_tau.py
- Right now, the .ipynb files contain some contradictory information on the bc ghost and free fermion, as we are testing various potential formula for the two.
- `covariant formalism/mathematica/` contains Mathematica notebooks for symbolic computations

# Claude-code-reasoning.md file
- claude-code-reasoning.md gives an outline to reasoning documents you have created with explanations of various topics
- Each subfile should contain complete details, not just summarization of the topic.
- When reasoning through something, look at claude-code-reasoning.md first for an outline of topics before choosing which files to read.
- Do not just claim answers/results, reason through things step-by-step.
- Make things human readable and add extra explanations here such that all algebra is shown explicitly.
- Format latex so that it renders correctly in the VSCode markdown environment.
