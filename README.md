# Ordinal Neural Network Transformation Models (ONTRAM)

This repository accompanies arXiv:????.?????

Contents:

`Experiments`: Contains experiments to reproduce all results from the paper
in `R` and `Python`.

- `wine`

	- `learning-efficiency`: Code for reproducing Figure 8
	- `models`: Code for fitting all models for the wine data listed
	  in Table 1 (MCC, QWK, POLR, CI<sub>x</sub>, SI-LS<sub>x</sub>)
	- `permuted-class-labels`: Code for reproducing Figure C1

- `UTKFace` and `DR`

	- `models`: Python notebooks for fitting the models listed in Table 1
	  (MCC, MCC-x, QWK, SI-LS<sub>x</sub>, CI<sub>B</sub>, 
	  CI<sub>B</sub>-LS<sub>x</sub>, SI-CS<sub>B</sub>, 
	  SI-CS<sub>B</sub>-LS<sub>x</sub>)
	- `simulate-tabular`: Code for simulating the tabular predictors as
	  illustrated in Figure 5.

`ontram`: `R` package that implements ordinal neural network transformation
models.

`Miscellaneous`: Miscellaneous scripts illustrating scoring rules and results
from the paper.

- `qwk-impropriety.R`: R-script that computes the numerical example from
  Appendix A to show impropriety of the QWK loss.

