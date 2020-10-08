# Ordinal Neural Network Transformation Models (ONTRAM)

This repository accompanies arXiv:????.?????

Contents:

`Experiments`: Contains experiments to reproduce all results from the paper
in `R` and `Python`.

- `wine`

	- `data`: Contains the wine quality (red) data split into the 20CV
	  folds mentioned in the paper. The data was taken from 
	  [here](https://github.com/tensorchiefs/dl_playr/tree/master/mlt).
	- `learning-efficiency`: Code for reproducing Figure 8
	- `models`: Code for fitting all models for the wine data listed
	  in Table 1 (MCC, QWK, POLR, CI<sub>x</sub>, SI-LS<sub>x</sub>,
	  SI-LS<sub>x</sub><sup>*</sup>)
	- `permuted-class-labels`: Code for reproducing Figure A1

- `UTKFace`

	- `models`: Python notebooks for fitting the models listed in Table 1
	  (MCC, MCC-x, QWK, SI-LS<sub>x</sub>, CI<sub>B</sub>, 
	  CI<sub>B</sub>-LS<sub>x</sub>, SI-CS<sub>B</sub>, 
	  SI-CS<sub>B</sub>-LS<sub>x</sub>) and the models to reproduce
	  Figure 11.
	- `simulate-tabular`: Code for simulating the tabular predictors as
	  illustrated in Figure 5.

`Miscellaneous`: Miscellaneous scripts illustrating scoring rules and results
from the paper.

- `qwk-impropriety.R`: R-script that computes the numerical example from
  Appendix D to show impropriety of the QWK loss.

The **ontram** `R` package:

- The `R` package implementing ordinal neural network transformation models
  lies in a separate repository and can be installed from within `R` via
  `remotes::install_github("LucasKookUZH/ontram-pkg")`.
