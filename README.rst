Ultralight axion cores in gravitational lensing time delays
==============================

This repository hosts the programs needed to obtain the results showed in the paper
**Ultralight axion cores in gravitational lensing time delays** by Kfir Blum and Luca Teodori.
We used the `lenstronomy <https://github.com/sibirrer/lenstronomy>`_ package to perform our analysis.

Abstract
--------
We consider the possibility that the cored ground state ("soliton") of ultralight dark matter (ULDM)
is responsible for the mismatch between the value of H_0 deduced from the cosmic microwave
background to that deduced from quasar time delays in galaxy lensing.

Main Results
-------


Notebooks
---------
For fast visualization and understanding of how we obtained our results,
we prepared dedicated jupyter notebooks:

* ``galaxy_plots.ipynb``: Showing the ULDM soliton parameter space allowed by lensing data
* ``Velocity_dispersion.ipynb``: Showing the velocity dispersion influence of a soliton core
* ``Mock_analysis_uldm2PL.ipynb``: Parameter inference by using a power law model for a mock lens obtained with a power law + soliton model.
* ``Mock_analysis_uldm2uldm_No_H0_prior.ipynb``: Parameter inference by using a power law + soliton
  model with flat prior on the Hubble constant for a mock lens obtained with a power law + soliton model.
* ``Mock_analysis_uldm2uldm_H0_prior.ipynb``: Parameter inference by using a power law + soliton
  model with gaussian prior on the Hubble constant centered on the CMB value, for a mock lens obtained with a power law + soliton model.

Authors
-------
- Kfir Blum; kfir.blum@weizmann.ac.il
- Luca Teodori; luca.teodori@weizmann.ac.il

Citations
---------
To cite our work,

```
@article{Blum:2020,
    author = "Blum, Kfir and Teodori, Luca",
    archivePrefix = "arXiv",
    title = "{Ultralight axion cores in gravitational lensing time delays}",
    year = "2021"
}
```




