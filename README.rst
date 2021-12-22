Gravitational lensing H_0 tension from ultralight axion galactic cores
==============================

This repository hosts the programs needed to obtain some of the results showed in the paper
**Gravitational lensing H_0 tension from ultralight axion galactic cores** by Kfir Blum and Luca Teodori.
We used the `lenstronomy <https://github.com/sibirrer/lenstronomy>`_ package to perform our analysis.

Abstract
--------
Gravitational lensing time delays offer an avenue to measure the Hubble parameter H_0, with some
analyses suggesting a tension with early-type probes of H_0. The lensing measurements must mitigate
systematic uncertainties due to the mass modelling of lens galaxies. In particular, a core component
in the lens density profile would form an approximate local mass sheet degeneracy and could bias H_0
in the right direction to solve the lensing tension. We consider ultralight dark matter as a possible
mechanism to generate such galactic cores. We show that cores of roughly the required properties could
arise naturally if an ultralight axion of mass $ m \sim 10^−25$ eV makes up a fraction of order ten percent of
the total cosmological dark matter density. A relic abundance of this order of magnitude could come from
vacuum misalignment. Stellar kinematics measurements of well-resolved massive galaxies (including the
Milky Way) may offer a way to test the scenario. Kinematics analyses aiming to test the core hypothesis
in massive elliptical lens galaxies should not, in general, adopt the perfect mass sheet limit, as ignoring
the finite extent of an actual physical core could lead to significant systematic errors.

Notebooks
---------
For understanding on how we obtained our results,
we prepared dedicated jupyter notebooks:

* ``Velocity_dispersion.ipynb``: Showing the velocity dispersion influence of a core
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
To cite our work::

  @article{Blum:2021oxj,
      author = "Blum, Kfir and Teodori, Luca",
      title = "{Gravitational lensing $H_0$ tension from ultralight axion galactic cores}",
      eprint = "2105.10873",
      archivePrefix = "arXiv",
      primaryClass = "astro-ph.CO",
      doi = "10.1103/PhysRevD.104.123011",
      journal = "Phys. Rev. D",
      volume = "104",
      number = "12",
      pages = "123011",
      year = "2021"
  }





