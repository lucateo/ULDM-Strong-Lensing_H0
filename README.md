=============================================
Hierarchical Time Delay Cosmography With ULDM 
=============================================

This repository hosts the hierarc and lenstronomy modules, to be modified to include ULDM in the hierarchical analysis made by `Birrer et al. 2020 <https://arxiv.org/abs/2007.02941>`_.

Our goal is to understand if the Hubble tension in measurements of strong gravitational lensing can be exèplained by the presence of large ULDM cores in elliptical galaxies. The idea is to set a sharp prior on the Hubble constant based on CMB data and see what are the ULDM parameter values that are compatible with data. To make the code run, copy the repository of the TDCOMSO analysis `here <https://github.com/TDCOSMO/hierarchy_analysis_2020_public>`_ and insert the ULDM/ folder inside.

 Structure of Repository
------------------------
In ULDM folder:
* ``hierarc``: hierarc module (to be modified by inserting ULDM). Original github repo `here <https://github.com/sibirrer/hierArc>`_
* ``lenstronomy``: lenstronomy module (to be modified by inserting ULDM). Original github repo `here <https://github.com/sibirrer/lenstronomy>`_
* ``Insert_ULDM``: main program
* ``Functions``: various secondary functions used in main program

