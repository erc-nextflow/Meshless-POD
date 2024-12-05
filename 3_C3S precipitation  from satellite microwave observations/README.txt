Codes of: "A meshless method to compute the proper orthogonal decomposition and its variants from scattered data"

This is the third test case: C3S precipitation  from satellite microwave observations.

DOI and reference of the original dataset:  Konrad H, Panegrossi G, Bagaglini L, Sanò P, Sikorski T, Cattani E, Schröder M, Mikalsen A,
Hollmann R. 2022 Precipitation monthly and daily gridded data from 2000 to 2017 derived
from satellite microwave observations. Copernicus Climate Change Service (C3S) Climate Data
Store (CDS). (10.24381/cds.ada9c583)


Files: Precipitation_meshless_fluct.py, d_fixed_comp.py, MeanCorrection_module,EPTV_module, Grid.npy,Tref.npy

Folders: Output, EPTV

- Precipitation_meshless_fluct.py is a main file that you can run for generate the data needed to perform the analysis like in the paper, here are explained all the parameters and steps of the algorithm.
-d_fixed_comp.py perform the same analysis presented in the paper and generate the same figures ( you can generate them using the main or use the files in Output already computed). This code is mainly for plotting and comparison purposes.
- MeanCorrection_module.py is a module to perform the mean correction as explained in tirelli et al 2023 " A simple trick..."
- EPTV_module.py is a module for ensamble averaging, to extract HR statistics
- Tref.npy reference data already on the output grid (available on Zenodo)
- Grid.npy reference grid

Output folder is here empty, but you can download the data from zenodo and run exactly the same results of the paper, otherwise you can run the code from scratch without download the data (you will need to download at least Tref on Zenodo) and use the configuration that you prefer. In Output will be different subfolder, the name is related to the number of sensor used for that specific case. Inside each folder you have:

- K_est.npy -> temporal correlation matrix (meshless POD)
- Phi_free,PSi_free and Sigma_free -> POD modes from meshless POD
- PTV -> sensors distribution
- t_binned-> binned distribution for gridded POD
- T_proj -> projection of the velocity fields on the output grid

For feedbacks and questions: itirelli@pa.uc3m.es. 