Codes of: "A meshless method to compute the proper orthogonal decomposition and its variants from scattered data"

This is the first test case, the analytic one.

Files: AnalyticalPOD_meshless_meshless_fluct, d_fixed, EPTV_module
Folders: Output

- AnalyticalPOD_meshless_fluct.py is a main file that you can run for a general case fixing the level of sparsity of the sensors.

-d_fixed.py perform the same analysis presented in the paper and generate the same figures.

-EPTV_module.py is a module for Ensemble PTV (I.Tirelli et al. 2024, "A simple trick to improve the accuracy of PIV/PTV data").

Output folder is here empty, but you can download the data from zenodo and run exactly the same results of the paper, otherwise you can run the code from scratch without download the data and use the configuration that you prefer. In Output will be different subfolder, the name is related to the number of sensor used for that specific case. Inside each folder you have:

- K_est.npy -> temporal correlation matrix (meshless POD)
- Phi_free,PSi_free and Sigma_free -> POD modes from meshless POD
- PTV -> sensors distribution
- t_binned -> binned distribution for gridded POD




