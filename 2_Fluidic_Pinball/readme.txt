Codes of: "A meshless method to compute the proper orthogonal decomposition and its variants from scattered data"
this is the second testcase: fluidic pinball.

DOI and reference of the original dataset: Deng N, Noack BR, Morzy ´nski M, Pastur LR. 2020 Low-order model for successive
bifurcations of the fluidic pinball. Journal of fluid mechanics 884, A37. (10.1017/jfm.2019.959)

Files: Pinball_main_meshlessPOD.py, d_fixed_comp.py, MeanCorrection_module,EPTV_module, dnsu.npy,dnsv.npy,GridX.mat,GridY.mat,U.mat,V.mat.

Folders: Output, EPTV

- Pinball_main_meshlessPOD.py is a main file that you can run for generate the data needed to perform the analysis like in the paper, here are explained all the parameters and steps of the algorithm.
-d_fixed_comp.py perform the same analysis presented in the paper and generate the same figures ( you can generate them using the main or use the files in Output already computed). This code is mainly for plotting and comparison purposes.
- MeanCorrection_module.py is a module to perform the mean correction as explained in tirelli et al 2023 " A simple trick..."
-EPTV_module.py is a module for ensamble averaging, to extract HR statistics
-dnsu.npy,dnsv.npy reference data already on the output grid (available on Zenodo)
- GridX.mat,GridY.mat DNS original mesh
- U.mat,V.mat particle distribution from DNS simulation (available on Zenodo)

Output folder is here empty, but you can download the data from zenodo and run exactly the same results of the paper, otherwise you can run the code from scratch without download the data (you will need to download at least Tref on Zenodo) and use the configuration that you prefer. In Output will be different subfolder, the name is related to the number of sensor used for that specific case. Inside each folder you have:

- K_est.npy -> temporal correlation matrix (meshless POD)
- Phi_free,PSi_free and Sigma_free -> POD modes from meshless POD
- PTV -> sensors distribution
- u_binned,v_binned -> binned distribution for gridded POD
- U_proj -> projection of the velocity fields on the output grid

For feedbacks and questions: itirelli@pa.uc3m.es. 