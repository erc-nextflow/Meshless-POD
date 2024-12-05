# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:44:14 2023

@author: itirelli
"""

import scipy.io
import numpy as np
from scipy.interpolate import RBFInterpolator
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from scipy.spatial import cKDTree
import os
import pickle
# import MeshlessPODfunction as MPOD
from numpy.polynomial.legendre import leggauss
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import EPTV_module
import MeanCorrection_module as MC
from concurrent.futures import ThreadPoolExecutor
import math
from scipy.interpolate import griddata


def Int_DNS(data_points, interpolation_points, U):
     #function that interpolate reference data
     interpolatorU = RBFInterpolator(data_points, U)
     U_result = interpolatorU(interpolation_points)
     
     
     return U_result    
      

def Int_PTV(Npart,data_points,interpolation_points,U):
    # function that create the PTV dictonary
    
    # Interpolate U values using linear interpolation
    U_linear = griddata((data_points[:,0], data_points[:,1]), U.flatten(), 
                        (interpolation_points[:,0], interpolation_points[:,1]), method='linear')
    
    # Interpolate U values using nearest neighbor method
    U_nearest = griddata((data_points[:,0], data_points[:,1]), U.flatten(), 
                         (interpolation_points[:,0], interpolation_points[:,1]), method='nearest')
    
    # Replace NaN values in linear interpolation result with nearest neighbor values
    U_result = np.where(np.isnan(U_linear), U_nearest, U_linear)
    U_result[U_result < 0] = 0
    
    xpart = interpolation_points[:,0]
    ypart = interpolation_points[:,1]
   
    return {'X': xpart.reshape([-1,1]), 'Y':ypart.reshape([-1,1]), 'T':U_result.reshape([-1,1])}
  

def Binning2D_worker(PTV, b, X, Y):
    #binning function  for the single snapshot
    xp, yp = PTV['X'], PTV['Y']
    u = PTV['t']
    tree = cKDTree(np.column_stack((xp, yp)))
    U = np.zeros(X.shape)
    
    for II in range(X.shape[0]):
        for JJ in range(Y.shape[1]):
            p = [X[II, JJ], Y[II, JJ]]
            idxs = tree.query_ball_point(p, round((b / 2), 3))
            if idxs:
                U[II, JJ] = np.mean(u[idxs])
                
    tree = None
    return U

def Binning2D_worker_wrapper(i, PTV, b, X, Y):
    #wrapped version of binning function
    Binning2D_worker_wrapped = partial(Binning2D_worker, PTV=PTV[i], b=b, X=X, Y=Y)
    return Binning2D_worker_wrapped()

def Binning2D(PTV, NImg, b, X, Y):
     # binning function main with parallelization
    num_snaps = len(NImg)
    Ub = np.zeros((X.shape[0], X.shape[1], num_snaps))
    
    with ThreadPoolExecutor(max_workers=60) as executor:
        results = list(tqdm(executor.map(lambda i: Binning2D_worker_wrapper(i, PTV, b, X, Y), NImg), 
                            total=len(NImg), 
                            desc='Processing snapshots', 
                            unit='snapshot', 
                            leave=True))
    
    for i, result in enumerate(tqdm(results, desc='Assembling Matrices', unit='snapshot')):
        Ub[:, :, i] = result
    
    return Ub

def Int_GL(args, XGL, PTV):
     #function to compute the value of velocity in the GL points
     i, _ = args
     data_points = np.column_stack((PTV[i+1]['X'], PTV[i+1]['Y']))
     interpolatorU = RBFInterpolator(data_points, PTV[i+1]['t'])
     U_int = interpolatorU(XGL)
     U_GL = U_int
     return U_GL

def compute_row_K(args,UGL,nt,WGL):
    # function to compute each row of the correlation matrix K
    i, row = args  # unpack the input arguments
    u_GLi = UGL[i]
    UU = np.squeeze(np.array(UGL[i:nt]))
    row[i:nt] = (WGL.reshape(-1, 1) * u_GLi).T @ UU.T
          
    return row

def U_projection(args,interp_grid,PTV):
    #function to compute the value of velocity in the GL points 
    i, row = args 
    data_points = np.column_stack((PTV[i+1]['X'],PTV[i+1]['Y']))
    interpolatorU = RBFInterpolator(data_points, PTV[i+1]['t'])
    
    U_int = interpolatorU(interp_grid)
    
    U_proj = U_int
    return U_proj

def rayleigh_quotient(x,A):
    return(x.T.dot(A).dot(x)/x.T.dot(x))


def process_PTV(i):
    #function to generate random sensor positions 
    Int_PTV_wrapped = partial(Int_PTV, Npart, data_points)
    xpart = np.random.uniform(xboundary[0], xboundary[1], size=Npart)
    ypart = np.random.uniform(yboundary[0], yboundary[1], size=Npart)
    interpolation_points = np.column_stack((xpart, ypart))
    return Int_PTV_wrapped(interpolation_points, Flow['T'][i-1, :])

     #%%  SETTING  
if __name__ == '__main__':    
  
    with open('Tref', 'rb') as f:
        TREF = pickle.load(f) #load reference precipitation data
        print(f'Loaded reference field!')
    Flow = {}
    Flow['T'] = TREF
    Nt = TREF.shape[0] #number of days
    NpRef = TREF.shape[1] # number of sensors for the reference
    NImg = list(np.arange(1,Nt+1))
    xboundary = [-50, 50] # Longitude
    yboundary = [0, 90] # latitude
    with open('Grid', 'rb') as f:
        GRID = pickle.load(f) # reference grid
        print(f'Loaded grid!')
    X = GRID['LON']
    Y = GRID['LAT']
    dx_ = X[0,1]-X[0,0] # grid distance
    data_points = np.column_stack((X.flatten(), Y.flatten()))

    Area_D = (xboundary[-1]-xboundary[0])*(yboundary[-1]-yboundary[0]) # computing area
    Area = (math.radians(xboundary[-1])-math.radians(xboundary[0]))*(math.radians(yboundary[-1])-math.radians(yboundary[0])) # in radiants
    d_hat = [0.015,0.02,0.025,0.03,0.035] #sparsity level (see paper)
    Npart_ = [int(np.floor(Area / dw**2)) for dw in d_hat] # number of sensor for each case
    # Npart_[0] = np.min([Npart_[0],NpRef])
    # Calculate the density
    Nppp_ = [Np / Area for Np in Npart_]
    
    

   
#%% for loop on the sparsity levels
    for dd in range(0,len(d_hat)):
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'Case d_hat = {d_hat[dd]}')
        Npart = Npart_[dd]
        Nppp = Npart/Area_D
        out_path = f'Output/Case_{Npart}/'  # Assuming you are using Python 3.6 or later for f-string

# Check if the directory exists, and if not, create it
        if not os.path.exists(out_path):
            os.makedirs(out_path)
#%% BUILDING SCATTERED DISTRIBUTION (PTV)
    
        if os.path.isfile(out_path +'PTV'):
            print(f'PTV already computed, loading...')
            with open(out_path +'PTV', 'rb') as f:
                    PTV = pickle.load(f)
                    print(f'Loaded!')
        else:
            PTV = {} # initialize dictonary of PTV
            with ThreadPoolExecutor(max_workers=60) as executor: # parallel computation of PTV snapshots
                results = list(tqdm(executor.map(process_PTV, NImg),total=len(NImg),desc='Processing snapshots',unit='snapshot', leave=True))
        
            # Assemble matrices
            for i, result in enumerate(tqdm(results, desc='Assembling Matrices', unit='snapshot')):
                PTV[i+1] = result
        
            # Save the results to a file
            with open(out_path +'PTV', 'wb') as f:
                pickle.dump(PTV, f)
                    
 #%% EPTV
     # in this section EPTV is performed, this is needed to avoid modulation effect on the mean flow (I.Tirelli et al 2023, A simple trick to improve the accuracy of PIV/PTV data)
        EPTV_name_stat = 'EPTV/EPTV_stat_Prep'  
        flag = '2D'
        if os.path.isfile(EPTV_name_stat):
             print(f'EPTV already performed, loading...')
             with open(EPTV_name_stat, 'rb') as f:
                 EPTV_stat = pickle.load(f)
                 print(f'Loaded!')
        else:
         print(f'EPTV not performed')
         print(f'Starting EPTV')
         EPTV_stat = EPTV_module.EPTV(NImg,Nppp,X,Y,0,PTV,dx_,EPTV_name_stat,flag)
         print(f'EPTV ended')
 #%% Mean correction (I.Tirelli et al 2023, A simple trick to improve the accuracy of PIV/PTV data)  
        print(f'Starting mean correction')
        PTV = MC.MeanCorrection (PTV,NImg,EPTV_stat,flag)
        print(f'Mean correction completed!')
    
 #%% BUILDING BINNED DISTRIBUTION (PIV)    
     # In this section the binned distribution are built whose POD modes will be the competitors of the meshless ones          
     
        if os.path.isfile(out_path +'t_binned.npy'):
                print(f'PIV already computed, loading...')
                t_b = np.load(out_path +'t_binned.npy')
                
                print(f'Loaded!')
        else:
                print(f'Computing binned...')
                t_b = np.zeros((X.shape[0], X.shape[1], len(NImg)))
                b = round((10/(Nppp))**(1/2),2) # BIN SIZE
                t_b = Binning2D(PTV, NImg, b,  X, Y)
                np.save(out_path +'t_binned',t_b)
             
    
    #%% POD Reference
        Nt = 5000
        import numpy.matlib
        print(f'Computing POD DNS')
        TmHR = EPTV_stat['TmEPTV'].flatten().reshape([1,-1])
        TmHR = np.tile(TmHR, (len(NImg), 1))
        tREF = TREF[0:5000,:]-TmHR
        Psi_ref, Sigma_ref,Phi_ref= np.linalg.svd(tREF[0:Nt,:], full_matrices=False)
        print(f'POD DNS computed')
        Phi_ref = Phi_ref.T
        
    #%% POD Binned
        print(f'Computing POD binned')
        t_b = t_b[:,:,0:Nt].reshape([X.shape[0]*X.shape[1],len(NImg)])
        t_b = t_b.T
        Psi_b, Sigma_b,Phi_b= np.linalg.svd(t_b, full_matrices=False)
        print(f'POD binned computed')
        Phi_b = Phi_b.T
    
    #%% Meshless POD
        if os.path.isfile(out_path +'Psi_free.npy'):
            print(f'POD already computed, loading...')
            Psi_free = np.load(out_path +'Psi_free.npy')
            Sigma_free = np.load(out_path +'Sigma_free.npy')
            print(f'Loaded!')
        else:
            print(f'Computing POD meshless')
            W = np.arange(xboundary[0],xboundary[-1]+1)
            H = np.arange(yboundary[0],yboundary[-1]+1)
            xGL,wGL = leggauss(60) # select number of G-L points and compute the corresponding weights
            xq, yq = np.meshgrid(xGL, xGL) # define grid of GL points
            xq = xq*(W[-1]-W[0])/2 + (W[-1]+W[0])/2 # scaling 
            yq = yq*(H[-1]-H[0])/2 + (H[-1]+H[0])/2 # scaling 
            XGL = np.column_stack((xq.flatten(), yq.flatten()))
            Int_GL_wrapped = partial(Int_GL, XGL = XGL,PTV = PTV)
            start_time = time.time()
            print(f"Parallel computation of all Gauss-Legenendre points")   
            num_iterations = len(NImg)
    
            with ThreadPoolExecutor(max_workers=60) as executor:
                UGL = list(tqdm(executor.map(Int_GL_wrapped, enumerate(NImg)), total=num_iterations,desc='Processing snapshots',unit='snapshot', leave=True))
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time} seconds")  
            # computing correaltion matrix
            K_est = np.zeros((len(NImg),len(NImg)))  
            # assembling GL weights matrix
            WGL = np.outer(wGL, wGL).reshape(-1,1)
            start_time = time.time()
            # Perform parallel computation
            # Wrap compute_row_K function call with additional arguments using partial
            compute_row_K_wrapped = partial(compute_row_K, UGL=UGL, nt=len(NImg),WGL=WGL)
            # Perform parallel computation
            print(f"Parallel computation of K_est")   
            with ThreadPoolExecutor(max_workers=60) as executor:
                # computing in parallel the rows of K

                results = list(tqdm(executor.map(compute_row_K_wrapped, enumerate(K_est)), total=num_iterations,desc='Processing snapshots',unit='snapshot', leave=True))
            print(f"Time taken: {elapsed_time} seconds")           
                    
            K_est = np.array(results) #from the list to the matrix 
            K_est = K_est/4 # divided by the GL domain [-1,1] ( the area is omitted because for the change of variable we have it in the numerator of the jacobian, while we have it in the denominator of the inner product)

            nt=len(NImg)
            print("Computing lower diagonal K")  
            lower_diag_indices = np.tril_indices(nt, k=-1) #taking the index of the upper diagonal elements
            K_est[lower_diag_indices] = K_est.T[lower_diag_indices] # flipping ( the matrix is symmetric)
            end_time = time.time()
            elapsed_time = end_time - start_time 
                    
            Psi_free, Sigma_square, _ = np.linalg.svd(K_est, full_matrices=False)
            Sigma_free = np.sqrt((Sigma_square)) 
          
            np.save(out_path +'Psi_free',Psi_free)
            np.save(out_path +'Sigma_free',Sigma_free)
            np.save(out_path +'K_est',K_est)
       
#%% Projection of the velocity component on the output grid
        # In order to extract the spatial modes we need to project the velocity fields on the grid
        if os.path.isfile(out_path +'T_proj.npy'):
            print(f'projections already computed, loading...')
            T_proj = np.load(out_path +'T_proj.npy')
            
            print(f'Loaded!')
        else:
            T_projection_wrapped = partial(U_projection, interp_grid = np.column_stack((X.flatten(),Y.flatten())),PTV = PTV)
            start_time = time.time()
            print(f'projections ...')
            num_iterations = len(NImg)
    
            with ThreadPoolExecutor(max_workers=60) as executor:
                # computing projection in parallel
                T_proj = list(tqdm(executor.map(T_projection_wrapped, enumerate(NImg)), total=num_iterations,desc='Processing snapshots',unit='snapshot', leave=True))
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time} seconds") 
            T_proj = np.array(T_proj)
            T_proj  = np.squeeze(T_proj)
            np.save(out_path + 'T_proj',T_proj)
        
        if os.path.isfile(out_path +'Phi_free.npy'):
           print(f'Phi already computed')
           
        else:
            # assembling matrices for rayleigh quotient sigma_i = psi_i'Kpsi_i/psi_i'psi_i
            A = T_proj@T_proj.T
            Sigma_proj = []
            for j in range(len(NImg)):
                eigenvector = Psi_free[:, j]
                eigenvalue = rayleigh_quotient(eigenvector,A)
                Sigma_proj.append(np.sqrt(eigenvalue))
    
            print(f"Extracting Phi")
            Phi_free = T_proj.T@Psi_free@ np.diag(Sigma_free)
            Phi_free_norm = np.zeros_like(Phi_free)
            ## normalization of PHI
            for n in range(0,len(NImg)):
                aa = Phi_free[:,n]/np.linalg.norm(Phi_free[:,n])
                # aa[mask_flat[0,:]] = 'nan'
                Phi_free_norm[:,n] = aa
            np.save(out_path +'Phi_free',Phi_free_norm)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')   
    
   