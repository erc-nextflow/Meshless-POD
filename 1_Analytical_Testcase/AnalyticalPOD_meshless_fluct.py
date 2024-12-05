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
from numpy.polynomial.legendre import leggauss
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor



def Int_DNS(data_points, interpolation_points, U):
     interpolatorU = RBFInterpolator(data_points, U)
     
     U_result = interpolatorU(interpolation_points)
     
     
     return U_result    
      

def Int_PTV(Npart,data_points,interpolation_points,U):
    
    
    interpolatorU = RBFInterpolator(data_points, U)
   
    U_result = interpolatorU(interpolation_points)
    
    xpart = interpolation_points[:,0]
    ypart = interpolation_points[:,1]
   
    return {'X': xpart.reshape([-1,1]), 'Y':ypart.reshape([-1,1]), 'T':U_result.reshape([-1,1])}
  

def Binning2D_worker(PTV, b, X, Y):
    xp, yp = PTV['X'], PTV['Y']
    u = PTV['T']
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
    Binning2D_worker_wrapped = partial(Binning2D_worker, PTV=PTV[i], b=b, X=X, Y=Y)
    return Binning2D_worker_wrapped()

def Binning2D(PTV, NImg, b, X, Y):
    
    num_snaps = len(NImg)
    Ub = np.zeros((X.shape[0], X.shape[1], num_snaps))
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda i: Binning2D_worker_wrapper(i, PTV, b, X, Y), NImg), 
                            total=len(NImg), 
                            desc='Processing snapshots', 
                            unit='snapshot', 
                            leave=True))
    
    for i, result in enumerate(tqdm(results, desc='Assembling Matrices', unit='snapshot')):
        Ub[:, :, i] = result
    
    return Ub

def Int_GL(args, XGL, PTV):
    i, _ = args
    data_points = np.column_stack((PTV[i+1]['X'], PTV[i+1]['Y']))
    interpolatorU = RBFInterpolator(data_points, PTV[i+1]['T'])
    U_int = interpolatorU(XGL)
    U_GL = U_int
    return U_GL

def compute_row_K(args,UGL,nt,WGL):
             # specify that x is a global variable
          i, row = args  # unpack the input arguments
          
          u_GLi = UGL[i]
          UU = np.squeeze(np.array(UGL[i:nt]))
          row[i:nt] = (WGL.reshape(-1, 1) * u_GLi).T @ UU.T
          
          return row

def U_projection(args,interp_grid,PTV):
    i, row = args 
    data_points = np.column_stack((PTV[i+1]['X'],PTV[i+1]['Y']))
    interpolatorU = RBFInterpolator(data_points, PTV[i+1]['T'])
    
    U_int = interpolatorU(interp_grid)
    
    U_proj = U_int
    return U_proj

def rayleigh_quotient(x,A):
    return(x.T.dot(A).dot(x)/x.T.dot(x))

def process_PTV(i):
    Int_PTV_wrapped = partial(Int_PTV, Npart, data_points)
    xpart = np.random.uniform(xboundary[0], xboundary[1], size=Npart)
    ypart = np.random.uniform(yboundary[0], yboundary[1], size=Npart)
    interpolation_points = np.column_stack((xpart, ypart))
   
    return Int_PTV_wrapped(interpolation_points, Dref[:,i-1])

     #%%  SETTING  
if __name__ == '__main__':    
    NImg = list(np.arange(1,501)) 
    L_x=12
    L_y=12
    n_x=100
    n_y=100
    k_x=3
    k_y=3
    x=np.linspace(0,L_x,n_x); y=np.linspace(0,L_y,n_y)
    X,Y=np.meshgrid(x,y)
    dx_ = X[0,1]-X[0,0]
    xboundary = [0, L_x]
    yboundary = [0, L_y]

    # Spatial structures orthogonal in space
    phi_1=np.sin(2*np.pi/L_x*k_x*X)*np.sin(2*np.pi/L_y*(k_y)*Y)
    phi_2=np.sin(2*np.pi/L_x*(2*k_x)*X)*np.sin(2*np.pi/L_y*(2*k_y)*Y)
    phi_1_v=phi_1.reshape(-1,1)
    phi_2_v=phi_2.reshape(-1,1)

    norm_phi_1_v=np.linalg.norm(phi_1_v)
    norm_phi_2_v=np.linalg.norm(phi_2_v)

    phi_1_v=phi_1_v/norm_phi_1_v
    phi_2_v=phi_2_v/norm_phi_2_v

    lambda_1=L_x/k_x
    lambda_2=(L_x/k_x)/2

#Temporal structures orthogonal in time
    T=10; n_t=500
    t=np.linspace(0,T,n_t)
    psi_1=np.sin(2*np.pi/T*2*t)
    psi_2=np.sin(2*np.pi/T*t)
    psi_1=psi_1/np.linalg.norm(psi_1)
    psi_2=psi_2/np.linalg.norm(psi_2)
    sigma_1=10; sigma_2= 5
    Dref = sigma_1*phi_1_v.dot(psi_1.reshape(-1,1).T)+sigma_2*phi_2_v.dot(psi_2.reshape(-1,1).T)
    Area = L_x*L_y
    Area_ = Area/(lambda_2**2)
    d_hat = [0.1  , 0.2, 0.3 , 0.4, 0.5   ] 
    Npart_ = [int(np.floor(Area_ / dw**2)) for dw in d_hat]
    Nppp_ = [Np / Area for Np in Npart_]
    Xflow = X.reshape([-1,])
    Yflow = Y.reshape([-1,])
    # Combine Xflow and Yflow into data_points
    data_points = np.column_stack((Xflow, Yflow))
   #%% BUILDING PTV
    for dd in range(0,len(d_hat)):
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'Case d_hat = {d_hat[dd]}')
        Npart = Npart_[dd]
        Nppp = Nppp_[dd]
        out_path = f'Output/Case_{Npart}/'  # Assuming you are using Python 3.6 or later for f-string

# Check if the directory exists, and if not, create it
        if not os.path.exists(out_path):
            os.makedirs(out_path)  
        #%% BUILDING PTV
 # in this part we are going to create the PTV distribution, the total number of particles is a fraction of the reference one, they are allocated randomly in the domain   
        if os.path.isfile(out_path + 'PTV'):
            print(f'PTV already computed, loading...')
            with open(out_path + 'PTV', 'rb') as f:
                    PTV = pickle.load(f)
                    print(f'Loaded!')
        else:
            PTV = {} # initialize dictonary of PTV
            with ThreadPoolExecutor(max_workers=(60)) as executor:  # parallel computation of PTV snapshots
                results = list(tqdm(executor.map(process_PTV, NImg),total=len(NImg),desc='Processing snapshots',unit='snapshot', leave=True))
        
            # Assemble matrices
            for i, result in enumerate(tqdm(results, desc='Assembling Matrices', unit='snapshot')):
                PTV[i+1] = result
        
            # Save the results to a file
            with open(out_path + 'PTV', 'wb') as f:
                pickle.dump(PTV, f)
                
 
#%% BUILDING PIV        
        if os.path.isfile(out_path + 't_binned.npy'):
            print(f'PIV already computed, loading...')
            t_b = np.load(out_path + 't_binned.npy')
            
            print(f'Loaded!')
        else:
            print(f'Computing binned...')
            t_b = np.zeros((X.shape[0], X.shape[1], len(NImg)))
            b = round((10/(Nppp))**(1/2),2)
            t_b = Binning2D(PTV, NImg, b,  X, Y)
            np.save(out_path + 't_binned',t_b)
         
#%% POD PIV
        print(f'Computing POD binned')
        t_b = t_b.reshape([X.shape[0]*X.shape[1],len(NImg)])
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
            xGL,wGL = leggauss(20)
            xq, yq = np.meshgrid(xGL, xGL)
            xq = xq*(W[-1]-W[0])/2 + (W[-1]+W[0])/2
            yq = yq*(H[-1]-H[0])/2 + (H[-1]+H[0])/2
            XGL = np.column_stack((xq.flatten(), yq.flatten()))
            Int_GL_wrapped = partial(Int_GL, XGL = XGL,PTV = PTV)
            start_time = time.time()
            print(f"Parallel computation of all Gauss-Legenendre points")   
            num_iterations = len(NImg)
    
            with ThreadPoolExecutor(max_workers=(60)) as executor:
                UGL = list(tqdm(executor.map(Int_GL_wrapped, enumerate(NImg)), total=num_iterations,desc='Processing snapshots',unit='snapshot', leave=True))
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time} seconds")  
            
            K_est = np.zeros((len(NImg),len(NImg)))  
            WGL = np.outer(wGL, wGL).reshape(-1,1)
            start_time = time.time()
            # Perform parallel computation
            # Wrap compute_row_K function call with additional arguments using partial
            compute_row_K_wrapped = partial(compute_row_K, UGL=UGL, nt=len(NImg),WGL=WGL)
            # Perform parallel computation
            print(f"Parallel computation of K_est")   
            # # Use ThreadPoolExecutor to parallelize the computation
            with ThreadPoolExecutor(max_workers=(60)) as executor:
                     results = list(tqdm(executor.map(compute_row_K_wrapped, enumerate(K_est)), total=num_iterations,desc='Processing snapshots',unit='snapshot', leave=True))
            print(f"Time taken: {elapsed_time} seconds")           
                      
            K_est = np.array(results)
            K_est = K_est/4 
            nt=len(NImg)
            print("Computing lower diagonal K")  
            lower_diag_indices = np.tril_indices(nt, k=-1)
            K_est[lower_diag_indices] = K_est.T[lower_diag_indices]
            end_time = time.time()
            elapsed_time = end_time - start_time 
            Psi_free, Sigma_square, _ = np.linalg.svd(K_est, full_matrices=False)
            Sigma_free = np.sqrt((Sigma_square)) 
            np.save(out_path +'Psi_free',Psi_free)
            np.save(out_path +'Sigma_free',Sigma_free)
            np.save(out_path +'K_est',K_est)
       
        #%%projection of the velocity component on the output grid
        if os.path.isfile(out_path +'T_proj.npy'):
            print(f'projections already computed, loading...')
            T_proj = np.load(out_path +'T_proj.npy')
            
            print(f'Loaded!')
        else:
            interpolation_points = np.column_stack((X.flatten(), Y.flatten()))
            T_projection_wrapped = partial(U_projection, interp_grid = interpolation_points,PTV = PTV)
            start_time = time.time()
            print(f'projections ...')
            num_iterations = len(NImg)
    
            with ThreadPoolExecutor(max_workers=(60)) as executor:
                T_proj = list(tqdm(executor.map(T_projection_wrapped, enumerate(NImg)), total=num_iterations,desc='Processing snapshots',unit='snapshot', leave=True))
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time} seconds") 
            T_proj = np.array(T_proj)
            T_proj  = np.squeeze(T_proj)
            np.save(out_path + 'T_proj',T_proj)
        
       
        A = T_proj@T_proj.T
        Sigma_proj = []
        for j in range(len(NImg)):
            eigenvector = Psi_free[:, j]
            eigenvalue = rayleigh_quotient(eigenvector,A)
            Sigma_proj.append(np.sqrt(eigenvalue))
    
        print(f"Extracting Phi")
        Phi_free = T_proj.T@Psi_free@ np.diag(Sigma_free)
        Phi_free_norm = np.zeros_like(Phi_free)
        for n in range(0,len(NImg)):
            aa = Phi_free[:,n]/np.linalg.norm(Phi_free[:,n])
            Phi_free_norm[:,n] = aa
        np.save(out_path +'Phi_free',Phi_free_norm)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')   