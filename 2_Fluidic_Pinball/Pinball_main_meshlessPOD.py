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
import EPTV_module
import MeanCorrection_module as MC
from concurrent.futures import ThreadPoolExecutor


def Int_DNS(data_points, interpolation_points, U, V):
     #function that interpolate DNS data
     interpolatorU = RBFInterpolator(data_points, U)
     interpolatorV = RBFInterpolator(data_points, V)
     U_result = interpolatorU(interpolation_points)
     V_result = interpolatorV(interpolation_points)
     
     return U_result, V_result    
      

def Int_PTV(Npart,data_points,interpolation_points,U,V):
    
    # function that create the PTV dictonary
    interpolatorU = RBFInterpolator(data_points, U)# thin plate spline as default
    interpolatorV = RBFInterpolator(data_points, V)
    U_result = interpolatorU(interpolation_points)
    V_result = interpolatorV(interpolation_points)
    xpart = interpolation_points[:,0]
    ypart = interpolation_points[:,1]
   
    return {'X': xpart.reshape([-1,1]), 'Y':ypart.reshape([-1,1]), 'U':U_result.reshape([-1,1]),'V':V_result.reshape([-1,1])}
  

def Binning2D_worker(PTV, b, X, Y):
    #binning function  for the single snapshot
    xp, yp = PTV['X'], PTV['Y']
    u,v = PTV['u'],PTV['v'],
    tree = cKDTree(np.column_stack((xp, yp)))
    U = np.zeros(X.shape)
    V  = np.zeros_like(U)
    
    for II in range(X.shape[0]):
        for JJ in range(Y.shape[1]):
            p = [X[II, JJ], Y[II, JJ]]
            idxs = tree.query_ball_point(p, round((b / 2), 3))
            if idxs:
                U[II, JJ] = np.mean(u[idxs])
                V[II, JJ] = np.mean(v[idxs])
                
    tree = None
    return U,V

def Binning2D_worker_wrapper(i, PTV, b, X, Y):
    #wrapped version of binning function
    Binning2D_worker_wrapped = partial(Binning2D_worker, PTV=PTV[i], b=b, X=X, Y=Y)
    return Binning2D_worker_wrapped()


def Binning2D(PTV, NImg, b, X, Y):
    # binning function main with parallelization
    num_snaps = len(NImg)
    Ub = np.zeros((X.shape[0], X.shape[1], num_snaps))
    Vb = np.zeros((X.shape[0], X.shape[1], num_snaps))
    with ThreadPoolExecutor(max_workers=60) as executor:
        results = list(tqdm(executor.map(lambda i: Binning2D_worker_wrapper(i, PTV, b, X, Y), NImg), 
                            total=len(NImg), 
                            desc='Processing snapshots', 
                            unit='snapshot', 
                            leave=True))
    
   
    for i, (U_result, V_result) in enumerate(tqdm(results, desc='Assembling Matrices', unit='snapshot')):
         Ub[:, :,i] = U_result
         Vb[:, :,i] = V_result
    return Ub,Vb

def Int_GL(args, XGL, PTV):
     #function to compute the value of velocity in the GL points
     i, _ = args
     data_points = np.column_stack((PTV[i+1]['X'], PTV[i+1]['Y']))
     interpolatorU = RBFInterpolator(data_points, PTV[i+1]['u'])
     interpolatorV = RBFInterpolator(data_points, PTV[i+1]['v'])
     U_int = interpolatorU(XGL)
     V_int = interpolatorV(XGL)
     U_GL = np.row_stack((U_int, V_int))
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
   interpolatorU = RBFInterpolator(data_points, PTV[i+1]['u'])
   interpolatorV = RBFInterpolator(data_points, PTV[i+1]['v'])
   U_int = interpolatorU(interp_grid)
   V_int = interpolatorV(interp_grid)
   U_proj = np.row_stack((U_int,V_int))
   return U_proj

def rayleigh_quotient(x,A):
    return(x.T.dot(A).dot(x)/x.T.dot(x))


def process_PTV(i):
    #function to generate random sensor positions 
    xpart = np.random.uniform(xboundary[0], xboundary[1], size=Npart)
    ypart = np.random.uniform(yboundary[0], yboundary[1], size=Npart)
    interpolation_points = np.column_stack((xpart, ypart))
    Int_PTV_wrapped = partial(Int_PTV, Npart, data_points)
    return Int_PTV_wrapped(interpolation_points, Flow['U'][i-1, index[0]], Flow['V'][i-1, index[0]])



     #%%  SETTING  
if __name__ == '__main__':    
  
    # Preparing all the data
    Flow = {}
    Flow_data = scipy.io.loadmat("U.mat")
    Flow['U'] = Flow_data['U']
    Flow_data = scipy.io.loadmat("V.mat")
    Flow['V'] = Flow_data['V']
    Flow_data = scipy.io.loadmat("GridX.mat")
    Flow['X'] = Flow_data['GridX']
    Flow_data = scipy.io.loadmat("GridY.mat")
    Flow['Y'] = Flow_data['GridY']
    
    # Define doamin
    xboundary = [1, 11] # D
    yboundary = [-2, 2] # D
    Area = (xboundary[-1]-xboundary[0])*(yboundary[-1]-yboundary[0]) # computing area
    # find particles in the domain
    index = np.where((Flow['X'] >= xboundary[0]) & (Flow['X'] <= xboundary[-1]) & (Flow['Y'] >= yboundary[0]) & (Flow['Y'] <= yboundary[-1]))
    
    NpDNS = len(index[0]) # number of particles for the reference
    Res = 32    # Resolution in pix/D
    NImg = list(np.arange(1,4737+1)) # number of snapshots
    dx = 4 # grid distance pixel
    dx_ = dx/Res # grid distance dimensionless
    # defining grid
    x = np.arange(xboundary[0], xboundary[-1] + dx_, dx_)
    y = np.arange(yboundary[0], yboundary[-1] + dx_, dx_)
    [X,Y] = np.meshgrid(x,y)
    # Reference DNS location particles
    Xflow = Flow['X'][index[0]].reshape([len(index[0]),])
    Yflow = Flow['Y'][index[0]].reshape([len(index[0]),])
    
    d_hat = [0.1,0.4,0.6,0.8,1] #sparsity level (see paper)
    Npart_ = [int(np.floor(Area / dw**2)) for dw in d_hat]
    Npart_[0] = np.min([Npart_[0],NpDNS])
    # Calculate the density for each case
    Nppp_ = [Np / Area for Np in Npart_]
     
     #%% BUILDING Reference
    
    # in this part we are going to create the reference distribution, we interpolate these data on the output grid
    
   # Combine Xflow and Yflow into data_points
    data_points = np.column_stack((Xflow, Yflow))
    
    # Generate interpolation points
    interpolation_points = np.column_stack((X.flatten(), Y.flatten()))
    
    if os.path.isfile('dnsu.npy'):
        print(f'Reference already computed, loading...')
        UDNS = np.load('dnsu.npy')
        VDNS = np.load('dnsv.npy')
    
    else:
        print(f'Computing reference...')
        # Initialize UDNS and VDNS arrays
        UDNS = np.zeros([len(NImg), len(interpolation_points)])
        VDNS = np.zeros_like(UDNS)
        
        # Define a partial function with fixed arguments
        Int_DNS_wrapped = partial(Int_DNS, data_points, interpolation_points)
        
        # Create a multiprocessing pool
        with Pool() as pool:
            results = []
            for i in tqdm(NImg, desc='Processing snapshots', unit='snapshot'):
                result = pool.apply_async(Int_DNS_wrapped, (Flow['U'][i-1, index[0]].reshape([len(index[0])]), Flow['V'][i-1, index[0]].reshape([len(index[0])])))
                results.append(result)
        
            for i, result in enumerate(tqdm(results, desc='Assembling Matrices', unit='snapshot')):
                  UDNS[i, :], VDNS[i, :] = result.get()
        
            # Close the pool
        pool.close()
        pool.join()
  
 #%% for loop on the sparsity levels
    for dd in range(0,len(d_hat)):
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'Case d_hat = {d_hat[dd]}')
        Npart = Npart_[dd]
        Nppp = Nppp_[dd]
        out_path = f'Output/Case_{Npart}/'  # Assuming you are using Python 3.6 or later for f-string

# Check if the directory exists, and if not, create it
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
#%% BUILDING SCATTERED DISTRIBUTION (PTV)
 # in this part we are going to create the PTV distribution, they are allocated randomly in the domain   

        if os.path.isfile(out_path + 'PTV'):
            print(f'PTV already computed, loading...')
            with open(out_path +'PTV', 'rb') as f:
                    PTV = pickle.load(f)
                    print(f'Loaded!')
        else:
            PTV = {} # initialize dictonary of PTV
            with ThreadPoolExecutor() as executor: # parallel computation of PTV snapshots
                results = list(tqdm(executor.map(process_PTV, NImg),total=len(NImg),desc='Processing snapshots',unit='snapshot', leave=True))
        
            # Assemble matrices
            for i, result in enumerate(tqdm(results, desc='Assembling Matrices', unit='snapshot')):
                PTV[i+1] = result
        
            # Save the results to a file
            with open(out_path +'PTV', 'wb') as f:
                pickle.dump(PTV, f)
                    
     #%% EPTV
     # in this section EPTV is performed, this is needed to avoid modulation effect on the mean flow (I.Tirelli et al 2023, A simple trick to improve the accuracy of PIV/PTV data)
        EPTV_name_stat = 'EPTV/EPTV_stat_pin'   
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
   
        if os.path.isfile(out_path +'u_binned.npy'):
                print(f'PIV already computed, loading...')
                uPIV = np.load(out_path +'u_binned.npy')
                vPIV = np.load(out_path +'v_binned.npy')
                
                print(f'Loaded!')
        else:
                print(f'Computing binned...')
                b = round((10/(Nppp))**(1/2),2) # BIN SIZE
                uPIV,vPIV = Binning2D(PTV, NImg, b,  X, Y)
                np.save(out_path +'u_binned',uPIV)
                np.save(out_path +'v_binned',vPIV)
    
    #%% POD Reference
        import numpy.matlib
        print(f'Computing POD DNS')
        UmHR = EPTV_stat['UmEPTV'].flatten().reshape([1,-1])
        UmHR = np.matlib.repmat(UmHR, len(NImg),1)
        VmHR = EPTV_stat['VmEPTV'].flatten().reshape([1,-1])
        VmHR = np.matlib.repmat(VmHR, len(NImg),1)
        uDNS = UDNS-np.mean(UDNS,0)
        vDNS = VDNS-np.mean(VDNS,0)
        DNS = np.concatenate((uDNS,vDNS),axis = 1)
        Psi_ref, Sigma_ref,Phi_ref= np.linalg.svd(DNS, full_matrices=False)
        print(f'POD DNS computed')
        Phi_ref = Phi_ref.T
        
    #%% POD Binned
        print(f'Computing POD PIV')
        uPIV = uPIV.reshape([X.shape[0]*X.shape[1],len(NImg)])
        vPIV = vPIV.reshape([X.shape[0]*X.shape[1],len(NImg)])
        uPIV = uPIV.T
        vPIV = vPIV.T
        PIV = np.concatenate((uPIV,vPIV),axis =1)
        Psi_PIV, Sigma_PIV,Phi_PIV= np.linalg.svd(PIV, full_matrices=False)
        print(f'POD PIV computed')
        Phi_PIV = Phi_PIV.T
    
    #%% Meshless POD
        if os.path.isfile(out_path +'Psi_free.npy'):
            print(f'POD already computed, loading...')
            Psi_free = np.load(out_path +'Psi_free.npy')
            Sigma_free = np.load(out_path +'Sigma_free.npy')
            print(f'Loaded!')
        else:
            print(f'Computing POD meshless')
            W = xboundary
            H = yboundary
            xGL,wGL = leggauss(20)  # select number of G-L points and compute the corresponding weights
            xq, yq = np.meshgrid(xGL, xGL) # define grid of GL points
            xq = xq*(W[-1]-W[0])/2 + (W[-1]+W[0])/2 # scaling 
            yq = yq*(H[-1]-H[0])/2 + (H[-1]+H[0])/2 # scaling 
            xq = xq*(W[-1]-W[0])/2 + (W[-1]+W[0])/2
            yq = yq*(H[-1]-H[0])/2 + (H[-1]+H[0])/2
            XGL = np.column_stack((xq.flatten(), yq.flatten()))
            Int_GL_wrapped = partial(Int_GL, XGL = XGL,PTV = PTV)
            start_time = time.time()
            print(f"Parallel computation of all Gauss-Legenendre points")   
            num_iterations = len(NImg)
    
            with ThreadPoolExecutor() as executor:
                UGL = list(tqdm(executor.map(Int_GL_wrapped, enumerate(NImg)), total=num_iterations,desc='Processing snapshots',unit='snapshot', leave=True))
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time} seconds")  
            # computing correaltion matrix
            K_est = np.zeros((len(NImg),len(NImg)))  
            # assembling GL weights matrix
            WGL = np.outer(wGL, wGL).reshape(-1,1)
            WGL = np.tile(WGL.reshape(-1,1), (2,1))
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
                    
            Psi_free, Sigma_square, _ = np.linalg.svd(K_est, full_matrices=False) #  K = PSI*SIGMA**2*PSI'
            Sigma_free = np.sqrt((Sigma_square)) 

            np.save(out_path +'Psi_free',Psi_free)
            np.save(out_path +'Sigma_free',Sigma_free)
            np.save(out_path +'K_est',K_est)
       
       #%% Projection of the velocity component on the output grid
        # In order to extract the spatial modes we need to project the velocity fields on the grid
        if os.path.isfile(out_path +'U_proj.npy'):
            print(f'projections already computed, loading...')
            U_proj = np.load(out_path +'U_proj.npy')
            
            print(f'Loaded!')
        else:
            U_projection_wrapped = partial(U_projection, interp_grid = interpolation_points,PTV = PTV)
            start_time = time.time()
            print(f'projections ...')
            num_iterations = len(NImg)
    
            with ThreadPoolExecutor(max_workers=60) as executor:
                # computing projection in parallel
                U_proj = list(tqdm(executor.map(U_projection_wrapped, enumerate(NImg)), total=num_iterations,desc='Processing snapshots',unit='snapshot', leave=True))
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time} seconds") 
            U_proj = np.array(U_proj)
            U_proj  = np.squeeze(U_proj)
            np.save(out_path +'U_proj',U_proj)
        
        # assembling matrices for rayleigh quotient sigma_i = psi_i'Kpsi_i/psi_i'psi_i
        A = U_proj@U_proj.T
        Sigma_proj = []
        for j in range(len(NImg)):
            eigenvector = Psi_free[:, j]
            eigenvalue = rayleigh_quotient(eigenvector,A)
            Sigma_proj.append(np.sqrt(eigenvalue))
    
        print(f"Extracting Phi")
        Phi_free = U_proj.T@Psi_free@ np.diag(Sigma_free)
        Phi_free_norm = np.zeros_like(Phi_free)
        ## normalization of PHI
        for n in range(0,len(NImg)):
            Phi_free_norm[:,n] = Phi_free[:,n]/np.linalg.norm(Phi_free[:,n])
        np.save(out_path +'Phi_free',Phi_free_norm)
        
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')  
    
