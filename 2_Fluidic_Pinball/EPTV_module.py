# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 19:10:32 2023

@author: itirelli
"""


######################################## FUNCTION FOR EPTV #################################
# In this module all the function necessary for the algorithm are collected.


import os
from scipy.spatial import cKDTree
import numpy as np
import time
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import pickle

def Binning3D(PTV, NImg, b,  EPTV_name_stat, X, Y, Z):
    os.makedirs(f"EPTV\\", exist_ok=True)
    bm = b / 2
    S1, S2, S3 = X.shape
   
    for i in NImg:
       
            print(f'Snap {i}')
            xp, yp, zp = PTV[i]['X'], PTV[i]['Y'], PTV[i]['Z']
            u, v, w = PTV[i]['U'], PTV[i]['V'], PTV[i]['W']
            tree = cKDTree(np.column_stack((xp, yp, zp)))
            U = np.zeros((X.shape[0], X.shape[1],X.shape[-1], len(NImg)))
            V = np.zeros_like(U)
            W = np.zeros_like(U)
            FlagPTV = np.zeros_like(U)
            for II in range(S1):
                for JJ in range(S2):
                    for KK in range(S3):
                        p = [X[II, JJ, KK], Y[II, JJ, KK], Z[II, JJ, KK]]
                        # range_ = np.column_stack((p - [bm, bm, bm], p + [bm, bm, bm]))
                        idxs = tree.query_ball_point(p, bm)
                        # idxs, dists = tree.query(p, k=None, distance_upper_bound=bm)
                        if idxs:
                            U[II, JJ, KK,i] = np.mean(u[idxs])
                            V[II, JJ, KK,i] = np.mean(v[idxs])
                            W[II, JJ, KK,i] = np.mean(w[idxs])
                            FlagPTV[II, JJ, KK,i] = 1
            tree = None
            return U,V,W
       
def Binning2D_worker(PTV,b,X,Y):
    
    xp, yp = PTV['X'], PTV['Y']
    u, v = PTV['U'], PTV['V']
    tree = cKDTree(np.column_stack((xp, yp)))
    U = np.zeros(X.shape)
    V = np.zeros_like(U)
    
    for II in range(X.shape[0]):
        for JJ in range(Y.shape[1]):
            p = [X[II, JJ], Y[II, JJ]]
            idxs = tree.query_ball_point(p, round((b / 2),3))
            if idxs:
                U[II, JJ] = np.mean(u[idxs])
                V[II, JJ] = np.mean(v[idxs])
                
                
    tree = None
    return U, V

def Binning2D(PTV, NImg, b, X, Y):
    os.makedirs(f"EPTV\\", exist_ok=True)
    num_snaps = len(NImg)
    Ub = np.zeros((X.shape[0], X.shape[1], num_snaps))
    Vb = np.zeros_like(Ub)
    Binning2D_worker_wrapped = partial(Binning2D_worker, b=b, X=X, Y=Y)
    with mp.Pool() as pool:
        results = []
        for i in tqdm(NImg, desc='Processing snapshots', unit='snapshot'):
            
            result = pool.apply_async(Binning2D_worker_wrapped, args=(PTV[i],))            
            results.append(result)
    
        for i, result in enumerate(tqdm(results, desc='Assemblying Matrices', unit='snapshot')):
            Ub[:, :, i], Vb[:, :, i] = result.get()
    pool.close()
    pool.join() 
    return Ub, Vb



def EPTV (NImg,Npp,X,Y,Z,PTV,dx,EPTV_name_stat,flag):
    # computing bin for EPTV
    EPTV_stat = {}
    if flag == '3D':
        beptv = ((1000/(len(NImg)*Npp))**(1/3))
        U,V,W = Binning3D(PTV,NImg, beptv, EPTV_name_stat, X, Y, Z)
        U[U == 0] = np.nan
        V[V == 0] = np.nan
        W[W == 0] = np.nan
        UmEPTV = np.nanmean(U, axis=3)
        VmEPTV = np.nanmean(V, axis=3)
        WmEPTV = np.nanmean(W, axis=3)
        UmEPTV[np.isnan(UmEPTV)] = 0
        VmEPTV[np.isnan(VmEPTV)] = 0
        WmEPTV[np.isnan(WmEPTV)] = 0
        EPTV_stat['UmEPTV'] = UmEPTV
        EPTV_stat['VmEPTV'] = VmEPTV
        EPTV_stat['WmEPTV'] = WmEPTV
        EPTV_stat['X'] = X
        EPTV_stat['Y'] = Y
        EPTV_stat['Z'] = Z
        # saving
        with open(EPTV_name_stat, 'wb') as f:
            pickle.dump(EPTV_stat, f)
    elif flag == '2D':
        beptv = round((1000/(len(NImg)*Npp))**(1/2),2)
        U,V = Binning2D(PTV, NImg, beptv,  X, Y)
        U[U == 0] = np.nan
        V[V == 0] = np.nan
        UmEPTV = np.nanmean(U, axis=2)
        VmEPTV = np.nanmean(V, axis=2)
        UmEPTV[np.isnan(UmEPTV)] = 0
        VmEPTV[np.isnan(VmEPTV)] = 0
        EPTV_stat['UmEPTV'] = UmEPTV
        EPTV_stat['VmEPTV'] = VmEPTV
        EPTV_stat['X'] = X
        EPTV_stat['Y'] = Y
        # saving
        with open(EPTV_name_stat, 'wb') as f:
            pickle.dump(EPTV_stat, f)
    else:
        raise ValueError(f"Invalid input shape: select flag  = 3D or 2D")
    
    return EPTV_stat

    