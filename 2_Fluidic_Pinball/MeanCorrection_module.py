# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:04:39 2023

@author: itirelli
"""
######################################## MODUL FUNCTION FOR MEAN CORRECTION (I.TIRELLI ET AL.2023) #################################

# In this module all the function necessary for the algorithm are collected.

from functools import partial
import multiprocessing as mp
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import numpy as np
from scipy.interpolate import griddata

def subtract_mean_vel3D(i, PTV, Fu, Fv, Fw):
    # Define the function that will be executed in parallel
    xp = PTV[i]['X']
    yp = PTV[i]['Y']
    zp = PTV[i]['Z']
    u = PTV[i]['U'] - Fu([xp, yp, zp])
    v = PTV[i]['V'] - Fv([xp, yp, zp])
    w = PTV[i]['W'] - Fw([xp, yp, zp])
    return u, v, w


def subtract_mean_velocities3D(PTV, x_grid, y_grid, z_grid, UmEPTV, VmEPTV, WmEPTV,NImg):
    # Define the main function that will call the parallel workers
    Fu = RegularGridInterpolator((x_grid, y_grid, z_grid), UmEPTV)
    Fv = RegularGridInterpolator((x_grid, y_grid, z_grid), VmEPTV)
    Fw = RegularGridInterpolator((x_grid, y_grid, z_grid), WmEPTV)

    with mp.Pool() as pool:
        subtract_mean_vel_worker = partial(subtract_mean_vel3D, PTV=PTV, Fu=Fu, Fv=Fv, Fw=Fw)
        results = []
        for i in tqdm(NImg, desc='Processing snapshots', unit='snapshot'):
            result = pool.apply_async(subtract_mean_vel_worker, args=(i,))
            results.append(result)
        pool.close()
        pool.join()

    for i, result in enumerate(tqdm(results, desc='Assembling matrices', unit='snapshot')):
        u, v, w = result.get()
        PTV[i]['u'] = u
        PTV[i]['v'] = v
        PTV[i]['w'] = w
        
    return PTV

def subtract_mean_vel2D(PTV, StartingGrid,UmEPTV, VmEPTV):
    # Define the function that will be executed in parallel
    
    FinalGrid = np.transpose(np.array([PTV['X'].flatten(),PTV['Y'].flatten()]))
    meanUEPTV_PTV = griddata(StartingGrid, UmEPTV.flatten(), FinalGrid, method='nearest')
    meanVEPTV_PTV = griddata(StartingGrid, VmEPTV.flatten(), FinalGrid, method='nearest')
    u = PTV['U'].flatten() - meanUEPTV_PTV
    v = PTV['V'].flatten() - meanVEPTV_PTV
    u = u.reshape(-1,1)
    v = v.reshape(-1,1)
    
    return u, v


def subtract_mean_velocities2D(PTV, StartingGrid, UmEPTV, VmEPTV,NImg):
    # Define the main function that will call the parallel workers
    

    with mp.Pool(processes = 60) as pool:
        subtract_mean_vel_worker = partial(subtract_mean_vel2D, StartingGrid=StartingGrid,UmEPTV=UmEPTV, VmEPTV=VmEPTV)
        results = []
        for i in tqdm(NImg, desc='Processing snapshots', unit='snapshot'):
            result = pool.apply_async(subtract_mean_vel_worker, args=(PTV[i],))
            results.append(result)
        pool.close()
        pool.join()

    for i, result in enumerate(tqdm(results, desc='Assembling matrices', unit='snapshot')):
        u, v = result.get()
        PTV[i+1]['u'] = u
        PTV[i+1]['v'] = v
        
    return PTV

def MeanCorrection (PTV,NImg,EPTV_stat,flag):
    
    if flag == '3D':
        UmEPTV = EPTV_stat['UmEPTV']
        VmEPTV = EPTV_stat['VmEPTV']
        WmEPTV = EPTV_stat['WmEPTV']
        X = EPTV_stat['X']
        Y = EPTV_stat['Y']
        Z = EPTV_stat['Z']
        # x_grid = X[1,:]
        # y_grid = Y[1,:]
        # z_grid = Z[1,:]
        PTV = subtract_mean_velocities3D(PTV, X, Y, Z, UmEPTV, VmEPTV, WmEPTV,NImg)
    elif flag == '2D':
        UmEPTV = EPTV_stat['UmEPTV']
        VmEPTV = EPTV_stat['VmEPTV']
        X = EPTV_stat['X']
        Y = EPTV_stat['Y']
        StartingGrid = np.transpose(np.array([X.flatten(),Y.flatten()]))
        PTV = subtract_mean_velocities2D(PTV, StartingGrid, UmEPTV, VmEPTV,NImg)
    else:
        raise ValueError(f"Invalid input shape: select flag  = 3D or 2D")
        
    
    return PTV