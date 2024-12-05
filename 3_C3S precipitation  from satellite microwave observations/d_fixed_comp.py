# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:42:01 2024

@author: itirelli
"""
import numpy as np
import os
import pickle
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import scipy.io
import math 
from matplotlib.ticker import AutoMinorLocator

def rayleigh_quotient(x,A):
    return(x.T.dot(A).dot(x)/x.T.dot(x))


with open('Tref', 'rb') as f:
     TREF = pickle.load(f)
     print(f'Loaded reference field!')
Nt = TREF.shape[0]
NpRef = TREF.shape[1]
NImg = list(np.arange(1,Nt+1))
xboundary = [-50, 50]
yboundary = [0, 90]
with open('Grid', 'rb') as f:
     GRID = pickle.load(f)
     print(f'Loaded grid!')
X = GRID['LON']
Y = GRID['LAT']

 
dx_ = X[0,1]-X[0,0]
Area_D = (xboundary[-1]-xboundary[0])*(yboundary[-1]-yboundary[0])
Area = (math.radians(xboundary[-1])-math.radians(xboundary[0]))*(math.radians(yboundary[-1])-math.radians(yboundary[0])) # rad
d_hat = [0.015,0.02,0.025,0.03,0.035]
 
Npart_ = [int(np.floor(Area / dw**2)) for dw in d_hat]

 # Calculate the density
Nppp_ = [Np / Area for Np in Npart_]
 
Flow = {}
 
Flow['T'] = TREF
data_points = np.column_stack((X.flatten(), Y.flatten()))

errPSI_free = np.zeros((len(d_hat),6))
errPSI_b = np.zeros((len(d_hat),6))
errPHI_free = np.zeros((len(d_hat),3))
errPHI_b = np.zeros((len(d_hat),3))
errSig = np.zeros((2,len(d_hat)))
errREC = np.zeros((2,len(d_hat)))

print(f'DNS already computed, loading...')
with open('Tref', 'rb') as f:
     TREF = pickle.load(f)
     print(f'Loaded reference field!')
    
#%% EPTV
EPTV_name_stat = 'EPTV/EPTV_stat_Prep'  
flag = '2D'
print(f'EPTV already performed, loading...')
with open(EPTV_name_stat, 'rb') as f:
            EPTV_stat = pickle.load(f)
            print(f'Loaded!')

#%% POD DNS

import numpy.matlib
print(f'Computing POD DNS')
TmHR = EPTV_stat['TmEPTV'].flatten().reshape([1,-1])
TmHR = np.matlib.repmat(TmHR, len(NImg),1)
tREF = TREF[0:Nt,:]-TmHR
Psi_ref, Sigma_ref,Phi_ref= np.linalg.svd(tREF[0:Nt,:], full_matrices=False)
print(f'POD DNS computed')
Phi_ref = Phi_ref.T

cont = 0
for dd in range(0,len(d_hat)):
    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'Case d_hat = {d_hat[dd]}')
    Npart = Npart_[dd]
    Nppp = Nppp_[dd]
    folder_path = f'Output/Case_{Npart}/' 
    
    print(f'binned already computed, loading...')
    t_b = np.load(folder_path +'t_binned.npy')
   
    print(f'Loaded!')
    print(f'Computing POD binned')
    t_b = t_b[:,:,0:Nt].reshape([X.shape[0]*X.shape[1],len(NImg)])
    t_b = t_b.T
    Psi_b, Sigma_b,Phi_b= np.linalg.svd(t_b, full_matrices=False)
    print(f'POD binned computed')
    Phi_b = Phi_b.T
    print(f'Meshless POD already computed, loading...')
    Psi_free = np.load(folder_path +'Psi_free.npy')
    Sigma_free = np.load(folder_path +'Sigma_free.npy')
    Phi_free_norm = np.load(folder_path +'Phi_free.npy')
    print(f'Loaded!')
    print(f'projections already computed, loading...')
    T_proj = np.load(folder_path +'T_proj.npy')
    A = T_proj@T_proj.T
    Sigma_proj = []
    for j in range(len(NImg)):
        eigenvector = Psi_free[:, j]
        eigenvalue = rayleigh_quotient(eigenvector,A)
        Sigma_proj.append(np.sqrt(eigenvalue))
        
    SIGMA_b = np.diag(Sigma_b)
    SIGMA_free = np.diag(Sigma_proj)
    cont2 = 0
   
     #%% flipping
    temp = np.arange(0,6)
    corr_threshold = 0.8  # You can adjust this threshold as needed 
    flipb = np.zeros(len(temp))
    flipfree = np.zeros_like(flipb)
    Psi_free_flipped = np.zeros_like(Psi_free)
    Psi_b_flipped = np.zeros_like(Psi_free)
    Phi_free_flipped = np.zeros_like(Phi_free_norm)
    Phi_b_flipped = np.zeros_like(Phi_free_norm)
    for n in range(0,len(temp)):
         
          corr_b = np.corrcoef(Psi_ref[0:20, n], Psi_b[0:20, n])[0, 1]
          corr_free = np.corrcoef(Psi_ref[0:20, n], Psi_free[0:20, n])[0, 1]
          if corr_b > 0:
               flipb[n] = 1
          else:
              flipb[n] = -1
          
          if corr_free > 0:
                   flipfree[n] = 1
          else:
                  flipfree[n] = -1
          Psi_free_flipped[:,n] = Psi_free[:,n]*flipfree[n]
          Psi_b_flipped[:,n] = Psi_b[:,n]*flipb[n]
          Phi_free_flipped[:,n] = Phi_free_norm[:,n]*flipfree[n]
          Phi_b_flipped[:,n] = Phi_b[:,n]*flipb[n]
         
             
    for n in range (0,6):
       errPSI_free[cont,n] = root_mean_squared_error(Psi_ref[:, n], Psi_free_flipped[:, n]) / np.sqrt(np.mean(Psi_ref[:, n]**2))
       errPSI_b[cont,n] = root_mean_squared_error(Psi_ref[:, n], Psi_b_flipped[:, n]) / np.sqrt(np.mean(Psi_ref[:, n]**2))
    
    for n in range(3):
        errPHI_free[cont,n] = root_mean_squared_error(Phi_ref[0:X.shape[0]*X.shape[-1], n],Phi_free_flipped[0:X.shape[0]*X.shape[-1], n])/np.nanstd(Phi_ref[0:X.shape[0]*X.shape[-1], n])
        errPHI_b[cont,n] = root_mean_squared_error(Phi_ref[0:X.shape[0]*X.shape[-1], n],Phi_b_flipped[0:X.shape[0]*X.shape[-1], n])/np.nanstd(Phi_ref[0:X.shape[0]*X.shape[-1], n])
        
    cont = cont +1
      
      
    
#%% Plotting parameters

plt.rcParams['font.size'] = 10
plt.rcParams.update({"text.usetex": True,"font.family": "Helvetica"})
fontsize = 10
saveplot = 0

    
#%% Plotting PSI
  
import matplotlib.gridspec as gridspec
  
 # Create a shared x-axis and y-axis for all subplots
fig, axs = plt.subplots(2, 3, sharex=True, sharey='row', figsize=(3.2, 3.2*0.56))
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.80, wspace=0.45, hspace=0.2)

for n in range(0, 6):
    row = n // 3
    col = n % 3

    axs[row,col].set_ylim([0, 1])  # Different y-axis scale for the first plot
    axs[row,col].set_ylabel(f'$\delta_{{RMS_{{\psi_{n+1}}}}}$')
    axs[row,col].plot(d_hat, errPSI_b[:,n], 's--b', label="$Gridded$ $POD$", markersize=3, linewidth=0.5)
    axs[row,col].plot(d_hat, errPSI_free[:,n], '*--r', label="$Meshless$ $POD$", markersize=3, linewidth=0.5)
    
    axs[row,col].minorticks_on()
    axs[row,col].xaxis.set_minor_locator(AutoMinorLocator())
    axs[row,col].yaxis.set_minor_locator(AutoMinorLocator())
    
    if row == 1:
        axs[row,col].set_xlabel('$\hat{d}$')
    
    axs[row,col].set_xticks([0.015, 0.025, 0.035])

    axs[row,col].grid(True)
 
 # Add a legend
handles, labels = axs[0, 0].get_legend_handles_labels()
# Shift the legend to the right by adjusting bbox_to_anchor
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))

if saveplot == 1:
    plt.savefig('Prep_psi_dw.png', dpi=300, bbox_inches='tight') 


#%%phi+downsampling


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error
import cartopy.crs as ccrs
import cartopy.feature as cfeature

case = 2
folder_path = f'Output/Case_{Npart_[case]}/'  
print(f'Case d_hat = {d_hat[case]}')
print(f'PIV already computed, loading...')
t_b = np.load(folder_path +'t_binned.npy')
print(f'Loaded!')
print(f'Computing POD binned')
t_b = t_b[:,:,0:Nt].reshape([X.shape[0]*X.shape[1],len(NImg)])
t_b = t_b.T

Psi_b, Sigma_b,Phi_b= np.linalg.svd(t_b, full_matrices=False)

print(f'POD binned computed')
Phi_b = Phi_b.T
print(f'Meshless POD already computed, loading...')
Psi_free = np.load(folder_path +'Psi_free.npy')
Sigma_free = np.load(folder_path +'Sigma_free.npy')
Phi_free_norm = np.load(folder_path +'Phi_free.npy')
print(f'Loaded!')
temp = np.arange(0,6)
flipPIV = np.zeros(len(temp))
flipfree = np.zeros_like(flipPIV)
Psi_free_flipped = np.zeros_like(Psi_free)
Psi_b_flipped = np.zeros_like(Psi_free)
Phi_free_flipped = np.zeros_like(Phi_free_norm)
Phi_b_flipped = np.zeros_like(Phi_free_norm)
corr_threshold = 0.8  # You can adjust this threshold as needed 
for n in range(0,len(temp)):
     
      corr_PIV = np.corrcoef(Psi_ref[0:20, n], Psi_b[0:20, n])[0, 1]
      corr_free = np.corrcoef(Psi_ref[0:20, n], Psi_free[0:20, n])[0, 1]
      if corr_PIV > corr_threshold:
           flipPIV[n] = 1
      else:
          flipPIV[n] = -1
      if corr_free > corr_threshold:
               flipfree[n] = 1
      else:
              flipfree[n] = -1
    # Set a threshold for correlation coefficient
      
      Psi_free_flipped[:,n] = Psi_free[:,n]*flipfree[n]
      Psi_b_flipped[:,n] = Psi_b[:,n]*flipPIV[n]
      Phi_free_flipped[:,n] = Phi_free_norm[:,n]*flipfree[n]
      Phi_b_flipped[:,n] = Phi_b[:,n]*flipPIV[n]

  #%% Spatial modes 
import cartopy.crs as ccrs
import cartopy.feature as cfeature  

fig, axs = plt.subplots(3, 3, sharex=True, sharey='row', figsize=(4.5, 3.5), subplot_kw={'projection': ccrs.PlateCarree()})
cmap = plt.cm.get_cmap('coolwarm', 16)
clim = [-1, 1]
for n in range(3):
        # First row: Phi_PIV
        axs[0, n].coastlines()
        im1 = axs[0, n].pcolormesh(X, Y, Phi_ref[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), clim=clim,cmap = cmap,shading='gouraud')
        axs[0, n].contour(X, Y, Phi_ref[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), colors='k',linestyles='dashed', linewidths=0.5)
        
       
        if n == 0:
            axs[0, n].set_ylabel('$\mathrm{LAT} (^{\circ})$')
            
        axs[0, n].axis('equal')
        axs[0, n].axis('tight')
        axs[0, n].set_aspect('equal', adjustable='box')
        # axs[0, n].set_yticks([0, 45, 90])
        # axs[0, n].set_xticks([1, 5, 10])
        # Second row: Phi_free
        axs[1, n].coastlines()
        im2 = axs[1, n].pcolormesh(X, Y, Phi_b_flipped[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), clim=clim,cmap = cmap,shading='gouraud')
        
        axs[1, n].contour(X, Y, Phi_b_flipped[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), colors='k',linestyles='dashed', linewidths=0.5)
        # axs[1, n].contour(X, Y, mask, levels=[0.5], colors=['k'], linewidths=0.8)
        # axs[1, n].set_title(f'Phi_free {n+1}')
        # axs[1, n].set_xlabel('X')
        if n == 0:
            axs[1, n].set_ylabel('$\mathrm{LAT} (^{\circ})$')
           
        axs[1, n].axis('equal')
        axs[1, n].axis('tight')
        axs[1, n].set_aspect('equal', adjustable='box')
        # axs[1, n].set_yticks([0, 45, 90])
        # axs[1, n].set_xticks([1, 5, 10])
        # Third row: Phi_ref
        im3 = axs[2, n].pcolormesh(X, Y, Phi_free_flipped[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), clim=clim,cmap = cmap,shading='gouraud')
        axs[2, n].coastlines()
        axs[2, n].contour(X, Y, Phi_free_flipped[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), colors='k',linestyles='dashed', linewidths=0.5)
        # axs[2, n].contour(X, Y, mask, levels=[0.5], colors=['k'], linewidths=0.8)
        # axs[2, n].set_title(f'Phi_ref {n+1}')
        axs[2, n].set_xlabel('$\mathrm{LON} (^{\circ})$')
        if n == 0:
            axs[2, n].set_ylabel('$\mathrm{LAT} (^{\circ})$')
            
            
        axs[2, n].axis('equal')
        axs[2, n].axis('tight')
        axs[2, n].set_aspect('equal', adjustable='box')
        axs[2, n].set_xticks([-50, 0, 50])
        axs[2, 0].set_yticks([0, 45, 90])
        axs[1, 0].set_yticks([0, 45, 90])
        axs[0, 0].set_yticks([0, 45, 90])
       
        
        errorphifree = root_mean_squared_error(Phi_ref[0:X.shape[0]*X.shape[-1], n],Phi_free_flipped[0:X.shape[0]*X.shape[-1], n])/np.nanstd(Phi_ref[0:X.shape[0]*X.shape[-1], n])
        errorphiPIV =  root_mean_squared_error(Phi_ref[0:X.shape[0]*X.shape[-1], n],Phi_b_flipped[0:X.shape[0]*X.shape[-1], n])/np.nanstd(Phi_ref[0:X.shape[0]*X.shape[-1], n])
      
        print(f"RMSE for Phi {n+1} (Meshless approach): {errorphifree:.4f}")
        print(f"RMSE for Phi {n+1} (PIV approach): {errorphiPIV:.4f}")
    # Add a colorbar (choose one of the axes to use as the colorbar axis)
    
cax = fig.add_axes([0.25, 0., 0.5, 0.02])  # Position of colorbar
cbar = fig.colorbar(im1, cax=cax, orientation='horizontal')
    # Set the number of levels in the colorbar
    # cbar.locator = MaxNLocator(nbins=16)
    # cbar.update_ticks()
cax.set_xlabel('$\phi_i\sqrt{N_p}$')
fig.subplots_adjust(left=0.09, right=0.96, bottom=0.2, top=0.99, wspace=0.2, hspace=0.3)
    # Adjust spacing between subplots
    # plt.tight_layout()
plt.show()
if saveplot == 1:
        plt.savefig('Prep_Spatial_modes_comparison_N.png', dpi=300, bbox_inches='tight')
        

  

#%% Plotting PHI 
   # Create a shared x-axis and y-axis for all subplots
fig, axs = plt.subplots(1, 3, sharex=True, sharey='row', figsize=(3.5, 1.5*0.56))
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.58, wspace=0.5, hspace=0.2)
   
for n in range(0, 3):
       row = 0
       col = n
 
       axs[col].set_ylim([0, 0.8])  # Different y-axis scale for the first plot
       axs[col].set_yticks([0,0.4,0.8])  
       axs[col].set_ylabel(f'$\delta_{{RMS_{{\phi_{n+1}}}}}$')
       axs[col].plot(d_hat,errPHI_b[:,n], 's--b', label="$Gridded$ $POD$ ", markersize=3, linewidth=0.5)
       axs[col].plot(d_hat,errPHI_free[:,n], '*--r', label="$Meshless$ $POD$ ", markersize=3, linewidth=0.5)
       axs[col].scatter(d_hat[case], errPHI_b[case,n], color='blue', marker='o', facecolors='none', edgecolors='blue', s=100, linewidth=0.5, linestyle='dashed')
       axs[col].scatter(d_hat[case], errPHI_free[case,n], color='red', marker='o', facecolors='none', edgecolors='red', s=100, linewidth=0.5, linestyle='dashed')
       axs[col].set_xlabel('$\hat{d}$')
       axs[col].set_xticks([0.015,0.025,0.035])   
       axs[col].minorticks_on()
       axs[col].xaxis.set_minor_locator(AutoMinorLocator())
       axs[col].yaxis.set_minor_locator(AutoMinorLocator())
 
       axs[col].grid()
 
 # Add a legend
handles, labels = axs[ 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3)
if saveplot == 1:
    plt.savefig('Prep_phi_dw.png', dpi=300, bbox_inches='tight') 



