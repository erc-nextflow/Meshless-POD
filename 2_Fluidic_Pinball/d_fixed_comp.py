# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:42:01 2024

@author: itirelli
"""
import numpy as np
import os
import pickle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.ticker import AutoMinorLocator

def rayleigh_quotient(x,A):
    return(x.T.dot(A).dot(x)/x.T.dot(x))


Flow = {}
Flow_data = scipy.io.loadmat("U.mat")
Flow['U'] = Flow_data['U']
Flow_data = scipy.io.loadmat("V.mat")
Flow['V'] = Flow_data['V']
Flow_data = scipy.io.loadmat("GridX.mat")
Flow['X'] = Flow_data['GridX']
Flow_data = scipy.io.loadmat("GridY.mat")
Flow['Y'] = Flow_data['GridY']

   
xboundary = [1, 11]
yboundary = [-2, 2]
Area = (xboundary[-1]-xboundary[0])*(yboundary[-1]-yboundary[0])
index = np.where((Flow['X'] >= xboundary[0]) & (Flow['X'] <= xboundary[-1]) & (Flow['Y'] >= yboundary[0]) & (Flow['Y'] <= yboundary[-1]))
NpDNS = len(index[0])
d_hat = [0.1,0.4,0.6,0.8,1]
Npart_ = [int(np.floor(Area / dw**2)) for dw in d_hat]
Npart_[0] = np.min([Npart_[0],NpDNS])
# Calculate the density
Nppp_ = [Np / Area for Np in Npart_]

errPSI_free = np.zeros((len(d_hat),6))
errPSI_b = np.zeros((len(d_hat),6))
errPHI_free = np.zeros((len(d_hat),3))
errPHI_b = np.zeros((len(d_hat),3))
errSig = np.zeros((2,len(d_hat)))
errREC = np.zeros((2,len(d_hat)))
NImg = list(np.arange(1,4737+1))
Res = 32    #pix/D
dx = 4 #pixel
dx_ = dx/Res
xboundary = [1, 11]
yboundary = [-2, 2]
x = np.arange(xboundary[0], xboundary[-1] + dx_, dx_)
y = np.arange(yboundary[0], yboundary[-1] + dx_, dx_)
[X,Y] = np.meshgrid(x,y)
print(f'DNS already computed, loading...')
UDNS = np.load('dnsu.npy')
VDNS = np.load('dnsv.npy')
#%% EPTV
EPTV_name_stat = 'EPTV/EPTV_stat_pin'   
flag = '2D'
print(f'EPTV already performed, loading...')
# EPTV_stat = np.load(EPTV_name_stat)
with open(EPTV_name_stat, 'rb') as f:
            EPTV_stat = pickle.load(f)
            print(f'Loaded!')

#%% POD DNS
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
cont = 0
for dd in range(0,len(d_hat)):
    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'Case d_hat = {d_hat[dd]}')
    Npart = Npart_[dd]
    Nppp = Nppp_[dd]
    folder_path = f'Output/Case_{Npart}/'  
    
    print(f'PIV already computed, loading...')
    uPIV = np.load(folder_path + 'u_binned.npy')
    vPIV = np.load(folder_path + 'v_binned.npy')
    print(f'Loaded!')
    print(f'Computing POD PIV')
    uPIV = uPIV.reshape([X.shape[0]*X.shape[1],len(NImg)])
    vPIV = vPIV.reshape([X.shape[0]*X.shape[1],len(NImg)])
    uPIV = uPIV.T
    vPIV = vPIV.T
    PIV = np.concatenate((uPIV,vPIV),axis =1)
    Psi_PIV, Sigma_PIV,Phi_PIV= np.linalg.svd(PIV, full_matrices=False)
    print(f'POD PIV computed')
    Phi_PIV = Phi_PIV.T
    print(f'Meshless POD already computed, loading...')
    Psi_free = np.load(folder_path +'Psi_free.npy')
    Sigma_free = np.load(folder_path +'Sigma_free.npy')
    Phi_free_norm = np.load(folder_path +'Phi_free.npy')
    print(f'Loaded!')
    print(f'projections already computed, loading...')
    U_proj = np.load(folder_path +'U_proj.npy')
    A = U_proj@U_proj.T
    Sigma_proj = []
    for j in range(len(NImg)):
        eigenvector = Psi_free[:, j]
        eigenvalue = rayleigh_quotient(eigenvector,A)
        Sigma_proj.append(np.sqrt(eigenvalue))

    print(f"Extracting Phi")
    Phi_free = U_proj.T@Psi_free@ np.diag(Sigma_free)
    Phi_free_norm = np.zeros_like(Phi_free)
    for n in range(0,len(NImg)):
        Phi_free_norm[:,n] = Phi_free[:,n]/np.linalg.norm(Phi_free[:,n])
        
     #%% flipping
    temp = np.arange(0,7)
    flipPIV = np.zeros(len(temp))
    flipfree = np.zeros_like(flipPIV)
    Psi_free_flipped = np.zeros_like(Psi_free)
    Psi_PIV_flipped = np.zeros_like(Psi_free)
    Phi_free_flipped = np.zeros_like(Phi_free_norm)
    Phi_PIV_flipped = np.zeros_like(Phi_free_norm)
    
    corr_threshold = 0.8  # You can adjust this threshold as needed 
    for n in range(0,len(temp)):
          corr_PIV = np.corrcoef(Psi_ref[0:20, n], Psi_PIV[0:20, n])[0, 1]
          corr_free = np.corrcoef(Psi_ref[0:20, n], Psi_free[0:20, n])[0, 1]
          if corr_PIV > 0:
               flipPIV[n] = 1
          else:
              flipPIV[n] = -1
          
          if corr_free > 0:
                   flipfree[n] = 1
          else:
                  flipfree[n] = -1
          Psi_free_flipped[:,n] = Psi_free[:,n]*flipfree[n]
          Psi_PIV_flipped[:,n] = Psi_PIV[:,n]*flipPIV[n]
          Phi_free_flipped[:,n] = Phi_free_norm[:,n]*flipfree[n]
          Phi_PIV_flipped[:,n] = Phi_PIV[:,n]*flipPIV[n]
              
             
    for n in range (0,6):
       errPSI_free[cont,n] = np.sqrt(mean_squared_error(Psi_ref[:, n], Psi_free_flipped[:, n])) / np.sqrt(np.mean(Psi_ref[:, n]**2))
       errPSI_b[cont,n] = np.sqrt(mean_squared_error(Psi_ref[:, n], Psi_PIV_flipped[:, n])) / np.sqrt(np.mean(Psi_ref[:, n]**2))
    
    for n in range(3):
        errPHI_free[cont,n] = np.sqrt(mean_squared_error(Phi_ref[0:X.shape[0]*X.shape[-1], n],Phi_free_flipped[0:X.shape[0]*X.shape[-1], n]))/ np.sqrt(np.mean(Phi_ref[0:X.shape[0]*X.shape[-1], n]**2))
        errPHI_b[cont,n] =  np.sqrt(mean_squared_error(Phi_ref[0:X.shape[0]*X.shape[-1], n],Phi_PIV_flipped[0:X.shape[0]*X.shape[-1], n]))/ np.sqrt(np.mean(Phi_ref[0:X.shape[0]*X.shape[-1], n]**2))
        
    cont = cont +1
      
    
#%% Plotting parameters

plt.rcParams['font.size'] = 10
plt.rcParams.update({"text.usetex": True,"font.family": "Helvetica"})
fontsize = 10
saveplot = 0

    
#%% Plotting PSI
  
import matplotlib.gridspec as gridspec
  
 # Create a shared x-axis and y-axis for all subplots
fig, axs = plt.subplots(2, 3, sharex=True, sharey='row', figsize=(3.0, 3.0*0.56))
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.80, wspace=0.45, hspace=0.2)

for n in range(0, 6):
     row = n // 3
     col = n % 3
 
     axs[row,col].set_ylim([0, 1.5])  # Different y-axis scale for the first plot
     axs[row,col].set_ylabel(f'$\delta_{{RMS_{{\psi_{n+1}}}}}$')
     axs[row,col].plot(d_hat,errPSI_b[:,n], 's--b', label="$Binned$ $POD$ ", markersize=3, linewidth=0.5)
     axs[row,col].plot(d_hat,errPSI_free[:,n], '*--r', label="$Meshless$ $POD$ ", markersize=3, linewidth=0.5)
     axs[row,col].minorticks_on()
     axs[row,col].xaxis.set_minor_locator(AutoMinorLocator())
     axs[row,col].yaxis.set_minor_locator(AutoMinorLocator())
     
     if row == 1 :
         axs[row,col].set_xlabel('$\hat{d}$')
     
     axs[row,col].set_xticks([0,0.5,1])
 
     axs[row,col].grid()
 
 # Add a legend
handles, labels = axs[0, 0].get_legend_handles_labels()
# Shift the legend to the right by adjusting bbox_to_anchor
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))

if saveplot == 1:
    plt.savefig('Pinball_psi_dw.png', dpi=300, bbox_inches='tight') 
    

#%%phi+downsampling


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

case = 2
folder_path = f'Output/Case_{Npart_[case]}/'  
print(f'Case d_hat = {d_hat[case]}')
print(f'PIV already computed, loading...')

uPIV = np.load(folder_path + 'u_binned.npy')
vPIV = np.load(folder_path + 'v_binned.npy')
print(f'Loaded!')
print(f'Computing POD PIV')
uPIV = uPIV.reshape([X.shape[0]*X.shape[1],len(NImg)])
vPIV = vPIV.reshape([X.shape[0]*X.shape[1],len(NImg)])
uPIV = uPIV.T
vPIV = vPIV.T
PIV = np.concatenate((uPIV,vPIV),axis =1)
Psi_PIV, Sigma_PIV,Phi_PIV= np.linalg.svd(PIV, full_matrices=False)
print(f'POD PIV computed')
Phi_PIV = Phi_PIV.T
print(f'Meshless POD already computed, loading...')
Psi_free = np.load(folder_path +'Psi_free.npy')
Sigma_free = np.load(folder_path +'Sigma_free.npy')
Phi_free_norm = np.load(folder_path +'Phi_free.npy')
print(f'Loaded!')
temp = np.arange(0,4737)
flipPIV = np.zeros(len(temp))
flipfree = np.zeros_like(flipPIV)
Psi_free_flipped = np.zeros_like(Psi_free)
Psi_PIV_flipped = np.zeros_like(Psi_free)
Phi_free_flipped = np.zeros_like(Phi_free_norm)
Phi_PIV_flipped = np.zeros_like(Phi_free_norm)
corr_threshold = 0.5  # You can adjust this threshold as needed 
for n in range(0,len(temp)):
      corr_PIV = np.corrcoef(Psi_ref[0:20, n], Psi_PIV[0:20, n])[0, 1]
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
      Psi_PIV_flipped[:,n] = Psi_PIV[:,n]*flipPIV[n]
      Phi_free_flipped[:,n] = Phi_free_norm[:,n]*flipfree[n]
      Phi_PIV_flipped[:,n] = Phi_PIV[:,n]*flipPIV[n]

  #%% Spatial modes
    
fig, axs = plt.subplots(3, 3, sharex=True, sharey='row', figsize=(5,5*0.56))
cmap = plt.cm.get_cmap('coolwarm', 16)
clim = [-1, 1]
for n in range(3):
        # First row: Phi_PIV
        
        im1 = axs[0, n].pcolormesh(X, Y, Phi_PIV_flipped[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), clim=clim,cmap = cmap,shading='gouraud')
        axs[0, n].contour(X, Y, Phi_PIV_flipped[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), colors='k',linestyles='solid', linewidths=0.5)
        # axs[0, n].set_title(f'Phi_PIV {n+1}')
        # axs[0, n].set_xlabel('$X/D$')
        if n == 0:
            axs[0, n].set_ylabel('$Y/D$')
        axs[0, n].axis('equal')
        axs[0, n].axis('tight')
        axs[0, n].set_aspect('equal', adjustable='box')
        axs[0, n].set_xticks([1, 5, 10])
        # Second row: Phi_free
        im2 = axs[1, n].pcolormesh(X, Y, Phi_free_flipped[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), clim=clim,cmap = cmap,shading='gouraud')
        axs[1, n].contour(X, Y, Phi_free_flipped[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), colors='k',linestyles='solid', linewidths=0.5)
        # axs[1, n].set_title(f'Phi_free {n+1}')
        # axs[1, n].set_xlabel('X')
        if n == 0:
            axs[1, n].set_ylabel('$Y/D$')
        axs[1, n].axis('equal')
        axs[1, n].axis('tight')
        axs[1, n].set_aspect('equal', adjustable='box')
        axs[1, n].set_xticks([1, 5, 10])
        # Third row: Phi_ref
        im3 = axs[2, n].pcolormesh(X, Y, Phi_ref[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), clim=clim,cmap = cmap,shading='gouraud')
        axs[2, n].contour(X, Y, Phi_ref[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), colors='k',linestyles='solid', linewidths=0.5)
        # axs[2, n].set_title(f'Phi_ref {n+1}')
        axs[2, n].set_xlabel('$X/D$')
        if n == 0:
            axs[2, n].set_ylabel('$Y/D$')
        axs[2, n].axis('equal')
        axs[2, n].axis('tight')
        axs[2, n].set_aspect('equal', adjustable='box')
        axs[2, n].set_xticks([1, 5, 10])
        errorphifree = np.sqrt(mean_squared_error(Phi_ref[0:X.shape[0]*X.shape[-1], n],Phi_free_flipped[0:X.shape[0]*X.shape[-1], n]))/ np.sqrt(np.mean(Phi_ref[0:X.shape[0]*X.shape[-1], n]**2))
        errorphiPIV =  np.sqrt(mean_squared_error(Phi_ref[0:X.shape[0]*X.shape[-1], n],Phi_PIV_flipped[0:X.shape[0]*X.shape[-1], n]))/ np.sqrt(np.mean(Phi_ref[0:X.shape[0]*X.shape[-1], n]**2))
        print(f"RMSE for Phi {n+1} (Meshless approach): {errorphifree:.4f}")
        print(f"RMSE for Phi {n+1} (PIV approach): {errorphiPIV:.4f}")
    # Add a colorbar (choose one of the axes to use as the colorbar axis)
    
cax = fig.add_axes([0.25, 0, 0.5, 0.02])  # Position of colorbar
cbar = fig.colorbar(im1, cax=cax, orientation='horizontal')
cax.set_xlabel('$\phi_i\sqrt{N_p}$')
fig.subplots_adjust(left=0.09, right=0.96, bottom=0.18, top=0.99, wspace=0.2, hspace=0.02)
plt.show()
if saveplot == 1:
        plt.savefig('Spatial_modes_comparison_N.png', dpi=300, bbox_inches='tight')  


#%% Plotting PHI 
   # Create a shared x-axis and y-axis for all subplots
fig, axs = plt.subplots(1, 3, sharex=True, sharey='row', figsize=(3.0, 1.5*0.56))
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.58, wspace=0.5, hspace=0.2)
   
for n in range(0, 3):
       row = 0
       col = n
 
       axs[col].set_ylim([0, 2])  # Different y-axis scale for the first plot
       axs[col].set_yticks([0,1,2])  
       axs[col].set_ylabel(f'$\delta_{{RMS_{{\phi_{n+1}}}}}$')
       axs[col].plot(d_hat,errPHI_b[:,n], 's--b', label="$Binned$ $POD$ ", markersize=3, linewidth=0.5)
       axs[col].plot(d_hat,errPHI_free[:,n], '*--r', label="$Meshless$ $POD$ ", markersize=3, linewidth=0.5)
       axs[col].scatter(d_hat[case], errPHI_b[case,n], color='blue', marker='o', facecolors='none', edgecolors='blue', s=100, linewidth=0.5, linestyle='dashed')
       axs[col].scatter(d_hat[case], errPHI_free[case,n], color='red', marker='o', facecolors='none', edgecolors='red', s=100, linewidth=0.5, linestyle='dashed')
       axs[col].set_xlabel('$\hat{d}$')
       axs[col].set_xticks([0,0.5,1])     
       axs[col].minorticks_on()
       axs[col].xaxis.set_minor_locator(AutoMinorLocator())
       axs[col].yaxis.set_minor_locator(AutoMinorLocator())
 
       axs[col].grid()
 
handles, labels = axs[ 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3)
if saveplot == 1:
    plt.savefig('Pinball_phi_dw.png', dpi=300, bbox_inches='tight') 


