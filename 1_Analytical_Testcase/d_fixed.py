# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:11:23 2024

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
d_hat = [0.1 , 0.2, 0.3 , 0.4, 0.5   ]
Npart_ = [int(np.floor(Area_ / dw**2)) for dw in d_hat]
Nppp_ = [Np / Area for Np in Npart_]
Xflow = X.reshape([-1,])
Yflow = Y.reshape([-1,])
# Combine Xflow and Yflow into data_points
data_points = np.column_stack((Xflow, Yflow))

errPSI_free = np.zeros((len(d_hat),2))
errPSI_b = np.zeros((len(d_hat),2))
errPHI_free = np.zeros((len(d_hat),2))
errPHI_b = np.zeros((len(d_hat),2))
errSig = np.zeros((2,len(d_hat)))
errREC = np.zeros((2,len(d_hat)))



#%% POD DNS
Nt =len(NImg)
print(f'Computing POD DNS')
K=Dref.T.dot(Dref)
Psi_P,Lam_P,_=np.linalg.svd(K)
Psi_ref = np.vstack([psi_1,psi_2]).T
Phi_ref = np.vstack([phi_1_v.T,phi_2_v.T]).T
print(f'POD DNS computed')


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
    # t_b[mask_flat] = 0
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



     #%% flipping
    temp = np.arange(0,2)
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
          
             
    for n in range (0,2):
       errPSI_free[cont,n] = np.sqrt(mean_squared_error(Psi_ref[:, n], Psi_free_flipped[:, n])) / np.sqrt(np.mean(Psi_ref[:, n]**2))
       errPSI_b[cont,n] = np.sqrt(mean_squared_error(Psi_ref[:, n], Psi_b_flipped[:, n])) / np.sqrt(np.mean(Psi_ref[:, n]**2))
    
    for n in range(2):
        errPHI_free[cont,n] = np.sqrt(np.nanmean(Phi_ref[0:X.shape[0]*X.shape[-1], n]-Phi_free_flipped[0:X.shape[0]*X.shape[-1], n])**2)/np.nanstd(Phi_ref[0:X.shape[0]*X.shape[-1], n])
        errPHI_b[cont,n] = np.sqrt(np.nanmean(Phi_ref[0:X.shape[0]*X.shape[-1], n]-Phi_b_flipped[0:X.shape[0]*X.shape[-1], n])**2)/np.nanstd(Phi_ref[0:X.shape[0]*X.shape[-1], n])
        
    cont = cont +1
      
      
    
#%% Plotting parameters

plt.rcParams['font.size'] = 10
plt.rcParams.update({"text.usetex": True,"font.family": "Helvetica"})
fontsize = 10
saveplot = 0


#%% PSI
import matplotlib.gridspec as gridspec
  

fig, axs = plt.subplots(1, 2, sharex=True, sharey='row', figsize=(3.0, 3*0.56))
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.58, wspace=0.5, hspace=0.2)
   
for n in range(0, 2):
       row = 0
       col = n
 
       # axs[col].set_ylim([0, 2])  # Different y-axis scale for the first plot
       axs[col].set_ylabel(f'$\delta_{{RMS_{{\psi_{n+1}}}}}$')
       axs[col].plot(d_hat,errPSI_b[:,n], 's--b', label="$Gridded$ $POD$ ", markersize=3, linewidth=0.5)
       axs[col].plot(d_hat,errPSI_free[:,n], '*--r', label="$Meshless$ $POD$ ", markersize=3, linewidth=0.5)

       axs[col].set_xlabel('$\hat{d}$')
       axs[col].set_xticks([0,0.25,0.5])
       axs[col].set_yscale('log')
       axs[col].grid()
       axs[col].minorticks_on()
       axs[col].xaxis.set_minor_locator(AutoMinorLocator())
       axs[col].yaxis.set_minor_locator(AutoMinorLocator())
 
 # Add a legend
handles, labels = axs[ 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3)
 # Add a legend
# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=3)
if saveplot == 1:
    plt.savefig('analytic_psi_dw.png', dpi=300, bbox_inches='tight') 

#%% Plotting parameters



import matplotlib.gridspec as gridspec
  

    
fig, axs = plt.subplots(1, 2, sharex=True, sharey='row', figsize=(2.0*2, 2*0.56))
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.58, wspace=0.5, hspace=0.2)
   
for n in range(0, 2):
       row = 0
       col = n
 
       # axs[col].set_ylim([0, 0.5])  # Different y-axis scale for the first plot
       axs[col].set_ylabel(f'$\delta_{{RMS_{{\phi_{n+1}}}}}$')
       axs[col].plot(d_hat,errPHI_b[:,n], 's--b', label="$Gridded$ $POD$ ", markersize=3, linewidth=0.5)
       axs[col].plot(d_hat,errPHI_free[:,n], '*--r', label="$Meshless$ $POD$ ", markersize=3, linewidth=0.5)

       axs[col].set_xlabel('$\hat{d}$')
       axs[col].set_xticks(d_hat)
       axs[col].set_xticks([0,0.25,0.5])
       axs[col].set_yscale('log')
       axs[col].grid()
 
 # Add a legend
handles, labels = axs[ 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3)
 # Add a legend
# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=3)
if saveplot == 1:
    plt.savefig('analytic_phi_dw.png', dpi=300, bbox_inches='tight') 
    
#%% Load case
case = 2
Npart = Npart_[case]
folder_path = f'Output/Case_{Npart_[case]}/'  
print(f'Case d_hat = {d_hat[case]}')
print(f'b already computed, loading...')

Nppp = Nppp_[case]
folder_path = f'Output/Case_{Npart}/' 

print(f'binned already computed, loading...')
t_b = np.load(folder_path +'t_binned.npy')

print(f'Loaded!')
print(f'Computing POD binned')
t_b = t_b[:,:,0:Nt].reshape([X.shape[0]*X.shape[1],len(NImg)])
t_b = t_b.T
# t_b[mask_flat] = 0
Psi_b, Sigma_b,Phi_b= np.linalg.svd(t_b, full_matrices=False)
print(f'POD binned computed')
Phi_b = Phi_b.T
print(f'Meshless POD already computed, loading...')
Psi_free = np.load(folder_path +'Psi_free.npy')
Sigma_free = np.load(folder_path +'Sigma_free.npy')
Phi_free_norm = np.load(folder_path +'Phi_free.npy')
print(f'Loaded!')
temp = np.arange(0,2)
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

#%%       
fig, axs = plt.subplots(2, 3, sharex=True, sharey='row', figsize=(5,5*0.56))
cmap = plt.cm.get_cmap('coolwarm', 16)
clim = [-1, 1]
for n in range(2):
        # First row: Phi_b
        
        im1 = axs[n, 1].pcolormesh(X/lambda_2, Y/lambda_2, Phi_b_flipped[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), clim=clim,cmap = cmap,shading='gouraud')
        axs[n, 1].contour(X/lambda_2, Y/lambda_2, Phi_b_flipped[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), colors='k',linestyles='solid', linewidths=0.5)
       
        # axs[0, n].set_xlabel('$X/D$')
        
        axs[n, 0].set_ylabel('$Y/\lambda^*$')
        axs[n, 1].axis('equal')
        axs[n, 1].axis('tight')
        axs[n, 1].set_aspect('equal', adjustable='box')
        axs[n, 1].set_xticks([0, 3, 6])
        axs[n, 1].set_yticks([0, 3, 6])
        if n == 1 :
            axs[n, 1].set_xlabel('$X/\lambda^*$')

        # Second row: Phi_free
        im2 = axs[n, 2].pcolormesh(X/lambda_2, Y/lambda_2, Phi_free_flipped[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), clim=clim,cmap = cmap,shading='gouraud')
        axs[n, 2].contour(X/lambda_2, Y/lambda_2, Phi_free_flipped[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), colors='k',linestyles='solid', linewidths=0.5)
        # axs[1, n].set_title(f'Phi_free {n+1}')
        # axs[1, n].set_xlabel('X')
        
        
        axs[n, 2].axis('equal')
        axs[n, 2].axis('tight')
        axs[n, 2].set_aspect('equal', adjustable='box')
        axs[n, 2].set_xticks([0, 3, 6])
        axs[n, 2].set_yticks([0, 3, 6])
        if n == 1 :
            axs[n, 2].set_xlabel('$X/\lambda^*$')
        # Third row: Phi_ref
        im3 = axs[n, 0].pcolormesh(X/lambda_2, Y/lambda_2, Phi_ref[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), clim=clim,cmap = cmap,shading='gouraud')
        axs[n, 0].contour(X/lambda_2, Y/lambda_2, Phi_ref[0:X.shape[0]*X.shape[-1], n].reshape(X.shape)*np.sqrt(X.shape[0]*X.shape[-1]), colors='k',linestyles='solid', linewidths=0.5)
        # axs[2, n].set_title(f'Phi_ref {n+1}')
        axs[n, 0].axis('equal')
        axs[n, 0].axis('tight')
        axs[n, 0].set_aspect('equal', adjustable='box')
        axs[n, 0].set_xticks([0, 3, 6])
        axs[n, 0].set_yticks([0, 3, 6])
        if n == 1 :
            axs[n, 0].set_xlabel('$X/\lambda^*$')

        # errorphifree = np.sqrt(mean_squared_error(Phi_ref[0:X.shape[0]*X.shape[-1], n],Phi_free_flipped[0:X.shape[0]*X.shape[-1], n]))/np.std(Phi_ref[0:X.shape[0]*X.shape[-1], n])
        # errorphib = np.sqrt(mean_squared_error(Phi_ref[0:X.shape[0]*X.shape[-1], n],Phi_b_flipped[0:X.shape[0]*X.shape[-1], n]))/np.std(Phi_ref[0:X.shape[0]*X.shape[-1], n])
        errorphifree = np.sqrt(mean_squared_error(Phi_ref[0:X.shape[0]*X.shape[-1], n],Phi_free_flipped[0:X.shape[0]*X.shape[-1], n]))/ np.sqrt(np.mean(Phi_ref[0:X.shape[0]*X.shape[-1], n]**2))
        errorphib =  np.sqrt(mean_squared_error(Phi_ref[0:X.shape[0]*X.shape[-1], n],Phi_b_flipped[0:X.shape[0]*X.shape[-1], n]))/ np.sqrt(np.mean(Phi_ref[0:X.shape[0]*X.shape[-1], n]**2))
        print(f"RMSE for Phi {n+1} (Meshless approach): {errorphifree:.4f}")
        print(f"RMSE for Phi {n+1} (b approach): {errorphib:.4f}")
    # Add a colorbar (choose one of the axes to use as the colorbar axis)
    
cax = fig.add_axes([0.25, 0, 0.5, 0.02])  # Position of colorbar
cbar = fig.colorbar(im1, cax=cax, orientation='horizontal')
    # Set the number of levels in the colorbar
    # cbar.locator = MaxNLocator(nbins=16)
    # cbar.update_ticks()
cax.set_xlabel('$\phi_i\sqrt{N_p}$')
fig.subplots_adjust(left=0.09, right=0.96, bottom=0.18, top=0.99, wspace=0.02, hspace=0.2)
    # Adjust spacing between subplots
    # plt.tight_layout()
# axs[0, 0].set_title(f'Reference POD')
# axs[0, 1].set_title(f'Gridded POD')
# axs[0, 2].set_title(f'Meshless POD')
plt.show()
if saveplot == 1:
        plt.savefig('Analytical_Spatial_modes_comparison_N.png', dpi=300, bbox_inches='tight')  


#%% Plotting PHI 
   # Create a shared x-axis and y-axis for all subplots
fig, axs = plt.subplots(1, 2, sharex=True, sharey='row', figsize=(3.0, 2*0.56))
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.58, wspace=0.5, hspace=0.2)
   
for n in range(0, 2):
       row = 0
       col = n
 
       axs[col].set_ylim([0.000001, 1])  # Different y-axis scale for the first plot
       axs[col].set_yticks([0.000001,0.0001,1])  
       axs[col].set_ylabel(f'$\delta_{{RMS_{{\phi_{n+1}}}}}$')
       axs[col].plot(d_hat,errPHI_b[:,n], 's--b', label="$Gridded$ $POD$ ", markersize=3, linewidth=0.5)
       axs[col].plot(d_hat,errPHI_free[:,n], '*--r', label="$Meshless$ $POD$ ", markersize=3, linewidth=0.5)
       axs[col].scatter(d_hat[case], errPHI_b[case,n], color='blue', marker='o', facecolors='none', edgecolors='blue', s=100, linewidth=0.5, linestyle='dashed')
       axs[col].scatter(d_hat[case], errPHI_free[case,n], color='red', marker='o', facecolors='none', edgecolors='red', s=100, linewidth=0.5, linestyle='dashed')
       axs[col].set_xlabel('$\hat{d}$')
       axs[col].set_xticks([0,0.25,0.5])     
       axs[col].set_yscale('log')
       axs[col].minorticks_on()
       axs[col].xaxis.set_minor_locator(AutoMinorLocator())
       axs[col].yaxis.set_minor_locator(AutoMinorLocator())
       axs[col].grid()
 
 # Add a legend
handles, labels = axs[ 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3)
if saveplot == 1:
    plt.savefig('analytic_phi_dw.png', dpi=300, bbox_inches='tight') 