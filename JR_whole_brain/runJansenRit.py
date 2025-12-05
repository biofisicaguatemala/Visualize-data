# -*- coding: utf-8 -*-
"""
Modified version of the Jansen and Rit Neural Mass Model [1,2]

[1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked 
potential generation in a mathematical model of coupled cortical columns. 
Biological cybernetics, 73(4), 357-366.

[2] Coronel-Oliveros, C., Cofré, R., & Orio, P. (2021). Cholinergic neuromodulation 
of inhibitory interneurons facilitates functional integration in whole-brain models. 
PLoS computational biology, 17(2), e1008737.


"""

import numpy as np
from scipy import signal
import time
import BOLDModel as BD
import JansenRitModel as JR
import matplotlib.pyplot as plt
import importlib
import seaborn as sns
import psutil,os
try:
    psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS)
except Exception:
    pass



#Simulation parameters
JR.dt = 1E-3 #Integration step
JR.teq = 60 #Simulation time for stabilizing the system
JR.tmax = 200 + JR.teq * 2 #Length of simulated signals
ttotal = JR.teq + JR.tmax #Total simulation time
JR.downsamp = 10 #Downsampling to reduce the number of points        
Neq = int(JR.teq / JR.dt / JR.downsamp) #Number of points to discard
Nmax = int(JR.tmax / JR.dt / JR.downsamp) #Number of points of simulated signals
Ntotal = Neq + Nmax #Total number of points of simulation

seed = 0 #Random Seed

#Network parameters
JR.alpha = 1.3#Long-range pyramidal to pyramidal coupling
deco_mat = np.loadtxt('structural_Deco_AAL.txt')
norm = np.mean(np.sum(deco_mat,0)) #Normalization factor
M = deco_mat/norm
JR.M = deco_mat

JR.nnodes = len(JR.M) #number of nodes

#Node parameters
JR.r0 = 0.56 #Slope of pyramidal neurons sigmoid function
JR.p = 4.8 #Input mean
JR.sigma = 1 * np.sqrt(JR.dt) #Input standard deviation
JR.C4 = (0.3 + JR.alpha * 0.3 / 0.5) * 135 # Feedback inhibition

JR.seed = seed
init1 = time.time()
y, time_vector = JR.Sim(verbose = True)
pyrm = JR.C2 * y[:,1] - JR.C4 * y[:,2] + JR.C * JR.alpha * y[:,3] #EEG-like output of the model
end1 = time.time()

print([end1 - init1])


#%%
#Plot EEG-like signals

plt.figure(1)
plt.clf()
plt.plot(time_vector[Neq:(Neq + 10000)], pyrm[Neq:(Neq + 10000),0])
plt.tight_layout()
print(np.mean(np.corrcoef(pyrm[Neq:,:].T)))

#%%
#Power Spectral Density

#Welch method to stimate power spectal density (PSD)
#Remember: dt = original integration step, dws = downsampling           
window_length = 20 #in seconds
PSD_window = int(window_length / JR.dt / JR.downsamp) #Welch time window
PSD = signal.welch(pyrm[Neq:,:] - np.mean(pyrm[Neq:,:], axis = 0), fs = 1 / JR.dt / JR.downsamp, 
                    nperseg = PSD_window, noverlap = PSD_window // 2, 
                    scaling = 'density', axis = 0)
freq_vector = PSD[0] #Frequency values
PSD_curves =  PSD[1] #PSD curves for each node

#Power spectral density functions
plt.figure(3)
plt.clf()
plt.plot(freq_vector[1:-2], np.mean(10 * np.log10 (PSD_curves[1:-2,:]),axis=1))
plt.tight_layout()


    
#%%
#fMRI-BOLD response
init3 = time.time()

rE = JR.s(pyrm, JR.r0)  #Firing rates. Be careful when r0 = 0

BOLD_signals = BD.Sim(rE, JR.nnodes, JR.dt * JR.downsamp)
BOLD_signals = BOLD_signals[Neq:,:]

BOLD_downsamp = 100
BOLD_dt = JR.dt * JR.downsamp * BOLD_downsamp
BOLD_signals = BOLD_signals[::BOLD_downsamp,:]

#Filter the BOLD-like signal between 0.01 and 0.1 Hz
Fmin, Fmax = 0.01, 0.1
a0, b0 = signal.bessel(3, [2 * BOLD_dt * Fmin, 2 * BOLD_dt * Fmax], btype = 'bandpass')
BOLDfilt = signal.filtfilt(a0, b0, BOLD_signals[:,:], axis = 0)
cut0, cut1 = Neq // BOLD_downsamp, (Nmax - Neq) // BOLD_downsamp
BOLDfilt = BOLDfilt[cut0:cut1,:]
    
#Static Functional Connectivity (sFC) Matrix
sFC = np.corrcoef(BOLDfilt.T)

end3 = time.time()

print([end3 - init3])


#%%

#Filtered BOLD-like signals
plt.figure(5)
plt.clf()
plt.plot(BOLDfilt)
plt.tight_layout()

#sFC matrix
plt.figure(6)
plt.clf()
sns.heatmap(sFC, cmap = 'jet', vmin = 0, vmax = 1)
plt.tight_layout()

##echar un ojo a la matriz de conectividad
plt.figure(7)
plt.clf()
sns.heatmap(deco_mat)
plt.show()

