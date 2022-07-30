# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:18:59 2022

@author: Antoine Moneyron
"""

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from scipy.optimize import bisect
import scipy.interpolate as spinter
import hes1_langevin_Antoine as langevin
import hes1_master_Antoine as master
from numba import jit

def resolve_stationary_state(alpha_m,mu_m,alpha_p,mu_p,h,P_0):             #find stationary state for M and P
    
    def optim_func(x):                                                     #found from stat ODE
        return (alpha_m/mu_m)*1/(1+(x/P_0)**h) - mu_p/alpha_p*x            #alpha_m*f(P) - mu_m*M = 0 and alpha_p*M - mu_p*P =0

    p_stat=bisect(optim_func,0,10**6)                                  
    m_stat=mu_p/alpha_p*p_stat
    return m_stat,p_stat

'''Finds the stationary concentrations of mRNA and Hes1 (for non-oscillating systems).

    Parameters
    ----------

    P_0 : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    h : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold

    mu_m : float
        Rate at which mRNA is degraded, in copynumber per minute

    mu_p : float
        Rate at which Hes1 protein is degraded, in copynumber per minute

    alpha_m : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    alpha_p : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute.

    Returns
    -------

    M_stat : mRNA stationary concentration 
    
    P_stat : Hes1 stationary concentration 

'''


def lna_power_spectrum(alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,             
                                                      lambda_s=1,       
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      Omega=1,
                                                      T=1000,
                                                      delta_t=1):    
    n_t=int(T/delta_t)
    
    M_stat,P_stat=resolve_stationary_state(alpha_m,mu_m,alpha_p,mu_p,h,P_0)
    
    f_P=1/(1+(P_stat/P_0)**h)
    df_P=-h/P_0*(P_stat/P_0)**(h-1)/(1+(P_stat/P_0)**h)**2
    K=-alpha_m/lambda_s*df_P*f_P
    
    sigma_m2= alpha_m/Omega*f_P + mu_m/Omega*M_stat + alpha_m**2/lambda_s*  2*(P_stat/P_0)**h* f_P**3
    sigma_p2= alpha_p/Omega*M_stat + mu_p/Omega*P_stat
    
    freq=np.fft.fftfreq(n_t,d=delta_t)
    omega=2*np.pi*freq
    
    Delta2 = (  mu_m*mu_p - alpha_m*alpha_p*df_P*np.cos(omega*tau) - omega**2  )**2 +                                                                    (  omega*(mu_m+mu_p) + alpha_m*alpha_p*df_P*np.sin(omega*tau)  )**2
    num1   = (omega**2+mu_p**2)*sigma_m2 + (alpha_m*df_P)**2*sigma_p2
    num2   =  alpha_p**2*sigma_m2 +  (omega**2 + mu_m**2)*sigma_p2
            
    Sm=num1 / Delta2
    Sp=num2 / Delta2
        
    return freq,Sm,Sp

'''Computes the power spectrum (deterministic function) in the linear noise approximation.
   
   Parameters
    ----------
    
    T : float
        duration of the trace in minutes
        
    delta_t : float
        time step of the time mesh
    
    P_0 : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    h : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold
        
    lambda_s :float
        rate at which the environment switches. Higher values make it switch more often and limit switching induced diffusion.
        Also increase computation time.
        
    Omega : int
        size of the system. Higher values reduce demographic diffusion. Also increase (significantly) computation time

    mu_m : float
        Rate at which mRNA is degraded, in copynumber per minute

    mu_p : float
        Rate at which Hes1 protein is degraded, in copynumber per minute

    alpha_m : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    alpha_p : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute.
    
    Returns
    -------

    omega : 1D ndarray of shape int(T/delta_t)
        Angular frequencies associated with the time mesh i.e spanning [0,2 pi /delta_t] with step 2 pi /T
        
    Sm : 1D ndarray of shape int(T/delta_t)
        LNA power spectrum for mRNA concentrations, taken at frequency values given in 'freq'.
        
    Sp : 1D ndarray of shape int(T/delta_t)
        LNA power spectrum for Hes1 concentrations, taken at frequency values given in 'freq'.

'''



@jit
def compute_power_spectrum(t,table):
    n_iter,n_t=np.shape(table)
    delta_t=t[1]-t[0]
    T=t[-1]-t[0]
    freq=np.fft.fftfreq(n_t,d=delta_t)
    
    power_spectrum=np.mean( abs(np.fft.fft(table))**2, axis=0 )
    
    return freq,power_spectrum

'''Computes the mean power spectrum of a quantity (typically, a chemical concentration) from multiple trajectories.
   
   Parameters
    ----------
   
   t : 1D ndarray of shape int(T/delta_t)
       Time mesh at which the values (typically, concentrations) in table are given.
   
   table : 2D ndarray with int(T/delta_t) columns
       Values (typically, concentrations) at the times given in t.
       
   Returns
    -------

    omega : 1D ndarray of shape int(T/delta_t)
        Angular frequencies associated with the time mesh i.e spanning [0,2 pi / delta_t] with step 2 pi /T
        
    power_spectrum : 1D ndarray of shape int(T/delta_t)
        Power at angular frequencies given in omega.
        
'''


@jit
def compute_fourier_transform_mean_and_std(n_iter=100,alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,
                                                      lambda_s=1,        
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=10,
                                                      M_init=20,
                                                      T=1000,
                                                      delta_t=1,
                                                      Omega=1):    
    
    n_t=int(T/delta_t)
    n_stat=n_t//2

    
    t_ref,table_Mm,table_Pm=master.multiple_trajectories(n_iter=n_iter,alpha_m=alpha_m, alpha_p=alpha_p,mu_m=mu_m,
                                                              mu_p=mu_p,lambda_s=lambda_s,
                                                              P_0=P_0,
                                                              h=h,
                                                              tau=tau,
                                                              P_init=P_init,
                                                              M_init=M_init,
                                                              sigma_init=1,
                                                              T=T,
                                                              delta_t=delta_t,
                                                              Omega=Omega)
    
    _,table_Ml,table_Pl=langevin.multiple_trajectories(n_iter=n_iter,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=P_init,
                                                      M_init=M_init,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega)    
    
    
    omega,power_spectrum_Mm=compute_power_spectrum(t_ref[n_stat:],table_Mm[:,n_stat:])
    _,power_spectrum_Ml=compute_power_spectrum(t_ref[n_stat:],table_Ml[:,n_stat:])
    
    _,power_spectrum_Pm=compute_power_spectrum(t_ref[n_stat:],table_Pm[:,n_stat:])
    _,power_spectrum_Pl=compute_power_spectrum(t_ref[n_stat:],table_Pl[:,n_stat:])
    
    pool_Mm=np.reshape(table_Mm[:,n_stat:], n_iter*(n_t-n_stat))   #pool everything together
    pool_Ml=np.reshape(table_Ml[:,n_stat:], n_iter*(n_t-n_stat))   #pool everything together
    
    pool_Pm=np.reshape(table_Pm[:,n_stat:], n_iter*(n_t-n_stat))
    pool_Pl=np.reshape(table_Pl[:,n_stat:], n_iter*(n_t-n_stat))
    
    output={"std Mm": np.std(pool_Mm),"std Ml": np.std(pool_Ml),
           "mean Mm": np.mean(pool_Mm), "mean Ml": np.mean(pool_Ml),
           "power spectrum Mm": power_spectrum_Mm, "power spectrum Ml": power_spectrum_Ml,
           "std Pm": np.std(pool_Pm),"std Pl": np.std(pool_Pl),
           "mean Pm": np.mean(pool_Pm), "mean Pl": np.mean(pool_Pl),
           "power spectrum Pm": power_spectrum_Pm, "power spectrum Pl": power_spectrum_Pl,
           "times":t_ref, "frequencies":omega}
    
    return output

'''Computes mean, std and power spectrum all at once.
   
   Parameters
    ----------
    
    n_iter : int
        number of realisations Hes1 that are run
    
    T : float
        duration of the trace in minutes
        
    delta_t : float
        time step of the time mesh

    P_0 : float
        repression threshold, Hes autorepresses itself if its copynumber is larger
        than this repression threshold. Corresponds to P0 in the Monk paper

    h : float
        exponent in the hill function regulating the Hes autorepression. Small values
        make the response more shallow, whereas large values will lead to a switch-like
        response if the protein concentration exceeds the repression threshold
        
    lambda_s :float
        rate at which the environment switches. Higher values make it switch more often and limit switching induced diffusion.
        Also increase computation time.
        
    Omega : int
        size of the system. Higher values reduce demographic diffusion. Also increase (significantly) computation time

    mu_m : float
        Rate at which mRNA is degraded, in copynumber per minute

    mu_p : float
        Rate at which Hes1 protein is degraded, in copynumber per minute

    alpha_m : float
        Rate at which mRNA is described, in copynumber per minute, if there is no Hes
        autorepression. If the protein copy number is close to or exceeds the repression threshold
        the actual transcription rate will be lower

    alpha_p : float
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    tau : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.
        
    M_init : int
        initial mRNA concentration
        
    P_init : int
        initial Hes1 concentration
       
   Returns
    -------

    output : dictionary
        Dictionary with means, stds, power spectra, time mesh and associated angular frequencies.

'''
