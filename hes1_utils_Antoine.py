# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:18:59 2022

@author: tonio
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

@jit
def second_argmax(power_spectrum):
    temp = power_spectrum.copy()
    temp[0]=0
    return np.argmax(temp)


def theoretic_power_spectrum(alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,             
                                                      lambda_s=1,       
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      Omega=1,
                                                      T=1000,
                                                      delta_t=1,
                                                      drift="classical"):
    
    n_t=int(T/delta_t)
    
    M_stat,P_stat=resolve_stationary_state(alpha_m,mu_m,alpha_p,mu_p,h,P_0)
    
    f_P=1/(1+(P_stat/P_0)**h)
    df_P=-h/P_0*(P_stat/P_0)**(h-1)/(1+(P_stat/P_0)**h)**2
    K=-alpha_m/lambda_s*df_P*f_P
    
    sigma_m2= alpha_m/Omega*f_P + mu_m/Omega*M_stat + alpha_m**2/lambda_s*  2*(P_stat/P_0)**h* f_P**3
    sigma_p2= alpha_p/Omega*M_stat + mu_p/Omega*P_stat
    
    omega=np.fft.fftfreq(n_t,d=delta_t)
    
    if drift=="classical":
            Delta2 = (  mu_m*mu_p - alpha_m*alpha_p*df_P*np.cos(omega*tau) - omega**2  )**2 +                                                                    (  omega*(mu_m+mu_p) + alpha_m*alpha_p*df_P*np.sin(omega*tau)  )**2
            num1   = (omega**2+mu_p**2)*sigma_m2 + (alpha_m*df_P)**2*sigma_p2
            num2   =  alpha_p**2*sigma_m2 +  (omega**2 + mu_m**2)*sigma_p2
            
            Sm=num1 / Delta2/n_t
            Sp=num2 / Delta2/n_t
           
            return Sm*np.sqrt(2*np.pi),Sp*np.sqrt(2*np.pi)

        
    elif drift=="new":
            Delta2 = (  mu_m*mu_p - alpha_m*alpha_p*df_P*np.cos(omega*tau) - omega**2 - omega*alpha_p*K*np.sin(omega*tau) )**2 +                                  (  omega*(mu_m+mu_p - alpha_p*K*np.sin(omega*tau)) + alpha_m*alpha_p*df_P*np.sin(omega*tau)  )**2
            num1   = (omega**2+mu_p**2)*sigma_m2 + (alpha_m*df_P - mu_p*K)**2*sigma_p2
            num2   =  alpha_p**2*sigma_m2 + ( (omega- alpha_p*K*np.sin(omega*tau))**2 + (mu_m- alpha_p*K*np.cos(omega*tau))**2 )*sigma_p2
            
            Sm=num1 / Delta2/n_t
            Sp=num2 / Delta2/n_t
           
            return Sm*np.sqrt(2*np.pi),Sp*np.sqrt(2*np.pi)

@jit
def compute_power_spectrum(t,table):
    n_iter,n_t=np.shape(table)
    delta_t=t[1]-t[0]
    T=t[-1]-t[0]
    omega=np.fft.fftfreq(n_t,d=delta_t)
    
    power_spectrum=delta_t/T*np.mean( abs(np.fft.fft(table))**2, axis=0 )*1/n_t
    
    return omega,power_spectrum



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
    n_stat=min(n_t//2, int(1000/delta_t))

    
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
                                                      Omega=Omega,                                            
                                                      drift="classical")
    
    _,table_Mln,table_Pln=langevin.multiple_trajectories(n_iter=n_iter,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=P_init,
                                                      M_init=M_init,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega,                                            
                                                      drift="new")
    
    omega,power_spectrum_Mm=compute_power_spectrum(t_ref[n_stat:],table_Mm[:,n_stat:])
    _,power_spectrum_Ml=compute_power_spectrum(t_ref[n_stat:],table_Ml[:,n_stat:])
    _,power_spectrum_Mln=compute_power_spectrum(t_ref[n_stat:],table_Mln[:,n_stat:])
    
    _,power_spectrum_Pm=compute_power_spectrum(t_ref[n_stat:],table_Pm[:,n_stat:])
    _,power_spectrum_Pl=compute_power_spectrum(t_ref[n_stat:],table_Pl[:,n_stat:])
    _,power_spectrum_Pln=compute_power_spectrum(t_ref[n_stat:],table_Pln[:,n_stat:])
    
    pool_Mm=np.reshape(table_Mm[:,n_stat:], n_iter*(n_t-n_stat))   #pool everything together
    pool_Ml=np.reshape(table_Ml[:,n_stat:], n_iter*(n_t-n_stat))   #pool everything together
    pool_Mln=np.reshape(table_Mln[:,n_stat:], n_iter*(n_t-n_stat))   #pool everything together

    pool_Pm=np.reshape(table_Pm[:,n_stat:], n_iter*(n_t-n_stat))
    pool_Pl=np.reshape(table_Pl[:,n_stat:], n_iter*(n_t-n_stat))
    pool_Pln=np.reshape(table_Pln[:,n_stat:], n_iter*(n_t-n_stat))
    
    output={"std Mm": np.std(pool_Mm),"std Ml": np.std(pool_Ml),"std Mln": np.std(pool_Mln),
           "mean Mm": np.mean(pool_Mm), "mean Ml": np.mean(pool_Ml), "mean Mln": np.mean(pool_Mln),
           "power spectrum Mm": power_spectrum_Mm, "power spectrum Ml": power_spectrum_Ml, "power spectrum Mln": power_spectrum_Mln,
           "std Pm": np.std(pool_Pm),"std Pl": np.std(pool_Pl),"std Pln": np.std(pool_Pln),
           "mean Pm": np.mean(pool_Pm), "mean Pl": np.mean(pool_Pl), "mean Pln": np.mean(pool_Pln),
           "power spectrum Pm": power_spectrum_Pm, "power spectrum Pl": power_spectrum_Pl, "power spectrum Pln": power_spectrum_Pln,
           "times":t_ref, "frequencies":omega}
    
    return output