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

def resolve_stationary_state(alpha_m,mu_m,alpha_p,mu_p,h,P_0):             #find stationary state for M and P found from stationary ODE
    def optim_func(x):                                                     
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
    
    sigma_m2= alpha_m/Omega*f_P + mu_m/Omega*M_stat + alpha_m**2/lambda_s*  2*(P_stat/P_0)**h* f_P**3
    sigma_p2= alpha_p/Omega*M_stat + mu_p/Omega*P_stat
    
    
    freq=np.fft.fftfreq(n_t,d=delta_t)
    if n_t%2 == 0: 
        no_real_frequencies = len(freq)//2
    else:
        no_real_frequencies = len(freq)//2-1
    freq = freq[:no_real_frequencies]
    omega=2*np.pi*freq
    
    Delta2 = ((  mu_m*mu_p - alpha_m*alpha_p*df_P*np.cos(omega*tau) - omega**2  )**2 +
              (  omega*(mu_m+mu_p) + alpha_m*alpha_p*df_P*np.sin(omega*tau)  )**2)
    num1   = (omega**2+mu_p**2)*sigma_m2 + (alpha_m*df_P)**2*sigma_p2
    num2   =  alpha_p**2*sigma_m2 +  (omega**2 + mu_m**2)*sigma_p2
            
    Sm=num1 / Delta2
    Sp=num2 / Delta2
        
    return omega,Sm,Sp

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

@jit(nopython = True)
def compute_power_spectrum_traj(t,traj):
    n_t=len(traj)
    delta_t=t[1]-t[0]
    T=t[-1]-t[0]
    trajectory_for_calculation = traj - np.mean(traj)
    
    freq=np.fft.fftfreq(n_t,d=delta_t)
    if n_t%2 == 0: 
        no_real_frequencies = len(freq)//2
    else:
        no_real_frequencies = len(freq)//2-1
    
    freq=freq[:no_real_frequencies]*2*np.pi
    
    power_spectrum = np.abs(np.fft.fft(trajectory_for_calculation))**2*T/(n_t**2)
    power_spectrum = power_spectrum[:no_real_frequencies]
    
    return freq, power_spectrum



@jit
def compute_power_spectrum(t,table):
    n_t = len(table)
    delta_t=t[1]-t[0]
    T=t[-1]-t[0]
    freq=np.fft.fftfreq(n_t,d=delta_t)
    freq=freq*2*np.i
    
    power_spectrum= abs(np.fft.fft(table))**2*T/(n_t**2)*np.sqrt(2/np.pi)
    
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
def compute_fourier_transform_mean_and_std(n_iter=100, alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,
                                                      lambda_s=1,        
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=1000,
                                                      delta_t=0.1,
                                                      Omega=1,
                                                      sampling_timestep = 1.0):    
    
    n_t=int(T/delta_t)
    n_stat=n_t//2
    
    t_ref=np.arange(0,T,delta_t)
    freq=np.fft.fftfreq(n_t-n_stat,d=delta_t)
    
    var_Mm, var_Ml, var_Mlna = np.zeros(n_iter),np.zeros(n_iter),np.zeros(n_iter)
    var_Pm, var_Pl, var_Plna = np.zeros(n_iter),np.zeros(n_iter),np.zeros(n_iter)
    mean_Mm, mean_Ml, mean_Mlna = np.zeros(n_iter),np.zeros(n_iter),np.zeros(n_iter)
    mean_Pm, mean_Pl, mean_Plna = np.zeros(n_iter),np.zeros(n_iter),np.zeros(n_iter)
    
    power_spectrum_Mm=np.zeros(n_t//2)
    power_spectrum_Ml=np.zeros(n_t//2)
    power_spectrum_Mlna=np.zeros(n_t//2)
    
    power_spectrum_Pm=np.zeros(n_t//2)
    power_spectrum_Pl=np.zeros(n_t//2)
    power_spectrum_Plna=np.zeros(n_t//2)
    
    
    
    for i in range(n_iter):
        tm,Mm,Pm=master.one_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p, lambda_s=lambda_s,
                                                              P_0=P_0,
                                                              h=h,
                                                              tau=tau,
                                                              P_init=P_init,
                                                              M_init=M_init,
                                                              sigma_init=1,
                                                              T=T,
                                                              Omega=Omega,
                                                              sampling_timestep = 1.0,
                                                              delta_t_sampling = delta_t)
        
        Mm=spinter.interp1d(tm,Mm,kind="zero")(t_ref)
        Pm=spinter.interp1d(tm,Pm,kind="zero")(t_ref)    
    
        _,Ml,Pl=langevin.one_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=P_init,
                                                      M_init=M_init,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega,
                                                      sampling_timestep = 1.0)
            
    
        _,Mlna,Plna=langevin.one_trajectory_LNA(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega,
                                                      sampling_timestep = 1.0)
                
        mean_Mm[i] = np.mean(Mm)
        mean_Ml[i] = np.mean(Ml)
        mean_Mlna[i] = np.mean(Mlna)
        
        mean_Pm[i] = np.mean(Pm)
        mean_Pl[i] = np.mean(Pl)
        mean_Plna[i] = np.mean(Plna)
        
        power_spectrum_Mm += compute_power_spectrum_traj(t_ref[n_stat:],Mm[n_stat:])/n_iter
        power_spectrum_Ml += compute_power_spectrum_traj(t_ref[n_stat:],Ml[n_stat:])/n_iter
        power_spectrum_Mlna += compute_power_spectrum_traj(t_ref[n_stat:],Mlna[n_stat:])/n_iter

        power_spectrum_Pm += compute_power_spectrum_traj(t_ref[n_stat:],Pm[n_stat:])/n_iter
        power_spectrum_Pl += compute_power_spectrum_traj(t_ref[n_stat:],Pl[n_stat:])/n_iter
        power_spectrum_Plna += compute_power_spectrum_traj(t_ref[n_stat:],Plna[n_stat:])/n_iter
        
        var_Mm[i] = np.var(Mm)
        var_Ml[i] = np.var(Ml)
        var_Mlna[i] = np.var(Mlna)
        
        var_Pm[i] = np.var(Pm)
        var_Pl[i] = np.var(Pl)
        var_Plna[i] = np.var(Plna)
        
        
    var_Mm_g = np.mean(var_Mm) + np.var(mean_Mm)
    var_Ml_g = np.mean(var_Ml) + np.var(mean_Ml)
    var_Mlna_g = np.mean(var_Mlna) + np.var(mean_Mlna)
    
    var_Pm_g = np.mean(var_Pm) + np.var(mean_Pm)
    var_Pl_g = np.mean(var_Pl) + np.var(mean_Pl)
    var_Plna_g = np.mean(var_Plna) + np.var(mean_Plna)
    
    
    output={"std Mm": np.sqrt(var_Mm_g),"std Ml": np.sqrt(var_Ml_g), "std Mlna": np.sqrt(var_Mlna_g),
           "mean Mm": np.mean(mean_Mm), "mean Ml": np.mean(mean_Ml), "mean Mlna": np.mean(mean_Mlna),
           "power spectrum Mm": power_spectrum_Mm[1:T//2+1], "power spectrum Ml": power_spectrum_Ml[1:T//2+1], "power spectrum Mlna": power_spectrum_Mlna[1:T//2+1],
           "std Pm": np.sqrt(var_Pm_g),"std Pl": np.sqrt(var_Pl_g), "std Plna": np.sqrt(var_Plna_g),
           "mean Pm": np.mean(mean_Pm), "mean Pl": np.mean(mean_Pl), "mean Plna": np.mean(mean_Plna),
           "power spectrum Pm": power_spectrum_Pm[1:T//2+1], "power spectrum Pl": power_spectrum_Pl[1:T//2+1], "power spectrum Plna": power_spectrum_Plna[1:T//2+1],
           "times":t_ref, "frequencies": freq}
    
    return output

'''Computes means, stds and power spectra all at once.
   
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

@jit
def generateIncrements( alpha_m=1, alpha_p=1,mu_m=0.03,mu_p=0.03,lambda_s=1,
                                                              P_0=1,
                                                              h=4.1,
                                                              tau=0.1,
                                                              P_init=10,
                                                              M_init=20,
                                                              sigma_init=1,
                                                              Omega=1,
                                                              T=1000,
                                                              incr_step=1,
                                                              N_data=1000):
    t=[0] #list of times when a reaction is started or ended
    P=[P_init] #list of Hes1 molecule numbers
    M=[M_init] #list of mRNA molecule numbers
    sigma=[sigma_init] #list of environment configuration
    d_react=[] #list (queue) of end times of delayed reactions
    
    def perform_reaction(a_0,t,M,P,sigma):
        rr=rd.uniform(0,1)
        a_1=mu_m*M[-1]
        a_2=mu_p*P[-1]
        a_3=alpha_p*M[-1]
        a_4=alpha_m*Omega*sigma[-1]

        if rr<a_1/a_0:                      #destruction of M
            M.append(M[-1]-1)
            P.append(P[-1])
            sigma.append(sigma[-1])
        elif rr < (a_1+a_2)/a_0:            #destruction of P
            M.append(M[-1])
            P.append(P[-1]-1)
            sigma.append(sigma[-1])
        elif rr < (a_1+a_2+a_3)/a_0:        #creation of P
            M.append(M[-1])
            P.append(P[-1]+1)
            sigma.append(sigma[-1])
        elif rr < (a_1+a_2+a_3+a_4)/a_0:    #plan delayed reaction for creation of M 
            d_react.append(t[-1]+tau)
            M.append(M[-1])
            P.append(P[-1])
            sigma.append(sigma[-1])
        else:                               #switch environment
            M.append(M[-1])
            P.append(P[-1])
            sigma.append(1-sigma[-1])
    
    
    def run_master():
        while t[-1]<T:
            a_0=mu_m*M[-1]+mu_p*P[-1]+alpha_p*M[-1]+sigma[-1]*alpha_m*Omega    #total propensity
            
            n_p0=P_0*Omega
            if sigma[-1]==1:                                          #add switching environment term
                a_0+=lambda_s*(P[-1]/n_p0)**h
            else:
                a_0+=lambda_s   

            r=rd.uniform(0,1)                                  #generate delta via inverse transform method (exponential distribution)
            delta=-np.log(r)/a_0

            if len(d_react)!=0 and d_react[0]<=t[-1]+delta:    #if a delayed reaction is planned, creation of M
                t.append(d_react[0])
                M.append(M[-1]+1)
                P.append(P[-1])
                sigma.append(sigma[-1])

                del d_react[0]                               #then remove the delayed reaction

            else:                                            #else perform a new reaction
                t.append(t[-1]+delta)
                perform_reaction(a_0,t,M,P,sigma)

    run_master()
    
    
    
    
    
    
    
    
    
    data_incr_M = np.zeros(N_data)
    data_incr_P = np.zeros(N_data)
    
    M_ref=M[-1]
    P_ref=P[-1]
    
    for k in range(N_data):
        t_temp=[t[-1]]
        M_temp=[M_ref]
        P_temp=[P_ref]
        sigma_temp = [sigma[-1]]
        d_react_temp = d_react.copy()
        while t_temp[-1]<T+incr_step:
            a_0=mu_m*M_temp[-1]+mu_p*P_temp[-1]+alpha_p*M_temp[-1]+sigma_temp[-1]*alpha_m*Omega    #total propensity
            
            n_p0=P_0*Omega
            if sigma_temp[-1]==1:                                          #add switching environment term
                a_0+=lambda_s*(P_temp[-1]/n_p0)**h
            else:
                a_0+=lambda_s   

            r=rd.uniform(0,1)                                  #generate delta via inverse transform method (exponential distribution)
            delta=-np.log(r)/a_0

            if len(d_react_temp)!=0 and d_react_temp[0]<=t_temp[-1]+delta:    #if a delayed reaction is planned, creation of M
                t_temp.append(d_react_temp[0])
                M_temp.append(M_temp[-1]+1)
                P_temp.append(P_temp[-1])
                sigma_temp.append(sigma_temp[-1])

                del d_react_temp[0]                               #then remove the delayed reaction

            else:                                            #else perform a new reaction
                t_temp.append(t_temp[-1]+delta)
                perform_reaction(a_0,t_temp,M_temp,P_temp,sigma_temp)
        
        data_incr_M[k] = (M_temp[-1]-M_ref)/Omega
        data_incr_P[k] = (P_temp[-1]-P_ref)/Omega
        
        
        
        
        
        
        c=-1
        while t[c] > t[-1] - tau:
            c += (-1)
            
        hill_function=1/(1+(P[c]/P_0)**h)                                    #value of the hill function f(P(t-tau))
        var_switch=(alpha_m**2/lambda_s)*2*(P[c]/P_0)**h*hill_function**3    #value of the switching induced diffusion
        
        mean_increment_M=alpha_m*hill_function - mu_m*M_ref
        std_increment_M =np.sqrt(alpha_m/Omega*hill_function + mu_m/Omega*M_ref + var_switch)
        mean_increment_P=alpha_p*M_ref - mu_p*P_ref
        std_increment_P =np.sqrt(alpha_p/Omega*M_ref + mu_p/Omega*P_ref)
                
    return M_ref, P_ref, data_incr_M,data_incr_P, mean_increment_M, std_increment_M, mean_increment_P, std_increment_P

