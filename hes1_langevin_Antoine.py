# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:18:59 2022

@author: Antoine Moneyron
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt
from scipy.optimize import bisect
import hes1_master_Antoine as master
from numba import jit

@jit(nopython = True)
def one_trajectory(alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,             #one trajectory of langevin equation, scheme Euler-Maruyama
                                                      lambda_s=1,       
                                                      P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=10,
                                                      M_init=20,
                                                      T=1000,
                                                      delta_t=1,
                                                      sampling_timestep  = 1,
                                                      Omega=1):
    
    n_t=int(T/delta_t)             #number of points in the time mesh
    k_delay=round(tau/delta_t)     #delayed shifting on indices
    t=np.linspace(0,T,n_t)         #time mesh
    P=np.zeros(n_t)                #array of Hes1 concentrations
    M=np.zeros(n_t)                #array of mRNA concentrations

    M[0]=M_init
    P[0]=P_init    
    
    for i in range(n_t-1):
        
        hill_function=0
        var_switch=0
        
        if i<k_delay:
            hill_function=1/(1+(P_init/P_0)**h)
            var_switch=(alpha_m**2/lambda_s)*(P_init/P_0)**h*hill_function**3
        elif i>= k_delay:
            hill_function=1/(1+(P[i-k_delay]/P_0)**h)                                    #value of the hill function f(P(t-tau))
            var_switch=(alpha_m**2/lambda_s)*2*(P[i-k_delay]/P_0)**h*hill_function**3    #value of the switching induced diffusion
           
        w_m=np.random.normal(0,np.sqrt(delta_t))
        w_p=np.random.normal(0,np.sqrt(delta_t))
        
        mean_increment_M=alpha_m*hill_function - mu_m*M[i]
        std_increment_M =np.sqrt(alpha_m/Omega*hill_function + mu_m/Omega*M[i] + var_switch)
        mean_increment_P=alpha_p*M[i] - mu_p*P[i]
        std_increment_P =np.sqrt(alpha_p/Omega*M[i] + mu_p/Omega*P[i])
        
        M[i+1]=abs(M[i] + mean_increment_M*delta_t + std_increment_M*w_m)  #reflective boundary conditions
        P[i+1]=abs(P[i] + mean_increment_P*delta_t + std_increment_P*w_p)
    
    
    #for this to make sense delta_t has to be less than one
    sampling_timestep_multiple = int(round(1.0/delta_t))

    t_to_return = t[::(sampling_timestep_multiple*sampling_timestep)]
    m_to_return = M[::(sampling_timestep_multiple*sampling_timestep)]
    p_to_return = P[::(sampling_timestep_multiple*sampling_timestep)]

    return t_to_return,m_to_return,p_to_return


'''Generate one trace of the Langevin SDE Hes1 model.

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
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    tau : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.
        
    M_init : int
        initial mRNA molecule number
        
    P_init : int
        initial Hes1 molecule number


    Returns
    -------

    t : 1D ndarray of shape int(T/delta_t)
        Times in the time mesh.
        
    M : 2D ndarray of shape n_iter * int(T/delta_t)
        mRNA concentrations, taken at time values given in 't'.
        
    P : 2D ndarray of shape n_iter * int(T/delta_t)
        Hes1 concentrations, taken at time values given in 't'.
'''



@jit(nopython = True)
def multiple_trajectories(n_iter=100,alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,        #perform many realisations
                                                      lambda_s=1,                    #and gather in a table
                                                      P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=10,
                                                      M_init=20,
                                                      T=1000,
                                                      delta_t=1,
                                                      Omega=1):    
    n_t=int(T/delta_t)
    
    table_M=np.zeros((n_iter,n_t))
    table_P=np.zeros((n_iter,n_t))
    
    for k in range(n_iter):
        t,M,P=one_trajectory(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=P_init,
                                                      M_init=M_init,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega)           #run one_trajectory multiple times
        
        table_M[k,:]=M    #gather independently of time
        table_P[k,:]=P
    
    return t,table_M,table_P

'''Generate mutliple traces of the Langevin SDE Hes1 model. Calls "one_trajectory".

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
        initial mRNA molecule number
        
    P_init : int
        initial Hes1 molecule number


    Returns
    -------

    t : 1D ndarray of shape int(T/delta_t)
        Times in the time mesh.
        
    table_M : 2D ndarray of shape n_iter * int(T/delta_t)
        mRNA concentrations, taken at time values given in 't'.
        
    table_P : 2D ndarray of shape n_iter * int(T/delta_t)
        Hes1 concentrations, taken at time values given in 't'.
'''

@jit(nopython = True)
def pool_values(n_iter=100,alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,      #pool many stationary realisations
                                                      lambda_s=1,        #inside of an array
                                                      P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=10,
                                                      M_init=20,
                                                      T=1000,
                                                      delta_t=1,
                                                      Omega=1):    
    n_t=int(T/delta_t)
    
    _,table_M,table_P=multiple_trajectories(n_iter=n_iter,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=P_init,
                                                      M_init=M_init,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega)        #compute many realisations from this state
    
    
    pool_M=np.reshape(table_M[:,(n_t//2):], n_iter*(n_t//2))   #pool everything together
    pool_P=np.reshape(table_P[:,(n_t//2):], n_iter*(n_t//2))
    
    return pool_M,pool_P


'''Generate multiple traces of the Langevin SDE Hes1 model, and pool the stationary values together.
   Note that the duration T must be large enough for the system to be stationary on the interval [T/2, T].
   Calls 'multiple_trajectories'.

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
        initial mRNA molecule number
        
    P_init : int
        initial Hes1 molecule number


    Returns
    -------

    table_M : 2D ndarray of shape n_iter * (int(T/delta_t)//2)
        Table of stationary mRNA concentrations.
        
    table_P : 2D ndarray of shape n_iter * (int(T/delta_t)//2)
        Table of stationary Hes1 concentrations.
        
'''

@jit
def resolve_ODE(alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=10,
                                                      M_init=20,
                                                      T=1000,
                                                      delta_t=1):
    
    return one_trajectory(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                      lambda_s=float('infinity'),
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=P_init,
                                                      M_init=M_init,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=float('infinity')) #Omega and lambda_s set to infinity



'''Generate one deterministic trace of the Hes1 model, by setting the switching rate lambda_s and the system size Omega to infinity.
   Calls 'one_trajectories'.

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
        initial mRNA molecule number
        
    P_init : int
        initial Hes1 molecule number


    Returns
    -------

    table_M : 2D ndarray of shape n_iter * (int(T/delta_t)//2)
        Table of stationary mRNA concentrations.
        
    table_P : 2D ndarray of shape n_iter * (int(T/delta_t)//2)
        Table of stationary Hes1 concentrations.
        
'''



def resolve_stationary_state(alpha_m,mu_m,alpha_p,mu_p,h,P_0):             #find stationary state for M and P found from stationary ODE
    def optim_func(x):                                                     
        return (alpha_m/mu_m)*1/(1+(x/P_0)**h) - mu_p/alpha_p*x            #alpha_m*f(P) - mu_m*M = 0 and alpha_p*M - mu_p*P =0

    p_stat=bisect(optim_func,0,10**6)                                  
    m_stat=mu_p/alpha_p*p_stat
    return m_stat,p_stat



@jit
def one_trajectory_LNA_with_steady_state(alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,             #one trajectory of langevin equation, scheme Euler-Maruyama
                                                      lambda_s=1,       
                                                      P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=1000,
                                                      delta_t=1,
                                                      Omega=1,
                                                      sampling_timestep = 1.0,
                                                      stationary_m = 10.0,
                                                      stationary_p = 10.0):
    
    n_t=int(T/delta_t)             #number of points in the time mesh
    k_delay=round(tau/delta_t)     #delayed shifting on indices
    t=np.linspace(0,T,n_t)         #time mesh
    P=np.zeros(n_t)                #array of Hes1 concentrations
    M=np.zeros(n_t)                #array of mRNA concentrations

    M[0]=M_init
    P[0]=P_init    
    
    M_stat,P_stat = stationary_m, stationary_p
    df_P=-h/P_0*(P_stat/P_0)**(h-1)/(1+(P_stat/P_0)**h)**2
    
    for i in range(n_t-1):
        
        if i<k_delay:
            mean_increment_M=alpha_m*df_P*P[0] - mu_m*M[i]
            mean_increment_P=alpha_p*M[i] - mu_p*P[i]
        elif i>= k_delay:
            mean_increment_M=alpha_m*df_P*P[i-k_delay] - mu_m*M[i]                  #increment in LNA
            mean_increment_P=alpha_p*M[i] - mu_p*P[i]
        
        hill_function_stat=1/(1+(P_stat/P_0)**h)
        var_switch=(alpha_m**2/(lambda_s))*2*(P_stat/P_0)**h*hill_function_stat**3    #value of the switching induced diffusion
           
        w_m=np.random.normal(0,np.sqrt(delta_t))
        w_p=np.random.normal(0,np.sqrt(delta_t))
        
        std_increment_M =np.sqrt(alpha_m/Omega*hill_function_stat + mu_m/Omega*M_stat + var_switch)
        std_increment_P =np.sqrt(alpha_p/Omega*M_stat + mu_p/Omega*P_stat)
        
        M[i+1]= M[i] + mean_increment_M*delta_t + std_increment_M*w_m  
        P[i+1]= P[i] + mean_increment_P*delta_t + std_increment_P*w_p
    
    M = M + M_stat*np.ones(n_t)
    P = P + P_stat*np.ones(n_t)
    
    #for this to make sense delta_t has to be less than the sampling timestep
    sampling_timestep_multiple = int(round(sampling_timestep/delta_t))

    t_to_return = t[::(sampling_timestep_multiple)]
    m_to_return = M[::(sampling_timestep_multiple)]
    p_to_return = P[::(sampling_timestep_multiple)]
    
    return t_to_return,m_to_return,p_to_return


@jit
def one_trajectory_LNA(alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,             #one trajectory of langevin equation, scheme Euler-Maruyama
                                                      lambda_s=1,       
                                                      P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=1000,
                                                      delta_t=1,
                                                      Omega=1,
                                                      sampling_timestep = 1.0):
    
    M_stat,P_stat = resolve_stationary_state(alpha_m,mu_m,alpha_p,mu_p,h,P_0)
    t,M,P = one_trajectory_LNA_with_steady_state(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,             #one trajectory of langevin equation, scheme Euler-Maruyama
                                                      lambda_s=lambda_s,       
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=P_init,
                                                      M_init=M_init,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega,
                                                      sampling_timestep = sampling_timestep,
                                                      stationary_m = M_stat,
                                                      stationary_p = P_stat)
    
    return t,M,P


'''Generate one trace of the LNA SDE Hes1 model.

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
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    tau : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.
        
    M_init : int
        initial mRNA molecule number
        
    P_init : int
        initial Hes1 molecule number


    Returns
    -------

    t : 1D ndarray of shape int(T/delta_t)
        Times in the time mesh.
        
    M : 2D ndarray of shape n_iter * int(T/delta_t)
        mRNA concentrations, taken at time values given in 't'.
        
    P : 2D ndarray of shape n_iter * int(T/delta_t)
        Hes1 concentrations, taken at time values given in 't'.
'''


@jit
def multiple_trajectories_LNA(n_iter=100, alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,        #perform many realisations
                                                      lambda_s=1,                    #and gather in a table
                                                      P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      T=1000,
                                                      delta_t=1,
                                                      Omega=1):    
    n_t=int(T/delta_t)
    
    table_M=np.zeros((n_iter,n_t))
    table_P=np.zeros((n_iter,n_t))
    
    for k in range(n_iter):
        t,M,P=one_trajectory_LNA(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega)           #run one_trajectory multiple times
        
        table_M[k,:]=M    #gather independently of time
        table_P[k,:]=P
    
    return t,table_M,table_P


'''Generate mutliple traces of the LNA SDE Hes1 model. Calls "one_trajectory_LNA".

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
        initial mRNA molecule number
        
    P_init : int
        initial Hes1 molecule number


    Returns
    -------

    t : 1D ndarray of shape int(T/delta_t)
        Times in the time mesh.
        
    table_M : 2D ndarray of shape n_iter * int(T/delta_t)
        mRNA concentrations, taken at time values given in 't'.
        
    table_P : 2D ndarray of shape n_iter * int(T/delta_t)
        Hes1 concentrations, taken at time values given in 't'.
'''





@jit(nopython = True)
def one_trajectory_PLP(alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,             #one trajectory of langevin equation, scheme Euler-Maruyama
                                                      lambda_s=1,       
                                                      P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=10,
                                                      M_init=20,
                                                      sigma_init=0,
                                                      T=1000,
                                                      delta_t=1,
                                                      sampling_timestep  = 1,
                                                      Omega=1):
    
    n_t=int(T/delta_t)             #number of points in the time mesh
    k_delay=round(tau/delta_t)     #delayed shifting on indices
    
    t=np.linspace(0,T,n_t)         #time mesh
    P=np.zeros(n_t)                #array of Hes1 concentrations
    M=np.zeros(n_t)                #array of mRNA concentrations
    sigma=np.zeros(n_t, dtype=np.int8)            #array environment configuration (ON/OFF)

    M[0]=M_init
    P[0]=P_init
    sigma[0]=sigma_init
    
    for i in range(n_t-1):
        w_m=np.random.normal(0,np.sqrt(delta_t))
        w_p=np.random.normal(0,np.sqrt(delta_t))
        
        if i < k_delay:
            mean_increment_M=alpha_m*sigma[0] - mu_m*M[i]
            std_increment_M =np.sqrt(alpha_m/Omega*sigma[0] + mu_m/Omega*M[i])
        else:
            mean_increment_M=alpha_m*sigma[i-k_delay] - mu_m*M[i]
            std_increment_M =np.sqrt(alpha_m/Omega*sigma[i-k_delay] + mu_m/Omega*M[i])
        
        mean_increment_P=alpha_p*M[i] - mu_p*P[i]
        std_increment_P =np.sqrt(alpha_p/Omega*M[i] + mu_p/Omega*P[i])
        
        M[i+1]=abs(M[i] + mean_increment_M*delta_t + std_increment_M*w_m)  #reflective boundary conditions
        P[i+1]=abs(P[i] + mean_increment_P*delta_t + std_increment_P*w_p)
        
        if sigma[i]==1:                                       
            a_0 = lambda_s*(P[i]/P_0)**h
        else:
            a_0 = lambda_s
        r=rd.uniform(0,1)
        delta= 2*T if a_0 ==0 else -np.log(r)/a_0
        
        if delta <= delta_t:
            sigma[i+1] = 1 - sigma[i]
        else:
            sigma[i+1] = sigma[i]
        

            
    #for this to make sense delta_t has to be less than one
    sampling_timestep_multiple = int(round(1.0/delta_t))

    t_to_return = t[::(sampling_timestep_multiple*sampling_timestep)]
    m_to_return = M[::(sampling_timestep_multiple*sampling_timestep)]
    p_to_return = P[::(sampling_timestep_multiple*sampling_timestep)]

    return t_to_return,m_to_return,p_to_return


'''Generate one trace of the PLP Hes1 model (piecewise Langevin process).

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
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    tau : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.
        
    M_init : int
        initial mRNA molecule number
        
    P_init : int
        initial Hes1 molecule number


    Returns
    -------

    t : 1D ndarray of shape int(T/delta_t)
        Times in the time mesh.
        
    M : 2D ndarray of shape n_iter * int(T/delta_t)
        mRNA concentrations, taken at time values given in 't'.
        
    P : 2D ndarray of shape n_iter * int(T/delta_t)
        Hes1 concentrations, taken at time values given in 't'.
'''


@jit(nopython = True)
def one_trajectory_PDMP(alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,             
                                                      lambda_s=1,       
                                                      P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=10,
                                                      M_init=20,
                                                      sigma_init=0,
                                                      T=1000,
                                                      delta_t=1,
                                                      sampling_timestep  = 1):
    
    return one_trajectory_PLP(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,             
                                                      lambda_s=lambda_s,       
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=P_init,
                                                      M_init=M_init,
                                                      sigma_init=sigma_init,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      sampling_timestep  = sampling_timestep,
                                                      Omega= float('infinity'))

'''Generate one trace of the PDMP Hes1 model (piecewise deterministic Markov process), by setting demographic noise to zero in the PLP (infinite Omega).

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
        rate at protein translation, in Hes copy number per mRNA copy number and minute,

    tau : float
        delay of the repression response to Hes protein in minutes. The rate of mRNA transcription depends
        on the protein copy number at this amount of time in the past.
        
    M_init : int
        initial mRNA molecule number
        
    P_init : int
        initial Hes1 molecule number


    Returns
    -------

    t : 1D ndarray of shape int(T/delta_t)
        Times in the time mesh.
        
    M : 2D ndarray of shape n_iter * int(T/delta_t)
        mRNA concentrations, taken at time values given in 't'.
        
    P : 2D ndarray of shape n_iter * int(T/delta_t)
        Hes1 concentrations, taken at time values given in 't'.
'''




def one_trajectory_PDMP_doubleCheck(alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,  #one trajectory of langevin equation, scheme Euler-Maruyama
                                                      lambda_s=1,       
                                                      P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=10,
                                                      M_init=20,
                                                      sigma_init=0,
                                                      T=1000,
                                                      delta_t=1,
                                                      sampling_timestep  = 1,
                                                      Omega=1):
    
    n_t=int(T/delta_t)             #number of points in the time mesh
    k_delay=round(tau/delta_t)     #delayed shifting on indices
    
    t=np.linspace(0,T,n_t)         #time mesh
    P=np.zeros(n_t)                #array of Hes1 concentrations
    M=np.zeros(n_t)                #array of mRNA concentrations
    sigma=np.zeros(n_t, dtype=np.int8)            #array environment configuration (ON/OFF)

    M[0]=M_init
    P[0]=P_init
    sigma[0]=sigma_init
    
    for i in range(n_t-1):   
        if i < k_delay:
            mean_increment_M=alpha_m*sigma[0] - mu_m*M[i]
        else:
            mean_increment_M=alpha_m*sigma[i-k_delay] - mu_m*M[i]
        
        mean_increment_P=alpha_p*M[i] - mu_p*P[i]
        
        M[i+1]=abs(M[i] + mean_increment_M*delta_t) #reflective boundary conditions
        P[i+1]=abs(P[i] + mean_increment_P*delta_t) 
        
        if sigma[i]==1:                                       
            a_0 = lambda_s*(P[i]/P_0)**h
        else:
            a_0 = lambda_s
        r=rd.uniform(0,1)
        delta= 2*T if a_0 ==0 else -np.log(r)/a_0
        
        if delta <= delta_t:
            sigma[i+1] = 1 - sigma[i]
        else:
            sigma[i+1] = sigma[i]
            
    #for this to make sense delta_t has to be less than one
    sampling_timestep_multiple = int(round(1.0/delta_t))

    t_to_return = t[::(sampling_timestep_multiple*sampling_timestep)]
    m_to_return = M[::(sampling_timestep_multiple*sampling_timestep)]
    p_to_return = P[::(sampling_timestep_multiple*sampling_timestep)]

    return t_to_return,m_to_return,p_to_return
    
