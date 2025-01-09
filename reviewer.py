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
from numba import jit
from numba import njit
import hes1_utils_Antoine as utils




@jit(nopython = True)
def perform_reaction(t, M, P, sigma, d_react, a_0, prop1, prop2, prop3, prop4, tau):
        
        rr = a_0 * rd.uniform(0,1)
               
        if rr < prop1:                         #destruction of M
            M=M-1
        elif rr < prop2:                       #destruction of P
            P=P-1
        elif rr < prop3:                       #creation of P
            P=P+1
        elif rr < prop4:                       #plan delayed reaction for creation of M 
            d_react.append(t+tau)
        else:                                  #switch environment
            sigma = (1-sigma)
        
        return M,P,sigma

    
    
@jit(nopython = True)
def run_master( alpha_m=1, alpha_p=1,mu_m=0.03,mu_p=0.03,lambda_s=1,
                                                              P_0=1,
                                                              h=4.1,
                                                              tau=0.1,
                                                              P_init=10,
                                                              M_init=20,
                                                              sigma_init=1,
                                                              Omega=1,
                                                              T=1000,
                                                              sampling_timestep = 1):

    t=0                     #time when a reaction is started or ended
    t_next=t
    
    P=P_init                #Hes1 molecule numbers
    M=M_init                #mRNA molecule numbers
    sigma=sigma_init        #environment configuration
    d_react=[0.1]              #list (queue) of end times of delayed reactions
    del d_react[0]

    coeff = alpha_m*Omega
    n_p0=P_0*Omega

    
    N=int(T/sampling_timestep)
    time_to_return = np.linspace(0,T,N)
    mRNA_to_return = np.zeros(N)
    Hes1_to_return = np.zeros(N)
    
    index_to_store = 1
    time_to_store = sampling_timestep
    
    while t<T:
        prop1 = mu_m*M
        prop2 = prop1 + mu_p*P
        prop3 = prop2 + alpha_p*M
        prop4 = prop3 + sigma*coeff
                
        if sigma==1:                                          #add switching environment term
            a_0 = prop4 + lambda_s
        else:
            a_0 = prop4 + lambda_s*(P/n_p0)**(-h)   

        r=rd.uniform(0,1)                                  #generate delta via inverse transform method (exponential distribution)
        delta=-np.log(r)/a_0
                        
        if d_react and d_react[0]<=t+delta:    #if a delayed reaction is planned, creation of M
            t_next=d_react[0]
            if t_next > time_to_store:
                mRNA_to_return[index_to_store] = M/Omega
                Hes1_to_return[index_to_store] = P/Omega
                index_to_store+=1
                time_to_store+=sampling_timestep
            M=M+1
            del d_react[0]                               #then remove the delayed reaction

        else:                                            #else perform a new reaction
            t_next=t+delta
            if t_next > time_to_store:
                mRNA_to_return[index_to_store] = M/Omega
                Hes1_to_return[index_to_store] = P/Omega
                index_to_store+=1
                time_to_store+=sampling_timestep
                
            M,P,sigma = perform_reaction(t_next, M, P, sigma, d_react, a_0, prop1, prop2, prop3, prop4, tau)
        
        t = t_next
    
    mRNA_to_return[0] = M_init/Omega
    Hes1_to_return[0] = P_init/Omega
    
    mRNA_to_return[-1] = M/Omega
    Hes1_to_return[-1] = P/Omega
    
    return time_to_return,mRNA_to_return,Hes1_to_return
                 
        

@jit(nopython = True)
def one_ssa_trajectory( alpha_m=1, alpha_p=1,mu_m=0.03,mu_p=0.03,lambda_s=1,
                                                              P_0=1,
                                                              h=4.1,
                                                              tau=0.1,
                                                              P_init=10,
                                                              M_init=20,
                                                              sigma_init=1,
                                                              Omega=1,
                                                              T=1000,
                                                              sampling_timestep = 1):
    
    
    return run_master( alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p, lambda_s=lambda_s,
                                                              P_0=P_0,
                                                              h=h,
                                                              tau=tau,
                                                              P_init=P_init,
                                                              M_init=M_init,
                                                              sigma_init=sigma_init,
                                                              Omega=Omega,
                                                              T=T,
                                                              sampling_timestep = sampling_timestep)
    
    

'''Generate one trace of the Hes1 model from the master equation (Gillespie algorithm).

    Parameters
    ----------

    T : float
        duration of the trace in minutes

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
        
    sigma_init : bool
        initial environment configuration
        
    Returns
    -------

    times : 1D ndarray
        Times at which the system-environment changes, i.e one reaction occurs.
        
    mRNA : 1D ndarray
        mRNA concentration taken at time values given in 'times'.
        
    Hes1 : 1D ndarray
        Hes1 concentration taken at time values given in 'times'.
'''


@jit(nopython = True)
def multiple_trajectories(n_iter=100,alpha_m=1, alpha_p=1,mu_m=0.03,mu_p=0.03,lambda_s=1,   
                                                              P_0=1,                        
                                                              h=4.1,
                                                              tau=0.1,
                                                              P_init=10,
                                                              M_init=20,
                                                              sigma_init=1,
                                                              T=1000,
                                                              Omega=1,
                                                              sampling_timestep=1.0):
    n_t=int(T/delta_t)
    table_M=np.zeros((n_iter,n_t))
    table_P=np.zeros((n_iter,n_t))
    
    for k in range(n_iter):
        t,M,P=one_trajectory(alpha_m=alpha_m, alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,lambda_s=lambda_s,
                                  P_0=P_0,
                                  h=h,
                                  tau=tau,
                                  Omega=Omega,
                                  P_init=P_init,
                                  M_init=M_init,
                                  sigma_init=sigma_init,
                                  T=T,
                                  sampling_timestep = sampling_timestep)                         #run one_trajectory n_iter times
        
        table_M[k,:]=M   #interpolate on the mesh and gather in a table
        table_P[k,:]=P
    
    return t,table_M,table_P


'''Generate multiple traces of the Hes1 model from the master equation (Gillespie algorithm). Calls "one_trajectory".

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

    t_ref : 1D ndarray of shape int(T/delta_t)
        Times at which the system-environment changes, i.e one reaction occurs.
        
    table_M : 2D ndarray of shape n_iter * int(T/delta_t)
        Table of mRNA concentrations, taken at time values given in 't_ref'.
        
    table_P : 2D ndarray of shape n_iter * int(T/delta_t)
        Table of Hes1 concentrations, taken at time values given in 't_ref'.
'''

@jit(nopython = True)
def pool_values(n_iter=100,alpha_m=1, alpha_p=1,mu_m=0.03,mu_p=0.03,lambda_s=1,            
                                                              P_0=1,                   
                                                              h=4.1,
                                                              tau=0.1,
                                                              P_init=10,
                                                              M_init=20,
                                                              sigma_init=1,
                                                              T=1000,
                                                              delta_t=1,
                                                              Omega=1):
    
    n_t=int(T/delta_t)
    
    t_ref,table_M,table_P=multiple_trajectories(n_iter=n_iter,alpha_m=alpha_m, alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,lambda_s=lambda_s,
                                                              P_0=P_0,
                                                              h=h,
                                                              tau=tau,
                                                              P_init=P_init,
                                                              M_init=M_init,
                                                              sigma_init=sigma_init,
                                                              T=T,
                                                              delta_t=delta_t,
                                                              Omega=Omega)               #run multiple trajectories

    pool_M=[]  
    pool_P=[]  
    for k in range(n_iter):
        pool_M=pool_M+list(table_M[k,(n_t//2):]) #gather stationary values, without interest in temporal organisation
        pool_P=pool_P+list(table_P[k,(n_t//2):])
    return np.array(pool_M),np.array(pool_P)


'''Generate multiple traces of the Hes1 model from the master equation (Gillespie algorithm), and pool the stationary values together.
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




@jit(nopython = True)
def perform_reaction_noSwitch(t, M, P, d_react, a_0, prop1, prop2, prop3, tau):
        
        rr = a_0 * rd.uniform(0,1)
               
        if rr < prop1:                         #destruction of M
            M=M-1
        elif rr < prop2:                       #destruction of P
            P=P-1
        elif rr < prop3:                       #creation of P
            P=P+1
        else:                       #plan delayed reaction for creation of M 
            d_react.append(t+tau)
        
        return M,P    


def one_trajectory_PDMP(alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,  #one trajectory of langevin equation, scheme Euler-Maruyama
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
            a_0 = lambda_s
        else:
            a_0 = lambda_s*(P[i]/P_0)**(-h)
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
    
@jit(nopython = True)
def one_langevin_trajectory(alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,             #one trajectory of langevin equation, scheme Euler-Maruyama
                                                      lambda_s=1.0,       
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
            var_switch=(alpha_m**2/lambda_s)*(P_init/P_0)**(2*h)*hill_function**3
        elif i>= k_delay:
            hill_function=1/(1+(P[i-k_delay]/P_0)**h)                                    #value of the hill function f(P(t-tau))
            var_switch=(alpha_m**2/lambda_s)*2*(P[i-k_delay]/P_0)**(2*h)*hill_function**3    #value of the switching induced diffusion
           
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

@njit
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
        var_switch=(alpha_m**2/(lambda_s))*2*(P_stat/P_0)**(2*h)*hill_function_stat**3    #value of the switching induced diffusion
           
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


def resolve_stationary_state(alpha_m,mu_m,alpha_p,mu_p,h,P_0):             #find stationary state for M and P found from stationary ODE
    def optim_func(x):                                                     
        return (alpha_m/mu_m)*1/(1+(x/P_0)**h) - mu_p/alpha_p*x            #alpha_m*f(P) - mu_m*M = 0 and alpha_p*M - mu_p*P =0

    p_stat=bisect(optim_func,0,10**6)                                  
    m_stat=mu_p/alpha_p*p_stat
    # print(m_stat)
    # print(p_stat)
    return m_stat,p_stat



# @jit
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


@jit(nopython = True)
def dummy_simulate_master_meanAndStd(n_iter=100, alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,
                                                      lambda_s=1,        
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=10000,
                                                      delta_t=1,
                                                      Omega=1,
                                                      sampling_timestep = 1.0):

    var_Mm = np.zeros(n_iter)
    var_Pm = np.zeros(n_iter)
    mean_Mm = np.zeros(n_iter)
    mean_Pm = np.zeros(n_iter)
    momentumFour_Mm = np.zeros(n_iter)
    momentumFour_Pm = np.zeros(n_iter)
    
    n_stat=int(2000/sampling_timestep)


    for i in range(n_iter):
        t,Mm,Pm=t,Mm,Pm=one_ssa_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p, lambda_s=lambda_s,
                                                              P_0=P_0,
                                                              h=h,
                                                              tau=tau,
                                                              P_init=P_init,
                                                              M_init=M_init,
                                                              sigma_init=1,
                                                              Omega=Omega,
                                                              T=T, sampling_timestep=sampling_timestep)

        mean_Mm[i] = np.mean(Mm[n_stat:])
        mean_Pm[i] = np.mean(Pm[n_stat:])
        var_Mm[i] = np.var(Mm[n_stat:])
        var_Pm[i] = np.var(Pm[n_stat:])
        momentumFour_Mm[i] = np.mean(Mm[n_stat:]**4)
        momentumFour_Pm[i] = np.mean(Pm[n_stat:]**4)
            
    var_Mm_g = np.mean(var_Mm) + np.var(mean_Mm)
    var_Pm_g = np.mean(var_Pm) + np.var(mean_Pm)
    
    return np.mean(mean_Mm), np.sqrt(var_Mm_g), np.mean(momentumFour_Mm), np.mean(mean_Pm), np.sqrt(var_Pm_g), np.mean(momentumFour_Pm)

@jit(nopython = True)
def simulate_master_meanAndStd(n_iter=100, alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,
                                                      lambda_s=1,        
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=10000,
                                                      delta_t=1,
                                                      Omega=1,
                                                      sampling_timestep = 1.0):
    
    temp = dummy_simulate_master_meanAndStd(n_iter=n_iter, alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                      lambda_s=lambda_s,        
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=P_init,
                                                      M_init=M_init,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega,
                                                      sampling_timestep = sampling_timestep)
    
    output={"mean Mm": temp[0],
           "std Mm": temp[1],
           "momentumFour Mm": temp[2],
           "mean Pm": temp[3],
           "std Pm": temp[4],
           "momentumFour Pm": temp[5]}
    
    return output



# @jit
def simulate_master_all(n_iter=100, alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,
                                                      lambda_s=1,        
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=10000,
                                                      delta_t=1,
                                                      Omega=1,
                                                      sampling_timestep = 1.0):
    
    var_Mm = np.zeros(n_iter)
    var_Pm = np.zeros(n_iter)
    mean_Mm = np.zeros(n_iter)
    mean_Pm = np.zeros(n_iter)
        
    t,Mm,Pm=one_ssa_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p, lambda_s=lambda_s,
                                                              P_0=P_0,
                                                              h=h,
                                                              tau=tau,
                                                              P_init=P_init,
                                                              M_init=M_init,
                                                              sigma_init=1,
                                                              Omega=Omega,
                                                              T=T)
    
    sampling_timestep_multiple = int(round(1.0/delta_t))
    t_ref=np.arange(0,T,delta_t)
    t_ref=t_ref[::(sampling_timestep_multiple*sampling_timestep)]
    
    Mm=spinter.interp1d(t,Mm,kind="zero")(t_ref)
    Pm=spinter.interp1d(t,Pm,kind="zero")(t_ref)

    n_stat=int(2000/sampling_timestep)
    freq_ref, test_power_spectrum = utils.compute_power_spectrum_traj(t_ref[n_stat:],Pm[n_stat:])
    power_spectrum_Mm=np.zeros_like(test_power_spectrum)
    power_spectrum_Pm=np.zeros_like(test_power_spectrum)
    
    for i in range(n_iter):
        t,Mm,Pm=t,Mm,Pm=one_ssa_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p, lambda_s=lambda_s,
                                                              P_0=P_0,
                                                              h=h,
                                                              tau=tau,
                                                              P_init=P_init,
                                                              M_init=M_init,
                                                              sigma_init=1,
                                                              Omega=Omega,
                                                              T=T)
        Mm=spinter.interp1d(t,Mm,kind="zero")(t_ref)
        Pm=spinter.interp1d(t,Pm,kind="zero")(t_ref)

        mean_Mm[i] = np.mean(Mm[n_stat:])
        mean_Pm[i] = np.mean(Pm[n_stat:])
        var_Mm[i] = np.var(Mm[n_stat:])
        var_Pm[i] = np.var(Pm[n_stat:])
        momentumFour_Mm = np.mean(Mm[n_stat:]**4)
        momentumFour_Pm = np.mean(Pm[n_stat:]**4)

        freq, this_power_spectrum = utils.compute_power_spectrum_traj(t_ref[n_stat:],Mm[n_stat:])
        power_spectrum_Mm += this_power_spectrum/n_iter
        freq, this_power_spectrum = utils.compute_power_spectrum_traj(t_ref[n_stat:],Pm[n_stat:])
        power_spectrum_Pm += this_power_spectrum/n_iter
        
    var_Mm_g = np.mean(var_Mm) + np.var(mean_Mm)
    var_Pm_g = np.mean(var_Pm) + np.var(mean_Pm)
    
    output={"std Mm": np.sqrt(var_Mm_g),
           "mean Mm": np.mean(mean_Mm),
           "power spectrum Mm": power_spectrum_Mm,
           "std Pm": np.sqrt(var_Pm_g),
           "mean Pm": np.mean(mean_Pm),
           "power spectrum Pm": power_spectrum_Pm,
           "times":t_ref[n_stat:], "frequencies": freq}
    
    return output


@jit(nopython = True)
def simulate_langevin_meanAndStd(n_iter=100, alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,
                                                      lambda_s=1,        
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=10000,
                                                      delta_t=1,
                                                      Omega=1,
                                                      sampling_timestep = 1.0):
    
    var_Ml = np.zeros(n_iter)
    var_Pl = np.zeros(n_iter)
    mean_Ml = np.zeros(n_iter)
    mean_Pl = np.zeros(n_iter)
    momentumFour_Ml = np.zeros(n_iter)
    momentumFour_Pl = np.zeros(n_iter)

    n_stat=int(2000/sampling_timestep)
    
    for i in range(n_iter):
        t,Ml,Pl=one_langevin_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega,
                                                      sampling_timestep = sampling_timestep)

        mean_Ml[i] = np.mean(Ml[n_stat:])
        mean_Pl[i] = np.mean(Pl[n_stat:])
        var_Ml[i] = np.var(Ml[n_stat:])
        var_Pl[i] = np.var(Pl[n_stat:])
        momentumFour_Ml[i] = np.mean(Ml[n_stat:]**4)
        momentumFour_Pl[i] = np.mean(Pl[n_stat:]**4)
        
    var_Ml_g = np.mean(var_Ml) + np.var(mean_Ml)
    var_Pl_g = np.mean(var_Pl) + np.var(mean_Pl)
    
    
    output={"std Ml": np.sqrt(var_Ml_g),
           "mean Ml": np.mean(mean_Ml),
           "momentumFour Ml": np.mean(momentumFour_Ml),
           "std Pl": np.sqrt(var_Pl_g),
           "mean Pl": np.mean(mean_Pl),
           "momentumFour Pl": np.mean(momentumFour_Pl)}
    
    
    return output


# @jit
def simulate_langevin_all(n_iter=100, alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,
                                                      lambda_s=1,        
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=10000,
                                                      delta_t=1,
                                                      Omega=1,
                                                      sampling_timestep = 1.0):
    
    var_Ml = np.zeros(n_iter)
    var_Pl = np.zeros(n_iter)
    mean_Ml = np.zeros(n_iter)
    mean_Pl = np.zeros(n_iter)
    
        
    t,Ml,Pl=one_langevin_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega,
                                                      sampling_timestep = sampling_timestep)

    n_stat=int(2000/sampling_timestep)
    freq_ref, test_power_spectrum = utils.compute_power_spectrum_traj(t[n_stat:],Pl[n_stat:])
    power_spectrum_Ml=np.zeros_like(test_power_spectrum)
    power_spectrum_Pl=np.zeros_like(test_power_spectrum)
    
    for i in range(n_iter):
        t,Ml,Pl=one_langevin_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega,
                                                      sampling_timestep = sampling_timestep)

        mean_Ml[i] = np.mean(Ml[n_stat:])
        mean_Pl[i] = np.mean(Pl[n_stat:])
        var_Ml[i] = np.var(Ml[n_stat:])
        var_Pl[i] = np.var(Pl[n_stat:])

        freq, this_power_spectrum = utils.compute_power_spectrum_traj(t[n_stat:],Ml[n_stat:])
        power_spectrum_Ml += this_power_spectrum/n_iter
        freq, this_power_spectrum = utils.compute_power_spectrum_traj(t[n_stat:],Pl[n_stat:])
        power_spectrum_Pl += this_power_spectrum/n_iter
        
    var_Ml_g = np.mean(var_Ml) + np.var(mean_Ml)
    var_Pl_g = np.mean(var_Pl) + np.var(mean_Pl)
    
    
    output={"std Ml": np.sqrt(var_Ml_g),
           "mean Ml": np.mean(mean_Ml),
           "power spectrum Ml": power_spectrum_Ml,
           "std Pl": np.sqrt(var_Pl_g),
           "mean Pl": np.mean(mean_Pl),
           "power spectrum Pl": power_spectrum_Pl,
           "times":t[n_stat:], "frequencies": freq}
    
    return output


# @jit
def simulate_lna_meanAndStd(n_iter=100, alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,
                                                      lambda_s=1,        
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=10000,
                                                      delta_t=1,
                                                      Omega=1,
                                                      sampling_timestep = 1.0):
    
    var_Mlna = np.zeros(n_iter)
    var_Plna = np.zeros(n_iter)
    mean_Mlna = np.zeros(n_iter)
    mean_Plna = np.zeros(n_iter)
    momentumFour_Mlna = np.zeros(n_iter)
    momentumFour_Plna = np.zeros(n_iter) 

    n_stat=int(2000/sampling_timestep)
    
    for i in range(n_iter):
        t,Mlna,Plna=one_trajectory_LNA(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega,
                                                      sampling_timestep = sampling_timestep)

        mean_Mlna[i] = np.mean(Mlna[n_stat:])
        mean_Plna[i] = np.mean(Plna[n_stat:])
        var_Mlna[i] = np.var(Mlna[n_stat:])
        var_Plna[i] = np.var(Plna[n_stat:])
        momentumFour_Mlna[i] = np.mean(Mlna[n_stat:]**4)
        momentumFour_Plna[i] = np.mean(Plna[n_stat:]**4)

    var_Mlna_g = np.mean(var_Mlna) + np.var(mean_Mlna)
    var_Plna_g = np.mean(var_Plna) + np.var(mean_Plna)  
    
    output={"std Mlna": np.sqrt(var_Mlna_g),
           "mean Mlna": np.mean(mean_Mlna),
           "momentumFour Mlna": np.mean(momentumFour_Mlna),
           "std Plna": np.sqrt(var_Plna_g),
           "mean Plna": np.mean(mean_Plna),
           "momentumFour Plna": np.mean(momentumFour_Plna)}
    
    return output


# @jit
def simulate_lna_all(n_iter=100, alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,
                                                      lambda_s=1,        
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=10000,
                                                      delta_t=1,
                                                      Omega=1,
                                                      sampling_timestep = 1.0):
    
    var_Mlna = np.zeros(n_iter)
    var_Plna = np.zeros(n_iter)
    mean_Mlna = np.zeros(n_iter)
    mean_Plna = np.zeros(n_iter)
        
    t,Mlna,Plna=one_trajectory_LNA(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega,
                                                      sampling_timestep = sampling_timestep)

    n_stat=int(2000/sampling_timestep)
    freq_ref, test_power_spectrum = utils.compute_power_spectrum_traj(t[n_stat:],Plna[n_stat:])
    power_spectrum_M=np.zeros_like(test_power_spectrum)
    power_spectrum_P=np.zeros_like(test_power_spectrum)
    
    for s in range(n_iter):
        t,Mlna,Plna=one_trajectory_LNA(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                      lambda_s=lambda_s,
                                                      P_0=P_0,
                                                      h=h,
                                                      tau=tau,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=T,
                                                      delta_t=delta_t,
                                                      Omega=Omega,
                                                      sampling_timestep = sampling_timestep)

        mean_Mlna[s] = np.mean(Mlna[n_stat:])
        mean_Plna[s] = np.mean(Plna[n_stat:])
        var_Mlna[s] = np.var(Mlna[n_stat:])
        var_Plna[s] = np.var(Plna[n_stat:])

        freq, this_power_spectrum = utils.compute_power_spectrum_traj(t[n_stat:],Mlna[n_stat:])
        power_spectrum_M += this_power_spectrum
        freq, this_power_spectrum = utils.compute_power_spectrum_traj(t[n_stat:],Plna[n_stat:])
        power_spectrum_P += this_power_spectrum
        
    var_Mlna_g = np.mean(var_Mlna) + np.var(mean_Mlna)
    var_Plna_g = np.mean(var_Plna) + np.var(mean_Plna)
    
    power_spectrum_M/=n_iter
    power_spectrum_P/=n_iter
    
    
    output={"std Mlna": np.sqrt(var_Mlna_g),
           "mean Mlna": np.mean(mean_Mlna),
           "power spectrum Mlna": power_spectrum_M,
           "std Plna": np.sqrt(var_Plna_g),
           "mean Plna": np.mean(mean_Plna),
           "power spectrum Plna": power_spectrum_P,
           "times":t[n_stat:], "frequencies": freq}
    
    return output

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
    
    sigma_m2= alpha_m/Omega*f_P + mu_m/Omega*M_stat + alpha_m**2/lambda_s*  2*(P_stat/P_0)**(2*h)* f_P**3
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

