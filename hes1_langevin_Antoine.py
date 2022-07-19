# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:18:59 2022

@author: tonio
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt
from scipy.optimize import bisect
import hes1_master_Antoine as master
from numba import jit

@jit
def one_trajectory(alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,             #one trajectory of langevin equation, scheme Euler-Maruyama
                                                      lambda_s=1,       #or Milstein
                                                      P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=10,
                                                      M_init=20,
                                                      T=1000,
                                                      delta_t=1,
                                                      Omega=1,
                                                      drift="classical"):
    n_t=int(T/delta_t)
    k_delay=round(tau/delta_t)
    t=np.linspace(0,T,n_t)
    P=np.zeros(n_t)
    M=np.zeros(n_t)

    M[0]=M_init/Omega
    P[0]=P_init/Omega    
    
    for i in range(n_t-1):
        
        hill_function=0
        mean_switch=0
        var_switch=0
        
        if i>= k_delay:
            hill_function=1/(1+(P[i-k_delay]/P_0)**h)
            mean_switch=(alpha_m*h/lambda_s)*(alpha_p*M[i-k_delay] - mu_p*P[i-k_delay])/P_0*((P[i-k_delay]/P_0)**(h-1))*hill_function**3
            var_switch=(alpha_m**2/lambda_s)*2*(P[i-k_delay]/P_0)**h*hill_function**3
           
        w_m=np.random.normal(0,np.sqrt(delta_t))
        w_m2=np.random.normal(0,np.sqrt(delta_t))
        w_p=np.random.normal(0,np.sqrt(delta_t))
        
        mean_increment_M=alpha_m*hill_function - mu_m*M[i] + mean_switch*(drift=="new")
        std_increment_M =np.sqrt(alpha_m/Omega*hill_function + mu_m/Omega*M[i] + var_switch)
        mean_increment_P=alpha_p*M[i] - mu_p*P[i]
        std_increment_P =np.sqrt(alpha_p/Omega*M[i] + mu_p/Omega*P[i])
        
        M[i+1]=abs(M[i] + mean_increment_M*delta_t + std_increment_M*w_m)
        P[i+1]=abs(P[i] + mean_increment_P*delta_t + std_increment_P*w_p)

    return t,M,P



@jit
def multiple_trajectories(n_iter=100,alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,        #perform many realisations
                                                      lambda_s=1,                    #and gather inside of a table
                                                      P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=10,
                                                      M_init=20,
                                                      T=1000,
                                                      delta_t=1,
                                                      Omega=1,
                                                      drift="classical"):
    
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
                                                      Omega=Omega,
                                                      drift=drift)
        table_M[k,:]=M
        table_P[k,:]=P
    
    return t,table_M,table_P



@jit
def pool_values(n_iter=100,alpha_m=1,alpha_p=1,mu_m=0.03,mu_p=0.03,      #pool many stationary realisations
                                                      lambda_s=1,        #inside of an array
                                                      P_0=1,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=10,
                                                      M_init=20,
                                                      T=1000,
                                                      delta_t=1,
                                                      Omega=1,
                                                      drift="classical"):
    
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
                                                      Omega=Omega,                                            
                                                      drift=drift)        #compute many realisations from this state
    
    
    pool_M=np.reshape(table_M[:,(n_t//2):], n_iter*(n_t//2))   #pool everything together
    pool_P=np.reshape(table_P[:,(n_t//2):], n_iter*(n_t//2))
    
    return pool_M,pool_P

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
                                                      Omega=float('infinity'),
                                                      drift='classical')


