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
from numba import jit


@jit
def one_trajectory( alpha_m=1, alpha_p=1,mu_m=0.03,mu_p=0.03,lambda_s=1,
                                                              P_0=1,
                                                              h=4.1,
                                                              tau=0.1,
                                                              P_init=10,
                                                              M_init=20,
                                                              sigma_init=1,
                                                              Omega=1,
                                                              T=1000):
    t=[0] #array of times when a reaction is started or ended
    P=[P_init] #array of P
    M=[M_init] #array of M
    sigma=[sigma_init] #array of environment sigma
    d_react=[] #list (queue) of end times of delayed reactions
    
    def perform_reaction(a_0,t,M,P):
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
            if sigma[-1]==1:                                          #add another term depending on state of environment
                a_0+=lambda_s*(P[-1]/n_p0)**h
            else:
                a_0+=lambda_s   

            r=rd.uniform(0,1)                      #generate delta via inverse transform method (exponential distribution)
            delta=-np.log(r)/a_0

            if len(d_react)!=0 and d_react[0]<=t[-1]+delta:    #if a delayed reaction is planned, creation of M
                t.append(d_react[0])
                M.append(M[-1]+1)
                P.append(P[-1])
                sigma.append(sigma[-1])

                del d_react[0]                               #then remove the delayed reaction

            else:                                            #else perform a new reaction
                t.append(t[-1]+delta)
                perform_reaction(a_0,t,M,P)

    run_master()
    
    return np.array(t),np.array(M)/Omega,np.array(P)/Omega




@jit
def multiple_trajectories(n_iter=100,alpha_m=1, alpha_p=1,mu_m=0.03,mu_p=0.03,lambda_s=1,   #perform many realisations
                                                              P_0=1,                      #and gather inside of a table
                                                              h=4.1,
                                                              tau=0.1,
                                                              P_init=10,
                                                              M_init=20,
                                                              sigma_init=1,
                                                              T=1000,
                                                              Omega=1,
                                                              delta_t=1):
    n_t=int(T/delta_t)
    t_ref=np.linspace(0,T,n_t)
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
                                                              T=T)
        table_M[k,:]=spinter.interp1d(t,M,kind="zero")(t_ref)
        table_P[k,:]=spinter.interp1d(t,P,kind="zero")(t_ref)
    
    return t_ref,table_M,table_P


@jit
def pool_values(n_iter=100,alpha_m=1, alpha_p=1,mu_m=0.03,mu_p=0.03,lambda_s=1,            #pool many stationary realisations
                                                              P_0=1,                   #inside of an array
                                                              h=4.1,
                                                              tau=0.1,
                                                              P_init=10,
                                                              M_init=20,
                                                              sigma_init=1,
                                                              T=1000,
                                                              delta_t=1,
                                                              Omega=1):
    
    n_t=int(T/delta_t)
    t=np.linspace(0,T,n_t)
    
    t_ref,table_M,table_P=multiple_trajectories(n_iter=n_iter,alpha_m=alpha_m, alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,lambda_s=lambda_s,
                                                              P_0=P_0,
                                                              h=h,
                                                              tau=tau,
                                                              P_init=P_init,
                                                              M_init=M_init,
                                                              sigma_init=sigma_init,
                                                              T=T,
                                                              delta_t=delta_t,
                                                              Omega=Omega)               #compute many realisations from 

    pool_M=[]  #np.reshape(table_M, n_iter*iterations) #pool everything together
    pool_P=[]  #np.reshape(table_P, n_iter*iterations)
    for k in range(n_iter):
        pool_M=pool_M+list(table_M[k,(n_t//2):])
        pool_P=pool_P+list(table_P[k,(n_t//2):])
    return np.array(pool_M),np.array(pool_P)

