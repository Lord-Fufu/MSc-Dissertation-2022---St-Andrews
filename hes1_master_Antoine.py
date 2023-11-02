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
    t=[0] #list of times when a reaction is started or ended
    P=[P_init] #list of Hes1 molecule numbers
    M=[M_init] #list of mRNA molecule numbers
    sigma=[sigma_init] #list of environment configuration
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
                perform_reaction(a_0,t,M,P)

    run_master()
    
    times=np.array(t)
    mRNA=np.array(M)/Omega
    Hes1=np.array(P)/Omega

    return times,mRNA,Hes1

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


@jit
def multiple_trajectories(n_iter=100,alpha_m=1, alpha_p=1,mu_m=0.03,mu_p=0.03,lambda_s=1,   
                                                              P_0=1,                        
                                                              h=4.1,
                                                              tau=0.1,
                                                              P_init=10,
                                                              M_init=20,
                                                              sigma_init=1,
                                                              T=1000,
                                                              Omega=1,
                                                              delta_t=1):
    n_t=int(T/delta_t)
    t_ref=np.linspace(0,T,n_t)             #define a time mesh
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
                                                              T=T)                         #run one_trajectory n_iter times
        
        table_M[k,:]=spinter.interp1d(t,M,kind="zero")(t_ref)   #interpolate on the mesh and gather in a table
        table_P[k,:]=spinter.interp1d(t,P,kind="zero")(t_ref)
    
    return t_ref,table_M,table_P


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

@jit
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

@jit
def one_trajectory_noSwitchNoise( alpha_m=1, alpha_p=1,mu_m=0.03,mu_p=0.03,
                                                              P_0=1,
                                                              h=4.1,
                                                              tau=0.1,
                                                              P_init=10,
                                                              M_init=20,
                                                              sigma_init=1,
                                                              Omega=1,
                                                              T=1000):
    t=[0] #list of times when a reaction is started or ended
    P=[P_init] #list of Hes1 molecule numbers
    M=[M_init] #list of mRNA molecule numbers
    d_react=[] #list (queue) of end times of delayed reactions
    
    n_p0=P_0*Omega
    
    def perform_reaction(a_0,t,M,P):
        rr=rd.uniform(0,1)
        a_1=mu_m*M[-1]
        a_2=mu_p*P[-1]
        a_3=alpha_p*M[-1]
        a_4=alpha_m*Omega/(1+(P[-1]/n_p0)**h)

        if rr<a_1/a_0:                      #destruction of M
            M.append(M[-1]-1)
            P.append(P[-1])
        elif rr < (a_1+a_2)/a_0:            #destruction of P
            M.append(M[-1])
            P.append(P[-1]-1)
        elif rr < (a_1+a_2+a_3)/a_0:        #creation of P
            M.append(M[-1])
            P.append(P[-1]+1)
        elif rr < (a_1+a_2+a_3+a_4)/a_0:    #plan delayed reaction for creation of M 
            d_react.append(t[-1]+tau)
            M.append(M[-1])
            P.append(P[-1])
    
    def run_master():
        while t[-1]<T:
            a_0=mu_m*M[-1]+mu_p*P[-1]+alpha_p*M[-1]+alpha_m*Omega/(1+(P[-1]/n_p0)**h)    #total propensity
            
            r=rd.uniform(0,1)                                  #generate delta via inverse transform method (exponential distribution)
            delta=-np.log(r)/a_0

            if len(d_react)!=0 and d_react[0]<=t[-1]+delta:    #if a delayed reaction is planned, creation of M
                t.append(d_react[0])
                M.append(M[-1]+1)
                P.append(P[-1])

                del d_react[0]                               #then remove the delayed reaction

            else:                                            #else perform a new reaction
                t.append(t[-1]+delta)
                perform_reaction(a_0,t,M,P)

    run_master()
    
    times=np.array(t)
    mRNA=np.array(M)/Omega
    Hes1=np.array(P)/Omega
    
    return times,mRNA,Hes1

'''Generate one trace of the Hes1 model from the master equation (Gillespie algorithm) with effective average rate (large lambda_s).

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

    times : 1D ndarray
        Times at which the system-environment changes, i.e one reaction occurs.
        
    mRNA : 1D ndarray
        mRNA concentration taken at time values given in 'times'.
        
    Hes1 : 1D ndarray
        Hes1 concentration taken at time values given in 'times'.
'''
