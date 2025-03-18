import hes1_langevin_Antoine as langevin
import hes1_master_Antoine as master
import hes1_utils_Antoine as utils
import hes1_utils_general as general
import reviewer
import toggle_switch
# import jochen_utils as jutils

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import scipy.interpolate as spinter
import scipy.fft
import time
import datetime as dt
import os
from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import wasserstein_distance
from numba import jit
from scipy.special import rel_entr
import scipy.integrate


font = {'size'   : 8,
        'sans-serif' : 'Arial'}
plt.rc('font', **font)

alpha_m=39.93
alpha_p=21.56
mu_m=np.log(2)/30
mu_p=np.log(2)/90
h=4.78
P_0=24201.01
T=10000

def make_noise_comparison_figure():
    # delta_t=0.0001
    delta_t=0.01
    
    n_t=int(T/delta_t)
    
    val_lambda=0.1
    val_Omega=1
    val_tau=33.0


    time_trajm,mRNA_trajm,Hes1_trajm = master.one_trajectory(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,h=h,P_0=P_0,
                                                          lambda_s=val_lambda,        
                                                          tau=val_tau,
                                                          P_init=0,
                                                          M_init=0,
                                                          T=T,
                                                          Omega=val_Omega)
    
    time_trajmns,mRNA_trajmns,Hes1_trajmns = master.one_trajectory_noSwitchNoise(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,
                                                          mu_p=mu_p,
                                                          h=h,
                                                          P_0=P_0,
                                                          tau=val_tau,
                                                          P_init=0,
                                                          M_init=0,
                                                          T=T,
                                                          Omega=val_Omega)
    
    time_trajmdet,mRNA_trajmdet,Hes1_trajmdet = langevin.resolve_ODE(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,h=h,
                                                          P_0=P_0,
                                                          tau=val_tau,
                                                          P_init=0,
                                                          M_init=0,
                                                          T=T,
                                                          delta_t=0.01)
    
    
    time_trajmnd,mRNA_trajmnd,Hes1_trajmnd = langevin.one_trajectory_PDMP(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,h=h,
                                                          P_0=P_0,
                                                          lambda_s=val_lambda,        
                                                          tau=val_tau,
                                                          P_init=0,
                                                          M_init=0,
                                                          T=T, delta_t=delta_t)
    
    # fig,ax=plt.subplots(3,1, figsize=(7*ratio,7))
    # fig, ax = plt.subplots(3,1, figsize=(3.15,5.25), constrained_layout = True) 
    # fig, ax = plt.subplots(3,1, figsize=(3.15,1.75), constrained_layout = True, sharex = True) 
    # fig, ax = plt.subplots(3,1, figsize=(3.15,2.0), sharex = True) 
    fig, ax = plt.subplots(3,1, figsize=(3.3,2.0), sharex = True) 
    # fig, ax = plt.subplots(3,1, figsize=(4.25,2.5), sharex = True) 
        
    # ax[0].set_title('Hes1 concentration over time',fontdict=font) 
    
    # ax[0].set_ylim(50000,70000)
    ax[0].set_ylim(20000,160000)
    # ax[0].set_yticks([55000, 60000, 65000])
    # ax[0].set_yticklabels(['5.5', '6.0', '6.5'])  # Set y-axis tick labels manually
    ax[0].set_yticks([50000, 100000, 150000])
    ax[0].set_yticklabels(['5', '10', '15'])  # Set y-axis tick labels manually
    ax[0].set_xlim(0,10000)
    # ax[0].set_xticks([])
    ax[0].tick_params(axis='x', bottom=False)  # Remove x-axis ticks for the upper plot
    
        
    linem, = ax[0].plot(time_trajm,Hes1_trajm, color = 'blue', label = 'Full model', lw = 0.2)
    # linedet, = ax[0].plot(time_trajmdet,Hes1_trajmdet,'--', color = 'red', alpha=0.3, label = 'Deterministic', lw = 0.2)
    linedet, = ax[0].plot(time_trajmdet,Hes1_trajmdet,'--', color = 'red', alpha = 0.6, label = 'Deterministic', lw = 1.0)
    # ax[0].legend([linem,linedet], ['Gillespie','Deterministic'],loc='lower right')
    
    
    
    ax[1].set_ylabel('$P(t)$ /(1e4 cu)')
    # ax[1].set_ylim(50000,70000)
    ax[1].set_ylim(20000,160000)
    # ax[1].set_yticks([55000, 60000, 65000])
    ax[1].set_yticks([50000, 100000, 150000])
    ax[1].set_yticklabels(['5', '10', '15'])  # Set y-axis tick labels manually
    # ax[1].set_yticklabels(['5.5', '6.0', '6.5'])  # Set y-axis tick labels manually
    ax[1].set_xlim(0,10000)
    # ax[1].set_xticks([])
    ax[1].tick_params(axis='x', bottom=False)  # Remove x-axis ticks for the upper plot
    
    linemnd, = ax[1].plot(time_trajmnd,Hes1_trajmnd, color = 'purple', label = 'Bursting noise only', lw = 0.2)
    linedet, = ax[1].plot(time_trajmdet,Hes1_trajmdet,'--', alpha = 0.6, color = 'red', lw = 1.0)
    # ax[1].legend([linemnd,linedet], ['PDMP','Deterministic'], loc='lower right')
    
    
    
    ax[2].set_xlabel('$t$ /min')
    # ax[2].set_ylim(50000,70000)
    ax[2].set_ylim(20000,160000)
    # ax[2].set_yticks([55000, 60000, 65000])
    # ax[2].set_yticklabels(['5.5', '6.0', '6.5'])  # Set y-axis tick labels manually
    ax[2].set_yticks([50000, 100000, 150000])
    ax[2].set_yticklabels(['5', '10', '15'])  # Set y-axis tick labels manually
    ax[2].set_xlim(0,10000)
    
    linemns, = ax[2].plot(time_trajmns,Hes1_trajmns, color = 'orange',label = 'Copy-number noise only', lw = 0.2)
    linedet, = ax[2].plot(time_trajmdet,Hes1_trajmdet,'--', color = 'red', alpha = 0.6, lw = 1.0)
    # ax[2].legend([linemns,linedet], ['Gillespie, $\lambda=\infty$','Deterministic'], loc='lower right')
    
    handles, labels = [], []
    for this_axis in ax:
        this_handle, this_label = this_axis.get_legend_handles_labels()
        handles.extend(this_handle)
        labels.extend(this_label)
    
    plt.subplots_adjust(hspace=0)
    fig.legend(handles, labels, loc='lower left', ncol =2, fontsize = 8,
               bbox_to_anchor=(0.1, 0.75))
            #    columnspacing = 1.0,  bbox_to_anchor=(0.01, 0.75))
    plt.tight_layout(h_pad = 0, rect = [0,-0.05,1,0.82])
    # plt.tight_layout(h_pad = 0)
    
    plt.savefig(os.path.join(os.path.dirname(__file__),'plots','comparison_full_with_vs_without_switching_alternative_new.pdf'))
    
def make_approximation_comparison_figure():
    delta_t=0.01
    
    n_t=int(T/delta_t)
    
    val_lambda=10
    val_Omega=100
    val_tau=33
    # val_lambda=0.1
    # val_Omega=1
    # val_tau=33
    
    m_stat,p_stat = utils.resolve_stationary_state(alpha_m,mu_m,alpha_p,mu_p,h,P_0)
    
    time_trajm,mRNA_trajm,Hes1_trajm = master.one_trajectory(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,
                                                          mu_p=mu_p,
                                                          h=h,
                                                          P_0=P_0,
                                                          lambda_s=val_lambda,
                                                          tau=val_tau,
                                                          P_init=0,
                                                          M_init=0,
                                                          T=T,
                                                          Omega=val_Omega)
            
    time_trajl,mRNA_trajl,Hes1_trajl = langevin.one_trajectory(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,
                                                          mu_p=mu_p,
                                                          h=h,
                                                          P_0=P_0,
                                                          lambda_s=val_lambda,
                                                          tau=val_tau,
                                                          P_init=0,
                                                          M_init=0,
                                                          T=T,
                                                          delta_t=delta_t,
                                                          Omega=val_Omega)
    
    time_trajlna,mRNA_trajlna,Hes1_trajlna = langevin.one_trajectory_LNA(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,
                                                          mu_p=mu_p,
                                                          h=h,
                                                          P_0=P_0,
                                                          lambda_s=val_lambda,
                                                          tau=val_tau,
                                                          T=T,
                                                          delta_t=delta_t,
                                                          Omega=val_Omega)
    
    fig,ax=plt.subplots(3,1,figsize=(3.3,2.0), sharex = True)
        
    # ax[0].set_ylim(20000,160000)
    ax[0].set_xlim(0,10000)
        
    linem,=ax[0].plot(time_trajm,Hes1_trajm, color = 'blue', label = 'Full model', lw = 0.2)
    # ax[0].legend([linem], ['Full model'], loc='upper right')
    ax[0].tick_params(axis='x', bottom=False)  # Remove x-axis ticks for the upper plot
    # ax[0].set_yticks([50000, 100000, 150000])
    # ax[0].set_yticklabels(['5', '10', '15'])  # Set y-axis tick labels manually
    ax[0].set_ylim(50000,70000)
    ax[0].set_yticks([55000, 60000, 65000])
    ax[0].set_yticklabels(['5.5', '6.0', '6.5'])  # Set y-axis tick labels manually
    
    
    ax[1].set_ylabel('$P(t)$ /(1e4 cu)') 
    
    # ax[1].set_ylim(20000,160000)
    # ax[1].set_yticks([50000, 100000, 150000])
    # ax[1].set_yticklabels(['5', '10', '15'])  # Set y-axis tick labels manually
    ax[1].set_ylim(50000,70000)
    ax[1].set_yticks([55000, 60000, 65000])
    ax[1].set_yticklabels(['5.5', '6.0', '6.5'])  # Set y-axis tick labels manually
    ax[1].set_xlim(0,10000)
        
    linel,=ax[1].plot(time_trajl,Hes1_trajl, color = 'green', label = 'CLE', lw = 0.2)
    # ax[1].legend([linel], ['Langevin model'], loc='upper right')
    ax[1].tick_params(axis='x', bottom=False)  # Remove x-axis ticks for the upper plot
    
    
    
    ax[2].set_xlabel('$t$ /min') 
    
    # ax[2].set_ylim(20000,160000)
    # ax[2].set_xlim(0,10000)
    # ax[2].set_yticks([50000, 100000, 150000])
    ax[2].set_ylim(50000,70000)
    ax[2].set_yticks([55000, 60000, 65000])
    ax[2].set_yticklabels(['5.5', '6.0', '6.5'])  # Set y-axis tick labels manually
    # ax[2].set_yticklabels(['5', '10', '15'])  # Set y-axis tick labels manually
        
    linelna,=ax[2].plot(time_trajlna,Hes1_trajlna, color = 'red', label = 'LNA', lw = 0.2, ls= '--')
    # ax[2].legend([linelna], ['LNA model'], loc='upper right')
    
    handles, labels = [], []
    for this_axis in ax:
        this_handle, this_label = this_axis.get_legend_handles_labels()
        handles.extend(this_handle)
        labels.extend(this_label)
    
    plt.subplots_adjust(hspace=0)
    fig.legend(handles, labels, loc='lower left', ncol =3, fontsize = 8,  bbox_to_anchor=(0.2, 0.8))
    plt.tight_layout(h_pad = 0, rect = [0,-0.07,1,0.85])
    
    fig.text(0.02,0.9,'A',size=9,weight='bold')
    plt.savefig(os.path.join(os.path.dirname(__file__),'plots','comparison_Full_vs_Langevin_new.pdf'))
        
def make_noise_comparison_figure_reviewer_switching():
    # delta_t=0.0001
    delta_t=0.01
    
    n_t=int(T/delta_t)
    
    val_lambda=5
    val_Omega=1
    val_tau=33.0


    time_trajm,mRNA_trajm,Hes1_trajm = reviewer.one_ssa_trajectory(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,h=h,P_0=P_0,
                                                          lambda_s=val_lambda,        
                                                          tau=val_tau,
                                                          P_init=0,
                                                          M_init=0,
                                                          T=T,
                                                          Omega=val_Omega)
    
    time_trajmns,mRNA_trajmns,Hes1_trajmns = master.one_trajectory_noSwitchNoise(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,
                                                          mu_p=mu_p,
                                                          h=h,
                                                          P_0=P_0,
                                                          tau=val_tau,
                                                          P_init=0,
                                                          M_init=0,
                                                          T=T,
                                                          Omega=val_Omega)
    
    time_trajmdet,mRNA_trajmdet,Hes1_trajmdet = langevin.resolve_ODE(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,h=h,
                                                          P_0=P_0,
                                                          tau=val_tau,
                                                          P_init=0,
                                                          M_init=0,
                                                          T=T,
                                                          delta_t=0.01)
    
    
    time_trajmnd,mRNA_trajmnd,Hes1_trajmnd = reviewer.one_trajectory_PDMP(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,h=h,
                                                          P_0=P_0,
                                                          lambda_s=val_lambda,        
                                                          tau=val_tau,
                                                          P_init=0,
                                                          M_init=0,
                                                          T=T, delta_t=delta_t)
    
    # fig,ax=plt.subplots(3,1, figsize=(7*ratio,7))
    # fig, ax = plt.subplots(3,1, figsize=(3.15,5.25), constrained_layout = True) 
    # fig, ax = plt.subplots(3,1, figsize=(3.15,1.75), constrained_layout = True, sharex = True) 
    # fig, ax = plt.subplots(3,1, figsize=(3.15,2.0), sharex = True) 
    fig, ax = plt.subplots(3,1, figsize=(3.3,2.0), sharex = True) 
    # fig, ax = plt.subplots(3,1, figsize=(4.25,2.5), sharex = True) 
        
    # ax[0].set_title('Hes1 concentration over time',fontdict=font) 
    
    # ax[0].set_ylim(50000,70000)
    ax[0].set_ylim(20000,160000)
    # ax[0].set_yticks([55000, 60000, 65000])
    # ax[0].set_yticklabels(['5.5', '6.0', '6.5'])  # Set y-axis tick labels manually
    ax[0].set_yticks([50000, 100000, 150000])
    ax[0].set_yticklabels(['5', '10', '15'])  # Set y-axis tick labels manually
    ax[0].set_xlim(0,10000)
    # ax[0].set_xticks([])
    ax[0].tick_params(axis='x', bottom=False)  # Remove x-axis ticks for the upper plot
    
        
    linem, = ax[0].plot(time_trajm,Hes1_trajm, color = 'blue', label = 'Full model', lw = 0.2)
    # linedet, = ax[0].plot(time_trajmdet,Hes1_trajmdet,'--', color = 'red', alpha=0.3, label = 'Deterministic', lw = 0.2)
    linedet, = ax[0].plot(time_trajmdet,Hes1_trajmdet,'--', color = 'red', alpha = 0.6, label = 'Deterministic', lw = 1.0)
    # ax[0].legend([linem,linedet], ['Gillespie','Deterministic'],loc='lower right')
    
    
    
    ax[1].set_ylabel('$P(t)$ /(1e4 cu)')
    # ax[1].set_ylim(50000,70000)
    ax[1].set_ylim(20000,160000)
    # ax[1].set_yticks([55000, 60000, 65000])
    ax[1].set_yticks([50000, 100000, 150000])
    ax[1].set_yticklabels(['5', '10', '15'])  # Set y-axis tick labels manually
    # ax[1].set_yticklabels(['5.5', '6.0', '6.5'])  # Set y-axis tick labels manually
    ax[1].set_xlim(0,10000)
    # ax[1].set_xticks([])
    ax[1].tick_params(axis='x', bottom=False)  # Remove x-axis ticks for the upper plot
    
    linemnd, = ax[1].plot(time_trajmnd,Hes1_trajmnd, color = 'purple', label = 'Bursting noise only', lw = 0.2)
    linedet, = ax[1].plot(time_trajmdet,Hes1_trajmdet,'--', alpha = 0.6, color = 'red', lw = 1.0)
    # ax[1].legend([linemnd,linedet], ['PDMP','Deterministic'], loc='lower right')
    
    
    
    ax[2].set_xlabel('$t$ /min')
    # ax[2].set_ylim(50000,70000)
    ax[2].set_ylim(20000,160000)
    # ax[2].set_yticks([55000, 60000, 65000])
    # ax[2].set_yticklabels(['5.5', '6.0', '6.5'])  # Set y-axis tick labels manually
    ax[2].set_yticks([50000, 100000, 150000])
    ax[2].set_yticklabels(['5', '10', '15'])  # Set y-axis tick labels manually
    ax[2].set_xlim(0,10000)
    
    linemns, = ax[2].plot(time_trajmns,Hes1_trajmns, color = 'orange',label = 'Copy-number noise only', lw = 0.2)
    linedet, = ax[2].plot(time_trajmdet,Hes1_trajmdet,'--', color = 'red', alpha = 0.6, lw = 1.0)
    # ax[2].legend([linemns,linedet], ['Gillespie, $\lambda=\infty$','Deterministic'], loc='lower right')
    
    handles, labels = [], []
    for this_axis in ax:
        this_handle, this_label = this_axis.get_legend_handles_labels()
        handles.extend(this_handle)
        labels.extend(this_label)
    
    plt.subplots_adjust(hspace=0)
    fig.legend(handles, labels, loc='lower left', ncol =2, fontsize = 8,
               bbox_to_anchor=(0.1, 0.75))
            #    columnspacing = 1.0,  bbox_to_anchor=(0.01, 0.75))
    plt.tight_layout(h_pad = 0, rect = [0,-0.05,1,0.82])
    # plt.tight_layout(h_pad = 0)
    
    plt.savefig(os.path.join(os.path.dirname(__file__),'plots','comparison_full_with_vs_without_switching_new_model.pdf'))
 
def make_approximation_comparison_figure_reviewer_switching():
    delta_t=0.01
    
    n_t=int(T/delta_t)
    
    val_lambda=100
    val_Omega=100
    val_tau=33
    # val_lambda=0.1
    # val_Omega=1
    # val_tau=33
    
    m_stat,p_stat = utils.resolve_stationary_state(alpha_m,mu_m,alpha_p,mu_p,h,P_0)
    
    time_trajm,mRNA_trajm,Hes1_trajm = reviewer.one_ssa_trajectory(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,
                                                          mu_p=mu_p,
                                                          h=h,
                                                          P_0=P_0,
                                                          lambda_s=val_lambda,
                                                          tau=val_tau,
                                                          P_init=0,
                                                          M_init=0,
                                                          T=T,
                                                          Omega=val_Omega)
            
    time_trajl,mRNA_trajl,Hes1_trajl = reviewer.one_langevin_trajectory(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,
                                                          mu_p=mu_p,
                                                          h=h,
                                                          P_0=P_0,
                                                          lambda_s=val_lambda,
                                                          tau=val_tau,
                                                          P_init=0,
                                                          M_init=0,
                                                          T=T,
                                                          delta_t=delta_t,
                                                          Omega=val_Omega)
    
    time_trajlna,mRNA_trajlna,Hes1_trajlna = reviewer.one_trajectory_LNA(alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,
                                                          mu_p=mu_p,
                                                          h=h,
                                                          P_0=P_0,
                                                          lambda_s=val_lambda,
                                                          tau=val_tau,
                                                          T=T,
                                                          delta_t=delta_t,
                                                          Omega=val_Omega)
    
    fig,ax=plt.subplots(3,1,figsize=(3.3,2.0), sharex = True)
        
    # ax[0].set_ylim(20000,160000)
    ax[0].set_xlim(0,10000)
        
    linem,=ax[0].plot(time_trajm,Hes1_trajm, color = 'blue', label = 'Full model', lw = 0.2)
    # ax[0].legend([linem], ['Full model'], loc='upper right')
    ax[0].tick_params(axis='x', bottom=False)  # Remove x-axis ticks for the upper plot
    # ax[0].set_yticks([50000, 100000, 150000])
    # ax[0].set_yticklabels(['5', '10', '15'])  # Set y-axis tick labels manually
    ax[0].set_ylim(50000,70000)
    ax[0].set_yticks([55000, 60000, 65000])
    ax[0].set_yticklabels(['5.5', '6.0', '6.5'])  # Set y-axis tick labels manually
    
    
    ax[1].set_ylabel('$P(t)$ /(1e4 cu)') 
    
    # ax[1].set_ylim(20000,160000)
    # ax[1].set_yticks([50000, 100000, 150000])
    # ax[1].set_yticklabels(['5', '10', '15'])  # Set y-axis tick labels manually
    ax[1].set_ylim(50000,70000)
    ax[1].set_yticks([55000, 60000, 65000])
    ax[1].set_yticklabels(['5.5', '6.0', '6.5'])  # Set y-axis tick labels manually
    ax[1].set_xlim(0,10000)
        
    linel,=ax[1].plot(time_trajl,Hes1_trajl, color = 'green', label = 'CLE', lw = 0.2)
    # ax[1].legend([linel], ['Langevin model'], loc='upper right')
    ax[1].tick_params(axis='x', bottom=False)  # Remove x-axis ticks for the upper plot
    
    
    
    ax[2].set_xlabel('$t$ /min') 
    
    # ax[2].set_ylim(20000,160000)
    # ax[2].set_xlim(0,10000)
    # ax[2].set_yticks([50000, 100000, 150000])
    ax[2].set_ylim(50000,70000)
    ax[2].set_yticks([55000, 60000, 65000])
    ax[2].set_yticklabels(['5.5', '6.0', '6.5'])  # Set y-axis tick labels manually
    # ax[2].set_yticklabels(['5', '10', '15'])  # Set y-axis tick labels manually
        
    linelna,=ax[2].plot(time_trajlna,Hes1_trajlna, color = 'red', label = 'LNA', lw = 0.2, ls= '--')
    # ax[2].legend([linelna], ['LNA model'], loc='upper right')
    
    handles, labels = [], []
    for this_axis in ax:
        this_handle, this_label = this_axis.get_legend_handles_labels()
        handles.extend(this_handle)
        labels.extend(this_label)
    
    plt.subplots_adjust(hspace=0)
    fig.legend(handles, labels, loc='lower left', ncol =3, fontsize = 8,  bbox_to_anchor=(0.2, 0.8))
    plt.tight_layout(h_pad = 0, rect = [0,-0.07,1,0.85])
    
    fig.text(0.02,0.9,'A',size=9,weight='bold')
    plt.savefig(os.path.join(os.path.dirname(__file__),'plots','comparison_Full_vs_Langevin_new_model.pdf'))
 
def compute_power_spectrum_data():
    delta_t = 0.01
    val_lambda=[10**x for x in np.arange(-2,2.1,0.2)]
    val_Omega=[0.001, 0.01, 0.1, 1, 10, 100]
    val_tau=[33]
    
    n_lambda=np.size(val_lambda)
    n_Omega=np.size(val_Omega)
    n_tau=np.size(val_tau)
    
    read_directory = os.path.join(os.path.dirname(__file__),'paper_plots_mean-std-power-spectrum') 
    
    T = 10000
    spec= [alpha_m, alpha_p, mu_m, mu_p, h, P_0, T, delta_t]
    #read_directory = "paper plots 2024-4-27-0-42-48 mean-std-power-spectrum"
    pd.DataFrame(spec,columns=['value']).to_csv(os.path.join(read_directory,'data','spec.csv'))
    
    # # Computing power spectrum
    
    n_iter=100
    
    sampling_timestep = 10
    n_t=int(T/sampling_timestep)
    
    i=15
    j=-1
    k=0
    
    valueOfLambda = 10
    valueOfOmega = 100
    valueOfTau = 33
    # valueOfLambda = 0.1
    # valueOfOmega = 1
    # valueOfTau = 33
    
    print("Value of lambda:", valueOfLambda)
    print("Value of Omega:", valueOfOmega)
    print("Value of tau:", valueOfTau)
    
    t0 = time.time()
    
    output=general.simulate_master_all(n_iter=n_iter,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                                    h=h,
                                                                    P_0=P_0,
                                                                    lambda_s=valueOfLambda,
                                                                    tau=valueOfTau,
                                                                    P_init=0,
                                                                    M_init=0,
                                                                    T=T,
                                                                    delta_t=delta_t,
                                                                    Omega=valueOfOmega,
                                                                    sampling_timestep = sampling_timestep)
    
    print("Computation time:", (time.time() - t0)//60, "min ", (time.time() - t0)%60, "s")
    
    
    t_ref=output["times"]
    
    freq_ref=output["frequencies"]
    
    
    std_Mm=output["std Mm"]
    
    std_Pm=output["std Pm"]
    
    mean_Mm=output["mean Mm"]
    
    mean_Pm=output["mean Pm"]
    
    power_spectrum_Mm=output["power spectrum Mm"]
              
    power_spectrum_Pm=output["power spectrum Pm"]
    
    output=general.simulate_langevin_all(n_iter=n_iter,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                                    h=h,
                                                                    P_0=P_0,
                                                                    lambda_s=valueOfLambda,
                                                                    tau=valueOfTau,
                                                                    P_init=0,
                                                                    M_init=0,
                                                                    T=T,
                                                                    delta_t=delta_t,
                                                                    Omega=valueOfOmega,
                                                                    sampling_timestep = sampling_timestep)            
                
    t_ref=output["times"]
    
    freq_ref=output["frequencies"]
    
    
    std_Ml=output["std Ml"]
    
    std_Pl=output["std Pl"]
    
    mean_Ml=output["mean Ml"]
    
    mean_Pl=output["mean Pl"]
    
    power_spectrum_Ml=output["power spectrum Ml"]
              
    power_spectrum_Pl=output["power spectrum Pl"]
    
    output=general.simulate_lna_all(n_iter=n_iter,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                                    h=h,
                                                                    P_0=P_0,
                                                                    lambda_s=valueOfLambda,
                                                                    tau=valueOfTau,
                                                                    P_init=0,
                                                                    M_init=0,
                                                                    T=T,
                                                                    delta_t=delta_t,
                                                                    Omega=valueOfOmega,
                                                                    sampling_timestep = sampling_timestep)            
                
    t_ref=output["times"]
    
    freq_ref=output["frequencies"]
    
    
    std_Mlna=output["std Mlna"]
    
    std_Plna=output["std Plna"]
    
    mean_Mlna=output["mean Mlna"]
    
    mean_Plna=output["mean Plna"]
    
    power_spectrum_Mlna=output["power spectrum Mlna"]
              
    power_spectrum_Plna=output["power spectrum Plna"]
    
    paramPoint = [valueOfLambda, valueOfOmega, valueOfTau, sampling_timestep]
    
    pd.DataFrame(paramPoint,columns=['value']).to_csv(os.path.join(read_directory,'data','paramPoint-spectrum.csv'))
    pd.DataFrame(t_ref,columns=['value']).to_csv(os.path.join(read_directory,'data','t-spectrum.csv'))
    pd.DataFrame(freq_ref,columns=['value']).to_csv(os.path.join(read_directory,'data','freq-spectrum.csv'))
    
    pd.DataFrame(power_spectrum_Mm,columns=['value']).to_csv(os.path.join(read_directory,'data','power-spectrum-Mm.csv'))
    pd.DataFrame(power_spectrum_Pm,columns=['value']).to_csv(os.path.join(read_directory,'data','power-spectrum-Pm.csv'))
    pd.DataFrame(power_spectrum_Ml,columns=['value']).to_csv(os.path.join(read_directory,'data','power-spectrum-Ml.csv'))
    pd.DataFrame(power_spectrum_Pl,columns=['value']).to_csv(os.path.join(read_directory,'data','power-spectrum-Pl.csv'))
    pd.DataFrame(power_spectrum_Mlna,columns=['value']).to_csv(os.path.join(read_directory,'data','power-spectrum-Mlna.csv'))
    pd.DataFrame(power_spectrum_Plna,columns=['value']).to_csv(os.path.join(read_directory,'data','power-spectrum-Plna.csv'))
    
    # # Compute mean & STD
    
    n_iter_master = 2
    n_iter_sde = 2
    
    T = 100000
    sampling_timestep = 1
    n_t=int(T/sampling_timestep)
    
    arr_mean_Mm=np.zeros(n_lambda)
    arr_mean_Pm=np.zeros(n_lambda)
    
    arr_mean_Ml=np.zeros(n_lambda)
    arr_mean_Pl=np.zeros(n_lambda)
    
    arr_mean_Mlna=np.zeros(n_lambda)
    arr_mean_Plna=np.zeros(n_lambda)
    
    arr_std_Mm=np.zeros(n_lambda)
    arr_std_Pm=np.zeros(n_lambda)
    
    arr_std_Ml=np.zeros(n_lambda)
    arr_std_Pl=np.zeros(n_lambda)
    
    arr_std_Mlna=np.zeros(n_lambda)
    arr_std_Plna=np.zeros(n_lambda)
    
    arr_momFour_Mm=np.zeros(n_lambda)
    arr_momFour_Pm=np.zeros(n_lambda)
    
    arr_momFour_Ml=np.zeros(n_lambda)
    arr_momFour_Pl=np.zeros(n_lambda)
    
    arr_momFour_Mlna=np.zeros(n_lambda)
    arr_momFour_Plna=np.zeros(n_lambda)
    
    j=-1
    k=0
    index_OmegaTau = [j,k]
    
    valueOfOmega = 100
    valueOfTau = 33
    pd.DataFrame(index_OmegaTau,columns=['value']).to_csv(os.path.join(read_directory,'data','index.csv'))
    
    for i in range(n_lambda):
        valueOfLambda = val_lambda[i]
        
        t0 = time.time()
    
        print("Value of lambda:", valueOfLambda)
        print("Value of Omega:", valueOfOmega)
        print("Value of tau:", valueOfTau)
        
        if valueOfLambda <= 0.1:
            n_iter_master = 2
            n_iter_sde = 2
        else:
            n_iter_master = 10
            n_iter_sde = 10
            
        output=general.simulate_master_meanAndStd(n_iter=n_iter_master, alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                                    h=h,
                                                                    P_0=P_0,
                                                                    lambda_s=valueOfLambda,
                                                                    tau=valueOfTau,
                                                                    P_init=0,
                                                                    M_init=0,
                                                                    T=T,
                                                                    delta_t=delta_t,
                                                                    Omega=valueOfOmega,
                                                                    sampling_timestep = sampling_timestep)
    
        arr_mean_Mm[i]=output["mean Mm"]
        arr_mean_Pm[i]=output["mean Pm"]
        arr_std_Mm[i]=output["std Mm"]
        arr_std_Pm[i]=output["std Pm"]
        arr_momFour_Mm[i]=output["momentumFour Mm"]
        arr_momFour_Pm[i]=output["momentumFour Pm"]
    
        output=general.simulate_langevin_meanAndStd(n_iter=n_iter_sde,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                                    h=h,
                                                                    P_0=P_0,
                                                                    lambda_s=valueOfLambda,
                                                                    tau=valueOfTau,
                                                                    P_init=0,
                                                                    M_init=0,
                                                                    T=T,
                                                                    delta_t=delta_t,
                                                                    Omega=valueOfOmega,
                                                                    sampling_timestep = sampling_timestep)            
        arr_mean_Ml[i]=output["mean Ml"]
        arr_mean_Pl[i]=output["mean Pl"]
        arr_std_Ml[i]=output["std Ml"]
        arr_std_Pl[i]=output["std Pl"]
        arr_momFour_Ml[i]=output["momentumFour Ml"]
        arr_momFour_Pl[i]=output["momentumFour Pl"]
    
        
        output=general.simulate_lna_meanAndStd(n_iter=n_iter_sde,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                                    h=h,
                                                                    P_0=P_0,
                                                                    lambda_s=valueOfLambda,
                                                                    tau=valueOfTau,
                                                                    P_init=0,
                                                                    M_init=0,
                                                                    T=T,
                                                                    delta_t=delta_t,
                                                                    Omega=valueOfOmega,
                                                                    sampling_timestep = sampling_timestep)            
        arr_mean_Mlna[i]=output["mean Mlna"]
        arr_mean_Plna[i]=output["mean Plna"]
        arr_std_Mlna[i]=output["std Mlna"]
        arr_std_Plna[i]=output["std Plna"]
        arr_momFour_Mlna[i]=output["momentumFour Mlna"]
        arr_momFour_Plna[i]=output["momentumFour Plna"]
        
        print("\n")
        print("Number of iterations: ", n_iter_master, n_iter_sde)
        print("Computation time: ", (time.time() - t0)//3600, "h ", ((time.time() - t0)//60)%60, "min ", (time.time() - t0)%60, "s")
        print("\n")
    
    
    pd.DataFrame(val_lambda,columns=['value']).to_csv(os.path.join(read_directory,'data','lambda.csv'))
    pd.DataFrame(val_Omega,columns=['value']).to_csv(os.path.join(read_directory,'data','Omega.csv'))
    pd.DataFrame(val_tau,columns=['value']).to_csv(os.path.join(read_directory,'data','tau.csv'))
    pd.DataFrame(index_OmegaTau,columns=['value']).to_csv(os.path.join(read_directory,'data','index.csv'))
    
    pd.DataFrame(arr_std_Mm,columns=['value']).to_csv(os.path.join(read_directory,'data','std-Mm.csv'))
    pd.DataFrame(arr_std_Pm,columns=['value']).to_csv(os.path.join(read_directory,'data','std-Pm.csv'))
    pd.DataFrame(arr_std_Ml,columns=['value']).to_csv(os.path.join(read_directory,'data','std-Ml.csv'))
    pd.DataFrame(arr_std_Pl,columns=['value']).to_csv(os.path.join(read_directory,'data','std-Pl.csv'))
    pd.DataFrame(arr_std_Mlna,columns=['value']).to_csv(os.path.join(read_directory,'data','std-Mlna.csv'))
    pd.DataFrame(arr_std_Plna,columns=['value']).to_csv(os.path.join(read_directory,'data','std-Plna.csv'))

def plot_power_spectrum_data():
    font = {'size'   : 8,
            'sans-serif' : 'Arial'}
    plt.rc('font', **font)
    
    read_directory = os.path.join(os.path.dirname(__file__),'paper_plots_mean-std-power-spectrum') 
    # read_directory = os.path.join(os.getcwd(),'paper_plots_mean-std-power-spectrum') 
    # write_directory = 'power_spectra_plots'
    
    spec = pd.read_csv(os.path.join(read_directory,'data','spec.csv'))['value'].values
    paramPoint=pd.read_csv(os.path.join(read_directory,'data','paramPoint-spectrum.csv'))['value'].values
    t_ref=pd.read_csv(os.path.join(read_directory,'data','t-spectrum.csv'))['value'].values
    freq_ref=pd.read_csv(os.path.join(read_directory,'data','freq-spectrum.csv'))['value'].values
    
    power_spectrum_Mm=pd.read_csv(os.path.join(read_directory,'data','power-spectrum-Mm.csv'))['value'].values
    power_spectrum_Pm=pd.read_csv(os.path.join(read_directory,'data','power-spectrum-Pm.csv'))['value'].values
    power_spectrum_Ml=pd.read_csv(os.path.join(read_directory,'data','power-spectrum-Ml.csv'))['value'].values
    power_spectrum_Pl=pd.read_csv(os.path.join(read_directory,'data','power-spectrum-Pl.csv'))['value'].values
    power_spectrum_Mlna=pd.read_csv(os.path.join(read_directory,'data','power-spectrum-Mlna.csv'))['value'].values
    power_spectrum_Plna=pd.read_csv(os.path.join(read_directory,'data','power-spectrum-Plna.csv'))['value'].values
    
    [alpha_m, alpha_p, mu_m, mu_p, h, P_0, T, delta_t] = spec
    [valueOfLambda, valueOfOmega, valueOfTau, sampling_timestep] = paramPoint
    
    fig = plt.figure(figsize=(3.3,2.0), constrained_layout = True) 
    
    ax1 = plt.gca()
    
    # ax1.set_title('Hes1 Power spectrum',fontdict=font) 
    # ax1.set_xlabel('Angular frequency') 
    ax1.set_xlabel('$\omega$ $/\mathrm{min}^{-1}$') 
    # ax1.set_ylabel('Power') 
    ax1.set_ylabel('$S_p(\omega)$ /$(\mathrm{cu}^2\mathrm{min})$') 
    
    freq_th,Sm,Sp=utils.lna_power_spectrum(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                    h=h,
                                                    P_0=P_0,
                                                    lambda_s=valueOfLambda,
                                                    tau=valueOfTau,
                                                    Omega=valueOfOmega,
                                                    T=T,
                                                    delta_t=sampling_timestep)
    
    fig.text(0.02, 0.9, 'B', size=9, weight='bold')
    linelna,=ax1.plot(freq_ref[1:], power_spectrum_Plna[1:], color = 'red', alpha = 0.2)
    linet,=ax1.plot(freq_th[1:], Sp[1:], '--', color = 'black', alpha = 0.2)
    linem,=ax1.plot(freq_ref[1:], power_spectrum_Pm[1:], color = 'blue')
    linel,=ax1.plot(freq_ref[1:], power_spectrum_Pl[1:], color = 'green')
    
    ax1.tick_params(axis ='y') 
    ax1.set_yscale('log')        
    ax1.legend([linem, linel, linelna, linet], ['Full model', 'CLE', 'LNA', 'LNA theory'], fontsize = 8)
    
    ax1.set_xlim(0, 0.12)
    ax1.set_ylim(10**5, 3*10**9)
    
    
    plt.savefig(os.path.join(read_directory,'plots','lpowerSpectrum-lambda'+str(valueOfLambda)[0:4]+'-Omega'+ str(valueOfOmega)
                                                                                  +'-tau'+ str(valueOfTau)  +'.pdf'))  
    # # Plotting mean & STD
    
    val_lambda=pd.read_csv(os.path.join(read_directory,'data','lambda.csv'))['value'].values
    val_Omega=pd.read_csv(os.path.join(read_directory,'data','Omega.csv'))['value'].values
    val_tau=pd.read_csv(os.path.join(read_directory,'data','tau.csv'))['value'].values
    index=pd.read_csv(os.path.join(read_directory,'data','index.csv'))['value'].values
    
    arr_std_Mm=pd.read_csv(os.path.join(read_directory,'data','std-Mm.csv'))['value'].values
    arr_std_Pm=pd.read_csv(os.path.join(read_directory,'data','std-Pm.csv'))['value'].values
    arr_std_Ml=pd.read_csv(os.path.join(read_directory,'data','std-Ml.csv'))['value'].values
    arr_std_Pl=pd.read_csv(os.path.join(read_directory,'data','std-Pl.csv'))['value'].values
    arr_std_Mlna=pd.read_csv(os.path.join(read_directory,'data','std-Mlna.csv'))['value'].values
    arr_std_Plna=pd.read_csv(os.path.join(read_directory,'data','std-Plna.csv'))['value'].values
    
    n_lambda=np.size(val_lambda)
    n_Omega=np.size(val_Omega)
    n_tau=np.size(val_tau)
    [j,k] = index
    
    arr_std_th_Plna=np.zeros(n_lambda)
    
    for i in range(n_lambda):
        
        valueOfLambda = val_lambda[i]
        valueOfOmega = val_Omega[j]
        valueOfTau = val_tau[k]
        
        freq_th,Sm,Sp=utils.lna_power_spectrum(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                    h=h,
                                                    P_0=P_0,
                                                    lambda_s=valueOfLambda,
                                                    tau=valueOfTau,
                                                    Omega=valueOfOmega,
                                                    T=T,
                                                    delta_t=sampling_timestep)
        dfreq = freq_th[1]-freq_th[0]
        arr_std_th_Plna[i] = np.sqrt(sum(Sp[1:])*dfreq/np.pi)
    
    fig = plt.figure(figsize=(3.3,2.0), constrained_layout = True) 
    
    ax1 = plt.gca()
    
    ax1.set_xlabel('$\lambda$ $/\mathrm{min}^{-1}$') 
    ax1.set_ylabel('$\Sigma_P~/\mathrm{cu}$')
    
    straight_line = 3e3/np.sqrt(val_lambda)
    
    linelna,=ax1.plot(val_lambda, arr_std_Plna, color = 'red', alpha = 0.5, lw = 1.0)
    linet,=ax1.plot(val_lambda, arr_std_th_Plna,'--', color = 'black', alpha = 0.5, lw = 1.0)
    linem,=ax1.plot(val_lambda, arr_std_Pm, color = 'blue', lw = 1.0)
    linel,=ax1.plot(val_lambda, arr_std_Pl, color = 'green', lw =1.0)
    lines,=ax1.plot(val_lambda, straight_line, color = 'black', lw =1.0)
    
    ax1.tick_params(axis ='y')
    ax1.legend([linem, linel, linelna, linet,lines], ['Full model', 'CLE', 'LNA', 'LNA theory', '$\Sigma_p \propto \sqrt{\lambda}$'], fontsize = 8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')        
    
    #ax1.set_xlim(1,100)
    #ax1.set_ylim(1.5*10**3, 6*10**3)
    fig.text(0.01, 0.9, 'A', size=9, weight='bold')
    
    
    plt.savefig(os.path.join(read_directory ,'plots','std-plots-Omega' + str(val_Omega[j]) + '-tau' + str(val_tau[k]) + '.pdf'))
    
    arr_rel_Pl = abs(arr_std_Pm - arr_std_Pl)/arr_std_Pm
    arr_rel_Plna = abs(arr_std_Pm - arr_std_Plna)/arr_std_Pm
    arr_rel_th_Plna = abs(arr_std_Pm - arr_std_th_Plna)/arr_std_Pm
    
    # fig = plt.figure(figsize=(5*ratio,5), constrained_layout = True) 
    fig = plt.figure(figsize=(3.3,2.0), constrained_layout = True) 
    
    ax1 = plt.gca()
    
    # ax1.set_title('Hes1 concentration STD relative error against lambda',fontdict=font) 
    ax1.set_xlabel('$\lambda$') 
    ax1.set_xlabel('$\lambda$ $/\mathrm{min}^{-1}$') 
    ax1.set_ylabel('$r_\Sigma$')
    
    linelim,=ax1.plot([0.01,100], [0.05, 0.05], color = 'grey', alpha = 0.6, lw = 1.0)
    linelna,=ax1.plot(val_lambda, arr_rel_Plna, color = 'red', alpha = 0.5, lw = 1.0)
    linet,=ax1.plot(val_lambda, arr_rel_th_Plna, '--', color = 'black', alpha = 0.5, lw = 1.0)
    linel,=ax1.plot(val_lambda, arr_rel_Pl, color = 'green', lw =1.0)

    ax1.legend([linel, linelna, linet,linelim], ['CLE', 'LNA', 'LNA theory', '5% error limit'], loc='upper right', fontsize = 8)
    
    ax1.tick_params(axis ='y') 
    ax1.set_xscale('log')
    #ax1.set_yscale('log')        
    
    #ax1.set_xlim(0, 0.15)
    #ax1.set_ylim(5*10**(-4), 2)
    fig.text(0.01, 0.9, 'B', size=9, weight='bold')
    
    plt.savefig(os.path.join(read_directory,'plots','error-std-plots-Omega' + str(val_Omega[j]) + '-tau' + str(val_tau[k]) + '.pdf'))
    
def compute_power_spectrum_data_review():
    delta_t = 0.01
    val_lambda=[10**x for x in np.arange(-1,3.1,0.2)]
    val_Omega=[0.001, 0.01, 0.1, 1, 10, 100]
    val_tau=[33]
    
    n_lambda=np.size(val_lambda)
    n_Omega=np.size(val_Omega)
    n_tau=np.size(val_tau)
    
    read_directory = os.path.join(os.path.dirname(__file__),'paper_plots_mean-std-power-spectrum') 
    
    T = 10000
    spec= [alpha_m, alpha_p, mu_m, mu_p, h, P_0, T, delta_t]
    #read_directory = "paper plots 2024-4-27-0-42-48 mean-std-power-spectrum"
    pd.DataFrame(spec,columns=['value']).to_csv(os.path.join(read_directory,'data_review','spec.csv'))
    
    # # Computing power spectrum
    
    n_iter=100
    
    sampling_timestep = 10
    n_t=int(T/sampling_timestep)
    
    i=15
    j=-1
    k=0
    
    valueOfLambda = 100
    valueOfOmega = 100
    valueOfTau = 33
    # valueOfLambda = 0.1
    # valueOfOmega = 1
    # valueOfTau = 33
    
    print("Value of lambda:", valueOfLambda)
    print("Value of Omega:", valueOfOmega)
    print("Value of tau:", valueOfTau)
    
    t0 = time.time()
    
    output=reviewer.simulate_master_all(n_iter=n_iter,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                                    h=h,
                                                                    P_0=P_0,
                                                                    lambda_s=valueOfLambda,
                                                                    tau=valueOfTau,
                                                                    P_init=0,
                                                                    M_init=0,
                                                                    T=T,
                                                                    delta_t=delta_t,
                                                                    Omega=valueOfOmega,
                                                                    sampling_timestep = sampling_timestep)
    
    print("Computation time:", (time.time() - t0)//60, "min ", (time.time() - t0)%60, "s")
    
    
    t_ref=output["times"]
    
    freq_ref=output["frequencies"]
    
    
    std_Mm=output["std Mm"]
    
    std_Pm=output["std Pm"]
    
    mean_Mm=output["mean Mm"]
    
    mean_Pm=output["mean Pm"]
    
    power_spectrum_Mm=output["power spectrum Mm"]
              
    power_spectrum_Pm=output["power spectrum Pm"]
    
    output=reviewer.simulate_langevin_all(n_iter=n_iter,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                                    h=h,
                                                                    P_0=P_0,
                                                                    lambda_s=valueOfLambda,
                                                                    tau=valueOfTau,
                                                                    P_init=0,
                                                                    M_init=0,
                                                                    T=T,
                                                                    delta_t=delta_t,
                                                                    Omega=valueOfOmega,
                                                                    sampling_timestep = sampling_timestep)            
                
    t_ref=output["times"]
    
    freq_ref=output["frequencies"]
    
    
    std_Ml=output["std Ml"]
    
    std_Pl=output["std Pl"]
    
    mean_Ml=output["mean Ml"]
    
    mean_Pl=output["mean Pl"]
    
    power_spectrum_Ml=output["power spectrum Ml"]
              
    power_spectrum_Pl=output["power spectrum Pl"]
    
    output=reviewer.simulate_lna_all(n_iter=n_iter,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                                    h=h,
                                                                    P_0=P_0,
                                                                    lambda_s=valueOfLambda,
                                                                    tau=valueOfTau,
                                                                    P_init=0,
                                                                    M_init=0,
                                                                    T=T,
                                                                    delta_t=delta_t,
                                                                    Omega=valueOfOmega,
                                                                    sampling_timestep = sampling_timestep)            
                
    t_ref=output["times"]
    
    freq_ref=output["frequencies"]
    
    
    std_Mlna=output["std Mlna"]
    
    std_Plna=output["std Plna"]
    
    mean_Mlna=output["mean Mlna"]
    
    mean_Plna=output["mean Plna"]
    
    power_spectrum_Mlna=output["power spectrum Mlna"]
              
    power_spectrum_Plna=output["power spectrum Plna"]
    
    paramPoint = [valueOfLambda, valueOfOmega, valueOfTau, sampling_timestep]
    
    pd.DataFrame(paramPoint,columns=['value']).to_csv(os.path.join(read_directory,'data_review','paramPoint-spectrum.csv'))
    pd.DataFrame(t_ref,columns=['value']).to_csv(os.path.join(read_directory,'data_review','t-spectrum.csv'))
    pd.DataFrame(freq_ref,columns=['value']).to_csv(os.path.join(read_directory,'data_review','freq-spectrum.csv'))
    
    pd.DataFrame(power_spectrum_Mm,columns=['value']).to_csv(os.path.join(read_directory,'data_review','power-spectrum-Mm.csv'))
    pd.DataFrame(power_spectrum_Pm,columns=['value']).to_csv(os.path.join(read_directory,'data_review','power-spectrum-Pm.csv'))
    pd.DataFrame(power_spectrum_Ml,columns=['value']).to_csv(os.path.join(read_directory,'data_review','power-spectrum-Ml.csv'))
    pd.DataFrame(power_spectrum_Pl,columns=['value']).to_csv(os.path.join(read_directory,'data_review','power-spectrum-Pl.csv'))
    pd.DataFrame(power_spectrum_Mlna,columns=['value']).to_csv(os.path.join(read_directory,'data_review','power-spectrum-Mlna.csv'))
    pd.DataFrame(power_spectrum_Plna,columns=['value']).to_csv(os.path.join(read_directory,'data_review','power-spectrum-Plna.csv'))
    
    # # Compute mean & STD
    
    n_iter_master = 2
    n_iter_sde = 2
    
    T = 100000
    sampling_timestep = 1
    n_t=int(T/sampling_timestep)
    
    arr_mean_Mm=np.zeros(n_lambda)
    arr_mean_Pm=np.zeros(n_lambda)
    
    arr_mean_Ml=np.zeros(n_lambda)
    arr_mean_Pl=np.zeros(n_lambda)
    
    arr_mean_Mlna=np.zeros(n_lambda)
    arr_mean_Plna=np.zeros(n_lambda)
    
    arr_std_Mm=np.zeros(n_lambda)
    arr_std_Pm=np.zeros(n_lambda)
    
    arr_std_Ml=np.zeros(n_lambda)
    arr_std_Pl=np.zeros(n_lambda)
    
    arr_std_Mlna=np.zeros(n_lambda)
    arr_std_Plna=np.zeros(n_lambda)
    
    arr_momFour_Mm=np.zeros(n_lambda)
    arr_momFour_Pm=np.zeros(n_lambda)
    
    arr_momFour_Ml=np.zeros(n_lambda)
    arr_momFour_Pl=np.zeros(n_lambda)
    
    arr_momFour_Mlna=np.zeros(n_lambda)
    arr_momFour_Plna=np.zeros(n_lambda)
    
    j=-1
    k=0
    index_OmegaTau = [j,k]
    
    valueOfOmega = 100
    valueOfTau = 33
    pd.DataFrame(index_OmegaTau,columns=['value']).to_csv(os.path.join(read_directory,'data_review','index.csv'))
    
    for i in range(n_lambda):
        valueOfLambda = val_lambda[i]
        
        t0 = time.time()
    
        print("Value of lambda:", valueOfLambda)
        print("Value of Omega:", valueOfOmega)
        print("Value of tau:", valueOfTau)
        
        if valueOfLambda <= 0.1:
            n_iter_master = 2
            n_iter_sde = 2
        else:
            n_iter_master = 10
            n_iter_sde = 10
            
        output=reviewer.simulate_master_meanAndStd(n_iter=n_iter_master, alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                                    h=h,
                                                                    P_0=P_0,
                                                                    lambda_s=valueOfLambda,
                                                                    tau=valueOfTau,
                                                                    P_init=0,
                                                                    M_init=0,
                                                                    T=T,
                                                                    delta_t=delta_t,
                                                                    Omega=valueOfOmega,
                                                                    sampling_timestep = sampling_timestep)
    
        arr_mean_Mm[i]=output["mean Mm"]
        arr_mean_Pm[i]=output["mean Pm"]
        arr_std_Mm[i]=output["std Mm"]
        arr_std_Pm[i]=output["std Pm"]
        arr_momFour_Mm[i]=output["momentumFour Mm"]
        arr_momFour_Pm[i]=output["momentumFour Pm"]
    
        output=reviewer.simulate_langevin_meanAndStd(n_iter=n_iter_sde,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                                    h=h,
                                                                    P_0=P_0,
                                                                    lambda_s=valueOfLambda,
                                                                    tau=valueOfTau,
                                                                    P_init=0,
                                                                    M_init=0,
                                                                    T=T,
                                                                    delta_t=delta_t,
                                                                    Omega=valueOfOmega,
                                                                    sampling_timestep = sampling_timestep)            
        arr_mean_Ml[i]=output["mean Ml"]
        arr_mean_Pl[i]=output["mean Pl"]
        arr_std_Ml[i]=output["std Ml"]
        arr_std_Pl[i]=output["std Pl"]
        arr_momFour_Ml[i]=output["momentumFour Ml"]
        arr_momFour_Pl[i]=output["momentumFour Pl"]
    
        
        output=reviewer.simulate_lna_meanAndStd(n_iter=n_iter_sde,alpha_m=alpha_m,alpha_p=alpha_p,mu_m=mu_m,mu_p=mu_p,
                                                                    h=h,
                                                                    P_0=P_0,
                                                                    lambda_s=valueOfLambda,
                                                                    tau=valueOfTau,
                                                                    P_init=0,
                                                                    M_init=0,
                                                                    T=T,
                                                                    delta_t=delta_t,
                                                                    Omega=valueOfOmega,
                                                                    sampling_timestep = sampling_timestep)            
        arr_mean_Mlna[i]=output["mean Mlna"]
        arr_mean_Plna[i]=output["mean Plna"]
        arr_std_Mlna[i]=output["std Mlna"]
        arr_std_Plna[i]=output["std Plna"]
        arr_momFour_Mlna[i]=output["momentumFour Mlna"]
        arr_momFour_Plna[i]=output["momentumFour Plna"]
        
        print("\n")
        print("Number of iterations: ", n_iter_master, n_iter_sde)
        print("Computation time: ", (time.time() - t0)//3600, "h ", ((time.time() - t0)//60)%60, "min ", (time.time() - t0)%60, "s")
        print("\n")
    
    
    pd.DataFrame(val_lambda,columns=['value']).to_csv(os.path.join(read_directory,'data_review','lambda.csv'))
    pd.DataFrame(val_Omega,columns=['value']).to_csv(os.path.join(read_directory,'data_review','Omega.csv'))
    pd.DataFrame(val_tau,columns=['value']).to_csv(os.path.join(read_directory,'data_review','tau.csv'))
    pd.DataFrame(index_OmegaTau,columns=['value']).to_csv(os.path.join(read_directory,'data_review','index.csv'))
    
    pd.DataFrame(arr_std_Mm,columns=['value']).to_csv(os.path.join(read_directory,'data_review','std-Mm.csv'))
    pd.DataFrame(arr_std_Pm,columns=['value']).to_csv(os.path.join(read_directory,'data_review','std-Pm.csv'))
    pd.DataFrame(arr_std_Ml,columns=['value']).to_csv(os.path.join(read_directory,'data_review','std-Ml.csv'))
    pd.DataFrame(arr_std_Pl,columns=['value']).to_csv(os.path.join(read_directory,'data_review','std-Pl.csv'))
    pd.DataFrame(arr_std_Mlna,columns=['value']).to_csv(os.path.join(read_directory,'data_review','std-Mlna.csv'))
    pd.DataFrame(arr_std_Plna,columns=['value']).to_csv(os.path.join(read_directory,'data_review','std-Plna.csv'))

def plot_power_spectrum_data_review():
    font = {'size'   : 8,
            'sans-serif' : 'Arial'}
    plt.rc('font', **font)
    
    read_directory = os.path.join(os.path.dirname(__file__),'paper_plots_mean-std-power-spectrum') 
    # read_directory = os.path.join(os.getcwd(),'paper_plots_mean-std-power-spectrum') 
    # write_directory = 'power_spectra_plots'
    
    spec = pd.read_csv(os.path.join(read_directory,'data_review','spec.csv'))['value'].values
    paramPoint=pd.read_csv(os.path.join(read_directory,'data_review','paramPoint-spectrum.csv'))['value'].values
    t_ref=pd.read_csv(os.path.join(read_directory,'data_review','t-spectrum.csv'))['value'].values
    freq_ref=pd.read_csv(os.path.join(read_directory,'data_review','freq-spectrum.csv'))['value'].values
    
    power_spectrum_Mm=pd.read_csv(os.path.join(read_directory,'data_review','power-spectrum-Mm.csv'))['value'].values
    power_spectrum_Pm=pd.read_csv(os.path.join(read_directory,'data_review','power-spectrum-Pm.csv'))['value'].values
    power_spectrum_Ml=pd.read_csv(os.path.join(read_directory,'data_review','power-spectrum-Ml.csv'))['value'].values
    power_spectrum_Pl=pd.read_csv(os.path.join(read_directory,'data_review','power-spectrum-Pl.csv'))['value'].values
    power_spectrum_Mlna=pd.read_csv(os.path.join(read_directory,'data_review','power-spectrum-Mlna.csv'))['value'].values
    power_spectrum_Plna=pd.read_csv(os.path.join(read_directory,'data_review','power-spectrum-Plna.csv'))['value'].values
    
    [alpha_m, alpha_p, mu_m, mu_p, h, P_0, T, delta_t] = spec
    [valueOfLambda, valueOfOmega, valueOfTau, sampling_timestep] = paramPoint
    
    fig = plt.figure(figsize=(3.3,2.0), constrained_layout = True) 
    
    ax1 = plt.gca()
    
    # ax1.set_title('Hes1 Power spectrum',fontdict=font) 
    # ax1.set_xlabel('Angular frequency') 
    ax1.set_xlabel('$\omega$ $/\mathrm{min}^{-1}$') 
    # ax1.set_ylabel('Power') 
    ax1.set_ylabel('$S_p(\omega)~/(\mathrm{cu}^2\mathrm{min})$') 
    
    freq_th,Sm,Sp=reviewer.lna_power_spectrum(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                    h=h,
                                                    P_0=P_0,
                                                    lambda_s=valueOfLambda,
                                                    tau=valueOfTau,
                                                    Omega=valueOfOmega,
                                                    T=T,
                                                    delta_t=sampling_timestep)
    
    fig.text(0.02, 0.9, 'B', size=9, weight='bold')
    linelna,=ax1.plot(freq_ref[1:], power_spectrum_Plna[1:], color = 'red', alpha = 0.2)
    linet,=ax1.plot(freq_th[1:], Sp[1:], '--', color = 'black', alpha = 0.2)
    linem,=ax1.plot(freq_ref[1:], power_spectrum_Pm[1:], color = 'blue')
    linel,=ax1.plot(freq_ref[1:], power_spectrum_Pl[1:], color = 'green')
    
    ax1.tick_params(axis ='y') 
    ax1.set_yscale('log')        
    ax1.legend([linem, linel, linelna, linet], ['Full model', 'CLE', 'LNA', 'LNA theory'], fontsize = 8)
    
    ax1.set_xlim(0, 0.12)
    # ax1.set_ylim(10**5, 3*10**9)
    
    
    plt.savefig(os.path.join(read_directory,'plots_review','lpowerSpectrum-lambda'+str(valueOfLambda)[0:4]+'-Omega'+ str(valueOfOmega)
                                                                                  +'-tau'+ str(valueOfTau)  +'.pdf'))  
    
    # # Plotting mean & STD
    
    val_lambda=pd.read_csv(os.path.join(read_directory,'data_review','lambda.csv'))['value'].values
    val_Omega=pd.read_csv(os.path.join(read_directory,'data_review','Omega.csv'))['value'].values
    val_tau=pd.read_csv(os.path.join(read_directory,'data_review','tau.csv'))['value'].values
    index=pd.read_csv(os.path.join(read_directory,'data_review','index.csv'))['value'].values
    
    arr_std_Mm=pd.read_csv(os.path.join(read_directory,'data_review','std-Mm.csv'))['value'].values
    arr_std_Pm=pd.read_csv(os.path.join(read_directory,'data_review','std-Pm.csv'))['value'].values
    arr_std_Ml=pd.read_csv(os.path.join(read_directory,'data_review','std-Ml.csv'))['value'].values
    arr_std_Pl=pd.read_csv(os.path.join(read_directory,'data_review','std-Pl.csv'))['value'].values
    arr_std_Mlna=pd.read_csv(os.path.join(read_directory,'data_review','std-Mlna.csv'))['value'].values
    arr_std_Plna=pd.read_csv(os.path.join(read_directory,'data_review','std-Plna.csv'))['value'].values
    
    n_lambda=np.size(val_lambda)
    n_Omega=np.size(val_Omega)
    n_tau=np.size(val_tau)
    [j,k] = index
    
    arr_std_th_Plna=np.zeros(n_lambda)
    
    for i in range(n_lambda):
        
        valueOfLambda = val_lambda[i]
        valueOfOmega = val_Omega[j]
        valueOfTau = val_tau[k]
        
        freq_th,Sm,Sp=reviewer.lna_power_spectrum(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
                                                    h=h,
                                                    P_0=P_0,
                                                    lambda_s=valueOfLambda,
                                                    tau=valueOfTau,
                                                    Omega=valueOfOmega,
                                                    T=T,
                                                    delta_t=sampling_timestep)
        dfreq = freq_th[1]-freq_th[0]
        arr_std_th_Plna[i] = np.sqrt(sum(Sp[1:])*dfreq/np.pi)
    
    fig = plt.figure(figsize=(3.3,2.0), constrained_layout = True) 
    
    ax1 = plt.gca()
    
    ax1.set_xlabel('$\lambda$ $/\mathrm{min}^{-1}$') 
    ax1.set_ylabel('$\Sigma_P$')
    
    straight_line = 3e3/np.sqrt(val_lambda)
    
    linelna,=ax1.plot(val_lambda, arr_std_Plna, color = 'red', alpha = 0.5, lw = 1.0)
    linet,=ax1.plot(val_lambda, arr_std_th_Plna,'--', color = 'black', alpha = 0.5, lw = 1.0)
    linem,=ax1.plot(val_lambda, arr_std_Pm, color = 'blue', lw = 1.0)
    linel,=ax1.plot(val_lambda, arr_std_Pl, color = 'green', lw =1.0)
    lines,=ax1.plot(val_lambda, straight_line, color = 'black', lw =1.0)
    
    ax1.tick_params(axis ='y')
    ax1.legend([linem, linel, linelna, linet,lines], ['Full model', 'CLE', 'LNA', 'LNA theory', '$\Sigma_p \propto \sqrt{\lambda}$'], fontsize = 8)
    ax1.set_xscale('log')
    ax1.set_yscale('log')        
    
    #ax1.set_xlim(1,100)
    #ax1.set_ylim(1.5*10**3, 6*10**3)
    fig.text(0.01, 0.9, 'A', size=9, weight='bold')
    
    
    plt.savefig(os.path.join(read_directory ,'plots_review','std-plots-Omega' + str(val_Omega[j]) + '-tau' + str(val_tau[k]) + '.pdf'))
    
    arr_rel_Pl = abs(arr_std_Pm - arr_std_Pl)/arr_std_Pm
    arr_rel_Plna = abs(arr_std_Pm - arr_std_Plna)/arr_std_Pm
    arr_rel_th_Plna = abs(arr_std_Pm - arr_std_th_Plna)/arr_std_Pm
    
    # fig = plt.figure(figsize=(5*ratio,5), constrained_layout = True) 
    fig = plt.figure(figsize=(3.3,2.0), constrained_layout = True) 
    
    ax1 = plt.gca()
    
    # ax1.set_title('Hes1 concentration STD relative error against lambda',fontdict=font) 
    ax1.set_xlabel('$\lambda$') 
    ax1.set_xlabel('$\lambda$ $/\mathrm{min}^{-1}$') 
    ax1.set_ylabel('$r_\Sigma$')
    
    linelim,=ax1.plot([0.01,100], [0.05, 0.05], color = 'grey', alpha = 0.6, lw = 1.0)
    linelna,=ax1.plot(val_lambda, arr_rel_Plna, color = 'red', alpha = 0.5, lw = 1.0)
    linet,=ax1.plot(val_lambda, arr_rel_th_Plna, '--', color = 'black', alpha = 0.5, lw = 1.0)
    linel,=ax1.plot(val_lambda, arr_rel_Pl, color = 'green', lw =1.0)

    ax1.legend([linel, linelna, linet,linelim], ['CLE', 'LNA', 'LNA theory', '5% error limit'], loc='upper right', fontsize = 8)
    
    ax1.tick_params(axis ='y') 
    ax1.set_xscale('log')
    #ax1.set_yscale('log')        
    
    #ax1.set_xlim(0, 0.15)
    #ax1.set_ylim(5*10**(-4), 2)
    fig.text(0.01, 0.9, 'B', size=9, weight='bold')
    
    plt.savefig(os.path.join(read_directory,'plots_review','error-std-plots-Omega' + str(val_Omega[j]) + '-tau' + str(val_tau[k]) + '.pdf'))

def plot_toggle_switch_gillespie():
    read_directory = os.path.join(os.path.dirname(__file__),'paper_plots_mean-std-power-spectrum') 
    trajectory = toggle_switch.generate_stochastic_trajectory( duration = 10000,
                                    repression_threshold = 3,
                                    hill_coefficient = 2,
                                    degradation_rate = 0.1,
                                    basal_transcription_rate = 1,
                                    switching_rate = 1,
                                    system_size = 100,
                                    initial_a = 1,
                                    initial_b = 1,
                                    equilibration_time = 0.0,
                                    sampling_timestep = 1.0)
 
    plt.figure(figsize=(3.3,2.0), constrained_layout = True) 
    plt.plot(trajectory[:,0], trajectory[:,1], label = 'A')
    plt.plot(trajectory[:,0], trajectory[:,2], label = 'B')
    plt.xlabel('Time (a.u)')
    plt.ylabel('Concentration (a.u)')
    plt.ylim(0,11)
    plt.legend()
    plt.savefig(os.path.join(read_directory,'plots_review','toggle_switch_start.pdf'))

def plot_toggle_switch_ODE():
    read_directory = os.path.join(os.path.dirname(__file__),'paper_plots_mean-std-power-spectrum') 
    # Define the parameters
    a_m = 1  # Maximum synthesis rate
    p_0 = 1   # Threshold parameter
    n = 2     # Hill coefficient
    mu = 0.1    # Degradation rate
    
    # Define the system of ODEs
    def toggle_switch(t, y):
        a, b = y
        da_dt = a_m / (1 + (b / p_0)**n) - mu * a
        db_dt = a_m / (1 + (a / p_0)**n) - mu * b
        return [da_dt, db_dt]
    
    # Initial conditions
    a0 = 1.1
    # a0 = 0.1
    b0 = 0.9    # b0 = 10
    initial_conditions = [a0, b0]
    
    # Time span for the simulation
    t_span = (0, 1000)  # From 0 to 50 time units
    t_eval = np.linspace(t_span[0], t_span[1], 500)  # Points at which to store the solution
    
    # Solve the ODEs
    solution = scipy.integrate.solve_ivp(toggle_switch, t_span, initial_conditions, t_eval=t_eval)
    
    # Extract the solution
    t = solution.t
    a = solution.y[0]
    b = solution.y[1]
    
    # Plot the results
    plt.figure(figsize=(3.3,2.0), constrained_layout = True) 
    plt.plot(t, a, label='a', linewidth=2)
    plt.plot(t, b, label='b', ls = '--', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(read_directory,'plots_review','toggle_switch_deterministic.pdf'))

def plot_toggle_switch_CLE():
    read_directory = os.path.join(os.path.dirname(__file__),'paper_plots_mean-std-power-spectrum') 
    trajectory = toggle_switch.generate_langevin_trajectory( duration = 10000,
                                    repression_threshold = 3,
                                    hill_coefficient = 2,
                                    degradation_rate = 0.1,
                                    basal_transcription_rate = 1,
                                    switching_rate = 1,
                                    system_size = 100,
                                    initial_a = 1,
                                    initial_b = 1,
                                    equilibration_time = 0.0,
                                    delta_t = 0.001)
                                    # sampling_timestep = 1)
    window_size = 100
    time_average_a = np.convolve(trajectory[:,1], np.ones(window_size) / window_size, mode='same')
 
    plt.figure(figsize=(3.3,2.0), constrained_layout = True) 
    plt.plot(trajectory[:,0], trajectory[:,1], label = 'A')
    plt.plot(trajectory[:,0], trajectory[:,2], label = 'B')
    plt.plot(trajectory[:,0], time_average_a, label = 'slide')
    plt.xlabel('Time (a.u)')
    plt.ylim(0,11)
    plt.ylabel('Concentration (a.u)')
    plt.legend()
    plt.savefig(os.path.join(read_directory,'plots_review','toggle_switch_CLE.pdf'))

def get_waiting_times_in_top_state(trajectory):
    
    # Assume `data` is your array with columns: time, A, B
    # data = np.array([...])  # Replace with your actual data

    # Identify the times when the state (A on, B off) is satisfied
    window_size = 100
    time_average_a = np.convolve(trajectory[:,1], np.ones(window_size) / window_size, mode='same')
    time_average_b = np.convolve(trajectory[:,2], np.ones(window_size) / window_size, mode='same')
    state_active = np.logical_and(time_average_a > 4 , time_average_b < 4)

    # Find the indices where the state changes
    state_changes = np.diff(state_active.astype(int))
    state_start_indices = np.where(state_changes == 1)[0]
    state_end_indices = np.where(state_changes == -1)[0]

    # Handle the case where the state is active at the end of the simulation
    if state_active[-1]:
        state_end_indices = np.append(state_end_indices, len(trajectory) - 1)
    if state_active[0]:
        state_start_indices = np.insert(state_start_indices, 0, 0, axis=0)

    # Calculate waiting times as differences between start and end times
    waiting_times = trajectory[state_end_indices, 0] - trajectory[state_start_indices, 0]
    return waiting_times

def compare_langevin_gillespie_toggle():
    # simulate a long langevin trajectory
    # simulate a long gillespie trajectory
    # make an figure panel of A stationary distribution in both
    # make a figure panel of top waiting time distributions in both
    langevin_trajectory = toggle_switch.generate_langevin_trajectory( duration = 100000000,
                                    repression_threshold = 3,
                                    hill_coefficient = 2,
                                    degradation_rate = 0.1,
                                    basal_transcription_rate = 1,
                                    switching_rate = 1,
                                    system_size = 100,
                                    initial_a = 1,
                                    initial_b = 1,
                                    equilibration_time = 0.0,
                                    delta_t = 0.001,
                                    sampling_timestep = 10)
                                    # sampling_timestep = 1)

    gillespie_trajectory = toggle_switch.generate_stochastic_trajectory( duration = 100000000,
                                    repression_threshold = 3,
                                    hill_coefficient = 2,
                                    degradation_rate = 0.1,
                                    basal_transcription_rate = 1,
                                    switching_rate = 1,
                                    system_size = 100,
                                    initial_a = 1,
                                    initial_b = 1,
                                    equilibration_time = 0.0,
                                    sampling_timestep = 10)
                                    # sampling_timestep = 1)
    
    gillespie_waiting_times = get_waiting_times_in_top_state(gillespie_trajectory)
    langevin_waiting_times = get_waiting_times_in_top_state(langevin_trajectory)
    print('number of gillespie waiting times is')
    print(len(gillespie_waiting_times))
    print('number of langevin waiting times is')
    print(len(langevin_waiting_times))
    
    print('mean of A molecules gillespie')
    print(np.mean(gillespie_trajectory[:,1]))
    print('mean of A molecules Langevin')
    print(np.mean(langevin_trajectory[:,1]))
    print('relative difference')
    print(np.abs(np.mean(langevin_trajectory[:,1]) - np.mean(gillespie_trajectory[:,1]))/
          np.mean(gillespie_trajectory[:,1]))
    print('variance of A molecules gillespie')
    print(np.std(gillespie_trajectory[:,1]))
    print('variance of A molecules Langevin')
    print(np.std(langevin_trajectory[:,1]))
    print(np.abs(np.std(langevin_trajectory[:,1]) - np.std(gillespie_trajectory[:,1]))/
          np.std(gillespie_trajectory[:,1]))
 
    print('mean of waiting times gillespie')
    print(np.mean(gillespie_waiting_times))
    print('mean of waiting times Langevin')
    print(np.mean(langevin_waiting_times))
    print('relative difference')
    print(np.abs(np.mean(langevin_waiting_times) - np.mean(gillespie_waiting_times))/
          np.mean(gillespie_waiting_times))

    print('std of waiting times gillespie')
    print(np.std(gillespie_waiting_times))
    print('std of waiting times Langevin')
    print(np.std(langevin_waiting_times))
    print('relative difference')
    print(np.abs(np.std(langevin_waiting_times) - np.std(gillespie_waiting_times))/
          np.std(gillespie_waiting_times))


    fig = plt.figure(figsize=(6.6,2.0), constrained_layout = True) 
    plt.subplot(121)
    plt.hist(langevin_trajectory[:,1],alpha =0.5, label = 'Langevin', density = True, bins = 20)
    plt.hist(gillespie_trajectory[:,1],alpha =0.5, label = 'Gillespie', density = True, bins = 20)
    plt.xlabel('Concentration /cu')
    plt.ylabel('Probability density /$\mathrm{cu}^{-1}$')
    plt.subplot(122)
    plt.hist(langevin_waiting_times, alpha =0.5, label = 'Langevin', bins = 20, density = True)
    plt.hist(gillespie_waiting_times, alpha =0.5, label = 'Gillespie', bins = 20, density = True)
    # plt.hist(langevin_waiting_times, range = (0,1000), alpha =0.5, label = 'Langevin')
    # plt.hist(gillespie_waiting_times,  range = (0,1000), alpha =0.5, label = 'Gillespie')
    # plt.xlim(0,1000)
    plt.xlabel('Waiting time /min')
    plt.ylabel('Probability density /$\mathrm{min}^{-1}$')
    plt.legend()
    fig.text(0.02,0.95,'A',size=9,weight='bold')
    fig.text(0.52,0.95,'B',size=9,weight='bold')
    read_directory = os.path.join(os.path.dirname(__file__),'paper_plots_mean-std-power-spectrum') 
    plt.savefig(os.path.join(read_directory,'plots_review','toggle_comparison.pdf'))

def illustrate_switching_effect():
    read_directory = os.path.join(os.path.dirname(__file__),'paper_plots_mean-std-power-spectrum') 
    trajectory_no_toggle_langevin = toggle_switch.generate_langevin_trajectory( duration = 10000,
                                    repression_threshold = 3,
                                    hill_coefficient = 2,
                                    degradation_rate = 0.1,
                                    basal_transcription_rate = 1,
                                    switching_rate = 100,
                                    system_size = 100,
                                    initial_a = 1,
                                    initial_b = 1,
                                    equilibration_time = 0.0,
                                    delta_t = 0.001)
                                    # sampling_timestep = 1)

    trajectory_no_toggle_gillespie = toggle_switch.generate_stochastic_trajectory( duration = 10000,
                                    repression_threshold = 3,
                                    hill_coefficient = 2,
                                    degradation_rate = 0.1,
                                    basal_transcription_rate = 1,
                                    switching_rate = 100,
                                    system_size = 100,
                                    initial_a = 1,
                                    initial_b = 1,
                                    equilibration_time = 0.0
                                    )
                                    # sampling_timestep = 1)
 
    trajectory_w_toggle_langevin = toggle_switch.generate_langevin_trajectory( duration = 10000,
                                    repression_threshold = 3,
                                    hill_coefficient = 2,
                                    degradation_rate = 0.1,
                                    basal_transcription_rate = 1,
                                    switching_rate = 1,
                                    system_size = 100,
                                    initial_a = 1,
                                    initial_b = 1,
                                    equilibration_time = 0.0,
                                    delta_t = 0.001)
                                    # sampling_timestep = 1)

    trajectory_w_toggle_gillespie = toggle_switch.generate_stochastic_trajectory( duration = 10000,
                                    repression_threshold = 3,
                                    hill_coefficient = 2,
                                    degradation_rate = 0.1,
                                    basal_transcription_rate = 1,
                                    switching_rate = 1,
                                    system_size = 100,
                                    initial_a = 1,
                                    initial_b = 1,
                                    equilibration_time = 0.0)
                                    # sampling_timestep = 1)
 
    fig = plt.figure(figsize=(6.6,4.0), constrained_layout = True) 
    plt.subplot(221)
    plt.plot(trajectory_no_toggle_gillespie[:,0], trajectory_no_toggle_gillespie[:,1], label = 'A')
    plt.plot(trajectory_no_toggle_gillespie[:,0], trajectory_no_toggle_gillespie[:,2], label = 'B')
    plt.xlabel('$t$ /min')
    plt.ylabel('Concentration /cu')
    plt.title('$\lambda = 100$, Gillespie', fontsize = 8)
    plt.ylim(0,11)
    plt.subplot(222)
    plt.plot(trajectory_w_toggle_gillespie[:,0], trajectory_w_toggle_gillespie[:,1], label = 'A')
    plt.plot(trajectory_w_toggle_gillespie[:,0], trajectory_w_toggle_gillespie[:,2], label = 'B')
    plt.xlabel('$t$ /min')
    plt.ylabel('Concentration /cu')
    plt.title('$\lambda = 1$, Gillespie', fontsize = 8)
    plt.ylim(0,11)
    plt.subplot(223)
    plt.plot(trajectory_no_toggle_langevin[:,0], trajectory_no_toggle_langevin[:,1], label = 'A')
    plt.plot(trajectory_no_toggle_langevin[:,0], trajectory_no_toggle_langevin[:,2], label = 'B')
    plt.xlabel('$t$ /min')
    plt.ylabel('Concentration /cu')
    plt.title('$\lambda = 100$, Langevin', fontsize = 8)
    plt.ylim(0,11)
    plt.subplot(224)
    plt.plot(trajectory_w_toggle_langevin[:,0], trajectory_w_toggle_langevin[:,1], label = 'A')
    plt.plot(trajectory_w_toggle_langevin[:,0], trajectory_w_toggle_langevin[:,2], label = 'B')
    plt.xlabel('$t$ /min')
    plt.ylabel('Concentration /cu')
    plt.title('$\lambda = 1$, Langevin', fontsize = 8)
    plt.ylim(0,11)
    plt.legend()
    fig.text(0.02,0.95,'A',size=9,weight='bold')
    fig.text(0.52,0.95,'B',size=9,weight='bold')
    fig.text(0.02,0.45,'C',size=9,weight='bold')
    fig.text(0.52,0.45,'D',size=9,weight='bold')
    plt.savefig(os.path.join(read_directory,'plots_review','toggle_illustration.pdf'))

if __name__ == "__main__":
    ## Figure 2 of the revised manuscript
    # make_noise_comparison_figure()

    ## Figure 3A of the revised manuscript
    # make_approximation_comparison_figure()

    ## Generate data for figure 3B and figure 4 of the revised manuscript
    # compute_power_spectrum_data()

    ## Figure 3B and figure 4 of the revised manuscript
    # plot_power_spectrum_data()
    
    ## Figure 5 can be generated using the scripts 
    ## generate_data_for_heatmap_plot.py and
    ## plot_heatmap_from_data.py

    ## Figure 7 of the revised manuscript
    # illustrate_switching_effect()

    ## Figure 8 of the revised manuscript
    compare_langevin_gillespie_toggle()

    ## Figure SF1 of the revised manuscript
    # make_noise_comparison_figure_reviewer_switching()
    
    ## Figure SF2A of the revised manuscript
    # make_approximation_comparison_figure_reviewer_switching()

    ## generate data for figure SF2B
    # compute_power_spectrum_data_review()
    
    ## Figure SF2B of the revised manuscript
    # plot_power_spectrum_data_review()

    ## old and redundant figures
    # plot_toggle_switch_gillespie()
    # plot_toggle_switch_ODE()
    # plot_toggle_switch_CLE()
    