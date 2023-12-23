import numpy as np
import hes1_master_Antoine as master
import scipy.interpolate as spinter
import hes1_langevin_Antoine as langevin
import hes1_utils_Antoine as utils
from numba import jit



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
        t,Mm,Pm=t,Mm,Pm=master.one_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p, lambda_s=lambda_s,
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



@jit
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
        
    t,Mm,Pm=master.one_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p, lambda_s=lambda_s,
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
    
    Mm=interpolator(t,t_ref,Mm)
    Pm=interpolator(t,t_ref,Pm)

    n_stat=int(2000/sampling_timestep)
    freq_ref, test_power_spectrum = utils.compute_power_spectrum_traj(t_ref[n_stat:],Pm[n_stat:])
    power_spectrum_Mm=np.zeros_like(test_power_spectrum)
    power_spectrum_Pm=np.zeros_like(test_power_spectrum)
    
    for i in range(n_iter):
        t,Mm,Pm=t,Mm,Pm=master.one_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p, lambda_s=lambda_s,
                                                              P_0=P_0,
                                                              h=h,
                                                              tau=tau,
                                                              P_init=P_init,
                                                              M_init=M_init,
                                                              sigma_init=1,
                                                              Omega=Omega,
                                                              T=T)
        Mm=interpolator(t,t_ref,Mm)
        Pm=interpolator(t,t_ref,Pm)

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
        t,Ml,Pl=langevin.one_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
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


@jit
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
    
        
    t,Ml,Pl=langevin.one_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
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
        t,Ml,Pl=langevin.one_trajectory(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
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


@jit
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
        t,Mlna,Plna=langevin.one_trajectory_LNA(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
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


@jit
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
        
    t,Mlna,Plna=langevin.one_trajectory_LNA(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
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
        t,Mlna,Plna=langevin.one_trajectory_LNA(alpha_m=alpha_m, alpha_p=alpha_p, mu_m=mu_m, mu_p=mu_p,
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

