import numpy as np
import hes1_master_Antoine as master
import scipy.interpolate as spinter
import hes1_langevin_Antoine as langevin
import hes1_utils_Antoine as utils




def simulate_master_all(n_iter=100, alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,
                                                      lambda_s=1,        
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=1000,
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
    
    Mm=spinter.interp1d(t,Mm,kind="zero")(t_ref)
    Pm=spinter.interp1d(t,Pm,kind="zero")(t_ref)

    n_stat=len(t_ref)//2
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
        Mm=spinter.interp1d(t,Mm,kind="zero")(t_ref)
        Pm=spinter.interp1d(t,Pm,kind="zero")(t_ref)

        mean_Mm[i] = np.mean(Mm)
        mean_Pm[i] = np.mean(Pm)
        var_Mm[i] = np.var(Mm)
        var_Pm[i] = np.var(Pm)

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
           "times":t[n_stat:], "frequencies": freq}
    
    return output




def simulate_langevin_all(n_iter=100, alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,
                                                      lambda_s=1,        
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=1000,
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

    n_stat=len(t)//2
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

        mean_Ml[i] = np.mean(Ml)
        mean_Pl[i] = np.mean(Pl)
        var_Ml[i] = np.var(Ml)
        var_Pl[i] = np.var(Pl)

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





def simulate_lna_all(n_iter=100, alpha_m=1, alpha_p=1, mu_m=0.03, mu_p=0.03,
                                                      lambda_s=1,        
                                                      P_0=100,
                                                      h=4.1,
                                                      tau=0.1,
                                                      P_init=0,
                                                      M_init=0,
                                                      T=1000,
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

    n_stat=len(t)//2
    freq_ref, test_power_spectrum = utils.compute_power_spectrum_traj(t[n_stat:],Plna[n_stat:])
    power_spectrum_M=np.zeros_like(test_power_spectrum)
    power_spectrum_P=np.zeros_like(test_power_spectrum)
    
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

        mean_Mlna[i] = np.mean(Mlna)
        mean_Plna[i] = np.mean(Plna)
        var_Mlna[i] = np.var(Mlna)
        var_Plna[i] = np.var(Plna)

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

