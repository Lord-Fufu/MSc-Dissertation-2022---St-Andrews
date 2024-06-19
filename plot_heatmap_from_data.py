import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ratio=(1+np.sqrt(5))/2


# font = {
    # 'family' : 'Arial',
    # 'color'  : 'black',
    # 'weight' : 'normal',
    # 'size'   : 9,
# }
font = {'size'   : 8,
        'sans-serif' : 'Arial'}
plt.rc('font', **font)

param_number = 2

k=0

read_directory = 'paper_plots' #Name here the directory you want to use for the plots
read_data=os.path.join(read_directory,'data',str(param_number))

val_lambda=pd.read_csv(read_data + '_lambda.csv')['value'].values
val_Omega=pd.read_csv(read_data + '_Omega.csv')['value'].values
val_tau=pd.read_csv(read_data + '_tau.csv')['value'].values


arr_std_Mm=pd.read_csv(read_data + '_std-Mm.csv').values
arr_std_Pm=pd.read_csv(read_data + '_std-Pm.csv').values
arr_std_Ml=pd.read_csv(read_data + '_std-Ml.csv').values
arr_std_Pl=pd.read_csv(read_data + '_std-Pl.csv').values
# arr_std_Mlna=pd.read_csv(read_data + '_std-Mlna.csv')['value'].values
# arr_std_Plna=pd.read_csv(read_data + '_std-Plna.csv')['value'].values

arr_rel_Pl = abs(arr_std_Pm - arr_std_Pl)/arr_std_Pm

inv_val_lambda = [1/x for x in val_lambda]
inv_val_Omega  = [1/x for x in val_Omega]


x, y = np.meshgrid(val_Omega, val_lambda)

fig = plt.figure(figsize=(5*ratio,5), constrained_layout = True) 

ax1 = plt.gca()

ax1.set_title('Heatmap of Hes1 concentration STD relative error \n against lambda and Omega') 
ax1.set_xlabel('Omega') 
ax1.set_ylabel('lambda')

cont=ax1.contour(val_Omega, val_lambda, arr_rel_Pl, levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], color = 'red')
hmap=ax1.pcolormesh(x, y, arr_rel_Pl, cmap='RdBu', shading = 'gouraud')
ax1.clabel(cont, inline=False, fontsize=10)
#ax1.fill_between(val_lambda[10:], 11*[0.25], 11*[100], color = 'red', alpha = 0.1)

#linelim, = ax1.plot(val_lambda, err_arr_varAll*30, '--', color = 'purple', alpha = 0.1)
#linelim, = ax1.plot(val_lambda, err_arr_varAll*20, '--', color = 'purple', alpha = 0.2)
#linelim, = ax1.plot(val_lambda, err_arr_varAll*10, '--', color = 'purple', alpha = 0.3)

ax1.legend([cont], ['Langevin model STD error'], loc='upper right')

#ax1.fill_between(val_lambda, err_arr_varAll*30, color = 'purple', alpha = 0.1)
#ax1.fill_between(val_lambda, err_arr_varAll*20, color = 'purple', alpha = 0.2)
#ax1.fill_between(val_lambda, err_arr_varAll*10, color = 'purple', alpha = 0.3)

ax1.tick_params(axis ='y') 
ax1.set_xscale('log')
ax1.set_yscale('log')        

ax1.set_xlim(0.01, 100)
ax1.set_ylim(0.03, 100)

plt.savefig(os.path.join(read_directory,'plots','heatmap-error-plot- tau' + str(val_tau[k]) + '.pdf'))

fig = plt.figure(figsize=(5*ratio,5), constrained_layout = True) 

ax1 = plt.gca()

ax1.set_title('Heatmap of Hes1 concentration STD relative error against lambda and Omega') 
ax1.set_xlabel('Omega') 
ax1.set_ylabel('lambda')

cont=ax1.contour(inv_val_Omega, inv_val_lambda, arr_rel_Pl, levels = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], color = 'red')
# cont=ax1.contour(inv_val_Omega, inv_val_lambda, arr_rel_Pl, levels = [0.05], color = 'red')
ax1.clabel(cont, inline=False, fontsize=10)
# ax1.fill_between(val_lambda[:12], 12*[0.01], 12*[4], color = 'red', alpha = 0.1)
ax1.fill_between(val_lambda[:2], 2*[0.01], 2*[4], color = 'red', alpha = 0.1)

ax1.legend([cont], ['Langevin model STD error'], loc='upper right')

ax1.tick_params(axis ='y') 
ax1.set_xscale('log')
ax1.set_yscale('log')        

ax1.set_xlim(0.01, 150)
ax1.set_ylim(0.01, 40)

plt.savefig(os.path.join(read_directory,'plots','heatmap-error-plot- tau' + str(val_tau[k]) + '.pdf'))

fig = plt.figure(figsize=(3.3,2.0), constrained_layout = True) 

ax1 = plt.gca()

# ax1.set_title('Heatmap of Hes1 concentration STD relative error against lambda and Omega') 
ax1.set_xlabel('$\Omega$') 
ax1.set_ylabel('$\lambda$ $/\mathrm{min^{-1}}$')

x, y = np.meshgrid(val_Omega, val_lambda)

print('lambda values are')
print(val_lambda)
print(val_lambda[5])
print('omega values are')
print(val_Omega)
print(val_Omega[12])
print('the relative error is')
print(x[5,12])
print(y[5,12])
print(arr_rel_Pl[5,12])
cont=ax1.pcolormesh(x, y, arr_rel_Pl, cmap='RdBu', rasterized = True)
plt.scatter(1,0.1,color = 'black', marker = 'x')
# cont=ax1.pcolormesh(x, y, arr_rel_Pl, rasterized = True)
# cont=ax1.pcolormesh(x, y, arr_rel_Pl, cmap='RdBu', shading = 'gouraud')
cbar = plt.colorbar(cont, ax=ax1)
cbar.set_label('$r_\Sigma$')
#ax1.fill_between(val_lambda[10:], 11*[0.25], 11*[100], color = 'red', alpha = 0.1)

#linelim, = ax1.plot(val_lambda, err_arr_varAll*30, '--', color = 'purple', alpha = 0.1)
#linelim, = ax1.plot(val_lambda, err_arr_varAll*20, '--', color = 'purple', alpha = 0.2)
#linelim, = ax1.plot(val_lambda, err_arr_varAll*10, '--', color = 'purple', alpha = 0.3)

# ax1.legend([cont], ['Langevin model STD error'], loc='upper right')

#ax1.fill_between(val_lambda, err_arr_varAll*30, color = 'purple', alpha = 0.1)
#ax1.fill_between(val_lambda, err_arr_varAll*20, color = 'purple', alpha = 0.2)
#ax1.fill_between(val_lambda, err_arr_varAll*10, color = 'purple', alpha = 0.3)

ax1.tick_params(axis ='y') 
ax1.set_xscale('log')
ax1.set_yscale('log')        


ax1.set_xlim(0.001, 100)
ax1.set_ylim(0.01, 100)

plt.savefig(os.path.join(read_directory,'plots','other_heatmap-error-plot-tau' + str(val_tau[k]) + '.pdf'), dpi = 600)  

# new heatmap plot
fig = plt.figure(figsize=(3.3,2.0), constrained_layout = True) 

ax1 = plt.gca()

# ax1.set_title('Heatmap of Hes1 concentration STD relative error against lambda and Omega') 
ax1.set_xlabel('$\Omega$') 
ax1.set_ylabel('$\lambda$ $/\mathrm{min^{-1}}$')

x, y = np.meshgrid(val_Omega, val_lambda)
# cont=ax1.pcolormesh(x, y, arr_rel_Pl>0.05, cmap='RdBu', rasterized = True)
cont=ax1.pcolormesh(x, y, arr_rel_Pl>0.05, cmap='RdBu', rasterized = True)
# cont=ax1.pcolormesh(x, y, arr_rel_Pl, cmap='RdBu', shading = 'gouraud')
cbar = plt.colorbar(cont, ax=ax1)
cbar.set_label('$r_\Sigma$')
#ax1.fill_between(val_lambda[10:], 11*[0.25], 11*[100], color = 'red', alpha = 0.1)

#linelim, = ax1.plot(val_lambda, err_arr_varAll*30, '--', color = 'purple', alpha = 0.1)
#linelim, = ax1.plot(val_lambda, err_arr_varAll*20, '--', color = 'purple', alpha = 0.2)
#linelim, = ax1.plot(val_lambda, err_arr_varAll*10, '--', color = 'purple', alpha = 0.3)

# ax1.legend([cont], ['Langevin model STD error'], loc='upper right')

#ax1.fill_between(val_lambda, err_arr_varAll*30, color = 'purple', alpha = 0.1)
#ax1.fill_between(val_lambda, err_arr_varAll*20, color = 'purple', alpha = 0.2)
#ax1.fill_between(val_lambda, err_arr_varAll*10, color = 'purple', alpha = 0.3)

ax1.tick_params(axis ='y') 
ax1.set_xscale('log')
ax1.set_yscale('log')        


ax1.set_xlim(0.001, 100)
ax1.set_ylim(0.01, 100)

plt.savefig(os.path.join(read_directory,'plots','heatmap_test_plot.pdf'), dpi = 600)  



