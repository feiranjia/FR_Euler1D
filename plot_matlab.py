import matplotlib.pyplot as plt
plt.style.use('sjc')
fig=plt.figure()
ax=fig.gca()
import numpy as np
from os.path import join

ref_fname="op_weno5_SSP_4_04_dt0.00100000.txt"
ref_data=np.loadtxt(ref_fname)
ref_X_arr=ref_data[:,1]
ref_Rho_arr=ref_data[:,2]
ax.plot(ref_X_arr,ref_Rho_arr,'-',lw=7,label="WENO5-RK4")

matlab_fdir="MATLAB"

#  limiter="MINMODTVB"
#  M=2000
#  fr_fname="ShuOsher_P3_NC360_RK3_CFL0.1_%s-M%d_Rho.dat"%(limiter,M)
#  fr_fpath=join(matlab_fdir,fr_fname)
#  fr_data=np.loadtxt(fr_fpath,delimiter=',')
#  fr_x_arr=fr_data[:,0]
#  fr_rho_arr=fr_data[:,1]

#  ax.plot(fr_x_arr,fr_rho_arr,'s',ms=1,label="FR,%s-M%d"%(limiter,M))

limiter="WENO"
M=10
matlab_fname="matlab_ShuOsher_P3_NC400_RK3_CFL0.1_%s-M%d.dat"%(limiter,M)
matlab_fpath=join(matlab_fdir,matlab_fname)
matlab_data=np.loadtxt(matlab_fpath,delimiter=',')
matlab_x_arr=matlab_data[:,0]
matlab_rho_arr=matlab_data[:,1]
ax.plot(matlab_x_arr,matlab_rho_arr,'s',ms=1,label="MATLAB,%s-M%d"%(limiter,M))

limiter="MINMODTVB"
M=100
matlab_fname="matlab_ShuOsher_P3_NC400_RK3_CFL0.1_%s-M%d.dat"%(limiter,M)
matlab_fpath=join(matlab_fdir,matlab_fname)
matlab_data=np.loadtxt(matlab_fpath,delimiter=',')
matlab_x_arr=matlab_data[:,0]
matlab_rho_arr=matlab_data[:,1]
ax.plot(matlab_x_arr,matlab_rho_arr,'s',ms=1,label="MATLAB,%s-M%d"%(limiter,M))

ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$\rho$")

ax.legend()

fig_fname="matlab_ShuOsher_P3_NC400_RK3_CFL0.1_%s-M%d_Rho.png"%(limiter,M)
fig.savefig(fig_fname)
print("%s is saved."%(fig_fname))
