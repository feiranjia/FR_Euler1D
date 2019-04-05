import numpy as np

n_cell=100
quadrature_type="GaussLegendre"

p_order=4

flux_type="ROE"
entropy_fix=1
#  flux_type="AUSMPlus"
#  BC_type="PERIODIC"
BC_type="EXTRAPOLATION"
IC_type=1
doLimiting=1
limiter_type="MINMOD"

if(IC_type==1):
    x_l=-1.0
    x_h=1.0
elif(IC_type==2):
    x_l=0.0
    x_h=1.0
    x_h=x_h*np.pi*2.0
else:
    exit("Not implemented!")

n_iter=100000
end_time=0.2

cfl=0.01

plot_init=0
plot_iter=1000

from Mesh1D import Mesh1D
mesh=Mesh1D([x_l,x_h], p_order, n_cell, quadrature_type)

from SpaceSolver import Euler1DEq
eq=Euler1DEq(mesh,flux_type,entropy_fix,BC_type,IC_type,doLimiting,limiter_type)

from TimeSolver import RK1,RK2,RK3
TimeScheme="RK3"

import matplotlib.pyplot as plt
plt.style.use('sjc')

plot_x_arr=np.reshape(mesh.GloCoor_Mat.T,(mesh.GloCoor_Mat.size,))
marker_size=8
line_width=4

if plot_init==1:
    fig=plt.figure()
    ax=fig.gca()
    from GasDynamics import U2V_mat3
    V_sp_mat=U2V_mat3(eq.U_sp_mat)
    ax.plot(plot_x_arr,np.reshape(V_sp_mat[:,:,0].T,(V_sp_mat[:,:,0].size,)),'o-',ms=marker_size,lw=line_width,label=r"$\rho$")
    ax.plot(plot_x_arr,np.reshape(V_sp_mat[:,:,1].T,(V_sp_mat[:,:,0].size,)),'s-',ms=marker_size,lw=line_width,label=r"$u$")
    ax.plot(plot_x_arr,np.reshape(V_sp_mat[:,:,2].T,(V_sp_mat[:,:,0].size,)),'v-',ms=marker_size,lw=line_width,label=r"$p$")
    ax.set_xlabel("X")
    #  ax.set_ylabel("U")
    ax.legend()
    plt.show()
    plt.close(fig)

i_iter=0
time=0
while(time<end_time and i_iter<n_iter):
    from GasDynamics import U2V_mat3,GAMMA,U2V_mat2
    V_sp_mat=U2V_mat3(eq.U_sp_mat)
    i_iter+=1
    c_sp_mat=np.sqrt(GAMMA*V_sp_mat[:,:,2]/V_sp_mat[:,:,0])
    #  if(time<0.01):
        #  cfl_used=0.001
    #  else:
        #  cfl_used=cfl
    #  dt=cfl_used*(mesh.CellSize/mesh.NodeInCellNMAX)/np.max(np.abs(V_sp_mat[:,:,1])+c_sp_mat)
    dt=cfl*(mesh.CellSize/mesh.NodeInCellNMAX)/np.max(np.abs(V_sp_mat[:,:,1])+c_sp_mat)
    if(time+dt>end_time):
        dt=end_time-time
    time+=dt
    print("iteration %d, time is %.1e"%(i_iter,time))
    if(TimeScheme=="RK1"):
        RK1(dt,eq)
    elif(TimeScheme=="RK2"):
        RK2(dt,eq)
    elif(TimeScheme=="RK3"):
        RK3(dt,eq)
    else:
        exit("Not implemented!")
    if(i_iter%plot_iter==0):
        fig=plt.figure()
        ax=fig.gca()
        V_sp_mat=U2V_mat3(eq.U_sp_mat)
        U_cell_mean_mat=eq.getWeightedAver_mat(eq.U_sp_mat)
        V_cell_mean_mat=U2V_mat2(U_cell_mean_mat)
        V_fp_mat=U2V_mat3(eq.U_fp_mat)
        #  plot_cell=np.arange(50-2,50+3,dtype=int)
        plot_cell=np.arange(100,dtype=int)
        plot_var=0
        ax.plot(mesh.GloCoor_Mat[:,plot_cell],eq.U_sp_mat[:,plot_cell,plot_var],'x-',ms=marker_size,lw=line_width)
        ax.plot(mesh.CellCenter_Vec[plot_cell],U_cell_mean_mat[plot_cell,plot_var],'o--',ms=marker_size,lw=line_width)
        ax.plot(mesh.FluxPts_Mat[0,plot_cell],eq.U_fp_mat[0,plot_cell,plot_var],'>',ms=marker_size,lw=line_width)
        ax.plot(mesh.FluxPts_Mat[1,plot_cell],eq.U_fp_mat[1,plot_cell,plot_var],'v',ms=marker_size,lw=line_width)
        ax.set_xlabel("X")
        if(plot_var==0):
            ax.set_ylabel(r"$\rho$")
        elif(plot_var==1):
            ax.set_ylabel(r"$\rho U$")
        elif(plot_var==2):
            ax.set_ylabel(r"$\rho E$")
        #  ax.legend()
        ax.set_title("it=%d,t=%.3f"%(i_iter,time))
        plt.show()
        plt.close(fig)

from shocktubecalc import sod
positions, regions, values = sod.solve(left_state=(1, 1, 0), right_state=(0.1, 0.125, 0.),\
    geometry=(x_l,x_h,0.5*(x_l+x_h)), t=end_time, gamma=1.4, npts=500)

fig=plt.figure()
ax=fig.gca()
from GasDynamics import U2V_mat3
V_sp_mat=U2V_mat3(eq.U_sp_mat)
ax.plot(plot_x_arr,np.reshape(V_sp_mat[:,:,0].T,(V_sp_mat[:,:,0].size,)),'o',color='C1',ms=marker_size,lw=line_width,label=r"$\rho$")
ax.plot(plot_x_arr,np.reshape(V_sp_mat[:,:,1].T,(V_sp_mat[:,:,0].size,)),'s',color='C2',ms=marker_size,lw=line_width,label=r"$u$")
ax.plot(plot_x_arr,np.reshape(V_sp_mat[:,:,2].T,(V_sp_mat[:,:,0].size,)),'v',color='C3',ms=marker_size,lw=line_width,label=r"$p$")
ax.plot(values['x'], values['rho'], '-',color='C1', label=r"Exact $\rho$")
ax.plot(values['x'], values['u'], '-',color='C2', label=r"Exact $u$")
ax.plot(values['x'], values['p'], '-',color='C3', label=r"Exact $p$")
ax.set_xlabel("X")
#  ax.set_ylabel("U")
ax.legend()
ax.set_title("it=%d,t=%.3f"%(i_iter,time))
#  plt.show()
plt.savefig("sol_it%d_t%.1f.png"%(i_iter,time))
plt.close(fig)


