import numpy as np

quadrature_type="GaussLegendre"

IC_type=1 # SOD
#  IC_type=2 # Shu-Osher

if(IC_type==1):
    n_cell=100 # IC=1
    #  n_cell=400 # IC=1
elif(IC_type==2):
    #  n_cell=180 # IC=2
    n_cell=360 # IC=2
    #  n_cell=720 # IC=2

p_order=3

flux_type="ROE"
entropy_fix=1
#  flux_type="AUSMPlus"

#  BC_type="PERIODIC"
BC_type="EXTRAPOLATION"

doLimiting=1
# M=0 is MINMOD
limiter={"type":"MINMODTVB","M":2500}
#  limiter={"type":"MINMODTVB_GRADIENT"}
#  limiter={"type":"SMOOTH","mid":-4.5,"wid":1.5}
#  limiter={"type":"SMOOTH_GOOCH","mid":-5.5,"wid":1.5} # -3,2 blows up.
#  limiter={"type":"MINMAX"}
#  limiter={"type":"MINMAX_GRADIENT"} # From hpMusic, not working for either case

TimeScheme="RK3"
cfl=0.1

plot_init=0
plot_iter=100
monitor_iter=100
##########################################

if(IC_type==1):
    x_l=-1.0
    x_h=1.0
elif(IC_type==2):
    x_l=-4.5
    x_h=4.5
else:
    exit("Not implemented!")

n_iter=100000
if(IC_type==1):
    end_time=0.2 # SOD
elif(IC_type==2):
    #  end_time=1.8 # Shu-Osher
    end_time=0.5
else:
    exit("IC %d is unknown."%(IC_type))

from Mesh1D import Mesh1D
mesh=Mesh1D([x_l,x_h], p_order, n_cell, quadrature_type)

from SpaceSolver import Euler1DEq
eq=Euler1DEq(mesh,flux_type,entropy_fix,BC_type,IC_type,doLimiting,limiter)

from TimeSolver import RK1,RK2,RK3

import matplotlib.pyplot as plt
plt.style.use('sjc')

plot_x_arr=np.reshape(mesh.GloCoor_Mat.T,(mesh.GloCoor_Mat.size,))
marker_size=2
line_width=2

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

trouble_cells_mat = np.zeros((1,n_cell+2))
trouble_cells_mat[0,2:] = mesh.CellCenter_Vec

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
    #  dt=cfl*(mesh.CellSize/mesh.NodeInCellNMAX)/np.max(np.abs(V_sp_mat[:,:,1])+c_sp_mat)
    dt = 2.8E-2/100
    if(time+dt>end_time):
        dt=end_time-time
    time+=dt
    if(i_iter%monitor_iter==0):
        print("iteration %d, time is %.1e"%(i_iter,time))
    if(TimeScheme=="RK1"):
        RK1(dt,eq)
    elif(TimeScheme=="RK2"):
        RK2(dt,eq)
    elif(TimeScheme=="RK3"):
        RK3(dt,eq)
    else:
        exit("Not implemented!")
    # Output the trouble cells
    trouble_cells_arr = np.append(np.array([i_iter, time]), eq.trouble_cells_arr)
    trouble_cells_arr = np.reshape(trouble_cells_arr, (1,trouble_cells_arr.size))
    trouble_cells_mat = np.append(trouble_cells_mat, trouble_cells_arr, axis=0)
    if(i_iter%plot_iter==0):
        fig=plt.figure()
        ax=fig.gca()
        V_sp_mat=U2V_mat3(eq.U_sp_mat)
        U_cell_mean_mat=eq.getWeightedAver_mat(eq.U_sp_mat)
        V_cell_mean_mat=U2V_mat2(U_cell_mean_mat)
        V_fp_mat=U2V_mat3(eq.U_fp_mat)
        #  plot_cell=np.arange(10-1,22+3,dtype=int) # Debug for IC=1
        plot_cell=np.where(np.logical_and(mesh.CellCenter_Vec>=-0.1,mesh.CellCenter_Vec<=0.1))[0] # Debug for IC=2
        #  plot_cell=np.where(np.logical_and(mesh.CellCenter_Vec>=-4.0,mesh.CellCenter_Vec<=-3.5))[0] # Debug for IC=2
        #  plot_cell=np.arange(n_cell,dtype=int)
        plot_var=0
        ax.plot(mesh.GloCoor_Mat[:,plot_cell],eq.U_sp_mat[:,plot_cell,plot_var],'x-',ms=marker_size,lw=line_width)
        ax.plot(mesh.CellCenter_Vec[plot_cell],U_cell_mean_mat[plot_cell,plot_var],'o--',ms=marker_size,lw=line_width)
        ax.plot(mesh.FluxPts_Mat[0,plot_cell],eq.U_fp_mat[0,plot_cell,plot_var],'>',ms=marker_size,lw=line_width)
        ax.plot(mesh.FluxPts_Mat[1,plot_cell],eq.U_fp_mat[1,plot_cell,plot_var],'v',ms=marker_size,lw=line_width)
        ax.set_xlabel("X")
        if(plot_var==0):
            ax.set_ylabel(r"$\rho$")
            if(IC_type==1):
                ax.set_ylim([-0.05,1.05]) # For SOD
            elif(IC_type==2):
                ax.set_ylim([0.5,5.0]) # Shu-Osher
        elif(plot_var==1):
            ax.set_ylabel(r"$\rho U$")
        elif(plot_var==2):
            ax.set_ylabel(r"$\rho E$")
        #  ax.legend()
        ax.set_title("it=%d,t=%.3f"%(i_iter,time))
        plt.show()
        plt.close(fig)

if((np.isnan(time)) or (time<end_time)):
    exit("Not finished. t = %.2f"%(time))

def plot_trouble_cells(in_trouble_cells_mat, in_fig_fname):
    fig=plt.figure()
    ax=fig.gca()
    # Filter the marked cells
    if(limiter["type"]=="SMOOTH" or limiter["type"]=="SMOOTH_GOOCH"):
        #  idx_mat = np.logical_and(in_trouble_cells_mat >= (limiter["mid"]-limiter["wid"]), in_trouble_cells_mat <= (limiter["mid"]+limiter["wid"]) )
        idx_mat = in_trouble_cells_mat >= (limiter["mid"]-limiter["wid"])
    elif(limiter["type"]=="MINMODTVB" or limiter["type"]=="MINMODTVB_GRADIENT" or limiter["type"]=="MINMAX" or limiter["type"]=="MINMAX_GRADIENT"):
        idx_mat = in_trouble_cells_mat > 0.5
    idx_mat = idx_mat[1:,2:]
    time_mat = in_trouble_cells_mat[1:,1]
    time_mat = np.tile(np.reshape(time_mat,(time_mat.size,1)), (1,idx_mat.shape[1]))
    x_mat = in_trouble_cells_mat[0,2:]
    x_mat = np.tile(np.reshape(x_mat, (1,x_mat.size)), (idx_mat.shape[0],1))
    ax.plot(x_mat[idx_mat], time_mat[idx_mat], 'o', ms=marker_size,lw=line_width)
    ax.set_xlabel("X")
    ax.set_ylabel("T")
    fig.savefig(in_fig_fname)
    print("%s is saved."%(in_fig_fname))

from GasDynamics import U2V_mat3,U2V_mat2
V_sp_mat=U2V_mat3(eq.U_sp_mat)

if(IC_type==1):
    fig=plt.figure()
    ax=fig.gca()
    from shocktubecalc import sod
    positions, regions, values = sod.solve(left_state=(1, 1, 0), right_state=(0.1, 0.125, 0.),\
        geometry=(x_l,x_h,0.5*(x_l+x_h)), t=end_time, gamma=1.4, npts=500)
    ax.plot(values['x'], values['rho'], '-',color='C1', label=r"Exact $\rho$")
    ax.plot(values['x'], values['u'], '-',color='C2', label=r"Exact $u$")
    ax.plot(values['x'], values['p'], '-',color='C3', label=r"Exact $p$")
    ax.plot(plot_x_arr,np.reshape(V_sp_mat[:,:,0].T,(V_sp_mat[:,:,0].size,)),'o',color='C1',ms=marker_size,lw=line_width,label=r"$\rho$")
    ax.plot(plot_x_arr,np.reshape(V_sp_mat[:,:,1].T,(V_sp_mat[:,:,0].size,)),'s',color='C2',ms=marker_size,lw=line_width,label=r"$u$")
    ax.plot(plot_x_arr,np.reshape(V_sp_mat[:,:,2].T,(V_sp_mat[:,:,0].size,)),'v',color='C3',ms=marker_size,lw=line_width,label=r"$p$")
    ax.set_ylim([-0.05,1.05])
    ax.set_xlabel("X")
    #  ax.set_ylabel("U")
    ax.legend()
    ax.set_title("it=%d,t=%.3f"%(i_iter,time))
    if(limiter["type"]=="MINMODTVB"):
        fig_fname="SOD_P%d_NC%d_%s_CFL%.1f_%s-M%d.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["M"])
        fig.savefig(fig_fname)
        print("%s is saved."%(fig_fname))
        data_fname="trouble_cells_history_SOD_P%d_NC%d_%s_CFL%.1f_%s-M%d.dat"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["M"])
        np.savetxt(data_fname, trouble_cells_mat)
        print("%s is saved."%(data_fname))
        fig_fname="trouble_cells_history_SOD_P%d_NC%d_%s_CFL%.1f_%s-M%d.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["M"])
        plot_trouble_cells(trouble_cells_mat,fig_fname)
    elif(limiter["type"]=="MINMODTVB_GRADIENT"):
        fig_fname="SOD_P%d_NC%d_%s_CFL%.1f_%s.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        fig.savefig(fig_fname)
        print("%s is saved."%(fig_fname))
        data_fname="trouble_cells_history_SOD_P%d_NC%d_%s_CFL%.1f_%s.dat"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        np.savetxt(data_fname, trouble_cells_mat)
        print("%s is saved."%(data_fname))
        fig_fname="trouble_cells_history_SOD_P%d_NC%d_%s_CFL%.1f_%s.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        plot_trouble_cells(trouble_cells_mat,fig_fname)
    elif(limiter["type"]=="SMOOTH"):
        fig_fname="SOD_P%d_NC%d_%s_CFL%.1f_%s-M%.1f-W%d.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["mid"],limiter["wid"])
        fig.savefig(fig_fname)
        print("%s is saved."%(fig_fname))
        data_fname="trouble_cells_history_SOD_P%d_NC%d_%s_CFL%.1f_%s-M%.1f-W%d.dat"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["mid"],limiter["wid"])
        np.savetxt(data_fname, trouble_cells_mat)
        print("%s is saved."%(data_fname))
        fig_fname="trouble_cells_history_SOD_P%d_NC%d_%s_CFL%.1f_%s-M%.1f-W%d.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["mid"],limiter["wid"])
        plot_trouble_cells(trouble_cells_mat,fig_fname)
    elif(limiter["type"]=="SMOOTH_GOOCH"):
        fig_fname="SOD_P%d_NC%d_%s_CFL%.1f_%s-M%.1f-W%d.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["mid"],limiter["wid"])
        fig.savefig(fig_fname)
        print("%s is saved."%(fig_fname))
        data_fname="trouble_cells_history_SOD_P%d_NC%d_%s_CFL%.1f_%s-M%.1f-W%d.dat"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["mid"],limiter["wid"])
        np.savetxt(data_fname, trouble_cells_mat)
        print("%s is saved."%(data_fname))
        fig_fname="trouble_cells_history_SOD_P%d_NC%d_%s_CFL%.1f_%s-M%.1f-W%d.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["mid"],limiter["wid"])
        plot_trouble_cells(trouble_cells_mat,fig_fname)
    elif(limiter["type"]=="MINMAX"):
        fig_fname="SOD_P%d_NC%d_%s_CFL%.1f_%s.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        fig.savefig(fig_fname)
        print("%s is saved."%(fig_fname))
        data_fname="trouble_cells_history_SOD_P%d_NC%d_%s_CFL%.1f_%s.dat"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        np.savetxt(data_fname, trouble_cells_mat)
        print("%s is saved."%(data_fname))
        fig_fname="trouble_cells_history_SOD_P%d_NC%d_%s_CFL%.1f_%s.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        plot_trouble_cells(trouble_cells_mat,fig_fname)
    elif(limiter["type"]=="MINMAX_GRADIENT"):
        fig_fname="SOD_P%d_NC%d_%s_CFL%.1f_%s.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        fig.savefig(fig_fname)
        print("%s is saved."%(fig_fname))
        data_fname="trouble_cells_history_SOD_P%d_NC%d_%s_CFL%.1f_%s.dat"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        np.savetxt(data_fname, trouble_cells_mat)
        print("%s is saved."%(data_fname))
        fig_fname="trouble_cells_history_SOD_P%d_NC%d_%s_CFL%.1f_%s.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        plot_trouble_cells(trouble_cells_mat,fig_fname)
elif(IC_type==2):
    # This data file is obtained from https://github.com/ketch/RK-WENO-Opt
    ref_fname="op_weno5_SSP_4_04_dt0.00100000.txt"
    ref_data=np.loadtxt(ref_fname)
    ref_X_arr=ref_data[:,1]
    ref_Rho_arr=ref_data[:,2]
    ref_RhoU_arr=ref_data[:,3]
    ref_RhoE_arr=ref_data[:,4]
    ref_U_mat=np.zeros((ref_Rho_arr.size,3))
    ref_U_mat[:,0]=ref_Rho_arr
    ref_U_mat[:,1]=ref_RhoU_arr
    ref_U_mat[:,2]=ref_RhoE_arr
    ref_V_mat=U2V_mat2(ref_U_mat)
    fig_Rho=plt.figure()
    fig_U=plt.figure()
    fig_P=plt.figure()
    ax_Rho=fig_Rho.gca()
    ax_U=fig_U.gca()
    ax_P=fig_P.gca()
    ax_Rho.plot(ref_X_arr,ref_V_mat[:,0],'-',lw=line_width,label="WENO5-RK4")
    ax_U.plot(ref_X_arr,ref_V_mat[:,1],'-',lw=line_width,label="WENO5-RK4")
    ax_P.plot(ref_X_arr,ref_V_mat[:,2],'-',lw=line_width,label="WENO5-RK4")
    ax_Rho.plot(plot_x_arr,np.reshape(V_sp_mat[:,:,0].T,(V_sp_mat[:,:,0].size,)),'o',ms=marker_size,lw=line_width,label="FR-P%d-%s"%(p_order,TimeScheme))
    #  out_data=np.zeros((plot_x_arr.size,2))
    #  out_data[:,0]=plot_x_arr
    #  out_data[:,1]=np.reshape(V_sp_mat[:,:,0].T,(V_sp_mat[:,:,0].size,))
    #  np.savetxt("rho.dat",out_data,delimiter=',')
    ax_U.plot(plot_x_arr,np.reshape(V_sp_mat[:,:,1].T,(V_sp_mat[:,:,0].size,)),'s',ms=marker_size,lw=line_width,label="FR-P%d-%s"%(p_order,TimeScheme))
    ax_P.plot(plot_x_arr,np.reshape(V_sp_mat[:,:,2].T,(V_sp_mat[:,:,0].size,)),'v',ms=marker_size,lw=line_width,label="FR-P%d-%s"%(p_order,TimeScheme))
    ax_Rho.set_ylim([0,4.5])
    ax_U.set_ylim([-0.2,3])
    ax_P.set_ylim([0,12])
    ax_Rho.set_xlabel("X")
    ax_U.set_xlabel("X")
    ax_P.set_xlabel("X")
    ax_Rho.set_ylabel(r"$\rho$")
    ax_U.set_ylabel(r"$U$")
    ax_P.set_ylabel(r"$P$")
    ax_Rho.legend()
    ax_U.legend()
    ax_P.legend()
    ax_Rho.set_title("it=%d,t=%.3f"%(i_iter,time))
    ax_U.set_title("it=%d,t=%.3f"%(i_iter,time))
    ax_P.set_title("it=%d,t=%.3f"%(i_iter,time))
    if(limiter["type"]=="MINMODTVB"):
        fig_Rho.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s-M%.1f_Rho.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["M"]))
        fig_U.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s-M%.1f_U.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["M"]))
        fig_P.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s-M%.1f_P.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["M"]))
        data_fname="trouble_cells_history_ShuOsher_P%d_NC%d_%s_CFL%.1f_%s-M%d.dat"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["M"])
        np.savetxt(data_fname, trouble_cells_mat)
        print("%s is saved."%(data_fname))
        fig_fname="trouble_cells_history_ShuOsher_P%d_NC%d_%s_CFL%.1f_%s-M%d.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["M"])
        plot_trouble_cells(trouble_cells_mat,fig_fname)
    elif(limiter["type"]=="MINMODTVB_GRADIENT"):
        fig_Rho.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s_Rho.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"]))
        fig_U.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s_U.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"]))
        fig_P.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s_P.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"]))
        data_fname="trouble_cells_history_ShuOsher_P%d_NC%d_%s_CFL%.1f_%s.dat"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        np.savetxt(data_fname, trouble_cells_mat)
        print("%s is saved."%(data_fname))
        fig_fname="trouble_cells_history_ShuOsher_P%d_NC%d_%s_CFL%.1f_%s.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        plot_trouble_cells(trouble_cells_mat,fig_fname)
    elif(limiter["type"]=="SMOOTH" or limiter["type"]=="SMOOTH_GOOCH"):
        fig_Rho.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s-M%.1f-W%.1f_Rho.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["mid"],limiter["wid"]))
        fig_U.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s-M%.1f-W%.1f_U.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["mid"],limiter["wid"]))
        fig_P.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s-M%.1f-W%.1f_P.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["mid"],limiter["wid"]))
        data_fname="trouble_cells_history_ShuOsher_P%d_NC%d_%s_CFL%.1f_%s-M%.1f-W%.1f.dat"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["mid"],limiter["wid"])
        np.savetxt(data_fname, trouble_cells_mat)
        print("%s is saved."%(data_fname))
        fig_fname="trouble_cells_history_ShuOsher_P%d_NC%d_%s_CFL%.1f_%s-M%.1f-W%.1f.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"],limiter["mid"],limiter["wid"])
        plot_trouble_cells(trouble_cells_mat,fig_fname)
    elif(limiter["type"]=="MINMAX"):
        fig_Rho.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s_Rho.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"]))
        fig_U.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s_U.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"]))
        fig_P.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s_P.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"]))
        data_fname="trouble_cells_history_ShuOsher_P%d_NC%d_%s_CFL%.1f_%s.dat"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        np.savetxt(data_fname, trouble_cells_mat)
        print("%s is saved."%(data_fname))
        fig_fname="trouble_cells_history_ShuOsher_P%d_NC%d_%s_CFL%.1f_%s.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        plot_trouble_cells(trouble_cells_mat,fig_fname)
    elif(limiter["type"]=="MINMAX_GRADIENT"):
        fig_Rho.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s_Rho.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"]))
        fig_U.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s_U.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"]))
        fig_P.savefig("ShuOsher_P%d_NC%d_%s_CFL%.1f_%s_P.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"]))
        data_fname="trouble_cells_history_ShuOsher_P%d_NC%d_%s_CFL%.1f_%s.dat"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        np.savetxt(data_fname, trouble_cells_mat)
        print("%s is saved."%(data_fname))
        fig_fname="trouble_cells_history_ShuOsher_P%d_NC%d_%s_CFL%.1f_%s.png"%(p_order,n_cell,TimeScheme,cfl,limiter["type"])
        plot_trouble_cells(trouble_cells_mat,fig_fname)

