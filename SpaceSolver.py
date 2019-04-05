###########################################################
class Euler1DEq():
    from GasDynamics import GAMMA,V2U_mat,U2V_mat3,getFlux_mat2,getFlux_mat3
###########################################################
# u_t + u * u_x = 0
###########################################################
    def __init__(self, in_mesh, in_flux_type, in_entropy_fix, in_BC_type, in_IC_type, in_doLimiting, in_limiter_type):
        self.mesh = in_mesh
        self.x_sp_mat = in_mesh.GloCoor_Mat
        #  self.p_order = in_mesh.PolyOrder
        #  self.n_cell = self.mesh.CellNMAX
        self.flux_type = in_flux_type
        self.EntropyFix = in_entropy_fix
        self.BC_type = in_BC_type
        self.IC_type = in_IC_type
        self.doLimiting = in_doLimiting
        self.limiter_type = in_limiter_type
        self.allocDOF()
        self.setIC()
        self.updateAfterRes()

    def allocDOF(self):
        from numpy import zeros
        self.U_sp_mat=zeros((self.x_sp_mat.shape[0],self.x_sp_mat.shape[1],3))
        self.F_sp_mat=zeros((self.x_sp_mat.shape[0],self.x_sp_mat.shape[1],3))
        self.U_fp_mat=zeros((2,self.mesh.CellNMAX,3))
        self.V_sp_mat=zeros((self.x_sp_mat.shape[0],self.x_sp_mat.shape[1],3))

    # Set IC
    def setIC(self):
        #  # Note: MATLAB behaves differently in this part.
        #  def heaviside(x):
            #  from numpy import sign
            #  y = 0.5 * ( sign(x) + 1.0 )
            #  return y
        #  from numpy import zeros,sqrt,logical_and,max,min,where,ones
        from GasDynamics import V2U_mat
        if self.IC_type==1:
            # Classic SOD shock tube
            rho_l_init=1.0
            u_l_init=0.0
            p_l_init=1.0
            rho_r_init=0.125
            u_r_init=0.0
            p_r_init=0.1
            idx_l_mat=self.x_sp_mat<=0.0
            idx_h_mat=self.x_sp_mat>0.0
            self.V_sp_mat[idx_l_mat,0]=rho_l_init
            self.V_sp_mat[idx_l_mat,1]=u_l_init
            self.V_sp_mat[idx_l_mat,2]=p_l_init
            self.V_sp_mat[idx_h_mat,0]=rho_r_init
            self.V_sp_mat[idx_h_mat,1]=u_r_init
            self.V_sp_mat[idx_h_mat,2]=p_r_init
            self.U_sp_mat=V2U_mat(self.V_sp_mat)
        else:
            from sys import exit
            exit('Initial Condition Error!')

    def updateFAtSp(self):
        from numpy import dot
        from GasDynamics import getFlux_mat3
        self.F_sp_mat=getFlux_mat3(self.U_sp_mat)

    def updateUAtFp(self):
        from numpy import dot
        for k in range(3):
            self.U_fp_mat[:,:,k]=dot(self.mesh.lag_poly4end_mat,self.U_sp_mat[:,:,k])

    #  def updateBeforeRes(self):
        #  self.updateFAtSp()

    def updateAfterRes(self):
        self.updateUAtFp()
        self.updateFAtSp()
        if(self.doLimiting):
            self.limitSol()

    def getWeightedAver_mat(self,in_var_sp_mat):
        from numpy import tile,reshape,sum,ones
        var_cell_mean_mat=sum(\
             tile(reshape(self.mesh.Weights_Vec,(self.mesh.NodeInCellNMAX,1,1)), \
                (1,in_var_sp_mat.shape[1],in_var_sp_mat.shape[2])) \
            * in_var_sp_mat \
            * (ones(in_var_sp_mat.shape)*self.mesh.Jacobian), \
            axis=0)
        #  vol=sum(self.mesh.Weights_Vec*self.mesh.Jacobian) # this is the integral of the cell volume
        var_cell_mean_mat/=self.mesh.CellSize
        return var_cell_mean_mat

    def limitSol(self):
        # Collect the max and min of the mean in the neighbouring cells
        from numpy import mean,max,min,zeros,ones,minimum,tile,isclose,logical_and,logical_not,reshape,sum,where,abs,append,unique
        U_cell_mean_mat=self.getWeightedAver_mat(self.U_sp_mat)
        U_cell_mean_diff_r_mat=zeros(U_cell_mean_mat.shape)
        U_cell_mean_diff_l_mat=zeros(U_cell_mean_mat.shape)
        U_cell_mean_diff_r_mat[:-1,:]=U_cell_mean_mat[1:,:]-U_cell_mean_mat[:-1,:]
        U_cell_mean_diff_l_mat[1:,:]=U_cell_mean_mat[1:,:]-U_cell_mean_mat[:-1,:]
        #  At the boundary, it depends on the boundary condition.
        #  Periodic
        U_cell_mean_diff_r_mat[-1,:]=U_cell_mean_mat[0,:]-U_cell_mean_mat[-1,:]
        U_cell_mean_diff_l_mat[0,:]=U_cell_mean_mat[0,:]-U_cell_mean_mat[-1,:]
        #  Calc the original U_diff between the cell averaged value and the FP
        #  value.
        U_diff_averfp_r_mat=U_cell_mean_mat-self.U_fp_mat[0,:,:]
        U_diff_averfp_l_mat=self.U_fp_mat[1,:,:]-U_cell_mean_mat
        #  Modify the U_diff by MINMOD
        from Utility import minmod
        from Utility import minmodTVB
        M=1E3
        P_order=self.U_sp_mat.shape[0]
        h=self.mesh.CellSize/(P_order+1)
        #  h=self.mesh.CellSize
        minmod_param_mat=zeros((3,U_diff_averfp_r_mat.size))
        minmod_param_mat[0,:]=U_diff_averfp_r_mat.flatten(order='C')
        minmod_param_mat[1,:]=U_cell_mean_diff_r_mat.flatten(order='C')
        minmod_param_mat[2,:]=U_cell_mean_diff_l_mat.flatten(order='C')
        if(self.limiter_type=="MINMOD"):
            U_diff_averfp_r_new_mat=minmod(minmod_param_mat)
        elif(self.limiter_type=="MINMOD_TVB"):
            U_diff_averfp_r_new_mat=minmodTVB(minmod_param_mat,M,h)
        else:
            exit("%s is undefined!"%(self.limiter_type))
        minmod_param_mat=zeros((3,U_diff_averfp_l_mat.size))
        minmod_param_mat[0,:]=U_diff_averfp_l_mat.flatten(order='C')
        minmod_param_mat[1,:]=U_cell_mean_diff_r_mat.flatten(order='C')
        minmod_param_mat[2,:]=U_cell_mean_diff_l_mat.flatten(order='C')
        if(self.limiter_type=="MINMOD"):
            U_diff_averfp_l_new_mat=minmod(minmod_param_mat)
        elif(self.limiter_type=="MINMOD_TVB"):
            U_diff_averfp_l_new_mat=minmodTVB(minmod_param_mat,M,h)
        else:
            exit("%s is undefined!"%(self.limiter_type))
        U_diff_averfp_r_new_mat=reshape(U_diff_averfp_r_new_mat,U_diff_averfp_r_mat.shape)
        U_diff_averfp_l_new_mat=reshape(U_diff_averfp_l_new_mat,U_diff_averfp_l_mat.shape)
        # Find the indexes needing limiting
        U_diff_averfp_eps=1E-8
        idx_r=where(abs(U_diff_averfp_r_mat[:,0]-U_diff_averfp_r_new_mat[:,0])>U_diff_averfp_eps)[0]
        idx_l=where(abs(U_diff_averfp_l_mat[:,0]-U_diff_averfp_l_new_mat[:,0])>U_diff_averfp_eps)[0]
        idx_arr=unique(append(idx_l,idx_r))
        # Do limiting
        if(self.limiter_type=="MINMOD" or self.limiter_type=="MINMOD_TVB"):
            # On FP
            self.U_fp_mat[0,idx_r,:]=U_cell_mean_mat[idx_r,:]-U_diff_averfp_r_new_mat[idx_r,:]
            self.U_fp_mat[1,idx_l,:]=U_cell_mean_mat[idx_l,:]+U_diff_averfp_l_new_mat[idx_l,:]
            # On SP
            for ic in idx_arr:
                for isp in range(self.U_sp_mat.shape[0]):
                    self.U_sp_mat[isp,ic,:]=U_cell_mean_mat[ic,:] # P0 MINMOD
                    #  self.U_sp_mat[isp,ic,:]=(self.x_sp_mat[isp,ic]-self.mesh.FluxPts_Mat[0,ic]) * (self.U_fp_mat[0,ic,:]-self.U_fp_mat[1,ic,:]) / (self.x_sp_mat[0,ic]-self.x_sp_mat[-1,ic]) + self.U_fp_mat[0,ic,:] # Biased P1 MINMOD will blow up.
                    #  self.U_sp_mat[isp,ic,:]=(self.x_sp_mat[isp,ic]-self.mesh.CellCenter_Vec[ic]) * (self.U_fp_mat[0,ic,:]-self.U_fp_mat[1,ic,:]) / (self.x_sp_mat[0,ic]-self.x_sp_mat[-1,ic]) + U_cell_mean_mat[ic,:] # Centered P1 MINMOD will oscillate
        elif(self.limiter_type=="MINMOD_EXT"): # Extrema preserving. NOT WORKING.
            from numpy import array,amin,amax,maximum,minimum
            V_min=array([0.125,0.0       ,0.1])
            V_max=array([1.0  ,0.92745263,1.0])
            from GasDynamics import V2U_mat1
            U_min=V2U_mat1(V_min)
            U_max=V2U_mat1(V_max)
            for ic in idx_arr:
                U_cell_min_arr=amin(self.U_sp_mat[:,ic,:],axis=0)
                U_cell_max_arr=amax(self.U_sp_mat[:,ic,:],axis=0)
                #  U_fp_min_arr=amin(self.U_fp_mat[:,ic,:],axis=0)
                #  U_fp_max_arr=amax(self.U_fp_mat[:,ic,:],axis=0)
                #  U_cell_min_arr=minimum(U_cell_min_arr,U_fp_min_arr)
                #  U_cell_max_arr=minimum(U_cell_max_arr,U_fp_max_arr)
                theta_max_arr=abs((U_max-U_cell_mean_mat[ic,:])/(U_cell_max_arr-U_cell_mean_mat[ic,:]))
                theta_min_arr=abs((U_min-U_cell_mean_mat[ic,:])/(U_cell_min_arr-U_cell_mean_mat[ic,:]))
                theta_arr=minimum(minimum(theta_max_arr,theta_min_arr),ones(theta_max_arr.shape))
                # SP
                for isp in range(self.U_sp_mat.shape[0]):
                    self.U_sp_mat[isp,ic,:]=U_cell_mean_mat[ic,:]+theta_arr*(self.U_sp_mat[isp,ic,:]-U_cell_mean_mat[ic,:])
                # FP
                for ifp in range(2):
                    self.U_fp_mat[ifp,ic,:]=U_cell_mean_mat[ic,:]+theta_arr*(self.U_fp_mat[ifp,ic,:]-U_cell_mean_mat[ic,:])
            #  self.updateUAtFp() # Finally U_fp_l and U_fp_r will intercept with each other
            # MINMOD update for FP is incompatible with the extrama preserving limiter of SP
            #  self.U_fp_mat[0,idx_r,:]=U_cell_mean_mat[idx_r,:]-U_diff_averfp_r_new_mat[idx_r,:]
            #  self.U_fp_mat[1,idx_l,:]=U_cell_mean_mat[idx_l,:]+U_diff_averfp_l_new_mat[idx_l,:]
        else:
            exit("%s is undefined!"%(self.limiter_type))

    #  def limitSol(self):
        #  # Collect the max and min of the mean in the neighbouring cells
        #  from numpy import mean,max,min,zeros,ones,minimum,tile,isclose,logical_and,logical_not,reshape,sum
        #  U_cell_mean_mat=self.getWeightedAver_mat(self.U_sp_mat)
        #  U_cell_mean_max_mat=zeros(U_cell_mean_mat.shape)
        #  U_cell_mean_min_mat=zeros(U_cell_mean_mat.shape)
        #  for i_cell in range(1,U_cell_mean_mat.shape[0]-1):
            #  U_cell_mean_max_mat[i_cell,:]=max(U_cell_mean_mat[i_cell-1:i_cell+2,:],axis=0)
            #  U_cell_mean_min_mat[i_cell,:]=min(U_cell_mean_mat[i_cell-1:i_cell+2,:],axis=0)
        #  U_cell_mean_max_mat[0,:]=max(U_cell_mean_mat[0:2,:],axis=0)
        #  U_cell_mean_min_mat[0,:]=min(U_cell_mean_mat[0:2,:],axis=0)
        #  U_cell_mean_max_mat[-1,:]=max(U_cell_mean_mat[-2:,:],axis=0)
        #  U_cell_mean_min_mat[-1,:]=min(U_cell_mean_mat[-2:,:],axis=0)
        #  #  #
        #  #  dFdx_sp_mat=self.getFluxDerivAtSp()
        #  #  dFdx_cell_mean_mat=self.getWeightedAver_mat(dFdx_sp_mat)
        #  # Calculate phi
        #  Phi_mat=ones(U_cell_mean_mat.shape)
        #  if(self.limiter_type=="MINMOD"):
            #  Phi0_max_mat=ones(U_cell_mean_mat.shape)
            #  Phi1_max_mat=ones(U_cell_mean_mat.shape)
            #  Phi0_min_mat=ones(U_cell_mean_mat.shape)
            #  Phi1_min_mat=ones(U_cell_mean_mat.shape)
            #  #  idx_fp_max_arr=argmax(self.U_fp_mat,axis=0)
            #  idx0_eq_max_mat=isclose(self.U_fp_mat[0,:,:],U_cell_mean_max_mat,rtol=1E-9)
            #  idx1_eq_max_mat=isclose(self.U_fp_mat[1,:,:],U_cell_mean_max_mat,rtol=1E-9)
            #  idx0_max_mat=self.U_fp_mat[0,:,:]>U_cell_mean_max_mat
            #  idx1_max_mat=self.U_fp_mat[1,:,:]>U_cell_mean_max_mat
            #  idx0_max_mat=logical_and(idx0_max_mat,logical_not(idx0_eq_max_mat))
            #  idx1_max_mat=logical_and(idx1_max_mat,logical_not(idx1_eq_max_mat))
            #  Phi0_max_mat[idx0_max_mat]=(U_cell_mean_max_mat[idx0_max_mat]-U_cell_mean_mat[idx0_max_mat])/(self.U_fp_mat[0,idx0_max_mat]-U_cell_mean_mat[idx0_max_mat])
            #  Phi1_max_mat[idx1_max_mat]=(U_cell_mean_max_mat[idx1_max_mat]-U_cell_mean_mat[idx1_max_mat])/(self.U_fp_mat[1,idx1_max_mat]-U_cell_mean_mat[idx1_max_mat])
            #  Phi_max_mat=minimum(Phi0_max_mat,Phi1_max_mat)
            #  idx0_eq_min_mat=isclose(self.U_fp_mat[0,:,:],U_cell_mean_min_mat,rtol=1E-9)
            #  idx1_eq_min_mat=isclose(self.U_fp_mat[1,:,:],U_cell_mean_min_mat,rtol=1E-9)
            #  idx0_min_mat=self.U_fp_mat[0,:,:]<U_cell_mean_min_mat
            #  idx1_min_mat=self.U_fp_mat[1,:,:]<U_cell_mean_min_mat
            #  idx0_min_mat=logical_and(idx0_min_mat,logical_not(idx0_eq_min_mat))
            #  idx1_min_mat=logical_and(idx1_min_mat,logical_not(idx1_eq_min_mat))
            #  Phi0_min_mat[idx0_min_mat]=(U_cell_mean_min_mat[idx0_min_mat]-U_cell_mean_mat[idx0_min_mat])/(self.U_fp_mat[0,idx0_min_mat]-U_cell_mean_mat[idx0_min_mat])
            #  Phi1_min_mat[idx1_min_mat]=(U_cell_mean_min_mat[idx1_min_mat]-U_cell_mean_mat[idx1_min_mat])/(self.U_fp_mat[1,idx1_min_mat]-U_cell_mean_mat[idx1_min_mat])
            #  Phi_min_mat=minimum(Phi0_min_mat,Phi1_min_mat)
            #  Phi_mat=minimum(Phi_max_mat,Phi_min_mat)
        #  else:
            #  from sys import exit
            #  exit("Not implemented!")
        #  # Do limiting
        #  # On SP
        #  self.U_sp_mat=tile(U_cell_mean_mat,(self.U_sp_mat.shape[0],1,1)) \
            #  + tile(Phi_mat,(self.U_sp_mat.shape[0],1,1))*(self.U_sp_mat-tile(U_cell_mean_mat,(self.U_sp_mat.shape[0],1,1)))
        #  # On FP
        #  self.U_fp_mat=tile(U_cell_mean_mat,(self.U_fp_mat.shape[0],1,1)) \
            #  + tile(Phi_mat,(self.U_fp_mat.shape[0],1,1))*(self.U_fp_mat-tile(U_cell_mean_mat,(self.U_fp_mat.shape[0],1,1)))

    def getRes(self):
        from numpy import reshape,matmul,zeros
        from GasDynamics import getFlux_mat2
        U_fp_l_mat,U_fp_r_mat=self.setBC()
        F_fp_comm_mat = self.getFluxCommAtFp(U_fp_l_mat,U_fp_r_mat)
        F_lfp_mat=getFlux_mat2(U_fp_r_mat[:-1,:])
        F_rfp_mat=getFlux_mat2(U_fp_l_mat[1:,:])
        dFdx_sp_mat=self.getFluxDerivAtSp()
        dFdx_sp_corr_mat = zeros(dFdx_sp_mat.shape)
        for k in range(3):
            dFdx_sp_corr_mat[:,:,k] = dFdx_sp_mat[:,:,k] \
                + matmul( reshape(self.mesh.radau_r_dpolydx_val_arr,(self.mesh.radau_r_dpolydx_val_arr.size,1)), \
                    reshape(F_fp_comm_mat[:-1,k]-F_lfp_mat[:,k],(1,F_lfp_mat[:,k].size)) ) \
                + matmul( reshape(self.mesh.radau_l_dpolydx_val_arr,(self.mesh.radau_l_dpolydx_val_arr.size,1)), \
                    reshape(F_fp_comm_mat[1: ,k]-F_rfp_mat[:,k],(1,F_rfp_mat[:,k].size)) )
        return dFdx_sp_corr_mat

    def setBC(self):
        from numpy import append,reshape
        if(self.BC_type=="PERIODIC"):
            U_fp_l_mat=append(reshape(self.U_fp_mat[1,-1,:],(1,3)),self.U_fp_mat[1,:,:],axis=0)
            U_fp_r_mat=append(self.U_fp_mat[0,:,:],reshape(self.U_fp_mat[0,0,:],(1,3)),axis=0)
        elif(self.BC_type=="EXTRAPOLATION"):
            U_fp_l_mat=append(reshape(self.U_fp_mat[1,0,:],(1,3)),self.U_fp_mat[1,:,:],axis=0)
            U_fp_r_mat=append(self.U_fp_mat[0,:,:],reshape(self.U_fp_mat[0,-1,:],(1,3)),axis=0)
        else:
            from sys import exit
            exit("Not implemented!")
        return U_fp_l_mat,U_fp_r_mat

    def getFluxCommAtFp(self,in_U_fp_l_mat,in_U_fp_r_mat):
        if(self.flux_type=="ROE"):
            from GasDynamics import getRoeFlux
            F_fp_comm_mat=getRoeFlux(in_U_fp_l_mat,in_U_fp_r_mat,self.EntropyFix)
        elif(self.flux_type=="AUSMPlus"):
            from GasDynamics import getAUSMPlusFlux
            F_fp_comm_mat=getAUSMPlusFlux(in_U_fp_l_mat,in_U_fp_r_mat)
        else:
            from sys import exit
            exit("Not implemented!")
        return F_fp_comm_mat

    def getFluxDerivAtSp(self):
        # Without Correction Procedure, Flux derivative is just the following line.
        from numpy import zeros,dot
        dFdx_sp_mat=zeros(self.F_sp_mat.shape)
        for k in range(3):
            dFdx_sp_mat[:,:,k] = dot(self.mesh.lag_dpolydx_mat, self.F_sp_mat[:,:,k])
        # Or use the following way: f = u**2 / 2, f_x = u * u_x
        # Then you do not need F_sp_mat
        #  dFdx_sp_mat = dot(self.mesh.lag_dpolydx_mat, self.U_sp_mat) * self.U_sp_mat
        return dFdx_sp_mat

