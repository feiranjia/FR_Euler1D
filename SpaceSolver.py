###########################################################
class Euler1DEq():
    from GasDynamics import GAMMA,V2U_mat,U2V_mat3,getFlux_mat2,getFlux_mat3
###########################################################
# u_t + u * u_x = 0
###########################################################
    def __init__(self, in_mesh, in_flux_type, in_entropy_fix, in_BC_type, in_IC_type, in_doLimiting, in_limiter):
        self.mesh = in_mesh
        self.x_sp_mat = in_mesh.GloCoor_Mat
        self.p_order = in_mesh.PolyOrder
        #  self.n_cell = self.mesh.CellNMAX
        self.flux_type = in_flux_type
        self.EntropyFix = in_entropy_fix
        self.BC_type = in_BC_type
        self.IC_type = in_IC_type
        self.doLimiting = in_doLimiting
        self.limiter = in_limiter
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
        elif self.IC_type==2:
            from numpy import sin
            # Shu-Osher
            rho_l_init=3.857143
            u_l_init=2.629369
            p_l_init=10.33333
            rho_r_init=1.0
            u_r_init=0.0
            p_r_init=1.0
            x_b=-4.0
            a_rho=0.2
            f_rho=5.0
            idx_l_mat=self.x_sp_mat<=x_b
            idx_h_mat=self.x_sp_mat>x_b
            self.V_sp_mat[idx_l_mat,0]=rho_l_init
            self.V_sp_mat[idx_l_mat,1]=u_l_init
            self.V_sp_mat[idx_l_mat,2]=p_l_init
            self.V_sp_mat[idx_h_mat,0]=rho_r_init*(1.0+a_rho*sin(f_rho*self.x_sp_mat[idx_h_mat]))
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
        from numpy import max,min,zeros,tile,reshape,sum,where,abs,append,unique,array
        U_cell_mean_mat=self.getWeightedAver_mat(self.U_sp_mat)
        U_cell_mean_diff_r_mat=zeros(U_cell_mean_mat.shape)
        U_cell_mean_diff_l_mat=zeros(U_cell_mean_mat.shape)
        U_cell_mean_diff_r_mat[:-1,:]=U_cell_mean_mat[1:,:]-U_cell_mean_mat[:-1,:]
        U_cell_mean_diff_l_mat[1:,:]=U_cell_mean_mat[1:,:]-U_cell_mean_mat[:-1,:]
        #  At the boundary, it depends on the boundary condition.
        #  Periodic
        U_cell_mean_diff_r_mat[-1,:]=U_cell_mean_mat[0,:]-U_cell_mean_mat[-1,:]
        U_cell_mean_diff_l_mat[0,:]=U_cell_mean_mat[0,:]-U_cell_mean_mat[-1,:]
        if(self.limiter["type"]=="SMOOTH_GOOCH"):
            #  Calc the original U_diff between the cell averaged value and the FP
            #  value.
            U_diff_averfp_r_mat=self.U_fp_mat[0,:,:]-U_cell_mean_mat
            U_diff_averfp_l_mat=self.U_fp_mat[1,:,:]-U_cell_mean_mat
            smth_idc_arr = self.getSmoothnessIndicator()
            U_cell_mean_diff_mat = array((U_cell_mean_diff_l_mat,U_cell_mean_diff_r_mat))
            U_diff_averfp_mat = array((U_diff_averfp_l_mat,U_diff_averfp_r_mat))
            phi_mat = self.getGoochLimiter(U_cell_mean_diff_mat,U_diff_averfp_mat,smth_idc_arr)
            # FP
            U_cell_mean_mat2 = tile(U_cell_mean_mat,(2,1,1))
            phi_mat2 = tile(phi_mat,(2,1,1))
            self.U_fp_mat = U_cell_mean_mat2 + phi_mat2*(self.U_fp_mat - U_cell_mean_mat2)
            # SP
            U_cell_mean_mat_sp = tile(U_cell_mean_mat,(self.p_order+1,1,1))
            phi_mat_sp = tile(phi_mat,(self.p_order+1,1,1))
            self.U_sp_mat = U_cell_mean_mat_sp + phi_mat_sp*(self.U_sp_mat - U_cell_mean_mat_sp)
        elif(self.limiter["type"]=="MINMODTVB"):
            #  Calc the original U_diff between the cell averaged value and the FP
            #  value.
            U_diff_averfp_r_mat=U_cell_mean_mat-self.U_fp_mat[0,:,:]
            U_diff_averfp_l_mat=self.U_fp_mat[1,:,:]-U_cell_mean_mat
            #  h=self.mesh.CellSize
            h=self.mesh.CellSize/(self.p_order+1)
            #  Modify the U_diff by MINMOD
            from Utility import minmod
            from Utility import minmodTVB
            minmod_param_mat=zeros((3,U_diff_averfp_r_mat.size))
            minmod_param_mat[0,:]=U_diff_averfp_r_mat.flatten(order='C')
            minmod_param_mat[1,:]=U_cell_mean_diff_r_mat.flatten(order='C')
            minmod_param_mat[2,:]=U_cell_mean_diff_l_mat.flatten(order='C')
            U_diff_averfp_r_new_mat=minmodTVB(minmod_param_mat,self.limiter["M"],h)
            minmod_param_mat=zeros((3,U_diff_averfp_l_mat.size))
            minmod_param_mat[0,:]=U_diff_averfp_l_mat.flatten(order='C')
            minmod_param_mat[1,:]=U_cell_mean_diff_r_mat.flatten(order='C')
            minmod_param_mat[2,:]=U_cell_mean_diff_l_mat.flatten(order='C')
            U_diff_averfp_l_new_mat=minmodTVB(minmod_param_mat,self.limiter["M"],h)
            U_diff_averfp_r_new_mat=reshape(U_diff_averfp_r_new_mat,U_diff_averfp_r_mat.shape)
            U_diff_averfp_l_new_mat=reshape(U_diff_averfp_l_new_mat,U_diff_averfp_l_mat.shape)
            # Find the indexes needing limiting
            U_diff_averfp_eps=1E-8
            idx_r=where(abs(U_diff_averfp_r_mat[:,0]-U_diff_averfp_r_new_mat[:,0])>U_diff_averfp_eps)[0]
            idx_l=where(abs(U_diff_averfp_l_mat[:,0]-U_diff_averfp_l_new_mat[:,0])>U_diff_averfp_eps)[0]
            idx_arr=unique(append(idx_l,idx_r))
            # Do limiting
            # On FP
            self.U_fp_mat[0,idx_r,:]=U_cell_mean_mat[idx_r,:]-U_diff_averfp_r_new_mat[idx_r,:]
            self.U_fp_mat[1,idx_l,:]=U_cell_mean_mat[idx_l,:]+U_diff_averfp_l_new_mat[idx_l,:]
            # On SP
            for ic in idx_arr:
                for isp in range(self.U_sp_mat.shape[0]):
                    self.U_sp_mat[isp,ic,:]=U_cell_mean_mat[ic,:] # P0 MINMOD
                    #  self.U_sp_mat[isp,ic,:]=(self.x_sp_mat[isp,ic]-self.mesh.FluxPts_Mat[0,ic]) * (self.U_fp_mat[0,ic,:]-self.U_fp_mat[1,ic,:]) / (self.x_sp_mat[0,ic]-self.x_sp_mat[-1,ic]) + self.U_fp_mat[0,ic,:] # Biased P1 MINMOD will blow up.
                    #  self.U_sp_mat[isp,ic,:]=(self.x_sp_mat[isp,ic]-self.mesh.CellCenter_Vec[ic]) * (self.U_fp_mat[0,ic,:]-self.U_fp_mat[1,ic,:]) / (self.x_sp_mat[0,ic]-self.x_sp_mat[-1,ic]) + U_cell_mean_mat[ic,:] # Centered P1 MINMOD will oscillate
        #  elif(self.limiter["type"]=="MINMOD_EXT"): # Extrema preserving. NOT WORKING.
            #  from numpy import array,amin,amax,maximum,minimum
            #  V_min=array([0.125,0.0       ,0.1])
            #  V_max=array([1.0  ,0.92745263,1.0])
            #  from GasDynamics import V2U_mat1
            #  U_min=V2U_mat1(V_min)
            #  U_max=V2U_mat1(V_max)
            #  for ic in idx_arr:
                #  U_cell_min_arr=amin(self.U_sp_mat[:,ic,:],axis=0)
                #  U_cell_max_arr=amax(self.U_sp_mat[:,ic,:],axis=0)
                #  #  U_fp_min_arr=amin(self.U_fp_mat[:,ic,:],axis=0)
                #  #  U_fp_max_arr=amax(self.U_fp_mat[:,ic,:],axis=0)
                #  #  U_cell_min_arr=minimum(U_cell_min_arr,U_fp_min_arr)
                #  #  U_cell_max_arr=minimum(U_cell_max_arr,U_fp_max_arr)
                #  theta_max_arr=abs((U_max-U_cell_mean_mat[ic,:])/(U_cell_max_arr-U_cell_mean_mat[ic,:]))
                #  theta_min_arr=abs((U_min-U_cell_mean_mat[ic,:])/(U_cell_min_arr-U_cell_mean_mat[ic,:]))
                #  theta_arr=minimum(minimum(theta_max_arr,theta_min_arr),ones(theta_max_arr.shape))
                #  # SP
                #  for isp in range(self.U_sp_mat.shape[0]):
                    #  self.U_sp_mat[isp,ic,:]=U_cell_mean_mat[ic,:]+theta_arr*(self.U_sp_mat[isp,ic,:]-U_cell_mean_mat[ic,:])
                #  # FP
                #  for ifp in range(2):
                    #  self.U_fp_mat[ifp,ic,:]=U_cell_mean_mat[ic,:]+theta_arr*(self.U_fp_mat[ifp,ic,:]-U_cell_mean_mat[ic,:])
            #  #  self.updateUAtFp() # Finally U_fp_l and U_fp_r will intercept with each other
            #  # MINMOD update for FP is incompatible with the extrama preserving limiter of SP
            #  #  self.U_fp_mat[0,idx_r,:]=U_cell_mean_mat[idx_r,:]-U_diff_averfp_r_new_mat[idx_r,:]
            #  #  self.U_fp_mat[1,idx_l,:]=U_cell_mean_mat[idx_l,:]+U_diff_averfp_l_new_mat[idx_l,:]
        else:
            exit("%s is undefined!"%(self.limiter["type"]))

    def getSmoothnessIndicator(self):
        from numpy import array,tile,matmul,sum,reshape,log10
        restrict_mat_dict={ \
            3: array([[ 0.7349768 , 0.37068122,-0.1445425 , 0.03888448], \
                         [-0.0923266 , 0.5923266 , 0.5923266 ,-0.0923266 ], \
                         [ 0.03888448,-0.1445425 , 0.37068122, 0.7349768 ]]), \
        }
        prolong_restrict_mat_dict={ \
            3: array([[ 1.17382422,-0.23592625, 0.06210202], \
                         [ 0.31577941, 0.80735482,-0.12313423], \
                         [-0.12313423, 0.80735482, 0.31577941], \
                         [ 0.06210202,-0.23592625, 1.17382422]]), \
        }
        restrict_mat = restrict_mat_dict[self.p_order]
        prolong_restrict_mat = prolong_restrict_mat_dict[self.p_order]
        # Integrate the differences between the original solution of the
        # restricted-prolongated solution.
        rho_mat = self.U_sp_mat[:,:,0]
        rho_rest_mat = matmul(restrict_mat, rho_mat)
        rho_rest_prol_mat = matmul(prolong_restrict_mat, rho_rest_mat)
        rho_diff2_mat = (rho_mat - rho_rest_prol_mat)**2
        weights_mat = reshape(self.mesh.Weights_Vec,(self.mesh.Weights_Vec.size,1))
        weights_mat = tile(weights_mat, (1,rho_diff2_mat.shape[1]))
        # The ingegration in fact requires the jacobian to be included in the
        # sum. However, since it is cancelled in the division, it is omitted.
        rho_diff2_int_arr = sum(weights_mat * rho_diff2_mat, axis=0)
        rho_int_arr = sum(weights_mat * rho_mat, axis=0)
        smth_idc_arr = log10(rho_diff2_int_arr / rho_int_arr)
        return smth_idc_arr

    def getGoochLimiter(self,in_U_cell_mean_diff_mat,in_U_diff_averfp_mat,in_smth_idc_arr):
        from numpy import amax,amin,tile,ones,sin,pi,logical_and,reshape
        # in_U_cell_mean_diff_mat consists of [U_cell_mean_diff_r_mat, U_cell_mean_diff_l_mat]
        # in_U_diff_averfp_mat consists of [U_diff_averfp_r_mat, U_diff_averfp_l_mat]
        U_cell_mean_diff_max_mat = amax(in_U_cell_mean_diff_mat, axis=0)
        U_cell_mean_diff_min_mat = amin(in_U_cell_mean_diff_mat, axis=0)
        U_cell_mean_diff_max_mat2 = tile(U_cell_mean_diff_max_mat, (2,1,1))
        U_cell_mean_diff_min_mat2 = tile(U_cell_mean_diff_min_mat, (2,1,1))
        eps_zero = 1E-12
        idx_pos_mat = in_U_diff_averfp_mat > eps_zero
        idx_neg_mat = in_U_diff_averfp_mat < -eps_zero
        U_diff_ratio_mat = ones(in_U_diff_averfp_mat.shape) * 1E3
        U_diff_ratio_mat[idx_pos_mat] = U_cell_mean_diff_max_mat2[idx_pos_mat] / in_U_diff_averfp_mat[idx_pos_mat]
        U_diff_ratio_mat[idx_neg_mat] = U_cell_mean_diff_min_mat2[idx_neg_mat] / in_U_diff_averfp_mat[idx_neg_mat]
        idx_mat = U_diff_ratio_mat < 1.5
        phi_mat = ones(U_diff_ratio_mat.shape)
        phi_mat[idx_mat] = -4.0/27.0 * U_diff_ratio_mat[idx_mat]**3 + U_diff_ratio_mat[idx_mat]
        phi_mat = amin(phi_mat, axis=0)
        # sigma
        sigma_arr = ones(in_smth_idc_arr.shape)
        idx_arr = logical_and(in_smth_idc_arr >= self.limiter["mid"]-self.limiter["wid"], in_smth_idc_arr <= self.limiter["mid"]+self.limiter["wid"])
        # Transition function affects the overshoots.
        #  sigma_mag = 0.5
        sigma_mag = 0.5*0.975
        sigma_freq = 0.5*pi # scale=4 fails. scale=1 should be correct.
        sigma_arr[idx_arr] = sigma_mag*(1-sin(sigma_freq*(in_smth_idc_arr[idx_arr]-self.limiter["mid"])/self.limiter["wid"]))
        #  from numpy import tanh
        #  sigma_arr[idx_arr]=0.5*(-tanh((in_smth_idc_arr[idx_arr]-self.limiter["mid"])*self.limiter["wid"])+1)
        idx_arr = in_smth_idc_arr > self.limiter["mid"]+self.limiter["wid"]
        sigma_arr[idx_arr] = 0.0
        sigma_arr3 = tile(reshape(sigma_arr,(sigma_arr.size,1)),(1,3))
        #  # Consider only the 2nd order. Not working. Phi from Michalak_JoCP2009
        #  # is not working in high order DG.
        #  # For the explanations of the 2nd order and the higher order, refer
        #  # to Michalak_JoCP2009.
        phi_mat = sigma_arr3 + (1.0 - sigma_arr3)*phi_mat
        #  # Treat the 2nd order and the higher order as the same.
        #  phi_mat = sigma_arr3 + (1.0 - sigma_arr3)*phi_mat/2.0 
        #  # Consider only the higher order
        #  phi_mat = sigma_arr3
        return phi_mat

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

