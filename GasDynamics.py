GAMMA=1.4

def V2U_mat(in_V_mat):
    from numpy import zeros
    U_mat=zeros(in_V_mat.shape)
    U_mat[:,:,0]=in_V_mat[:,:,0]
    U_mat[:,:,1]=in_V_mat[:,:,0]*in_V_mat[:,:,1]
    U_mat[:,:,2]=in_V_mat[:,:,2]/(GAMMA-1)+0.5*U_mat[:,:,1]*in_V_mat[:,:,1]
    return U_mat

def U2V_mat3(in_U_mat):
    from numpy import zeros
    V_mat=zeros(in_U_mat.shape)
    V_mat[:,:,0]=in_U_mat[:,:,0]
    V_mat[:,:,1]=in_U_mat[:,:,1]/V_mat[:,:,0]
    V_mat[:,:,2]=(in_U_mat[:,:,2]-0.5*V_mat[:,:,0]*V_mat[:,:,1]**2)*(GAMMA-1)
    return V_mat

def U2V_mat2(in_U_mat):
    from numpy import zeros
    V_mat=zeros(in_U_mat.shape)
    V_mat[:,0]=in_U_mat[:,0]
    V_mat[:,1]=in_U_mat[:,1]/V_mat[:,0]
    V_mat[:,2]=(in_U_mat[:,2]-0.5*V_mat[:,0]*V_mat[:,1]**2)*(GAMMA-1)
    return V_mat

def getFlux_mat2(in_U_mat):
    from numpy import zeros
    # 1D Euler equation
    F_mat=zeros(in_U_mat.shape)
    F_mat[:,0]=in_U_mat[:,1]
    RhoU2_arr=in_U_mat[:,1]**2/in_U_mat[:,0]
    p_arr=(in_U_mat[:,2]-0.5*RhoU2_arr)*(GAMMA-1)
    F_mat[:,1]=RhoU2_arr+p_arr
    F_mat[:,2]=(in_U_mat[:,2]+p_arr)*in_U_mat[:,1]/in_U_mat[:,0]
    return F_mat

def getFlux_mat3(in_U_mat):
    from numpy import zeros
    # 1D Euler equation
    F_mat=zeros(in_U_mat.shape)
    F_mat[:,:,0]=in_U_mat[:,:,1]
    RhoU2_arr=in_U_mat[:,:,1]**2/in_U_mat[:,:,0]
    p_arr=(in_U_mat[:,:,2]-0.5*RhoU2_arr)*(GAMMA-1)
    F_mat[:,:,1]=RhoU2_arr+p_arr
    F_mat[:,:,2]=(in_U_mat[:,:,2]+p_arr)*in_U_mat[:,:,1]/in_U_mat[:,:,0]
    return F_mat

def getRoeFlux(in_U_l_mat,in_U_r_mat,in_EntropyFix):
    from numpy import sqrt,ones,abs,zeros,maximum,tile
    F_l_mat=getFlux_mat2(in_U_l_mat)
    F_r_mat=getFlux_mat2(in_U_r_mat)
    rho_l_arr=in_U_l_mat[:,0]
    rho_r_arr=in_U_r_mat[:,0]
    u_l_arr=in_U_l_mat[:,1]/rho_l_arr
    u_r_arr=in_U_r_mat[:,1]/rho_r_arr
    p_l_arr=F_l_mat[:,1]-rho_l_arr*u_l_arr**2
    p_r_arr=F_r_mat[:,1]-rho_r_arr*u_r_arr**2
    H_l_arr=(in_U_l_mat[:,2]+p_l_arr)/rho_l_arr
    H_r_arr=(in_U_r_mat[:,2]+p_r_arr)/rho_r_arr
    R_arr=sqrt(rho_r_arr/rho_l_arr)
    rho_aver_arr=R_arr*rho_l_arr
    u_aver_arr=(R_arr*u_r_arr+u_l_arr)/(R_arr+1)
    H_aver_arr=(R_arr*H_r_arr+H_l_arr)/(R_arr+1)
    c_aver_arr=sqrt((GAMMA-1)*(H_aver_arr-0.5*u_aver_arr**2))
    lambda_1_aver_arr=u_aver_arr*ones(u_aver_arr.shape) # otherwise they are linked.
    lambda_2_aver_arr=u_aver_arr+c_aver_arr
    lambda_3_aver_arr=u_aver_arr-c_aver_arr
    lambda_1_aver_arr=abs(lambda_1_aver_arr)
    lambda_2_aver_arr=abs(lambda_2_aver_arr)
    lambda_3_aver_arr=abs(lambda_3_aver_arr)
    if in_EntropyFix==1:
        lambda_1_l_arr=u_l_arr*ones(u_l_arr.shape)
        lambda_1_r_arr=u_r_arr*ones(u_r_arr.shape)
        epsilon_1_arr=maximum(maximum(lambda_1_aver_arr-lambda_1_l_arr,lambda_1_r_arr-lambda_1_aver_arr),zeros(lambda_1_aver_arr.shape))
        lambda_1_aver_arr=(lambda_1_aver_arr>=epsilon_1_arr)*lambda_1_aver_arr+(lambda_1_aver_arr<epsilon_1_arr)*epsilon_1_arr
        c_l_arr=sqrt(GAMMA*p_l_arr/rho_l_arr)
        c_r_arr=sqrt(GAMMA*p_r_arr/rho_r_arr)
        lambda_2_l_arr=u_l_arr+c_l_arr
        lambda_2_r_arr=u_r_arr+c_r_arr
        epsilon_2_arr=maximum(maximum(lambda_2_aver_arr-lambda_2_l_arr,lambda_2_r_arr-lambda_2_aver_arr),zeros(lambda_2_aver_arr.shape))
        lambda_2_aver_arr=(lambda_2_aver_arr>=epsilon_2_arr)*lambda_2_aver_arr+(lambda_2_aver_arr<epsilon_2_arr)*epsilon_2_arr
        lambda_3_l_arr=u_l_arr-c_l_arr
        lambda_3_r_arr=u_r_arr-c_r_arr
        epsilon_3_arr=maximum(maximum(lambda_3_aver_arr-lambda_3_l_arr,lambda_3_r_arr-lambda_3_aver_arr),zeros(lambda_3_aver_arr.shape))
        lambda_3_aver_arr=(lambda_3_aver_arr>=epsilon_3_arr)*lambda_3_aver_arr+(lambda_3_aver_arr<epsilon_3_arr)*epsilon_3_arr
    eigvec_1_aver_mat=ones((lambda_1_aver_arr.size,3))
    eigvec_2_aver_mat=ones((lambda_2_aver_arr.size,3))
    eigvec_3_aver_mat=ones((lambda_3_aver_arr.size,3))
    eigvec_1_aver_mat[:,1]=u_aver_arr*ones(u_aver_arr.shape)
    eigvec_1_aver_mat[:,2]=0.5*u_aver_arr**2
    eigvec_2_aver_mat[:,1]=u_aver_arr+c_aver_arr
    eigvec_2_aver_mat[:,2]=H_aver_arr+u_aver_arr*c_aver_arr
    tmp_arr=0.5*rho_aver_arr/c_aver_arr
    eigvec_2_aver_mat[:,0]*=tmp_arr
    eigvec_2_aver_mat[:,1]*=tmp_arr
    eigvec_2_aver_mat[:,2]*=tmp_arr
    eigvec_3_aver_mat[:,1]=u_aver_arr-c_aver_arr
    eigvec_3_aver_mat[:,2]=H_aver_arr-u_aver_arr*c_aver_arr
    eigvec_3_aver_mat[:,0]*=-tmp_arr
    eigvec_3_aver_mat[:,1]*=-tmp_arr
    eigvec_3_aver_mat[:,2]*=-tmp_arr
    du_aver_arr=u_r_arr-u_l_arr
    dp_aver_arr=p_r_arr-p_l_arr
    drho_aver_arr=rho_r_arr-rho_l_arr
    wave_1_mag_arr=drho_aver_arr-dp_aver_arr/c_aver_arr**2
    wave_2_mag_arr=du_aver_arr+dp_aver_arr/(rho_aver_arr*c_aver_arr)
    wave_3_mag_arr=du_aver_arr-dp_aver_arr/(rho_aver_arr*c_aver_arr)
    tmp_mat=tile(lambda_1_aver_arr,(3,1)).T * tile(wave_1_mag_arr,(3,1)).T * eigvec_1_aver_mat
    tmp_mat+=tile(lambda_2_aver_arr,(3,1)).T * tile(wave_2_mag_arr,(3,1)).T * eigvec_2_aver_mat
    tmp_mat+=tile(lambda_3_aver_arr,(3,1)).T * tile(wave_3_mag_arr,(3,1)).T * eigvec_3_aver_mat
    F_mat=0.5*(F_l_mat+F_r_mat-tmp_mat)
    return F_mat

def getAUSMPlusFlux(in_U_l_mat,in_U_r_mat):
    from numpy import minimum,maximum,abs,sqrt,zeros,sign,tile,reshape
    V_l_mat=U2V_mat2(in_U_l_mat)
    V_r_mat=U2V_mat2(in_U_r_mat)
    c_l_sonic_arr=(GAMMA*V_l_mat[:,2]/V_l_mat[:,0])/(GAMMA-1)+0.5*V_l_mat[:,1]**2
    c_l_sonic_arr=sqrt(c_l_sonic_arr*2*(GAMMA-1)/(GAMMA+1))
    # c_l_sonic_arr will never be 0. So I do not need to worry about abs(V_l_mat[:,1])
    c_l_tilde_arr=c_l_sonic_arr**2/maximum(c_l_sonic_arr,abs(V_l_mat[:,1]))
    c_r_sonic_arr=(GAMMA*V_r_mat[:,2]/V_r_mat[:,0])/(GAMMA-1)+0.5*V_r_mat[:,1]**2
    c_r_sonic_arr=sqrt(c_r_sonic_arr*2*(GAMMA-1)/(GAMMA+1))
    c_r_tilde_arr=c_r_sonic_arr**2/maximum(c_r_sonic_arr,abs(V_r_mat[:,1]))
    c_arr=minimum(c_l_tilde_arr,c_r_tilde_arr)
    Ma_l_arr=V_l_mat[:,1]/c_arr
    Ma_r_arr=V_r_mat[:,1]/c_arr
    # P_ma
    alpha=0.1875
    P_ma_l_up_arr=zeros(Ma_l_arr.shape)
    idx_sup_arr=abs(Ma_l_arr)>=1.0
    P_ma_l_up_arr[idx_sup_arr]=0.5*(1+sign(Ma_l_arr[idx_sup_arr]))
    idx_sub_arr=abs(Ma_l_arr)<1.0
    P_ma_l_up_arr[idx_sub_arr]=0.25*(Ma_l_arr[idx_sub_arr]+1.0)**2*(2.0-Ma_l_arr[idx_sub_arr])\
        +alpha*Ma_l_arr[idx_sub_arr]*(Ma_l_arr[idx_sub_arr]**2-1.0)**2
    P_ma_r_down_arr=zeros(Ma_r_arr.shape)
    idx_sup_arr=abs(Ma_r_arr)>=1.0
    P_ma_r_down_arr[idx_sup_arr]=0.5*(1-sign(Ma_r_arr[idx_sup_arr]))
    idx_sub_arr=abs(Ma_r_arr)<1.0
    P_ma_r_down_arr[idx_sub_arr]=0.25*(Ma_r_arr[idx_sub_arr]-1.0)**2*(2.0+Ma_r_arr[idx_sub_arr])\
        -alpha*Ma_r_arr[idx_sub_arr]*(Ma_r_arr[idx_sub_arr]**2-1.0)**2
    P_arr=V_l_mat[:,2]*P_ma_l_up_arr + V_r_mat[:,2]*P_ma_r_down_arr
    P_mat=zeros(in_U_l_mat.shape)
    P_mat[:,1]=P_arr
    # M_ma
    beta=0.125
    M_ma_l_up_arr=zeros(Ma_l_arr.shape)
    M_ma_r_down_arr=zeros(Ma_l_arr.shape)
    idx_sup_arr=abs(Ma_l_arr)>1.0
    M_ma_l_up_arr[idx_sup_arr]=0.5*(Ma_l_arr[idx_sup_arr]+abs(Ma_l_arr[idx_sup_arr]))
    idx_sub_arr=abs(Ma_l_arr)<=1.0
    M_ma_l_up_arr[idx_sub_arr]=0.25*(Ma_l_arr[idx_sub_arr]+1.0)**2\
        +beta*(Ma_l_arr[idx_sub_arr]**2-1.0)**2
    idx_sup_arr=abs(Ma_r_arr)>1.0
    M_ma_r_down_arr[idx_sup_arr]=0.5*(Ma_r_arr[idx_sup_arr]-abs(Ma_r_arr[idx_sup_arr]))
    idx_sub_arr=abs(Ma_r_arr)<=1.0
    M_ma_r_down_arr[idx_sub_arr]=-0.25*(Ma_r_arr[idx_sub_arr]-1.0)**2\
        -beta*(Ma_r_arr[idx_sub_arr]**2-1.0)**2
    M_arr=M_ma_l_up_arr+M_ma_r_down_arr
    M_ma_up_arr=0.5*(M_arr+abs(M_arr))
    M_ma_down_arr=0.5*(M_arr-abs(M_arr))
    F_l_mat=getFlux_mat2(in_U_l_mat)
    F_r_mat=getFlux_mat2(in_U_r_mat)
    Phi_l_mat=zeros(in_U_l_mat.shape)
    Phi_l_mat[:,0]=V_l_mat[:,0]
    Phi_l_mat[:,1]=in_U_l_mat[:,1]
    Phi_l_mat[:,2]=(in_U_l_mat[:,2]+V_l_mat[:,2])
    Phi_r_mat=zeros(in_U_r_mat.shape)
    Phi_r_mat[:,0]=V_r_mat[:,0]
    Phi_r_mat[:,1]=in_U_r_mat[:,1]
    Phi_r_mat[:,2]=(in_U_r_mat[:,2]+V_r_mat[:,2])
    Phi_mat=tile(reshape(M_ma_up_arr,(M_ma_up_arr.size,1)),(1,3))*Phi_l_mat \
           +tile(reshape(M_ma_down_arr,(M_ma_down_arr.size,1)),(1,3))*Phi_r_mat
    Fc_mat=tile(reshape(c_arr,(c_arr.size,1)),(1,3))*Phi_mat
    F_mat=Fc_mat+P_mat
    return F_mat

