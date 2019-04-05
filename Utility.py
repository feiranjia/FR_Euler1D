def minmod(in_mat):
    # in_mat is 2D matrix: [n_arr, n_pts_in_arr]
    from numpy import sign,zeros,sum,abs,where,around,amin
    n_arr=in_mat.shape[0]
    sign_mat=sign(around(in_mat,decimals=12))
    sign_flag_mat=abs(sum(sign_mat,axis=0))
    update_flag_mat=where(sign_flag_mat==n_arr)[0]
    out_mat=zeros(sign_flag_mat.shape)
    if(update_flag_mat.size==0):
        return out_mat
    else:
        out_mat[update_flag_mat]=sign_mat[0,update_flag_mat]*amin(abs(in_mat[:,update_flag_mat]),axis=0)
    return out_mat

def minmodTVB(in_mat,in_M,in_h):
    from numpy import abs
    idx_arr=abs(in_mat[0,:])>in_M*in_h**2
    out_mat=in_mat[0,:]
    out_mat[idx_arr]=minmod(in_mat[:,idx_arr])
    return out_mat

