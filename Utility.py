def minmod(in_mat):
    from numpy import sign,zeros,sum,abs,where,around,amin
    n_arr=in_mat.shape[0]
    sign_mat=sign(around(in_mat,decimals=12))
    sign_flag_mat=abs(sum(sign_mat,axis=0))
    update_flag_mat_list=where(sign_flag_mat==n_arr)
    update_flag_mat1=update_flag_mat_list[0]
    update_flag_mat2=update_flag_mat_list[1]
    out_mat=zeros(sign_flag_mat.shape)
    if(update_flag_mat1.size==0):
        return out_mat
    else:
        for i in range(update_flag_mat1.size):
            j1=update_flag_mat1[i]
            j2=update_flag_mat2[i]
            out_mat[j1,j2]=sign_mat[0,j1,j2]*amin(abs(in_mat[:,j1,j2]),axis=0)
    return out_mat

