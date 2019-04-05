###########################################################
# Governing equation is the convection equation.
#                       u_t = -f_x, f = f(u)
###########################################################
#==========================================================
def RK1(dt, in_eq):
    #######################################################
    # u^{n+1} = u^{n} - \DeltaT * dF, dF = f_x
    #######################################################
    #  in_eq.updateBeforeRes()
    res_mat = in_eq.getRes()
    in_eq.U_sp_mat = in_eq.U_sp_mat - dt * res_mat
    in_eq.updateAfterRes()

def RK2(dt, in_eq):
    from numpy import copy
    U_sp_mat_old = copy(in_eq.U_sp_mat)
    # 1st stage
    #  in_eq.updateBeforeRes()
    res_mat = in_eq.getRes()
    in_eq.U_sp_mat = in_eq.U_sp_mat - dt * res_mat
    in_eq.updateAfterRes()
    # 2nd Stage
    #  in_eq.updateBeforeRes()
    res_mat = in_eq.getRes()
    in_eq.U_sp_mat = in_eq.U_sp_mat - dt * res_mat
    in_eq.U_sp_mat = 0.5*(in_eq.U_sp_mat + U_sp_mat_old)
    in_eq.updateAfterRes()

def RK3(dt, in_eq):
    from numpy import copy
    U_sp_mat_old = copy(in_eq.U_sp_mat)
    # 1st stage
    #  in_eq.updateBeforeRes()
    res_mat = in_eq.getRes()
    in_eq.U_sp_mat = U_sp_mat_old - dt * res_mat
    in_eq.updateAfterRes()
    # 2nd Stage
    #  in_eq.updateBeforeRes()
    res_mat = in_eq.getRes()
    in_eq.U_sp_mat = 0.75*U_sp_mat_old + 0.25*(in_eq.U_sp_mat-dt*res_mat)
    in_eq.updateAfterRes()
    # 3rd stage
    #  in_eq.updateBeforeRes()
    res_mat = in_eq.getRes()
    in_eq.U_sp_mat = (U_sp_mat_old + 2.0*(in_eq.U_sp_mat-dt*res_mat))/3.0
    in_eq.updateAfterRes()

#  #==========================================================
#  def RungeKutta54_LS(U, TimeStep, Eq, Mesh, Poly, t):
    #  from numpy import zeros
    #  #######################################################
    #  # Reference:
    #  # 1994, ...
    #  #######################################################
    #  # Low storage Runge-Kutta coefficients
    #  CoefA_Vec = [0.0, \
        #  -567301805773.0/1357537059087.0, \
        #  -2404267990393.0/2016746695238.0, \
        #  -3550918686646.0/2091501179385.0, \
        #  -1275806237668.0/842570457699.0]
    #  CoefB_Vec = [1432997174477.0/9575080441755.0, \
        #  5161836677717.0/13612068292357.0, \
        #  1720146321549.0/2090206949498.0, \
        #  3134564353537.0/4481467310338.0, \
        #  2277821191437.0/14882151754819.0]
    #  CoefC_Vec = [0.0, \
        #  1432997174477.0/9575080441755.0, \
        #  2526269341429.0/6820363962896.0, \
        #  2006345519317.0/3224310063776.0, \
        #  2802321613138.0/2924317926251.0]
    #  ResU = zeros(U.shape)
    #  for Ind in range(5):
        #  dF = Eq.getdF(Eq.ArtDiffuFlag, Mesh, Poly)
        #  ResU = CoefA_Vec[Ind] * ResU - TimeStep * (-dF)
        #  U = U - CoefB_Vec[Ind] * ResU
        #  Eq.update(U, t)
    #  return U
