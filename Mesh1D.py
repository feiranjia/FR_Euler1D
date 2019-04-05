def GaussLegendreRootsWeights(in_order):
    from numpy import arange,sqrt,diag,argsort
    from numpy.linalg import eig
    idx_arr=arange(1,in_order+1)
    beta_arr=idx_arr/(sqrt(4*idx_arr**2-1))
    T_mat=(diag(beta_arr,1)+diag(beta_arr,-1))
    [eigval_arr,eigvec_mat]=eig(T_mat)
    idx_sort=argsort(eigval_arr)
    roots_arr=eigval_arr[idx_sort]
    eigvec_mat=eigvec_mat[:,idx_sort]
    weights_arr=2*eigvec_mat[0,:]**2
    return roots_arr,weights_arr

class Mesh1D(object):

    def __init__(self, Range, OrderNMAX, CellNMAX, QuadType):
        from Poly import Poly
        from numpy import dot
        self.Range = Range
        self.PolyOrder = OrderNMAX
        self.NodeInCellNMAX = OrderNMAX + 1
        self.CellNMAX = CellNMAX
        self.QuadType = QuadType
        self.CellSize = self.getCellSize()
        self.FluxPts_Mat = self.getFluxPoints()
        self.CellCenter_Vec = self.getCellCenter()
        self.Jacobian = self.getJacobian()
        self.GloCoor_Mat = self.getGlobalCoordinates()
        self.SolPts_Vec = self.getSolutionPoints()
        self.Weights_Vec = self.getWeights()
        poly = Poly(self.SolPts_Vec)
        self.lag_dpolydx_mat = poly.getLagrangePolyDeriv() / self.Jacobian
        self.radau_r_dpolydx_val_arr = poly.getRadauRightPolyDeriv() / self.Jacobian
        self.radau_l_dpolydx_val_arr = poly.getRadauLeftPolyDeriv() / self.Jacobian
        self.lag_poly4end_mat = poly.getLagrangePoly4End()

    def getCellSize(self):
        CellSize = (self.Range[1] - self.Range[0]) / self.CellNMAX
        return CellSize

    def getFluxPoints(self):
        from numpy import linspace,zeros
        Face_Vec = linspace(self.Range[0], self.Range[1], self.CellNMAX+1)
        FluxPts_Mat = zeros((2,Face_Vec[0:-1].size))
        FluxPts_Mat[0,:] = Face_Vec[0:-1]
        FluxPts_Mat[1,:] = Face_Vec[1:]
        return FluxPts_Mat

    def getCellCenter(self):
        FluxPts_Mat = self.getFluxPoints()
        CellCenter_Vec = ( FluxPts_Mat[0,:] + FluxPts_Mat[1,:] ) / 2.0
        return CellCenter_Vec

    def getJacobian(self):
        # PROBLEM
        Jacobian = self.getCellSize() / 2.0
        return Jacobian

    #  def getSolutionPoints(self):
        #  from scipy.special import legendre
        #  from numpy.polynomial.polynomial import polyroots
        #  from numpy import zeros
        #  # Currently, there is only 1 type available.
        #  # LGL
        #  ###########################################################
        #  # Construct legendre polynomials of order OrderNMAX
        #  Poly = legendre(self.PolyOrder+1)
        #  Poly_Vec = Poly.coeffs
        #  #  PolyDeriv = Poly.deriv(1)
        #  # Note: In numpy.polynomial.polynomial module, the first element in the
        #  #       coefficients array is the constant value. While in scipy.special
        #  #       module, the coefficient array is arranged in descending order.
        #  #  PolyDeriv_Vec = PolyDeriv.coeffs
        #  # Compute Legendre-Gauss-Lobatto integration points
        #  # LGL points are local coordinates
        #  #  SolPts_Vec = zeros((self.NodeInCellNMAX,))
        #  #  SolPts_Vec[0] = -1.0
        #  #  SolPts_Vec[-1] = 1.0
        #  #  SolPts_Vec[1:-1] = polyroots(PolyDeriv_Vec[::-1])
        #  SolPts_Vec = polyroots(Poly_Vec[::-1])
        #  return SolPts_Vec

    def getGlobalCoordinates(self):
        from numpy import meshgrid
        # Transform to global coordinates
        Temp1_Mat, Temp2_Mat = \
            meshgrid(self.getCellCenter(), self.getJacobian()*self.getSolutionPoints())
        GloCoor_Mat = Temp1_Mat + Temp2_Mat
        return GloCoor_Mat

    #  def getWeights(self):
        #  from scipy.special import eval_legendre
        #  # Compute Legendre-Gauss-Lobatto weights
        #  Weights_Vec = 2.0 / (self.PolyOrder*(self.PolyOrder+1)) \
            #  / eval_legendre(self.PolyOrder, self.getSolutionPoints())**2
        #  return Weights_Vec

    def getSolutionPoints(self):
        SolPts_Vec,comp_sp_wi_arr=GaussLegendreRootsWeights(self.PolyOrder)
        return SolPts_Vec

    def getWeights(self):
        comp_sp_loc_arr,Weights_Vec=GaussLegendreRootsWeights(self.PolyOrder)
        return Weights_Vec

