class Poly(object):

    def __init__(self, Node_Vec):
        self.Order = len(Node_Vec) - 1
        self.Nodes = Node_Vec

    def getLagrangeBasis(self):
        from numpy.polynomial.polynomial import Polynomial
        #########################
        # Construct Lagrange Basis Functions in the form of the Polynomial class
        # imported from numpy.
        # Output:
        #           List of Polynomial object. Each Polynomial object has 3 default
        #           parameters. The first is the coefficients, second is the domain,
        #           third is the window size. Details about the latter 2 parameters
        #           are in the definition of Polynomial class in numpy.
        x = self.Nodes
        PolyList_List = []
        for j in range(self.Order+1):
            Poly_Poly = 1.0
            for k in range(self.Order+1):
                if k == j:
                    continue
                Poly_Poly *= Polynomial([-x[k], 1.0]) / (x[j] - x[k])
            PolyList_List.append(Poly_Poly)
        return PolyList_List

    def getLagrangePoly4End(self):
        from numpy import zeros
        from numpy.polynomial.polynomial import polyval
        ###########################################################
        # Construct Lagrange Polynomial of order OrderNMAX
        NodesNMAX = self.Order + 1
        Basis_List = self.getLagrangeBasis()
        BasisCoef_List = [Basis_List[i].coef for i in range(NodesNMAX)]
        Basis4End_Mat = zeros((2, NodesNMAX))
        for Ind in range(NodesNMAX):
            Basis4End_Mat[:, Ind] = polyval([-1,1], BasisCoef_List[Ind])
        return Basis4End_Mat

    def getLagrangePolyDeriv(self):
        from numpy import zeros
        from numpy.polynomial.polynomial import polyval
        ###########################################################
        # Construct Lagrange Polynomial of order OrderNMAX
        NodesNMAX = self.Order + 1
        Basis_List = self.getLagrangeBasis()
        BasisDeriv_List = \
            [Basis_List[i].deriv(1) for i in range(NodesNMAX)]
        BasisDerivCoef_List = \
            [BasisDeriv_List[i].coef for i in range(NodesNMAX)]
        BasisDeriv_Mat = zeros((NodesNMAX, NodesNMAX))
        for Ind in range(NodesNMAX):
            BasisDeriv_Mat[:, Ind] = \
                polyval(self.Nodes, BasisDerivCoef_List[Ind])
        return BasisDeriv_Mat

    def getRadauRightPoly(self):
        from scipy.special import legendre
        from numpy.polynomial.polynomial import polyval
        from numpy import insert
        ###########################################################
        # Construct Right Radau Polynomial
        Temp1 = legendre(self.Order).coeffs
        Temp2 = legendre(self.Order+1).coeffs
        RadauRPoly_Vec = (-1)**self.Order / 2.0 * (insert(Temp1, 0, 0) - Temp2)
        RadauRPolyValue_Vec = polyval(self.Nodes, RadauR_Vec[::-1])
        return RadauRPolyValue_Vec

    def getRadauRightPolyDeriv(self):
        from scipy.special import legendre
        from numpy.polynomial.polynomial import polyval
        from numpy import insert
        ###########################################################
        # Construct Right Radau Polynomial
        Temp1 = legendre(self.Order).deriv(1).coeffs
        Temp2 = legendre(self.Order+1).deriv(1).coeffs
        RadauRPolyDeriv_Vec = (-1)**self.Order / 2.0 * (insert(Temp1, 0, 0) - Temp2)
        RadauRPolyDerivValue_Vec = polyval(self.Nodes, RadauRPolyDeriv_Vec[::-1])
        return RadauRPolyDerivValue_Vec

    def getRadauLeftPolyDeriv(self):
        RadauRPolyDerivValue_Vec = self.getRadauRightPolyDeriv()
        return -RadauRPolyDerivValue_Vec[::-1]

    def getVandermondeLegendre(self):
        from numpy.polynomial.legendre import legvander
        return legvander(self.Nodes, self.Order)
