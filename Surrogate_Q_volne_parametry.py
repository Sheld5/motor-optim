import numpy as np

class Param():

    def __init__(self, nk, dk, D1, D2, l, Q, t, alt, alfa_0, lam_fe, A1, A2, A3):
        self.nk = nk
        self.dk = dk
        self.D1 = D1
        self.D2 = D2
        self.alfa_0 = alfa_0
        self.lam_fe = lam_fe
        self.l = l
        self.Q = Q
        self.t = t
        self.alt = alt
        self.Ok = np.pi*self.dk
        self.Sk = np.pi/4*self.dk**2
        
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        
        self.material_properties()


    def material_properties(self):
        # VentCalc Properties
        self.Pressure = 101325 * \
            (((273.15 + self.t) - 0.0065 * self.alt) / (273.15 + self.t)) ** 5.2559
        self.Density = 1.276 / (1 + 0.00366 * self.t) * \
            self.Pressure / 101325
        self.Viscosity = 9.81 * (1.478 * 10 ** -7 * (273.15 + self.t)
                                 ** 0.5) / (1 + 110.4 / (273.15 + self.t)) / self.Density

        # TempCalc Properties
        self.cv = (0.0116 * self.t ** 2 - 4.5615 * self.t
                   + 1299.7) * (self.Pressure / 101325)
        self.lam = 0.0243 * (1 + 0.00306 * self.t)
        self.k_alf = self.A1 * self.cv * self.Viscosity ** self.A2 \
            * (self.cv * self.Viscosity / self.lam) ** (-self.A3)
            
class VentCalc():

    def r_cont(self, D1, D2, Sk, nk, dens):

        S1 = np.pi/4*(D2**2 - D1**2)
        S2 = Sk * nk

        mi = 12.174 * (S2/S1) ** 6 - 36.685 * (S2/S1) ** 5 + 44.366 * (S2/S1) ** 4 - \
            27.069 * (S2/S1) ** 3 + 8.7337 * \
            (S2/S1) ** 2 - 1.2192*(S2/S1) + 0.6797

        if mi > 1:
            mi = 1

        Ksi_cont = (1 / mi - 1) ** 2
        K_cont = 0.5 * dens * (Ksi_cont + 1 - (S2 / S1) ** 2) / S2 ** 2

        return K_cont

    def r_duct(self, Sk, dk, nk, l, Re, dens):
        S = Sk * nk
        e = 1.2e-04 / dk  # 1e-04
        f = 0.02 #0.02
        it = 0
        F_ = (1 / (1.74 - 2 * np.log10(2 * e + 18.7 / (Re * np.sqrt(f))))) ** 2
        while (abs(F_ - f) > 0.0001):
            f = F_
            F_ = (1 / (1.74 - 2 * np.log10(2 * e + 18.7 / (Re * np.sqrt(f))))) ** 2

            if it >= 100:
                break
            else:
                it += 1

        coeff_duct = f * l / dk
        K_duct = 0.5 * dens * coeff_duct / S ** 2

        return K_duct

    def r_exp(self, D1, D2, nk, Sk, dens):
        S1 = Sk
        S2 = np.pi/4*(D1 ** 2 - D2 ** 2)
        K_exp = 0.5 * dens * (1 / S1 - 1 / S2) ** 2

        return K_exp

    def frict(self, Re, dk, e):
        K = e / dk
        L = 0.04
        L2 = (1 / (1.74 - 2 * np.log10(2 * K + 18.7 / (Re * L ** (0.5))))) ** 2
        while (abs(L2 - L) > 0.01): # 0.0001
            L = L2
            L2 = (1 / (1.74 - 2 * np.log10(2 * K + 18.7 / (Re * L ** (0.5))))) ** 2

        return L

class TempCalc():

    def g_cond(self, lam, L, S):
        G = (lam*S) / L
        return G

    def g_conv(self, alfa, S):
        G = alfa * S
        return G

    def heat_tranfer(self, dk, l, v, fl, fe, Kalf):
        Kl = 1+(dk/l)**0.67
        alfa_duct = Kalf * v**0.8 * dk**(-0.2) * Kl * np.sqrt(fe/fl)

        return alfa_duct
    
def check_feas(D1, D2, nk, dk, Ds):
    # Constrains
    duct_gap = 0.003
    D1_gap = 0.003
    D2_gap = 0.003
    
    hju = (D2 - Ds) / 2
    hjd = (Ds - D1) / 2

    const_gap = ( ( Ds / nk )*np.pi ) - (dk + duct_gap)
    const_D2 = hju - (dk/2 +  D2_gap)
    const_D1 = hjd - (dk/2  + D1_gap)

    print(const_gap, const_D1, const_D2)
    
    if  (const_gap or const_D1 or const_D2) <= 0:
        return "Unfeasible"      

    else:
        return "Feasible"

def calc(nk, dk, Ds, Q, A1=0.022, A2=0.2, A3=0.67):
    # Input values
    D1 = 0.41
    D2 = 0.52
    l = 0.23
    t = 30
    alt = 325
    alfa_0 = 16
    lam_fe = 29
    

    feas = check_feas(D1, D2, nk, dk, Ds)
 
    par = Param(nk, dk, D1, D2, l, Q, t, alt, alfa_0, lam_fe, A1, A2, A3)
    Pl = 5000  # Power loss
    V = VentCalc()
    T = TempCalc()
    
    # init Values
    dP_ = 100
    dP = 0
    Vs = 0.1

    # Vent calculation
    while abs(dP_ - dP) > 1:
        dP_ = dP

        K_cont = V.r_cont(par.D1, par.D2, par.nk, par.Sk, par.Density)
        Re = Vs * par.dk / par.Viscosity
        K_duct = V.r_duct(par.Sk,par.dk, par.nk, par.l, Re, par.Density)
        K_exp = V.r_exp(par.D1, par.D2, par.Sk, par.nk, par.Density)
        Ks = K_cont + K_duct + K_exp

        dP = Ks * par.Q ** 2  # Pressure drop
        Vs = np.sqrt(dP / Ks) / (par.Sk *  par.nk)  # Mean velocity
        

    #Temp calculation

    # Heat transfer coefficient calculation for duct
    fl = V.frict(Re, par.dk, 0.00001)
    fe = V.frict(Re, par.dk, 0.0001)
    alfa_duct = T.heat_tranfer(par.dk, par.l, Vs, fl, fe, par.k_alf)


    # Geom for stator radial conductivity
    h1 = (Ds - par.D1) / 2
    h2 = (par.D2 - Ds) / 2

    Sgp1 = np.pi * (Ds+par.D1)/2 * par.l * par.lam_fe
    Sgp2 = np.pi * (Ds+par.D2)/2 * par.l * par.lam_fe
    Sgd = par.Ok * par.nk * par.l
    S0 = np.pi * par.D2 * par.l

    Gp1 = T.g_cond(par.lam_fe, h1, Sgp1)
    Gp2 = T.g_cond(par.lam_fe, h2, Sgp2)
    Gd = T.g_conv(alfa_duct, Sgd)
    G0 = T.g_conv(par.alfa_0, S0)

    Gm = np.array([[Gp1,   -Gp1,   0],
                   [-Gp1, Gp1+Gp2+Gd, -Gp2],
                   [0,   -Gp2,  Gp2+G0],
                   ])
    Pm = np.array([Pl, Gd*30, G0*30])
    T = np.linalg.solve(Gm, Pm)
    T_av = np.mean(T)

    return [dP, T_av, feas]


if __name__ == "__main__":
    
    A1 = 0.01 # 0.01 - 0.05 - vliv na teplotu
    A2 =  0.15 # 0.1 - 0.2 - vliv na teplotu
    A3 = 0.1 # 0.1 - 1 - vliv na teplotu
    
    nk = 45 
    dk = 0.03
    Ds = 0.48 
    Q = 0.3
    
    print(calc(nk, dk, Ds, Q, A1, A2, A3))
