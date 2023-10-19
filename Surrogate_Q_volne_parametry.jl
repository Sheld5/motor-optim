module ModelParam
using LinearSolve
using Distributions


y_max() = [600., Inf]

function domain()  # nk, dk, Ds, Q
    return [20, 0.01, 0.374, 0.3], [80, 0.03, 0.520, 0.8]
end

discrete_dims() = [true, false, false, false]

param_range() = [0.01, 0.1, 0.1], [0.05, 0.2, 1.]


# - - - - - - MODEL - - - - - -

const D1 = 0.41
const D2 = 0.52

const _A1 = 0.022
const _A2 = 0.2
const _A3 = 0.67

struct Param{NUM<:Real}
    nk::NUM
    dk::NUM
    D1::NUM
    D2::NUM
    alfa_0::NUM
    lam_fe::NUM
    l::NUM
    Q::NUM
    t::NUM
    alt::NUM
    Ok::NUM
    Sk::NUM
    A1::NUM
    A2::NUM
    A3::NUM
    Pressure::NUM
    Density::NUM
    Viscosity::NUM
    cv::NUM
    lam::NUM
    k_alf::NUM


    function Param(nk, dk, D1, D2, l, Q, t, alt, alfa_0, lam_fe, A1, A2, A3)
        Ok = π * dk
        Sk = π/4 * dk^2

        Pressure, Density, Viscosity, cv, lam, k_alf = material_properties(t, alt, A1, A2, A3)
        
        fields = [nk, dk, D1, D2, alfa_0, lam_fe, l, Q, t, alt, Ok, Sk, A1, A2, A3,
            Pressure, Density, Viscosity, cv, lam, k_alf]
        NUM = promote_type(typeof.(fields)...)
        new{NUM}(fields...)
    end
end

function material_properties(t, alt, A1, A2, A3)
    # VentCalc Properties
    Pressure = 101325 * (((273.15 + t) - 0.0065 * alt) / (273.15 + t)) ^ 5.2559
    Density = 1.276 / (1 + 0.00366 * t) * Pressure / 101325
    Viscosity = 9.81 * (1.478 * 10 ^ -7 * (273.15 + t) ^ 0.5) / (1 + 110.4 / (273.15 + t)) / Density

    # TempCalc Properties
    cv = (0.0116 * t ^ 2 - 4.5615 * t
               + 1299.7) * (Pressure / 101325)
    lam = 0.0243 * (1 + 0.00306 * t)
    k_alf = A1 * cv * Viscosity ^ A2 * (cv * Viscosity / lam) ^ (-A3)

    return Pressure, Density, Viscosity, cv, lam, k_alf
end
            
module VentCalc

    function r_cont(D1, D2, Sk, nk, dens)

        S1 = π/4*(D2^2 - D1^2)
        S2 = Sk * nk

        mi = 12.174 * (S2/S1) ^ 6 - 36.685 * (S2/S1) ^ 5 + 44.366 * (S2/S1) ^ 4 -
            27.069 * (S2/S1) ^ 3 + 8.7337 *
            (S2/S1) ^ 2 - 1.2192*(S2/S1) + 0.6797

        if mi > 1
            mi = 1
        end

        Ksi_cont = (1 / mi - 1) ^ 2
        K_cont = 0.5 * dens * (Ksi_cont + 1 - (S2 / S1) ^ 2) / S2 ^ 2

        return K_cont
    end

    function r_duct(Sk, dk, nk, l, Re, dens)
        S = Sk * nk
        e = 1.2e-04 / dk  # 1e-04
        f = 0.02 #0.02
        it = 0
        F_ = (1 / (1.74 - 2 * log10(2 * e + 18.7 / (Re * sqrt(f))))) ^ 2
        while (abs(F_ - f) > 0.0001)
            f = F_
            F_ = (1 / (1.74 - 2 * log10(2 * e + 18.7 / (Re * sqrt(f))))) ^ 2

            if it >= 100
                break
            else
                it += 1
            end
        end

        coeff_duct = f * l / dk
        K_duct = 0.5 * dens * coeff_duct / S ^ 2

        return K_duct
    end

    function r_exp(D1, D2, nk, Sk, dens)
        S1 = Sk
        S2 = π/4*(D1 ^ 2 - D2 ^ 2)
        K_exp = 0.5 * dens * (1 / S1 - 1 / S2) ^ 2

        return K_exp
    end

    function frict(Re, dk, e)
        K = e / dk
        L = 0.04
        L2 = (1 / (1.74 - 2 * log10(2 * K + 18.7 / (Re * L ^ (0.5))))) ^ 2
        while (abs(L2 - L) > 0.01) # 0.0001
            L = L2
            L2 = (1 / (1.74 - 2 * log10(2 * K + 18.7 / (Re * L ^ (0.5))))) ^ 2
        end

        return L
    end
end

module TempCalc

    function g_cond(lam, L, S)
        G = (lam*S) / L
        return G
    end

    function g_conv(alfa, S)
        G = alfa * S
        return G
    end

    function heat_tranfer(dk, l, v, fl, fe, Kalf)
        Kl = 1+(dk/l)^0.67
        alfa_duct = Kalf * v^0.8 * dk^(-0.2) * Kl * sqrt(fe/fl)

        return alfa_duct
    end
end

function check_feas(nk, dk, Ds, Q=nothing)
    # Constrains
    duct_gap = 0.003
    D1_gap = 0.003
    D2_gap = 0.003
    
    hju = (D2 - Ds) / 2
    hjd = (Ds - D1) / 2

    const_gap = ( ( Ds / nk )*π ) - (dk + duct_gap)
    const_D2 = hju - (dk/2 +  D2_gap)
    const_D1 = hjd - (dk/2  + D1_gap)

    # return (const_gap > 0) && (const_D1 > 0) && (const_D2 > 0)
    return const_gap, const_D1, const_D2
end

function calc(nk, dk, Ds, Q, A1=_A1, A2=_A2, A3=_A3)
    # Input values
    l = 0.23
    t = 30
    alt = 325
    alfa_0 = 16
    lam_fe = 29
    
    # feas = check_feas(nk, dk, Ds)
 
    par = Param(nk, dk, D1, D2, l, Q, t, alt, alfa_0, lam_fe, A1, A2, A3)
    Pl = 5000  # Power loss
    # V = VentCalc()
    # T = TempCalc()
    
    # init Values
    dP_ = 100
    dP = 0
    Vs = 0.1
    Re = 0.

    # Vent calculation
    while abs(dP_ - dP) > 1
        dP_ = dP

        K_cont = VentCalc.r_cont(par.D1, par.D2, par.nk, par.Sk, par.Density)
        Re = Vs * par.dk / par.Viscosity
        K_duct = VentCalc.r_duct(par.Sk,par.dk, par.nk, par.l, Re, par.Density)
        K_exp = VentCalc.r_exp(par.D1, par.D2, par.Sk, par.nk, par.Density)
        Ks = K_cont + K_duct + K_exp

        dP = Ks * par.Q ^ 2  # Pressure drop
        Vs = sqrt(dP / Ks) / (par.Sk *  par.nk)  # Mean velocity
    end
        

    #Temp calculation

    # Heat transfer coefficient calculation for duct
    fl = VentCalc.frict(Re, par.dk, 0.00001)
    fe = VentCalc.frict(Re, par.dk, 0.0001)
    alfa_duct = TempCalc.heat_tranfer(par.dk, par.l, Vs, fl, fe, par.k_alf)


    # Geom for stator radial conductivity
    h1 = (Ds - par.D1) / 2
    h2 = (par.D2 - Ds) / 2

    Sgp1 = π * (Ds+par.D1)/2 * par.l * par.lam_fe
    Sgp2 = π * (Ds+par.D2)/2 * par.l * par.lam_fe
    Sgd = par.Ok * par.nk * par.l
    S0 = π * par.D2 * par.l

    Gp1 = TempCalc.g_cond(par.lam_fe, h1, Sgp1)
    Gp2 = TempCalc.g_cond(par.lam_fe, h2, Sgp2)
    Gd = TempCalc.g_conv(alfa_duct, Sgd)
    G0 = TempCalc.g_conv(par.alfa_0, S0)

    Gm = [
        Gp1   -Gp1        0
        -Gp1  Gp1+Gp2+Gd  -Gp2
        0     -Gp2        Gp2+G0
    ]
    Pm = [Pl, Gd*30, G0*30]
    T = solve(LinearProblem(Gm, Pm))
    T_av = mean(T)

    return [dP, T_av]
end

end # ModelParam


# if __name__ == "__main__":
    
#     A1 = 0.01 # 0.01 - 0.05 - vliv na teplotu
#     A2 =  0.15 # 0.1 - 0.2 - vliv na teplotu
#     A3 = 0.1 # 0.1 - 1 - vliv na teplotu
    
#     nk = 45 
#     dk = 0.03
#     Ds = 0.48 
#     Q = 0.3
    
#     print(calc(nk, dk, Ds, Q, A1, A2, A3))
