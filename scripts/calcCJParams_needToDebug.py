#Given JWL EOS parameters, calculate and output the CJ parameters.
#Haena Lee, October 2025

import numpy as np
from scipy.optimize import fsolve

#Calculate and return the JWL pressure
def p_jwl(rho, e, A, B, R1, R2, omega, rho0):
    t1 = A*(1.0-(omega*rho)/(R1*rho0)) * np.exp(-R1*rho0/rho)
    t2 = B*(1.0-(omega*rho)/(R2*rho0)) * np.exp(-R2*rho0/rho)
    t3 = omega*rho*e
    return t1+t2+t3

#Need to check this method
def sound_speed_squared_numeric(rho, e, A, B, R1, R2, omega, rho0, rel_eps=1e-6):
    # Approximate c^2 ≈ (dp/drho)_approx (finite difference keeping e constant)
    # NOTE: this is a practical approximation; see notes below for isentropic accuracy.
    dr = max(abs(rho)*rel_eps, 1e-12)
    p_plus = p_jwl(rho+dr, e, A, B, R1, R2, omega, rho0)
    p_minus = p_jwl(rho-dr, e, A, B, R1, R2, omega, rho0)
    dp_drho = (p_plus-p_minus)/(2.0*dr)
    # ensure non-negative
    return max(dp_drho, 0.0)

#Calculate and return residuals
def residuals_PU(x, params):
    P_CJ, U_CJ = x
    rho0, p0, e0, u0, A, B, R1, R2, omega = params
    u_CJ = u0 + ((P_CJ - p0)/(U_CJ - u0))

    denom = U_CJ - u_CJ
    if denom <= 0:  #unphysical, return large residuals to steer solver away
        return [1e12, 1e12]
    rho_CJ = rho0*((U_CJ-u0)/denom)

    e_CJ = e0 + ((P_CJ+p0)/2) * (1.0/rho0 - 1.0/rho_CJ)
    P_JWL = p_jwl(rho_CJ, e_CJ, A, B, R1, R2, omega, rho0) #the JWL pressure at the calculated state

    #residuals
    r1 = P_CJ-P_JWL
    c2 = sound_speed_squared_numeric(rho_CJ, e_CJ, A, B, R1, R2, omega, rho0)
    r2 = c2-(U_CJ-u_CJ)**2
    return [r1, r2]

#Use fsolve to calculate CJ parameters given JWL parameters
def compute_CJ(jwl_params, initial_guess=None):
    A = jwl_params['A']; B = jwl_params['B']; R1 = jwl_params['R1']; R2 = jwl_params['R2']
    omega = jwl_params['omega']; rho0 = jwl_params['rho0']; e0 = jwl_params.get('e0', 0.0)
    p0 = jwl_params.get('p0', 0.0)
    u0 = jwl_params.get('u0', 0.0)
    params = (rho0, p0, e0, u0, A, B, R1, R2, omega)

    if initial_guess is None:
        #typical explosive CJ pressure & speed magnitudes
        P_guess = 3     #Mbar
        U_guess = 0.6   #cm/microsecond
    else:
        P_guess, U_guess = initial_guess

    #Need to check this fsolve
    sol, infodict, ier, mesg = fsolve(residuals_PU, x0=[P_guess, U_guess], args=(params,), full_output=True, xtol=1e-8)
    if ier != 1:
        raise RuntimeError("CJ solver did not converge: " + mesg)

    P_CJ, U_CJ = sol
    #calculate full state again
    u_CJ = u0 + ((P_CJ-p0)/(U_CJ - u0))
    rho_CJ = rho0 * ((U_CJ-u0)/(U_CJ - u_CJ))
    e_CJ = e0 + ((P_CJ+p0)/2) * (1.0/rho0 - 1.0/rho_CJ)
    c_CJ = np.sqrt(sound_speed_squared_numeric(rho_CJ, e_CJ, A, B, R1, R2, omega, rho0))

    return {
        'P_CJ': P_CJ,
        'U_CJ': U_CJ,
        'u_CJ': u_CJ,
        'rho_CJ': rho_CJ,
        'e_CJ': e_CJ,
        'c_CJ': c_CJ
    }

#Fill out; make sure units are consistent (using: cm, g, microsecond)
if __name__ == "__main__":
    jwl = {
        'A': 3.71,      #Mbar
        'B': 3.23e-2,
        'R1': 4.15,
        'R2': 0.95,
        'omega': 0.30,
        'e0': 4.3e-2,   #(g*cm^2)/microsecond^2
        'rho0': 1.63,   #g/cm^3
        'u0': 0.0,
        'p0': 0
    }
    out = compute_CJ(jwl, initial_guess=(0.3, 0.8))  #P_CJ, U_CJ
    print("CJ result:")
    for k, v in out.items():
        if isinstance(v, (float, np.floating)):
            print(f" {k:7s}: {v:.2e}")
        else:
            print(f" {k:7s}: {v}")
