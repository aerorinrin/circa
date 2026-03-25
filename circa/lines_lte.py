# FUNCTION:     Compute population distributions and linestrengths for molecules in 
#               Local Thermodynamic Equilibrium (LTE).
# AUTHOR:       I.C. Dumitrescu

# Import external libraries
import numpy as np
from scipy.constants import c


####### CONSTANTS #######
c = c*100                   # [cm s^-1] speed of light
C2 = 1.4387769              # [cm K] C2 = h*c/k
P_ATM = 101325.             # [Pa] atmosphere to Pascal conversion
T_REF = 296.                # [K]
h = 6.62607015e-34          # [J s] Planck's constant


####### LINE STRENGTH & POPULATION DISTRIBUTIONS #######

# Linestrength equilibrium conditions (see Simeckova et al., 2006)
def lines_lte(nu_j, E_l, S_ref, T, Q, Q_ref):
    S_j = S_ref * (Q_ref / Q) * np.exp( - C2 * E_l * (1/T - 1/T_REF) ) * \
          ( (1-np.exp(-C2*nu_j/T)) / (1-np.exp(-C2*nu_j/T_REF) ) )
    return S_j

# Rovib state populations for LTE conditions
def population_lte(g, Q, E, T, I_a):
    exponent = - C2* E / T
    return I_a * (g/Q) * np.exp(exponent)

