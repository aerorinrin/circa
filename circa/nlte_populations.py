# FUNCTION:     File for computing energy levels for Non-Local Thermal Equilibrium (NLTE) simulations. Expand as needed.
# AUTHOR:       I.C. Dumitrescu
# REFERENCES:
#   - rotational degeneracy weights (gi & gs) from Šimečková et al. (2006)
#   - CO & CO2 vibrational and rotational constants from Klarenaar et al. (2017)
#   - for info on linear Bose systems, read Gamache et al. (2016) and Herzberg's IR & Raman spectra (1945)
#   - coupling constants for CO2 from Suzuki (1968)

# Import external libraries
import numpy as np
from numpy.typing import NDArray
from scipy.constants import c

####### CONSTANTS #######
c = c*100                   # [cm s^-1] speed of light
C2 = 1.4387769              # [cm K] C2 = h*c/k
T_REF = 296                 # [K] database reference temperature
h = 6.62607015e-34          # [J s] Planck's constant

# Simulation parameters
J_max = 150
v_max_CO = 35
v_max_CO2 = [ 40, 70, 30 ]


# General parameters for non-local thermodynamic equilibrium calculations
NLTE_PARAMETERS = {
    'CO': {
        1: {  # Constants for CO isotopologue 1
            'gi': 1,                # [-] state-independent weight for rotational degeneracy
            'gs': 1,                # [-] state-dependent weight for rotational degeneracy
            'G_1': 2143.24,         # [cm^-1] energy spacing between ground & first vib level
            'wexe': 13.29,          # [cm^-1] vibrational anharmonicity
            'B': 1.9225,            # [cm^-1] rotational constant
            'D': 6.121e-6,          # [cm^-1] centrifugal distortion
            'H': 5.7e-12,           # [cm^-1] third-order rotational correction factor
            'omega': 2169.81,       # [cm^-1] vibrational frequency for energy level calculation
            'E_0': 1074.9425        # [cm^-1] energy difference between first and zeroth level
        },
        2: {
            'gi': 1,
            'gs': 1,
            'G_1': 2096.03,
            'wexe': 12.70, 
            'B': 1.8380,  
            'D': 5.593e-6,  
            'H': 5.5e-12,
            'omega': 2121.43,
            'E_0': 1051.19
        },
        3: {
            'gi': 1,
            'gs': 1,
            'G_1': 2092.09,
            'wexe': 12.65, 
            'B': 1.8310,  
            'D': 5.550e-6,  
            'H': 4.9e-12,
            'omega': 2117.40,
            'E_0': 1049.2075
        },
    },
    'CO2': {
        1: {  # Constants for CO2 isotopologue 1
            'gi': 1,
            'gs': 1,
            'G_1': [1333.93, 667.47, 2349.16],      # nu_1, nu_2, nu_3
            'wexe': [2.93, -0.38, 12.47],           # nu_2, nu_2, nu_3
            'B': 0.39022,                           # rotational  energy constants
            'D': 0.1333e-6, 
            'H': 0.0090e-12,
            'omega': [1354.31, 672.85, 2396.32],    # vibrational energy: omega_1, omega_2, omega_3
            'x_1j': [-2.93, -4.61, -19.82],         # x_11, x_12, x_13
            'x_2j': [0., 1.35, -12.31],             # x_21, x_22, x_23
            'x_3j': [0., 0., -12.47],               # x_31, x_32, x_33
            'x_l2l2': -0.97,
            'E_0': 2532.25                          # energy difference between first and zeroth level

        },
        2: {
            'gi': 2,
            'gs': 1,
            'G_1': [1334.32, 648.63, 2283.49], 
            'wexe': [2.93, -0.37, 11.71], 
            'B': 0.39024, 
            'D': 0.1332e-6, 
            'H': 0.0090e-12,
            'omega': [1354.31, 653.70, 2328.12],
            'x_1j': [-2.93, -4.46, -19.33],
            'x_2j': [0., 1.28, -11.54],
            'x_3j': [0., 0., -11.71],
            'x_l2l2': -0.91,
            'E_0': 2479.7025
        },
        3: {
            'gi': 1,
            'gs': 1,
            'G_1': [1296.04, 643.44, 2265.98], 
            'wexe': [2.76, -0.36, 11.59], 
            'B': 0.36819, 
            'D': 0.1187e-6, 
            'H': 0.0075e-12,
            'omega': [1315.21, 667.72, 2378.53],
            'x_1j': [-2.76, -4.45, -19.04],
            'x_2j': [0., 1.34, -12.18],
            'x_3j': [0., 0., -12.34],
            'x_l2l2': -0.95,
            'E_0': 2485.06
        }
    }
}


# Isotopologues of different molecules which constitute linear Bose systems
BOSE_LINEAR_SYSTEMS = {
    'CO2'   : [ 1, 2, 7, 9, 10 ],
    'O3'    : [ 1, 3, 5 ]
    # Expand dictionary as needed
}


####### VIBRATIONAL ENERGY LEVEL COMPUTATION #######

# CARBON MONOXIDE (CO)
def CO_Evib(v, omega, omega_x, E_0):
    '''
    Calculates the energy levels of carbon monoxide (CO) as a function of
    vibrational quantum number v (int).

    Returns:
        delta_E_v (float) : energy of the level as wavenumber [cm-1]
    '''
    harmonic = omega * ( v + 0.5 )
    anharmonic = omega_x * ( v + 0.5 ) ** 2
    E_v = harmonic - anharmonic
    delta_E_v = E_v - E_0
    return delta_E_v


# CARBON DIOXIDE (CO2)
def CO2_Evib(v: NDArray, omega, x_ij, x_l2l2, E_0):
    '''
    Calculates the energy levels of carbon dioxide (CO2) as a function of
    vibrational quantum numbers v.
    The equation assumes v2=l2 (AFGL quantum notation).

    Returns:
        nu_CO2 (array) : wavenumber [cm-1]
    '''
    # Degeneracy of the different vibrational modes
    DEGENERACY = np.array([[1], [2], [1]])
    k = v + DEGENERACY / 2

    E_vib_lst = []
    for m in np.transpose(k):
        # Harmonic oscillator part: sum of omega and quantum numbers [v1_i, v2_i, v3_i]
        harmonic = np.sum(omega * m)

        # Anharmonic oscillator part: weird stuff
        anharmonic = 0
        for i in range(3):
            for j in range(i, 3):
                anharmonic += x_ij[i, j] * m[i] * m[j]

        # Total wavenumber of the vibration: sum of harmonic and anharmonic oscillation
        nu_CO2 = harmonic + anharmonic + x_l2l2 * (m[1]-1)

        # Energy of vibration: difference to energy of the ground level
        E_vib_lst.append(nu_CO2 - E_0)
    return np.array(E_vib_lst)



####### ROVIBRATIONAL STATE DISTRIBUTION FUNCTIONS #######

# Partition sums for the vibrational mode distribution
def partition_sum(factors):
    # factors : array (1D)
    # partition_sums : float
    if factors.size == 0 or np.all(np.isnan(factors)):
        raise ValueError("Empty or NaN-only array passed to partition_sum.")
    summed = np.sum(factors)
    return summed

# Population of individual modes
def population(J, phi_vib, phi_rot):
    return phi_vib*phi_rot[J]

# Rovib state populations for LTE conditions
def population_lte(g, Q, E, T):
    exponent = - C2* E / T
    return (g/Q) * np.exp(exponent)

# Define 1-Temp-Boltzmann distribution function
def dist_vib_boltz1(E_vib_i, g_vib_i, T_vib_i):
    factors_v = g_vib_i * np.exp( - C2 * E_vib_i / T_vib_i ) 
    phi_v = factors_v / partition_sum(factors_v)
    return phi_v

# 2-Temp-Boltzmann distribution function for vibrational modes (see Stewig et al., 2020)
def dist_vib_boltz2(E_vib_i, g_vib_i, T_vib1, T_vib2, d_1, d_2):
    boltz_A = np.exp( - C2 * E_vib_i / T_vib1 ) 
    boltz_B = np.exp( - C2 * E_vib_i / T_vib2 ) 
    factors_v = d_1 * boltz_A + d_2 * boltz_B   
    phi_v = g_vib_i * factors_v / partition_sum(factors_v)
    return phi_v

# Treanor distribution (see Klarenaar et al., 2017)
def dist_vib_treanor(v_i, g_vib_i, G_i, WEXE_i, T_vib_i, T_trans):
    factors_v = g_vib_i * np.exp( -C2 * (v_i*G_i/T_vib_i - v_i*(v_i-1)*WEXE_i/T_trans) )
    phi_v = factors_v / partition_sum(factors_v)
    return phi_v

# Distribution function for rotational modes (normal Fermi statistics)
def dist_rot(J, T_J, B, D, H, gs, gi):
    E_rot_J = B * J * (J+1) - D * J**2 * (J+1)**2 + H * J**3 * (J+1)**3
    g_rot_J = (2*J + 1) * gs * gi                   # [-] rotational state degeneracy
    exponent = - C2 * E_rot_J / T_J
    factors_J = g_rot_J * np.exp( exponent )        # [-] Maxwell-Boltzmann factor
    phi_J = factors_J / partition_sum(factors_J)    # [-] normalised with partition sum
    return phi_J

# Rotational state distribution for Bose statistics
def dist_rot_bose(J, T_J, B, D, H, gs, gi):
    E_rot_J = B * J * (J+1) - D * J**2 * (J+1)**2 + H * J**3 * (J+1)**3
    g_rot_J = (2*J + 1) * gs * gi                   # [-] rotational state degeneracy
    # Odd rotational states drop out in the vibrationless ground state
    J = J.astype(int)
    g_rot_J[J % 2 != 0] = 0  
    exponent = - C2 * E_rot_J / T_J
    factors_J = g_rot_J * np.exp( exponent )        # [-] Maxwell-Boltzmann factor
    phi_J = factors_J / partition_sum(factors_J)    # [-] normalised with partition sum
    return phi_J


####### COMPUTE POPULATION DISTRIBUTIONS #######

# Compute populations of CO (one vib. mode)
def compute_populations_CO(iso, df, distribution, T_arr, I_a):
    # Fetch parameters for the isotopologue from NLTE_PARAMETERS dictionary
    params = NLTE_PARAMETERS['CO'][iso]
    GI_CO = params['gi']
    GS_CO = params['gs']
    G_CO = params['G_1']
    WEXE_CO = params['wexe']
    B_CO = params['B']
    D_CO = params['D']
    H_CO = params['H']
    OMEGA_CO = params['omega']
    E_0_CO = params['E_0']

    # Initiate rovib state arrays
    J_arr = np.arange(0, J_max+1)
    v_arr = np.arange(0, v_max_CO+1)

    # Compute vibrational state distribution
    if distribution == 'boltzmann':
        T_rot, T_vib = T_arr
        E_vib = CO_Evib(v=v_arr, omega=OMEGA_CO, omega_x=WEXE_CO, E_0=E_0_CO)
        phi_v = dist_vib_boltz1(E_vib_i=E_vib, g_vib_i=1, T_vib_i=T_vib)
    elif distribution == 'boltzmann2':
        T_rot, T_vib_1, T_vib_2, dist_1, dist_2 = T_arr
        E_vib = CO_Evib(v=v_arr, omega=OMEGA_CO, omega_x=WEXE_CO, E_0=E_0_CO)
        phi_v = dist_vib_boltz2(v_i=v_arr, g_vib_i=1, G_i=G_CO, T_vib1=T_vib_1, T_vib2=T_vib_2, d_1=dist_1, d_2=dist_2)
    elif distribution == 'treanor':
        T_rot, T_vib = T_arr
        phi_v = dist_vib_treanor(v_i=v_arr, g_vib_i=1, G_i=G_CO, WEXE_i=WEXE_CO, T_vib_i=T_vib, T_trans=T_rot)
    else:
        raise ValueError("Unknown vibrational state distribution function:" + str(distribution))
        
    # Compute rotational state distribution
    phi_J = dist_rot(J=J_arr, T_J=T_rot, B=B_CO, D=D_CO, H=H_CO, gs=GS_CO, gi=GI_CO)

    # Compute total rovibrational state populations
    df['p_l'] = population(J=df['J_l'], phi_rot=phi_J, phi_vib=phi_v[df['v_l']]) * I_a
    df['p_u'] = population(J=df['J_u'], phi_rot=phi_J, phi_vib=phi_v[df['v_u']]) * I_a
    
    return df


# Populations for linear Bose systems (e.g. symmetric CO2 isotopologues)
def apply_population_bose(row, phi_J_v0, phi_J_normal):
    if row['v1_l'] == 0 and row['v2_l'] == 0 and row['v3_l'] == 0:
        # Odd rotations drop out for vibrationless ground state
        return population(row['J_l'], row['phi_v_l'], phi_J_v0)
    else:
        return population(row['J_l'], row['phi_v_l'], phi_J_normal)
    

# Compute populations CO2 (three vib. modes, one doubly degenerate)
def compute_populations_CO2(iso, df, distribution, T_arr, I_a):
    # Fetch parameters for the isotopologue from NLTE_PARAMETERS dictionary
    params = NLTE_PARAMETERS['CO2'][iso]
    GI_CO2 = params['gi']
    GS_CO2 = params['gs']
    G_CO2 = params['G_1']
    WEXE_CO2 = params['wexe']
    B_CO2 = params['B']
    D_CO2 = params['D']
    H_CO2 = params['H']
    OMEGA_CO2 = params['omega']
    X_IJ_CO2 = np.array([ params['x_1j'], params['x_2j'], params['x_3j'] ])
    X_L2L2_CO2 = params['x_l2l2']
    E_0_CO2 = params['E_0']

    # Fetch list of isotopologues that follow Bose statistics
    BOSE_ISO_CO2 = BOSE_LINEAR_SYSTEMS['CO2']
    
    # Initiate rovib state arrays
    J_arr = np.arange(0, J_max+1)
    v1_arr = np.arange(0, v_max_CO2[0]+1)
    v2_arr = np.arange(0, v_max_CO2[1]+1)
    v3_arr = np.arange(0, v_max_CO2[2]+1)
    zero_v1 = np.zeros(len(v1_arr))
    zero_v2 = np.zeros(len(v2_arr))
    zero_v3 = np.zeros(len(v3_arr))

    if distribution == 'boltzmann':
        T_rot, T_vib12, T_vib3 = T_arr
        E_vib_1 = CO2_Evib(v=[v1_arr,zero_v1,zero_v1], omega=OMEGA_CO2, x_ij=X_IJ_CO2, x_l2l2=X_L2L2_CO2, E_0=E_0_CO2)
        E_vib_2 = CO2_Evib(v=[zero_v2,v2_arr,zero_v2], omega=OMEGA_CO2, x_ij=X_IJ_CO2, x_l2l2=X_L2L2_CO2, E_0=E_0_CO2)
        E_vib_3 = CO2_Evib(v=[zero_v3,zero_v3,v3_arr], omega=OMEGA_CO2, x_ij=X_IJ_CO2, x_l2l2=X_L2L2_CO2, E_0=E_0_CO2)
        phi_v1 = dist_vib_boltz1(E_vib_i=E_vib_1, g_vib_i=1, T_vib_i=T_vib12)
        phi_v2 = dist_vib_boltz1(E_vib_i=E_vib_2, g_vib_i=v2_arr+1, T_vib_i=T_vib12)
        phi_v3 = dist_vib_boltz1(E_vib_i=E_vib_3, g_vib_i=1, T_vib_i=T_vib3)
    elif distribution == 'boltzmann2':
        T_rot, T_vib_1, T_vib_2, dist_1, dist_2 = T_arr
        E_vib_1 = CO2_Evib(v=[v1_arr,zero_v1,zero_v1], omega=OMEGA_CO2, x_ij=X_IJ_CO2, x_l2l2=X_L2L2_CO2, E_0=E_0_CO2)
        E_vib_2 = CO2_Evib(v=[zero_v2,v2_arr,zero_v2], omega=OMEGA_CO2, x_ij=X_IJ_CO2, x_l2l2=X_L2L2_CO2, E_0=E_0_CO2)
        E_vib_3 = CO2_Evib(v=[zero_v3,zero_v3,v3_arr], omega=OMEGA_CO2, x_ij=X_IJ_CO2, x_l2l2=X_L2L2_CO2, E_0=E_0_CO2)
        phi_v1 = dist_vib_boltz2(E_vib_i=E_vib_1, g_vib_i=1, T_vib1=T_vib_1, T_vib2=T_vib_2, d_1=dist_1, d_2=dist_2)
        phi_v2 = dist_vib_boltz2(E_vib_i=E_vib_2, g_vib_i=v2_arr+1, T_vib1=T_vib_1, T_vib2=T_vib_2, d_1=dist_1, d_2=dist_2)
        phi_v3 = dist_vib_boltz2(E_vib_i=E_vib_3, g_vib_i=1, T_vib1=T_vib_1, T_vib2=T_vib_2, d_1=dist_1, d_2=dist_2)
    elif distribution == 'treanor':
        T_rot, T_vib12, T_vib3 = T_arr
        phi_v1 = dist_vib_treanor(v_i=v1_arr, g_vib_i=1, G_i=G_CO2[0], WEXE_i=WEXE_CO2[0], T_vib_i=T_vib12, T_trans=T_rot)
        phi_v2 = dist_vib_treanor(v_i=v2_arr, g_vib_i=v2_arr+1, G_i=G_CO2[1], WEXE_i=WEXE_CO2[1], T_vib_i=T_vib12, T_trans=T_rot)
        phi_v3 = dist_vib_treanor(v_i=v3_arr, g_vib_i=1, G_i=G_CO2[2], WEXE_i=WEXE_CO2[2], T_vib_i=T_vib3, T_trans=T_rot)
    else:
        raise ValueError("Unsupported vibrational state distribution function: " + str(distribution))
    
    # Compute total vibrational distribution for each line
    df['phi_v_l'] = phi_v1[df['v1_l']] * phi_v2[df['v2_l']] * phi_v3[df['v3_l']]
    df['phi_v_u'] = phi_v1[df['v1_u']] * phi_v2[df['v2_u']] * phi_v3[df['v3_u']]
    
    # Compute rotational distribution and total state population
    if iso in BOSE_ISO_CO2:
        print("NOTE:  This CO2 isotopologue is a linear Bose system.\
             \n       Odd rotational transitions of the vibrationless ground state drop out.")
        phi_J_v0 = dist_rot_bose(J=J_arr, T_J=T_rot, B=B_CO2, D=D_CO2, H=H_CO2, gs=GS_CO2, gi=GI_CO2) 
        phi_J_normal = dist_rot(J=J_arr, T_J=T_rot, B=B_CO2, D=D_CO2, H=H_CO2, gs=GS_CO2, gi=GI_CO2)
        df['p_l'] = df.apply(apply_population_bose, axis=1,
                             phi_J_v0=phi_J_v0, phi_J_normal=phi_J_normal) * I_a
        df['p_u'] = population(J=df['J_u'], phi_rot=phi_J_normal, phi_vib=df['phi_v_u']) * I_a
    else:
        phi_J = dist_rot(J=J_arr, T_J=T_rot, B=B_CO2, D=D_CO2, H=H_CO2, gs=GS_CO2, gi=GI_CO2)
        df['p_l'] = population(J=df['J_l'], phi_rot=phi_J, phi_vib=df['phi_v_l']) * I_a
        df['p_u'] = population(J=df['J_u'], phi_rot=phi_J, phi_vib=df['phi_v_u']) * I_a

    return df
