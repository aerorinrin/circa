# FUNCTION:     Main CIRCA simulator programme code.
#               This file defines the circa.spectrum() function.
# AUTHOR:       I.C. Dumitrescu

# Import external libraries
import numpy as np
import pandas as pd
from scipy.constants import k, c

# Import internal functions
from circa.read_database import get_molecule_id, get_dataframe
from circa.isotopologues import iso_abundance, iso_mass, iso_Qref
from circa.partition_sums import iso_QT
from circa.broadening import compute_broadening_parameters, broadening_full
from circa.lines_lte import lines_lte, population_lte
from circa.lines_nlte import lines_nlte, compute_populations_CO, compute_populations_CO2


####### CONSTANTS #######
c = c*100                   # [cm s^-1] speed of light
C2 = 1.4387769              # [cm K] C2 = h*c/k
P_ATM = 101325.             # [Pa] atmosphere to Pascal conversion
T_REF = 296.                # [K]
h = 6.62607015e-34          # [J s] Planck's constant

# Relate OPQRS branches to change in rotational quantum number
DELTA_J = {'O': -2, 'P': -1, 'Q': 0, 'R': 1, 'S': 2}


####### ABSORPTION & EMISSION SPECTRA #######

# Emission coefficient of a line
def line_emission(nu_j, A_ul, p_u, molec_density):
    """
    Computes the emission cross-section of an individual transition line
    Args: 
        nu_j = wavenumber of line centre [1 cm^-1]
        A_ul = Einstein-A coefficient of spontaneous emission [1 s^-1]
        p_u = population distribution of the upper state of the transition [-]
        molec_density = gas molecule density (N/V) [molecules cm^-3]
    Returns: unbroadened integral emission coefficient of the line [mW sr^-1 cm^-3]
    """
    delta_E = h * c * nu_j         # [J] energy of the transition
    return delta_E * A_ul * p_u * molec_density / (4 * np.pi) * 1e3     # mW correction

# Absorbance (just the Beer-Lambert law)
def absorbance(sigma, molec_density, L):
    """
    Computes the spectral absorbance as per the Beer-Lambert law
    Args: 
        sigma = absorption cross-section [cm^2 molecule^-1]
        molec_density = gas molecule density (N/V) [molecules cm^-3]
        L = length of optical path [cm]
    Returns: absorbance [-]
    """
    return sigma * molec_density * L

# Transmittance from Beer-Lambert law 
def transmittance(sigma, molec_density, L):
    """
    Args: 
        sigma = absorption cross-section [cm^2 molecule^-1]
        molec_density = gas molecule density (N/V) [molecules cm^-3]
        L = length of optical path [cm]
    Returns: transmittance [-]
    """
    return np.exp( - sigma * molec_density * L ) 

# Radiance (see Pannier & Laux, 2018)
def radiance(j, sigma, molec_density, L):
    """
    Computes the radiance of a spectrum from the emission and absorption cross-sections
    Args: 
        j = emission cross-section [mW sr^-1 cm^-3]
        sigma = absorption cross-section [cm^2 molecule^-1]
        molec_density = gas molecule density (N/V) [molecules cm^-3]
        L = length of optical path [cm]
    Returns: radiance of the gas column [mW sr^-1 cm^-2]
    """
    return j * ( 1 - np.exp(- sigma * molec_density * L) ) / (sigma * molec_density )


####### COMPUTE LTE AND NLTE SPECTRA #######

# LTE spectra
def spectrum_eq(df, molec_id, iso_lst, T_eq):
    # Create copy of the dataframe to return at the end
    df_eq = df.copy()
    
    # Fetch additional isotopologue information and put it into a dataframe
    I_a_lst = []
    Q_ref_lst = []
    Q_T_lst = []
    for iso in iso_lst:
        I_a_iso = iso_abundance(molec_id, iso)  # isotopologue abundance
        Q_ref_iso = iso_Qref(molec_id, iso)     # reference partition sum Q_ref given for T_ref=296K
        Q_T_iso = iso_QT(molec_id, iso, T_eq)   # LTE partition sum interpolated from tabulated data for other temperatures
        
        I_a_lst.append(I_a_iso)
        Q_ref_lst.append(Q_ref_iso)
        Q_T_lst.append(Q_T_iso)
    df_iso_data = pd.DataFrame(index=iso_lst, data={"I_a": I_a_lst, "Q_ref": Q_ref_lst, "Q_T": Q_T_lst})
    
    # Compute upper state populations (needed for emission coefficients)
    df_eq['p_u'] = population_lte(g=df_eq['g_u'],
                                  Q=df_eq['isotopologue'].map(df_iso_data['Q_T']),
                                  E=df_eq['E_l']+df_eq['nu'],
                                  T=T_eq,
                                  I_a=df_eq['isotopologue'].map(df_iso_data['I_a']))
    df_eq['p_l'] = population_lte(g=df_eq['g_l'],
                                  Q=df_eq['isotopologue'].map(df_iso_data['Q_T']),
                                  E=df_eq['E_l'],
                                  T=T_eq,
                                  I_a=df_eq['isotopologue'].map(df_iso_data['I_a']))
    
    # Compute line strength (scaling of the reference line strengths)
    df_eq['S_j'] = lines_lte(nu_j=df_eq['nu'],
                             E_l=df_eq['E_l'],
                             S_ref=df_eq['sw'],
                             T=T_eq,
                             Q=df_eq['isotopologue'].map(df_iso_data['Q_T']),
                             Q_ref=df_eq['isotopologue'].map(df_iso_data['Q_ref']))

    # Return dataframe with LTE linestrengths
    return df_eq


# NLTE spectra of CO and CO2
def spectrum_noneq(df, molec_id, iso_lst, dist, T):
    """
    Compute linestrength and broadening parameters for nonequilibrium (NLTE) spectra;
    Currently, only the three most abundant isotopologues of CO2 and CO are supported!

    Args:
        df (pandas dataframe) : Spectral line dataframe
        molec_id (int)        : HITRAN molecule ID; Currently only 2 (CO2) and 5 (CO) supported
        iso_lst (lst)         : Sorted list of local isotopologue IDs being simulated
        dist (str)            : Vibrational state distribution function ('boltzmann', 'boltzmann2', 'treanor')
        T (float)             : [K] List of rovibrational state temperatures
            CO2 :
                'boltzmann'  : [ T_rot, T_vib_12, T_vib_3 ]
                'boltzmann2' : [ T_rot, T_vib1, T_vib2, d_1, d_2 ]
                'treanor'    : [ T_rot, T_vib_12, T_vib_3 ]
            CO  :
                'boltzmann'  : [ T_rot, T_vib_CO ]
                'boltzmann2' : [ T_rot, T_vib1, T_vib2, d_1, d_2 ]
                'treanor'    : [ T_rot, T_vib_CO ]

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # Create copy of the dataframe to return at the end
    df_noneq = df.copy()

    # Add upper state rotational quantum numbers J_u from branch info
    df_noneq['J_u'] = df_noneq['J_l'] + df_noneq['branch'].apply(lambda opqrs: DELTA_J.get(opqrs))

    # Fetch additional isotopologue information
    for iso in iso_lst:
        I_a_iso = iso_abundance(molec_id, iso)

        # Compute populations # TODO: fix isotopologue abundance issue
        if molec_id == 2:
            df_noneq = compute_populations_CO2(iso=iso, df=df_noneq, 
                                               distribution=dist, T_arr=T, 
                                               I_a=I_a_iso)
        elif molec_id == 5:
            df_noneq = compute_populations_CO(iso=iso, df=df_noneq,
                                              distribution=dist, T_arr=T,
                                              I_a=I_a_iso)
        else:
            raise ValueError("NLTE line computation is currently only supported for CO and CO2.")

    # Compute line strength
    df_noneq['S_j'] = lines_nlte(nu_j=df['nu'],
                                 A_ul=df['a'],
                                 p_l=df['p_l'],
                                 p_u=df['p_u'],
                                 g_l=df['g_l'],
                                 g_u=df['g_u'])
    
    # Return dataframe with NLTE linestrengths
    return df_noneq


### MAIN SPECTRUM FUNCTION ###
def spectrum(equilibrium, molecule, nu_min, nu_max, nu_step, T, L, distribution='boltzmann', iso_number=1, p_gas=101325., x=1, database='hitran', get_absorbance=False, get_computation=False):
    """
    Main spectrum computation function

    Args:
        equilibrium (bool): 
            True/False statement about whether the gas is in LTE or not
        molecule (str): 
            Name or chemical formula of the molecule to be simulated (e.g. 'carbon dioxide')
        nu_min (float): 
            [cm^-1] Minimum simulation wavenumber
        nu_max (float): 
            [cm^-1] Maximum simulation wavenumber
        nu_step (float): 
            [cm^-1] Wavenumber step size
        T (float): 
            [K] Gas temperature (LTE) or list of rovibrational state temperatures (NLTE)
        L (float): 
            [cm] Optical path length
        distribution (str, optional): 
            Vibrational state distribution function ('boltzmann', 'boltzmann2', 'treanor'); Defaults to 'boltzmann'
        iso_number (int or lst, optional): 
            Local isotopologue numbers to simulate (e.g. [1, 2, 3]); 
            Defaults to 1
        p_gas (int, optional): [Pa] Gas pressure; Defaults to 101325 Pa (= 1 atm)
        x (int, optional): 
            [-] Mole fraction of the investigated species in the gas; Defaults to 1.
        database (str, optional): 
            Database to use for the computation ('hitan' or 'hitemp'); Defaults to 'hitran'
        get_absorbance (bool, optional): 
            Return absorbance as a result; Defaults to False
        get_computation (bool, optional):
            Return line dataframe as a result (includes, for instance, upper and lower state populations); Defaults to False

    Raises:
        ValueError: User must indicate if the gas is in Local Thermodynamic Equilibrium (LTE) or not

    Returns:
        df_spectrum (pandas dataframe): 
            Spectrum containing 'nu' [cm^-1], 'transmittance' [-], 'radiance' [mW sr^-1 cm^-2], 
            'absorption_cross_section' [cm^2 molecule^-1], and 'emission_cross_section' [mW sr^-1 cm^-3]
    """
    
    print("CIRCA: Beginning spectrum simulation.")

    # Determine molecule selected (ID and chemical formula)
    mol_id, mol_form = get_molecule_id(molecule)

    # Sort isotopologues into a list
    iso_lst = sorted( [iso_number] if isinstance(iso_number, int) else list(iso_number) )

    # Pick out gas temperature from temperature input
    if isinstance(T, float):
        T_gas = T
    elif isinstance(T, int):
        T_gas = float(T)
    else:
        T_gas = T[0]

    # Compute concentration
    n = (p_gas*x) / (k*T_gas) * 1e-6    # convert to [cm^-3]

    # Convert pressure input to unit atmosphere [atm] for broadening computations
    p_gas_atm = p_gas / P_ATM           # [atm]
    
    # Get isotopologue dataframe for requested wavenumber regime
    df = get_dataframe(mol_form, mol_id, iso_lst, nu_min, nu_max, database)

    # Compute spectrum for either LTE or NLTE conditions
    if equilibrium:
        print("\nCIRCA: Computing equilibrium spectrum.")
        df_lines = spectrum_eq(df, mol_id, iso_lst, T_gas)
    elif not equilibrium:
        print("\nCIRCA: Computing nonequilibrium spectrum.")
        df_lines = spectrum_noneq(df, mol_id, iso_lst, distribution, T)
    else:
        raise ValueError("Please indicate if the gas is in local thermodynamic equilibrium (LTE) or not. \
                          \nExample: \t spec = spectrum(equilibrium=True, molecule='CO', ...)")

    # Compute line emission coefficients
    print("CIRCA: Computing line emission coefficients.")
    df_lines['epsilon'] = line_emission(nu_j=df_lines['nu'],
                                        A_ul=df_lines['a'],
                                        p_u=df_lines['p_u'],
                                        molec_density=n)
    
    # Compute spectral broadening parameters and apply approximate Voigt broadening function
    print("CIRCA: Computing broadening of the spectrum.")
    M_lst = []
    for iso in iso_lst:
        M_iso = iso_mass(mol_id, iso)
        M_lst.append(M_iso)
    series_M = pd.Series(index=iso_lst, data=M_lst)
    df_lines = compute_broadening_parameters(df=df_lines, 
                                             T_rot=T_gas, 
                                             p_ambient=p_gas_atm, 
                                             p_self=p_gas_atm*x, 
                                             M_self=df_lines['isotopologue'].map(series_M))
    nu_arr, sigma_nu, j_nu = broadening_full(df_lines, nu_min, nu_max, nu_step)

    if get_absorbance:
        # Compute spectral absorbance
        print("CIRCA: Computing absorbance.")
        absorbance_nu = absorbance(sigma=sigma_nu, molec_density=n, L=L)

    # Spectral transmittance
    print("CIRCA: Computing spectral transmittance & radiance.")
    transmittance_nu = transmittance(sigma=sigma_nu, molec_density=n, L=L)

    # Spectral radiance
    radiance_nu = radiance(j=j_nu, sigma=sigma_nu, molec_density=n, L=L)

    print("CIRCA: Spectrum computed.\n")

    # Return wavenumber array, transmittance, and radiance as dataframe
    df_spectrum = pd.DataFrame({'nu': nu_arr, 
                                'transmittance': transmittance_nu, 
                                'radiance': radiance_nu,
                                'absorption_cross_section': sigma_nu,
                                'emission_cross_section': j_nu})
    if get_computation:
        if get_absorbance:
            return df_spectrum, df_lines, absorbance_nu
        return df_spectrum, df_lines
    else:
        if get_absorbance:
            return df_spectrum, absorbance_nu
        return df_spectrum
