# FUNCTION:     CIRCA helper functions
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


# Planck's law
def planck(nu, T, epsilon=1):
    exponent = ( C2 * nu) / T
    I_blackbody = 2 * h * c**2 * nu**3 / (np.exp(exponent) - 1)
    I_epsilon = epsilon * I_blackbody
    return I_epsilon * 1e3      # convert to [mW sr^-1 cm^-2]


# Attenuation (see Rutten, 2015)
def attenuation(I_0, trans):
    '''
    INPUTS: 
        I_0 = radiative intensity at entry into the volume of gas
        T = transmittance of the volume of gas [-]
        L = length of optical path [cm]
    OUTPUT: intensity at exit from the volume of gas [mW sr^-1 cm^-3]
    '''
    return I_0 * np.exp(- trans)


# Function for combining spectra from serial volumes of gas
def combine_spectra(I_0, next_slab):
    """
    Combine two spectra from slabs in series to one another;
    Radiance I_0 is attenuated over the optical path of the next slab, which may also emit itself.

    Args:
        I_0 (_type_): _description_
        next_slab (_type_): _description_

    Returns:
        _type_: _description_
    """
    I_L = attenuation(I_0, next_slab['transmittance']) + next_slab['radiance']
    return I_L


# Function for combining spectra of joint volumes of gas
def merge_spectra(spectrum_1, spectrum_2):
    """
    Combine two spectra within the same volume of gas

    Args:
        spectrum_1 (pandas dataframe): Spectrum of one gas with 'radiance' and 'transmittance'
        spectrum_2 (pandas dataframe): Spectrum of second gas with 'radiance' and 'transmittance'

    Returns:
        (array): Intensity
    """
    I_1 = spectrum_1['radiance'] * (1-spectrum_2['transmittance'])
    I_2 = spectrum_2['radiance'] * (1-spectrum_1['transmittance'])
    return I_1 + I_2
