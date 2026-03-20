# FUNCTION:     Define line broadening functions (Lorentzian, Gaussian, approximate Voigt),
#               common Instrument Line Shape (ILS) functions, 
#               and functions to apply broadening to spectral dataframes.
# AUTHOR:       I.C. Dumitrescu
# REFERENCES:   RADIS, NEQAIR96, HAPI user manual, Whiting (1968), Olivero & Longbothum (1977)

import numpy as np
from scipy.constants import N_A, c, k
from scipy.integrate import trapezoid


####### CONSTANTS #######
c = c*100                   # [cm s^-1] speed of light
C2 = 1.4387769              # [cm*K]
P_ATM = 101325.             # [Pa] atmosphere to Pascal conversion
T_REF = 296.                # [K]


####### LINE BROADENING FUNCTIONS: GAUSSIAN, LORENTZIAN, APPROXIMATE VOIGT #######

# Lorentz: Half-width at half maximum (HWHM) for self- and pressure-broadening from HITRAN
def hwhm_lorentz(T, n_air, p, p_self, gamma_air, gamma_self):
    """
    Half width at half max (HWHM) for a Lorentzian (pressure) broadening function.

    Args:
        T (float)                   : [K] gas temperature (translational)
        n_air (float, array)        : [-] coefficient of temperature dependence of air-broadened half width
        p (float)                   : [atm] gas pressure
        p_self (float)              : [atm] partial pressure of the molecule investigated
        gamma_air (float, array)    : [cm^-1 atm^-1] air-broadened HWHM for T_ref=296K and p_ref=1atm
        gamma_self (float, array)   : [cm^-1 atm^-1] self-broadened HWHM for T_ref=296K and p_ref=1atm

    Returns:
        (float, array)              : [cm^-1] pressure broadening HWHM
    """
    return (T_REF/T)**n_air * ( gamma_air*(p-p_self) + gamma_self*p_self )

# Doppler: HWHM function from HITRAN
def hwhm_gauss(nu_ij, T, M):
    """
    Half width at half max (HWHM) for a Gaussian (Doppler) broadening function.

    Args:
        nu_ij (float, array)        : [cm^-1] wavenumber of spectral line transition in vacuum
        T (float)                   : [K] gas temperature (translational)
        M (float)                   : [g mol^-1] molar mass of the molecule investigated

    Returns:
        (float, array)              : [cm^-1] Doppler broadening HWHM
    """
    return (nu_ij/c) * np.sqrt( 2 * np.log(2) * N_A * k * (T/M) )

# Olivero-Longbothum (1977) approximate Voigt broadening as in NEQAIR96 and RADIS
def hwhm_olivero(w_l, w_g):
    """
    Half width at half max (HWHM) for the approximate Voigt profile by Whiting (1968) as given by 
    Olivero & Longbothhum (1977) based on empirical fits. The expression should be accurate within 0.01%.
    Remark that the original expression is given for full width at half max (FWHM), with FWHM=2*HWHM,
    but using HWHM everywhere should yield the approximate Voigt HWHM.

    Args:
        w_l (float) : Lorentzian (pressure) broadened HWHM
        w_g (float) : Gaussian (Doppler) broadening HWHM

    Returns:
        w_v (float) : Approximate Voigt broadening HWHM
    """
    d = (w_l - w_g) / (w_l + w_g)
    alpha = 0.18121
    beta = 0.023665 * np.exp(0.6 * d) + 0.00418 * np.exp(-1.9 * d)
    R = 1 - alpha * (1 - d**2) - beta * np.sin( np.pi * d )
    w_v = R * (w_l + w_g)
    return w_v

# Voigt profile approximation from NEQAIR96 user manual (also used by RADIS)
def broaden_whiting1968(nu, nu_0, w_v, w_l):
    """
    Approximate Voigt profile broadening function, as implemented into RADIS and NEQAIR96.
    Exact for pure Gaussian or pure Lorentzian broadening.
    """
    # Note from RADIS: The expression from Oliviero & Longbothum uses FWHM, *not* HWHM!
    w_v_FWHM = 2 * w_v
    w_l_FWHM = 2 * w_l
    delta_nu = np.abs(nu - nu_0)
    x = w_l_FWHM / w_v_FWHM
    y = (delta_nu / w_v_FWHM) ** 2
    z = (delta_nu / w_v_FWHM) ** 2.25
    lineshape = ( 
                  (1 - x) * np.exp(-2.772*y) 
                  + x / (1 + 4 * y)         
                  + 0.016 * x * (1-x) * ( np.exp(-0.4*z) - 10 / (10+z) )  # 2nd order correction
                )
    
    # Integrate over the lineshape to perform normalisation
    # (Tip from RADIS: Integration is more consistently reliable than the integral approximation from Whiting)
    integral = trapezoid(lineshape, nu)

    # Normalise 
    lineshape = lineshape / integral
    return lineshape


####### COMMON INSTRUMENT LINE SHAPES #######

# Rectangular slit function
def ils_rectangular(x, gamma):
    """
    Instrumental (slit) function.
    B(x) = 1/gamma , if |x| <= gamma/2 & B(x) = 0, if |x| > gamma/2,
    where gamma is a slit width or the instrumental resolution.
    """
    index_inner = np.abs(x) <= gamma/2
    index_outer = ~index_inner
    y = np.zeros(len(x))
    y[index_inner] = 1/gamma
    y[index_outer] = 0
    return y

# Triangular slit function
def ils_triangular(x, gamma):
    """
    Instrumental (slit) function.
    B(x) = 1/gamma*(1-|x|/gamma), if |x| ≤ gamma & B(x) = 0, if |x| > gamma,
    where gamma is the line width equal to the half base of the triangle.
    """
    index_inner = np.abs(x) <= gamma
    index_outer = ~index_inner
    y = np.zeros(len(x))
    y[index_inner] = 1/gamma * (1 - np.abs(x[index_inner])/gamma)
    y[index_outer] = 0
    return y

# Gaussian slit function
def ils_gaussian(x, gamma):
    """
    Gaussian line broadening function, where gamma/2 is the Gaussian half-width at half-maximum.
    """
    gamma /= 2
    return np.sqrt(np.log(2))/(np.sqrt(np.pi)*gamma) * np.exp(-np.log(2) * (x/gamma)**2)

# dispersion slit function
def ils_dispersion(x, gamma):
    """
    Dispersion line broadening function, where gamma/2 is the Lorentzian half-width at half-maximum.
    """
    gamma /= 2
    return gamma / np.pi / (x**2 + gamma**2)

# Cosine slit function
def ils_cosine(x, gamma):
    """
    Cosine line broadening function.
    """
    return (np.cos( np.pi/gamma*x ) + 1) / (2*gamma)

# Diffraction slit function
def ils_diffraction(x, gamma):
    """
    Diffraction line broadening function.
    """
    y = np.zeros(len(x))
    index_zero = x==0
    index_nonzero = ~index_zero
    dk_ = np.pi / gamma
    x_ = dk_*x[index_nonzero]
    w_ = np.sin(x_)
    r_ = w_**2/x_**2
    y[index_zero] = 1
    y[index_nonzero] = r_ / gamma
    return y

# Apparatus function of the ideal Michelson interferometer
def ils_michelson(x, gamma):
    """
    Michelson line broadening function.
    B(x) = 2/gamma*sin(2pi*x/gamma)/(2pi*x/gamma) if x!=0 else 1,
    where 1/gamma is the maximum optical path difference.
    """
    y = np.zeros(len(x))
    index_zero = x==0
    index_nonzero = ~index_zero
    dk_ = 2 * np.pi / gamma
    x_ = dk_ * x[index_nonzero]
    y[index_zero] = 1
    y[index_nonzero] = 2 / gamma * np.sin(x_) / x_
    return y


####### COMPUTE LINE BROADENING FOR SPECTRA #######

# Compute broadening parameters
def compute_broadening_parameters(df, T_rot, p_ambient, p_self, M_self):
    # Self-broadening (Lorentz profile)
    df['w_l'] = hwhm_lorentz(T=T_rot,
                             n_air=df['n_air'],
                             p=p_ambient,
                             p_self=p_self,
                             gamma_air=df['gamma_air'],
                             gamma_self=df['gamma_self'])
                
    # Doppler broadening (Gauss profile)
    df['w_g'] = hwhm_gauss(nu_ij=df['nu'],
                           T=T_rot,
                           M=M_self)
    
    # Approximate Voigt broadening (Olivero & Longbothum expression for Whiting approximation)
    df['w_v'] = hwhm_olivero(w_l=df['w_l'],
                             w_g=df['w_g'])

    return df


# Compute broadened spectrum
def broadening_full(df, nu_min, nu_max, nu_step, wing=20.):
    """
    Full line broadening function; applies approximate Voigt broadening to all lines in a given dataframe
    and adds up the absorption and emission cross-sections for each wavenumber in a given range.

    Args:
        df (pandas dataframe)   : dataframe containing spectral lines & HWHM parameters w_l and w_v
        nu_min (float)          : [cm^-1] minimum wavenumber to perform computation for
        nu_max (float)          : [cm^-1] maximum wavenumber to perform computation for
        nu_step (float)         : [cm^-1] wavenumber step size
        wing (float, optional)  : [cm^-1] lineshape wing to account for; defaults to 20.

    Returns:
        nu_arr (array)      : [cm^-1] wavenumber array
        sigma_nu (array)    : [?] broadened & combined absorption cross-sections
        j_nu (array)        : [?] broadened & combined emission cross-sections
    """
    # Abbreviate dataframe
    df_short = df[ (df['nu'] >= nu_min) & (df['nu'] <= nu_max) ]

    # Initiate wavenumber array & list for broadening cross-sections
    nu_arr = np.arange(nu_min, nu_max, nu_step)
    sigma_nu = np.zeros_like(nu_arr)
    j_nu = np.zeros_like(nu_arr)

    # Pre-compute a grid with relative wavenumbers, centred at 0
    nu_half = int( wing / nu_step )                             # converts wing width into number of grid steps
    nu_centred = np.arange(-nu_half, nu_half + 1) * nu_step     # converts grid points to corresponding wavenumbers

    # Compute broadening for each line j at all wavenumbers nu
    for _, row in df_short.iterrows():
        # Pick out central wavenumber for line j
        nu_j = row['nu']

        # Compute lineshapes on centred grid
        broadening_j = broaden_whiting1968(nu=nu_centred, nu_0=0.,
                                           w_v=row['w_v'], w_l=row['w_l'])

        # Find where nu_j maps onto nu_arr
        idx_centre = np.searchsorted(nu_arr, nu_j)

        # Compute slice indices, clamped to array bounds
        i_start = idx_centre - nu_half
        i_end = idx_centre + nu_half + 1        # +1 because nu_centred has an odd length
        
        # Corresponding slice into nu_centred (edge lines near nu_min/nu_max)
        k_start = max(0, -i_start)
        k_end = len(nu_centred) - max(0, i_end - len(nu_arr))

        i_start = max(0, i_start)
        i_end = i_start + (k_end - k_start)     # derive i_end from k length

        sigma_nu[i_start:i_end] += row['S_j'] * broadening_j[k_start:k_end]
        j_nu[i_start:i_end] += row['epsilon'] * broadening_j[k_start:k_end]

    return nu_arr, sigma_nu, j_nu


####### APPLY INSTRUMENTAL LINE SHAPE BROADENING #######
def apply_ils(nu, intensity, ils_function, resolution=0.1, wing=5.):
    """
    Convolves an existing spectrum with an Instrumental Line Shape (ILS) function,
    thus broadening the spectral lines.

    nu (arr)            : Wavenumber array [cm^-1]
    ils_function (str)  : ILS to convolve with, select from existing functions or define your own
                          ('ils_rectangular', 'ils_triangular', 'ils_gaussian',
                           'ils_dispersion', 'ils_cosine', 'ils_diffraction', 'ils_michelson')
    resolution          : Instrument resolution parameter gamma in [cm^-1]; default: 0.1cm^-1
    wing                : Instrument function wing [cm^-1]; default: 5cm^-1 (total grid width: 10cm^-1)
    """
    # Determine wavenumber array step size
    step = nu[1] - nu[0]
    if step >= resolution:
        raise Exception(f"Step size must be smaller than the resolution. \
                          \nCurrently step = {step} cm^-1 and resolution = {resolution} cm^-1.")

    # Define slit function & renormalise it to ensure preservation of total intensity
    x_slit = np.arange(-wing, wing+step, step)
    slit = ils_function(x=x_slit, gamma=resolution)
    slit /= np.sum(slit) * step

    # Define convolution bounds, AKA subtract half the slit length from each end
    left_bound = int( len(slit)/2 )     # avoid floats
    right_bound = len(nu) - int( len(slit)/2 )

    # Broaden spectrum by convolving with the slit function
    broadened_spectrum = np.convolve( intensity, slit, 'same' ) * step

    return nu[left_bound:right_bound], broadened_spectrum[left_bound:right_bound]
