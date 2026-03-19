# FUNCTION:     Define line broadening functions (Lorentzian, Gaussian, approximate Voigt),
#               common Instrument Line Shape (ILS) functions, 
#               and functions to apply broadening to spectral dataframes.
# AUTHOR:       I.C. Dumitrescu
# REFERENCES:   RADIS, NEQAIR96, HAPI user manual, Whiting (1968), Olivero & Longbothum (1977)

import numpy as np
from scipy.constants import N_A, c, k


####### CONSTANTS #######
c = c*100                   # [cm s^-1] speed of light
C2 = 1.4387769              # [cm*K]
P_ATM = 101325.             # [Pa] atmosphere to Pascal conversion
T_REF = 296.                # [K]


####### LINE BROADENING FUNCTIONS: GAUSSIAN, LORENTZIAN, APPROXIMATE VOIGT #######

# Lorentz: Half-width at half maximum (HWHM) for self- and pressure-broadening from HITRAN
def hwhm_lorentz(T, n_air, p, p_self, gamma_air, gamma_self):
    return (T_REF/T)**n_air * ( gamma_air*(p-p_self) + gamma_self*p_self )

# Doppler: HWHM function from HITRAN
def hwhm_gauss(nu_ij, T, M):
    return (nu_ij/c) * np.sqrt( 2 * np.log(2) * N_A * k * (T/M) )

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
                  + x * (1 + 4 * y)         
                  + 0.016 * x * (1-x) * ( np.exp(-0.4*z) - 10 / (10+z) )  # 2nd order correction
                )
    return lineshape    # TODO: implement normalisation

############ FROM RADIS #####################

def voigt_lineshape(w_centered, hwhm_lorentz, hwhm_voigt):
    """Calculates Voigt lineshape using the approximation of the Voigt profile
    of [NEQAIR-1996]_, [Whiting-1968]_ that maintains a good accuracy in the far wings.
    Exact for a pure Gaussian and pure Lorentzian.

    Parameters
    ----------
    w_centered: 2D array       [one per line: shape W x N]
        waverange (nm / cm-1) (centered on 0)
    hwhm_lorentz: array   (cm-1)        [length N]
        half-width half maximum coefficient (HWHM) for Lorentzian broadening
    hwhm_voigt: array   (cm-1)        [length N]
        half-width half maximum coefficient (HWHM) for Voigt broadening,
        calculated by :py:func:`~radis.lbl.broadening.voigt_broadening_HWHM`

    Other Parameters
    ----------------
    jit: boolean
        if ``True``, use just in time compiler. Usually faster when > 10k lines.
        Default ``True``.

    Returns
    -------
    lineshape: pandas Series        [shape N x W]
        line profile

    References
    ----------
    .. [NEQAIR-1996] `NEQAIR 1996 User Manual, Appendix D <https://ntrs.nasa.gov/search.jsp?R=19970004690>`_

    See Also
    --------
    :py:func:`~radis.lbl.broadening.voigt_broadening_HWHM`
    :py:func:`~radis.lbl.broadening.whiting1968`
    """

    # Note: Whiting and Olivero use FWHM. Here we keep HWHM in all public function
    # arguments for consistency.
    wl = 2 * hwhm_lorentz  # HWHM > FWHM
    wv = 2 * hwhm_voigt  # HWHM > FWHM

    lineshape = whiting1968(w_centered, wl, wv)

    # Normalization
    #    integral = wv*(1.065+0.447*(wl/wv)+0.058*(wl/wv)**2)
    # ... approximation used by Whiting, equation (7)
    # ... performance: ~ 6µs vs ~84µs for np.trapz(lineshape, w_centered) ):
    # ... But not used because:
    # ... - it may yield wrong results when the broadening range is not refined enough
    # ... - it is defined for wavelengths only. Here we may have wavenumbers as well

    integral = np.trapz(lineshape, w_centered, axis=0)

    # Normalize
    lineshape /= integral # TODO: implement this integration approach into circa, since the gharavi one is clearly not working

    return lineshape


def whiting1968(w_centered, wl, wv):
    r"""A pseudo-voigt analytical approximation.

    .. math::
        \Phi(w)=\left(1-\frac{w_l}{w_v}\right) \operatorname{exp}\left(-2.772{\left(\frac{w}{w_v}\right)}^{2.25}\right)+\frac{1\frac{w_l}{w_v}}{1+4{\left(\frac{w}{w_v}\right)}^{2.25}}+0.016\left(1-\frac{w_l}{w_v}\right) \frac{w_l}{w_v} \left(\operatorname{exp}\left(-0.4w_{wv,225}\right)-\frac{10}{10+{\left(\frac{w}{w_v}\right)}^{2.25}}\right)

    Parameters
    ----------
    w_centered: 2D array
        broadening spectral range for all lines
    wl: array
        Lorentzian FWHM
    wv: array
        Voigt FWHM

    References
    ----------
    .. [Whiting-1968] `Whiting 1968 "An empirical approximation to the Voigt profile", JQSRT <https://www.sciencedirect.com/science/article/pii/0022407368900812>`_

    Used in the expression of [Olivero-1977]_

    Notes
    -----
    Performances:

    using @jit yield a performance increase from 8.9s down to 5.1s
    on a 50k lines, 250k wavegrid case (performances.py)

    See Also
    --------
    :py:func:`~radis.lbl.broadening.olivero_1977`

    """
    # Calculate some temporary arrays
    # ... fasten up the calculation by 25% (ex: test on 20 cm-1, ~6000 lines:
    # ... 20.5.s > 16.5s) on the total eq_spectrum calculation
    # ... w_wv is typically a (10.001, 1997) array
    w_wv = w_centered / wv  # w_centered can be ~500 Mb
    w_wv_2 = w_wv**2
    wl_wv = wl / wv
    w_wv_225 = np.abs(w_wv) ** 2.25

    # Calculate!  (>>> this is the performance bottleneck <<< : ~ 2/3 of the time spent
    #              on lineshape equation below + temp array calculation above
    #              In particular exp(...) and ()**2.25 are very expensive <<< )
    # ... Voigt 1st order approximation
    lineshape = (
        (1 - wl_wv) * np.exp(-2.772 * w_wv_2)
        + wl_wv * 1 / (1 + 4 * w_wv_2)
        # ... 2nd order correction
        + 0.016 * (1 - wl_wv) * wl_wv * (np.exp(-0.4 * w_wv_225) - 10 / (10 + w_wv_225))
    )
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
    return df


# Compute broadened spectrum
def broadening_full(df, nu_min, nu_max, nu_steps):
    # Abbreviate dataframe
    df_short = df[ (df['nu'] >= nu_min) & (df['nu'] <= nu_max) ]

    # Initiate wavenumber array & list for broadening cross-sections
    nu_arr = np.arange(nu_min, nu_max, nu_steps)
    sigma_nu = np.zeros_like(nu_arr)
    j_nu = np.zeros_like(nu_arr)

    # Compute broadening for each line j at all wavenumbers nu
    for _, row in df_short.iterrows():
        # Narrow down wavenumber array
        nu_j = row['nu']
        nu_j_min = nu_j - 20.
        nu_j_max = nu_j + 20.
        nu_mask = (nu_arr>nu_j_min) & (nu_arr<nu_j_max)
        nu_j_arr = nu_arr [ nu_mask ]

        # Line broadening function (approximate Voigt shape)
        broadening_j = broaden_whiting1968(nu=nu_j_arr, nu_0=nu_j,
                                           w_v=row['w_v'], w_l=row['w_l'])
        
        # Absorption cross-section
        sigma_nu_j = row['sigma_v0'] * broadening_j
        sigma_nu[nu_mask] = sigma_nu[nu_mask] + sigma_nu_j

        # Emission cross-section
        j_nu_j = row['epsilon'] * broadening_j
        j_nu[nu_mask] = j_nu[nu_mask] + j_nu_j

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

    return nu[left_bound:right_bound], broadened_spectrum[left_bound:right_bound], 
