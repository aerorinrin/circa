# CIRCA

CIRCA, the Comprehensive InfraRed Carbon Absorption simulator, is a Python-based molecular spectrum simulation programme.
It can be used to generate equilibrium spectra of molecules represented in the HITRAN or HITEMP spectral databases and nonequilibrium spectra of the three most abundant isotopologues of carbon monoxide (CO) and carbon dioxide (CO<sub>2</sub>).

This code was developed as part of my master's thesis at the Institute of Space Systems, titled, "Application of Infrared Spectroscopy for In-Situ Characterisation of a CO<sub>2</sub> Plasma in PWK3".
The structure and approach implemented in CIRCA are heavily based on the works of Klarenaar et al. (2017) [1], Urbanietz et al. (2018) [2], and Stewig et al. (2020) [3], as well as RADIS (the RADIative Solver) originally developed by Pannier & Laux (2018) [4].
RADIS was a particularly valuable resource for figuring out some of the more obscure simulation issues, and the RADIS developers on Slack were very kind to answer my questions pertaining to the code and point me towards further relevant spectroscopy textbooks.


## Quickstart

### Installation

Given that Python and `pip` are already installed, CIRCA can be installed from this GitHub repository using:

```
pip install pip@git+https://github.com/aerorinrin/circa
```

### Simulating spectra

Spectra can be simulated for given input conditions using the `circa.compute_spectrum.spectrum` function.
Note that CIRCA simulations distinguish between spectra of gases in Local Thermodynamic Equilibrium (LTE) and spectra in Non-Local Thermodynamic Equilibrium (NLTE).
The user is required to input whether the gas is in LTE (`equilibrium=True`) or NLTE (`equilibrium=False`), which influences the kind of temperature input accepted by the function.

The output of the `circa.compute_spectrum.spectrum` function is a Pandas dataframe containing the wavenumber (`'nu'`) in [cm<sup>-1</sup>], the unitless transmittance (`'transmittance'`), the radiance (`'radiance'`) in [mW sr<sup>-1</sup> cm<sup>-2</sup>], the absorption cross-section (`'absorption_cross_section'`) in [cm<sup>2</sup> molecule<sup>-1</sup>], and the emission cross-section (`'emission_cross_section'`) in [mW sr<sup>-1</sup> cm<sup>-3</sup>].


#### LTE (equilibrium)

The LTE spectrum simulation in CIRCA is based on the line scaling approach for HITRAN-like databased as outlined by Šimečková et al. (2006) [5], as well as on the HITRANonline website (https://hitran.org/).

In the following code snippet, an equilibrium spectrum of CO is simulated using the HITRAN database:

```
from circa.compute_spectrum import spectrum

spec_CO = spectrum( equilibrium = True,           # molecule is in LTE
                    molecule = 'carbon monoxide', # simulate carbon monoxide
                    iso_number = [1, 2],          # isotopologues 1 and 2
                    nu_min = 2000,                # [cm^-1] minimum wavenumber
                    nu_max = 2300,                # [cm^-1] maximum wavenumber
                    nu_step = 0.005,              # [cm^-1] wavenumber step size
                    T = 500,                      # [K] gas temperature
                    p_gas = 0.1,                  # [bar] gas pressure
                    x = 0.52,                     # [-] mole fraction of CO
                    L = 20,                       # [cm] optical path length
                    database = 'hitran' )         # use HITRAN line data
```

#### NLTE (nonequilibrium)

The following code snippet is used to simulate a nonequilibrium spectrum of CO<sub>2</sub> using the HITEMP database, which is suitable for higher temperatures than HITRAN:

```
from circa.compute_spectrum import spectrum

spec_CO2 = spectrum( equilibrium = False,       # molecule is in NLTE
                     molecule = 'CO2',          # simulate carbon dioxide
                     iso_number = 1,            # only first isotopologue
                     nu_min = 2200,             # [cm^-1] minimum wavenumber
                     nu_max = 2500,             # [cm^-1] maximum wavenumber
                     nu_step = 0.01,            # [cm^-1] wavenumber step size
                     vibration = 'treanor',     # vibrational state distribution
                     T = [800, 1200, 1500],     # [K] T_rot, T_vib12, T_vib3
                     p_gas = 0.32,              # [bar] gas pressure
                     x = 0.9,                   # [-] mole fraction of CO2
                     L = 40,                    # [cm] optical path length
                     database = 'hitemp' )      # use HITEMP line data
```

### Applying instrumental broadening

The Instrument Line Shape (ILS) functions in CIRCA are adapted from the HITRAN Application Programming Interface (HAPI) [6, 7]. 
Theoretical instrumental broadening can be applied to a molecule spectrum via a convolution with the ILS function.
An example is seen below:

```
from circa.broadening import apply_ils, ils_michelson

nu_broad, trans_broad = apply_ils( nu = spec_CO['nu'],
                                   intensity = spec_CO['transmittance'],
                                   ils_function = ils_michelson,
                                   resolution = 0.4 )   # [cm^-1] instrument resolution parameter gamma
```


## References

[1] Klarenaar, B. L. M., Engeln, R., van den Bekerom, D. C. M., van de Sanden, M. C. M., Morillo-Candas, A. S., and Guaitella, O., “Time evolution of vibrational temperatures in a CO2 glow discharge measured with infrared absorption spectroscopy,” *Plasma Sources Science and Technology*, Vol. 26, No. 11, 2017, p. 115008. https://doi.org/10.1088/1361-6595/aa902e.

[2] Urbanietz, T., Böke, M., Gathen, V., and Keudell, A., “Non-equilibrium excitation of CO2 in an atmospheric pressure helium plasma jet,” *Journal of Physics D: Applied Physics*, Vol. 51, 2018. https://doi.org/10.1088/1361-6463/aad4d3.

[3] Stewig, C., Schüttler, S., Urbanietz, T., Böke, M., and von Keudell, A., “Excitation and dissociation of CO2 heavily diluted in noble gas atmospheric pressure plasma,” *Journal of Physics D: Applied Physics*, Vol. 53, No. 12, 2020, p. 125205. https://doi.org/10.1088/1361-6463/ab634f.

[4] Pannier, E., and Laux, C. O., “RADIS: A nonequilibrium line-by-line radiative code for CO2 and HITRAN-like database species,” *Journal of Quantitative Spectroscopy and Radiative Transfer*, Vol. 222-223, 2019, pp. 12–25. https://doi.org/10.1016/j.jqsrt.2018.09.027.

[5] Šimečková, M., Jacquemart, D., Rothman, L., Gamache, R., and Goldman, A., “Einstein A-coefficients and statistical weights for molecular absorption transitions in the HITRAN database,” *Journal of Quantitative Spectroscopy and Radiative Transfer*, Vol. 98, 2006, pp. 130 – 155. https://doi.org/10.1016/j.jqsrt.2005.07.003.

[6] Kochanov, R., Gordon, I., Rothman, L., Wcis, P., Hill, C., and Wilzewski, J., “HITRAN Application Programming Interface (HAPI): A comprehensive approach to working with spectroscopic data,” *Journal of Quantitative Spectroscopy and Radiative Transfer*, Vol. 177, 2016, pp. 15–30. https://doi.org/10.1016/j.jqsrt.2016.03.005.

[7] Kochanov, R. V., *HITRAN Application Programming Interface (HAPI): User Guide*, HITRAN; Harvard-Smithsonian Center for Astrophysics, April 2019.
