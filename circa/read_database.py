# FUNCTION:     Parse saved spectral line databases into Pandas dataframes
# AUTHOR:       I.C. Dumitrescu

# Import external libraries
import json
import os
import pandas as pd

# Import internal functions
from circa.isotopologues import local_to_global_iso
from circa.networking import fetch_hitran, fetch_hitemp

# Define HITRANonline website domain
HITRAN_HOST = 'https://hitran.org'

# API endpoint paths for HITRAN and HITEMP databases
DB_ENDPOINTS = {
    'hitran': '/lbl/api',           # real API endpoint
    'hitemp': '/files/HITEMP/',     # URL from which different compressed database files can be downloaded
}

# HITRAN molecule IDs, chemical formulae, formal names
MOLECULE_NAMES = {
    1: {"formula": "H2O",       "name": "Water"},
    2: {"formula": "CO2",       "name": "Carbon Dioxide"},
    3: {"formula": "O3",        "name": "Ozone"},
    4: {"formula": "N2O",       "name": "Nitrous Oxide"},
    5: {"formula": "CO",        "name": "Carbon Monoxide"},
    6: {"formula": "CH4",       "name": "Methane"},
    7: {"formula": "O2",        "name": "Oxygen"},
    8: {"formula": "NO",        "name": "Nitric Oxide"},
    9: {"formula": "SO2",       "name": "Sulfur Dioxide"},
    10: {"formula": "NO2",      "name": "Nitrogen Dioxide"},
    11: {"formula": "NH3",      "name": "Ammonia"},
    12: {"formula": "HNO3",     "name": "Nitric Acid"},
    13: {"formula": "OH",       "name": "Hydroxyl"},
    14: {"formula": "HF",       "name": "Hydrogen Fluoride"},
    15: {"formula": "HCl",      "name": "Hydrogen Chloride"},
    16: {"formula": "HBr",      "name": "Hydrogen Bromide"},
    17: {"formula": "HI",       "name": "Hydrogen Iodide"},
    18: {"formula": "ClO",      "name": "Chlorine Monoxide"},
    19: {"formula": "OCS",      "name": "Carbonyl Sulfide"},
    20: {"formula": "H2CO",     "name": "Formaldehyde"},
    21: {"formula": "HOCl",     "name": "Hypochlorous Acid"},
    22: {"formula": "N2",       "name": "Nitrogen"},
    23: {"formula": "HCN",      "name": "Hydrogen Cyanide"},
    24: {"formula": "CH3Cl",    "name": "Methyl Chloride"},
    25: {"formula": "H2O2",     "name": "Hydrogen Peroxide"},
    26: {"formula": "C2H2",     "name": "Acetylene"},
    27: {"formula": "C2H6",     "name": "Ethane"},
    28: {"formula": "PH3",      "name": "Phosphine"},
    29: {"formula": "COF2",     "name": "Carbonyl Fluoride"},
    30: {"formula": "SF6",      "name": "Sulfur Hexafluoride"},
    31: {"formula": "H2S",      "name": "Hydrogen Sulfide"},
    32: {"formula": "HCOOH",    "name": "Formic Acid"},
    33: {"formula": "HO2",      "name": "Hydroperoxyl"},
    34: {"formula": "O",        "name": "Oxygen Atom"},
    35: {"formula": "ClONO2",   "name": "Chlorine Nitrate"},
    36: {"formula": "NO+",      "name": "Nitric Oxide Cation"},
    37: {"formula": "HOBr",     "name": "Hypobromous Acid"},
    38: {"formula": "C2H4",     "name": "Ethylene"},
    39: {"formula": "CH3OH",    "name": "Methanol"},
    40: {"formula": "CH3Br",    "name": "Methyl Bromide"},
    41: {"formula": "CH3CN",    "name": "Acetonitrile"},
    42: {"formula": "CF4",      "name": "PFC-14"},
    43: {"formula": "C4H2",     "name": "Diacetylene"},
    44: {"formula": "HC3N",     "name": "Cyanoacetylene"},
    45: {"formula": "H2",       "name": "Hydrogen"},
    46: {"formula": "CS",       "name": "Carbon Monosulfide"},
    47: {"formula": "SO3",      "name": "Sulfur Trioxide"},
    48: {"formula": "C2N2",     "name": "Cyanogen"},
    49: {"formula": "COCl2",    "name": "Phosgene"},
    50: {"formula": "SO",       "name": "Sulfur Monoxide"},
    51: {"formula": "CH3F",     "name": "Methyl Fluoride"},
    52: {"formula": "GeH4",     "name": "Germane"},
    53: {"formula": "CS2",      "name": "Carbon Disulfide"},
    54: {"formula": "CH3I",     "name": "Methyl Iodide"},
    55: {"formula": "NF3",      "name": "Nitrogen Trifluoride"},
    56: {"formula": "H3+",      "name": "Trihydrogen Cation"},
    57: {"formula": "CH3",      "name": "Methyl Radical"},
    58: {"formula": "S2",       "name": "Sulfur Dimer"},
    59: {"formula": "COFCl",    "name": "Carbonyl Chlorofluoride"},
    60: {"formula": "HONO",     "name": "Nitrous Acid"},
    61: {"formula": "ClNO2",    "name": "Nitryl Chloride"},
}

# Isotopologues available for HITEMP molecules (as of March 2026)
# H2O is a special case: Currently partitioned by wavenumber range into several .zip files
HITEMP_ISOTOPOLOGUES = {
    1:  {"iso_max":  6, "url": "HITEMP-2010/H2O_line_list/",        "partitioned": True},  # H2O
    2:  {"iso_max": 12, "url": "bzip2format/02_HITEMP2024.par.bz2", "partitioned": False}, # CO2
    4:  {"iso_max":  5, "url": "bzip2format/04_HITEMP2019.par.bz2", "partitioned": False}, # N2O
    5:  {"iso_max":  6, "url": "bzip2format/05_HITEMP2019.par.bz2", "partitioned": False}, # CO
    6:  {"iso_max":  4, "url": "bzip2format/06_HITEMP2020.par.bz2", "partitioned": False}, # CH4
    8:  {"iso_max":  3, "url": "bzip2format/08_HITEMP2019.par.bz2", "partitioned": False}, # NO
    10: {"iso_max":  1, "url": "bzip2format/10_HITEMP2019.par.bz2", "partitioned": False}, # NO2
    13: {"iso_max":  3, "url": "bzip2format/13_HITEMP2020.par.bz2", "partitioned": False}, # OH
}

# Column widths and headers for the 160-character HITRAN .par fixed-width format.
# Molecule-specific formats unpack the quantum-number fields differently.
HITRAN_FORMAT = {
    'general': {
        'molec_id': 2, 'isotopologue': 1, 
        'nu': 12, 'sw': 10, 'a': 10, 
        'gamma_air': 5, 'gamma_self': 5, 
        'E_l': 10, 'n_air': 4, 'delta_air': 8, 
        'V_u': 15, 'V_l': 15,                                   # global upper and lower quanta
        'Q_u': 15, 'Q_l': 15,                                   # local upper and lower quanta
        'i_err': 6, 'i_ref': 12, 'flag': 1, 
        'g_u': 7, 'g_l': 7
        },
    'CO': {
        'molec_id': 2, 'isotopologue': 1,
        'nu': 12, 'sw': 10, 'a': 10, 
        'gamma_air': 5, 'gamma_self': 5, 
        'E_l': 10, 'n_air': 4, 'delta_air': 8, 
        'v_u': 15, 'v_l': 15,                                   # upper and lower vibrational quantum numbers
        'F_u': 15,                                              # upper state total angular momentum quantum number (incl. nuclear spin)
        'branch': 6, 'J_l': 3, 'sym_l': 1,                      # branch, rotational quantum number, symmetry/parity
        'F_l': 5,                                               # lower state total angular momentum quantum number (incl. nuclear spin)
        'i_err': 6, 'i_ref': 12, 'flag': 1, 
        'g_u': 7, 'g_l': 7
        },
    'CO2': {
        'molec_id': 2, 'isotopologue': 1, 
        'nu': 12, 'sw': 10, 'a': 10, 
        'gamma_air': 5, 'gamma_self': 5, 
        'E_l': 10, 'n_air': 4, 'delta_air': 8, 
        'v1_u': 8, 'v2_u': 2, 'l2_u': 2, 'v3_u': 2, 'r_u': 1,   # global upper quanta: v1, v2, l2, v3, r
        'v1_l': 8, 'v2_l': 2, 'l2_l': 2, 'v3_l': 2, 'r_l': 1,   # global lower quanta: v1, v2, l2, v3, r
        'F_u': 15,                                              # upper state total angular momentum quantum number (incl. nuclear spin)
        'branch': 6, 'J_l': 3, 'sym_l': 1,                      # branch, rotational quantum number, symmetry/parity
        'F_l': 5,                                               # lower state total angular momentum quantum number
        'i_err': 6, 'i_ref': 12, 'flag': 1, 
        'g_u': 7, 'g_l': 7
        },
}

# ---------------------------------------------------------------------------
# Check cache for existing files
# ---------------------------------------------------------------------------
def _find_covering_cache(database, mol_form, requested_global_ids, nu_min, nu_max):
    """
    Return (data_path, header) for a cached file that covers that covers requested wavenumber range 
    AND contains  superset of the requested isotopologues, or None if no such file exists.
    """
    if not os.path.exists(database):
        return None
    
    requested = set(requested_global_ids)

    for fname in os.listdir(database):
        # Header file must start with the molecule formula and end with .header
        if not fname.endswith('.header'):
            continue
        if not fname.startswith(f'{mol_form}_iso'):
            continue
        
        # Load .header file
        with open(os.path.join(database, fname)) as f:
            h = json.load(f)

        # Check isotopologues: caches set must contain all requested ones
        if database == 'hitran':
            cached_ids = set(h.get('iso_ids', []))
            if not requested.issubset(cached_ids):
                continue
        # TODO: Extend once filtering by isotopologue IDs has been implemented (right now they are all saved in the .par file)

        # Check wavenumber range
        if h.get('nu_min', float('inf')) <= nu_min and h.get('nu_max', float('-inf')) >= nu_max:
            data_path = os.path.join(database, fname.replace('.header', '.data'))
            if os.path.exists(data_path):
                return data_path, h     # If a cached file exists, return file path and header
    
    return None                         # If no corresponding cached file exists, return None


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def get_dataframe(mol_form, mol_id, iso, nu_min, nu_max, database='hitran'):
    """
    Return a DataFrame of HITRAN/HITEMP lines for requested isotopologue in a wavenumber range
    Downloads the data from HITRANonline if the local file does not yet exist

    Parameters
    ----------
    mol_form   : str            Chemical formula, e.g. 'CO2'. Used to select the parse format
    mol_id     : int            HITRAN molecule ID
    iso        : int or lst     HITRAN isotopologue number (1 = most abundant)
    nu_min     : float          Lower wavenumber bound (cm⁻¹)
    nu_max     : float          Upper wavenumber bound (cm⁻¹)
    database   : str            Database to query, 'hitran' (default) or 'hitemp'; local directory for cached .data
    """
    database = database.lower()
    if database not in DB_ENDPOINTS:
        raise ValueError(f"CIRCA: 'database' must be 'hitran' or 'hitemp', got {database!r}.")
    
    # Check requests for HITEMP
    if database == 'hitemp' and mol_id not in HITEMP_ISOTOPOLOGUES.keys():
        mol_formula = MOLECULE_NAMES.get(mol_id, {}).get('formula', str(mol_id))
        raise ValueError(
            f"{mol_formula} (molecule ID {mol_id}) is not currently available in HITEMP. "
            f"HITEMP molecules available: "
            f"{[MOLECULE_NAMES[m]['formula'] for m in sorted(HITEMP_ISOTOPOLOGUES.keys())]}"
        )
    
    # Handle isotopologue input by sorting it all into a list and forming a string for the file name ('1-2-3')
    iso_lst = sorted( [iso] if isinstance(iso, int) else list(iso) )
    iso_str = '-'.join(map(str, iso_lst))
    
    # For HITEMP requests, check if the isotopologues are even in the database
    if database == 'hitemp':
        for i in iso_lst:
            max_iso = HITEMP_ISOTOPOLOGUES.get(mol_id)['iso_max']
            if i > max_iso:
                raise ValueError(f"{mol_form} isotopologue {i} is not available in HITEMP (maximum local_iso={max_iso}).")

    # Convert requested local isotopologues to global IDs
    requested_global_ids = [ local_to_global_iso(mol_id, i) for i in iso_lst ]

    # Check for existing files encompassing the requested wavenumber range
    cache = _find_covering_cache(database, mol_form, requested_global_ids, nu_min, nu_max)
    
    if cache is not None: 
        data_path = cache[0]
        print(f"CIRCA: Using cached file: {data_path}")
    elif database == 'hitemp':
        # Check if the user ACTUALLY wants to use HITEMP
        print(f"\nCIRCA: There is no cached HITEMP data for {mol_form}, isotopologue(s) {iso_str}, wavenumber range {nu_min}-{nu_max} cm^-1.")
        print("       Downloading the data from HITRANonline will require a login and may take several minutes.")
        print("       Note that HITRAN is generally sufficient for most cases with temperatures up to approx. 1500K.")
        while True:
            patience = input(f"\nCIRCA: Do you want to download the HITEMP lines for {mol_form} ({nu_min}-{nu_max} cm^-1)? (Y/N) ")
            if patience[0].lower() == 'y':
                break
            elif patience[0].lower() == 'n':
                return
            else:
                continue
        print(f"CIRCA: Downloading HITEMP data for {mol_form} ({nu_min}-{nu_max}  cm^-1) ...")
        data_path, _ = fetch_hitemp(mol_form, mol_id, nu_min, nu_max, output_dir=database)
    else:
        print(f"CIRCA: Downloading HITRAN data for {mol_form} isotopologue(s) {iso} ({nu_min}–{nu_max} cm^-1) ...")
        table_name = f'{mol_form}_iso-{iso_str}_{nu_min}-{nu_max}cm-1'  # Creates a file name for the data downloaded
        data_path  = os.path.join(database, table_name + '.data')
        fetch_hitran(table_name, iso_ids=requested_global_ids, nu_min=nu_min, nu_max=nu_max, output_dir=database)
    
    # Read .par file into a pandas dataframe
    table_format = HITRAN_FORMAT.get(mol_form, HITRAN_FORMAT['general'])        # Get table format, return general if there's no special one
    df = pd.read_fwf(data_path, widths=table_format.values(), header=None)      # Read out fixed-width data
    df.columns = table_format.keys()                                            # Assign column headers
    df = df[ df['isotopologue'].isin(iso_lst) & (df['nu'] >= nu_min) & (df['nu'] <= nu_max)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Molecule lookup
# ---------------------------------------------------------------------------

def get_molecule_id(search_term):
    """
    Resolve a molecule name, formula, or numeric ID string to (mol_id, formula)

    Parameters
    ----------
    search_term : str
        Molecule name (e.g. 'water'), formula (e.g. 'H2O'), or integer ID (e.g. '1')

    Returns
    -------
    (mol_id, formula) : (int, str)

    Raises
    ------
    ValueError if the molecule is not found
    """
    if search_term.isdigit():
        mol_id = int(search_term)
        if mol_id in MOLECULE_NAMES:
            return mol_id, MOLECULE_NAMES[mol_id]['formula']
        raise ValueError(
            f"Molecule ID {mol_id!r} not recognised. "
            f"Valid IDs: {sorted(MOLECULE_NAMES)}."
        )

    term = search_term.lower()
    for mol_id, data in MOLECULE_NAMES.items():
        if data['name'].lower() == term or data['formula'].lower() == term:
            return mol_id, data['formula']

    raise ValueError(
        f"Molecule {search_term!r} not recognised. "
        f"Valid names/formulae: {[d['formula'] for d in MOLECULE_NAMES.values()]}."
    )


# Test example
if __name__ == '__main__':
    mol_id, mol_form = get_molecule_id('n2o')

    # Read from HITRAN (default)
    print("\nHITRAN:")
    df_hitran = get_dataframe(mol_form, mol_id, iso=1,
                              nu_min=2300, nu_max=2400)
    print(df_hitran.head())

    # Read from HITEMP
    print("\nHITEMP:")
    df_hitemp = get_dataframe(mol_form, mol_id, iso=1,
                              nu_min=2300, nu_max=2400, database='hitemp')
    print(df_hitemp.head())
