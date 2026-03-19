# FUNCTION:     Fetch HITRAN & HITEMP spectral line databases from HITRANonline;
#               Contains all the network interaction functions
# AUTHOR:       I.C. Dumitrescu

# Import external libraries
import bz2
import getpass
import json
import os
import zipfile
import urllib.request
import requests
from tqdm import tqdm

# Define HITRANonline website domain
HITRAN_HOST = 'https://hitran.org'

# API endpoint paths for HITRAN and HITEMP databases
DB_ENDPOINTS = {
    'hitran': '/lbl/api',           # real API endpoint
    'hitemp': '/files/HITEMP/',     # URL from which different compressed database files can be downloaded
}

# Isotopologues available for HITEMP molecules (as of March 2026)
# H2O is a special case: Currently partitioned by wavenumber range into several .zip files
HITEMP_ISOTOPOLOGUES = {
    1:  {"mol_form": "H2O", "iso_max":  6, "url": "HITEMP-2010/H2O_line_list/",        "partitioned": True},
    2:  {"mol_form": "CO2", "iso_max": 12, "url": "bzip2format/02_HITEMP2024.par.bz2", "partitioned": False},
    4:  {"mol_form": "N2O", "iso_max":  5, "url": "bzip2format/04_HITEMP2019.par.bz2", "partitioned": False},
    5:  {"mol_form": "CO",  "iso_max":  6, "url": "bzip2format/05_HITEMP2019.par.bz2", "partitioned": False},
    6:  {"mol_form": "CH4", "iso_max":  4, "url": "bzip2format/06_HITEMP2020.par.bz2", "partitioned": False},
    8:  {"mol_form": "NO",  "iso_max":  3, "url": "bzip2format/08_HITEMP2019.par.bz2", "partitioned": False},
    10: {"mol_form": "NO2", "iso_max":  1, "url": "bzip2format/10_HITEMP2019.par.bz2", "partitioned": False},
    13: {"mol_form": "OH",  "iso_max":  3, "url": "bzip2format/13_HITEMP2020.par.bz2", "partitioned": False},
}

# H2O partition files for different wavenumber ranges (nu_min, nu_max, filename)
# TODO: Update if and when this is bundled into a bzip2 archive (like CO2)
HITEMP_H2O_PARTITIONS = [
    (     0,   500, "01_HITEMP2010_00000-00500_H2O.zip"),
    (   500,  1000, "01_HITEMP2010_00500-01000_H2O.zip"),
    (  1000,  2000, "01_HITEMP2010_01000-02000_H2O.zip"),
    (  2000,  3000, "01_HITEMP2010_02000-03000_H2O.zip"),
    (  3000,  4000, "01_HITEMP2010_03000-04000_H2O.zip"),
    (  4000,  5000, "01_HITEMP2010_04000-05000_H2O.zip"),
    (  5000,  6000, "01_HITEMP2010_05000-06000_H2O.zip"),
    (  6000,  7000, "01_HITEMP2010_06000-07000_H2O.zip"),
    (  7000,  8000, "01_HITEMP2010_07000-08000_H2O.zip"),
    (  8000,  9000, "01_HITEMP2010_08000-09000_H2O.zip"),
    (  9000, 11000, "01_HITEMP2010_09000-11000_H2O.zip"),
    ( 11000, 30000, "01_HITEMP2010_11000-30000_H2O.zip"),
]

# ---------------------------------------------------------------------------
# Helper function: log in to HITRANonline
# ---------------------------------------------------------------------------
def _hitran_login(email=None, password=None):
    """
    Log in to HITRANonline and return and authenticates requests.Session;
    Prompts for credentials interactively if not already provided.
    """
    print("\nCIRCA: In order to download data from the HITEMP database, a HITRANonline login is required.")

    if email is None:
        email = input("\t HITRANonline email:    ")
    if password is None:
        password = getpass.getpass("\t HITRANonline password: ")
    
    session = requests.Session()
    login_url = f"{HITRAN_HOST}/login/"

    # Get the login page to retrieve the CSRF token
    resp = session.get(login_url)
    resp.raise_for_status()

    # Django embeds csrfmiddlewaretoken as a hidden input field
    import re
    match = re.search(r'csrfmiddlewaretoken["\s]+value="([^"]+)"', resp.text)
    if not match:
        # Fallback: Token may be in the cookie
        csrf = session.cookies.get('csrftoken', '')
    else:
        csrf = match.group(1)

    payload = {
        'csrfmiddlewaretoken': csrf,
        'email': email,
        'password': password,
    }
    headers = {'Referer': login_url}
    resp = session.post(login_url, data=payload, headers=headers)
    resp.raise_for_status()

    # HITRANonline redirects to '/' on sucess, and stays on '/login/' on failure
    if '/login/' in resp.url:
        raise RuntimeError("HITRANonline login failed. Check your email and password.")

    print("\n\t Successfully logged in to HITRANonline. \n")
    return session

# ---------------------------------------------------------------------------
# Fetch HITRAN over API endpoint
# ---------------------------------------------------------------------------
def fetch_hitran(table_name, iso_ids, nu_min, nu_max, params=None, output_dir='.'):
    """
    Download line-by-line data from HITRANonline API and save to local files

    Parameters
    ----------
    table_name (str): Base name for the output files (e.g. 'CO2_iso1-2-3_4000-4100cm-1')
    iso_ids (list of int): Global isotopologue IDs (see https://hitran.org/docs/iso-meta/)
    nu_min, nu_max (float): Wavenumber range (cm⁻¹)
    db (str): Database to query: 'hitran' (default) or 'hitemp'
    params (list of str, optional): Extra parameter names for custom output, otherwise use standard 160-character .par format
    output_dir (str): Directory for output files; created if absent
    """
    os.makedirs(output_dir, exist_ok=True)

    iso_ids_str = ','.join(str(i) for i in iso_ids)
    db = 'hitran'
    endpoint = DB_ENDPOINTS[db]

    if params:
        url = (
            f"{HITRAN_HOST}{endpoint}?"
            f"iso_ids_list={iso_ids_str}&"
            f"numin={nu_min}&numax={nu_max}&"
            f"fixwidth=0&sep=[comma]&"
            f"request_params={','.join(params)}"
        )
    else:
        url = (
            f"{HITRAN_HOST}{endpoint}?"
            f"iso_ids_list={iso_ids_str}&"
            f"numin={nu_min}&numax={nu_max}"
        )

    data_path   = os.path.join(output_dir, table_name + '.data')
    header_path = os.path.join(output_dir, table_name + '.header')

    print(f"Fetching ({db.upper()}): {url}")
    try:
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: failed to retrieve data for {table_name}.") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot connect to {HITRAN_HOST}: {e.reason}") from e

    with open(data_path, 'w', newline='\n') as f:
        while True:
            chunk = response.read(64*1024)
            if not chunk:
                break
            f.write(chunk.decode('utf-8'))
    print(f"  Data -> {data_path}")

    with open(header_path, 'w') as f:
        json.dump({
            'table_name': table_name,
            'iso_ids':    list(iso_ids),
            'nu_min':     nu_min,
            'nu_max':     nu_max,
            'params':     params,
            'source_url': url,
        }, f, indent=2)
    print(f"  Header -> {header_path}")


# ---------------------------------------------------------------------------
# Fetch HITEMP data
# ---------------------------------------------------------------------------
def fetch_hitemp(mol_form, mol_id, nu_min, nu_max, output_dir='hitemp', email=None, password=None):
    """
    Download compressed HITEMP line data from HITRANonline to the output directory, decompress, 
    filter for requested wavenumber regime, and save as .data and .header file pair.
    Note: All molecules except water are available as bzip2 files as of March 2026!
    TODO: Add filtering by isotopologue IDs!

    Args:
        mol_form (str)      : Chemical formula of the molecule (e.g. 'CO2')
        mol_id (int)        : HITRAN molecule ID
        nu_min (float)      : Lower bound wavenumber (1/cm)
        nu_max (float)      : Upper bound wavenumber (1/cm)
        output_dir (str)    : Directory for output files. Defaults to 'hitemp'

    Returns: (data_path, table_name) (str, str)
    """
    if mol_id not in HITEMP_ISOTOPOLOGUES:
        raise ValueError(
            f"{mol_form} (molecule ID {mol_id}) is not currently available in HITEMP. "
            f"HITEMP molecules available: {[HITEMP_ISOTOPOLOGUES[m]['mol_form'] for m in HITEMP_ISOTOPOLOGUES]}"
        )

    # Make sure output directory and compressed data folder actually exist, create if not
    os.makedirs(output_dir, exist_ok=True)
    compressed_dir = os.path.join(output_dir, 'compressed')
    os.makedirs(compressed_dir, exist_ok=True)

    meta = HITEMP_ISOTOPOLOGUES[mol_id]
    HITEMP_BASE_URL = HITRAN_HOST + DB_ENDPOINTS['hitemp']

    table_name = f'{mol_form}_iso-all_{nu_min}-{nu_max}cm-1'
    data_path = os.path.join(output_dir, table_name + '.data')
    header_path = os.path.join(output_dir, table_name + '.header')

    line_count = 0

    # Check for partitions (AKA only applicable for H2O)
    if meta['partitioned']:
        partitions = [ p for p in HITEMP_H2O_PARTITIONS if p[0] < nu_max and p[1] > nu_min ]    # Figure out required partitions
        if not partitions:
            raise ValueError(f"No H2O lines found in HITEMP for {nu_min}-{nu_max} cm^-1.")      # Error if wavenumber range not represented
    
        urls = [ HITEMP_BASE_URL + meta['url'] + fname for _, _, fname in partitions ] 
        print(f"Fetching (HITEMP H2O): {len(urls)} partition file(s) ...")

        # Download .zip archives to compressed data folder
        zip_paths = []
        for url in urls:
            fname = url.split('/')[-1]
            zip_path = os.path.join(compressed_dir, fname)
            zip_paths.append(zip_path)

            # Check for existing downloads before downloading more
            if os.path.exists(zip_path):
                print(f"  Skipping download, file already exists on disk: {zip_path}")
            else:
                session = _hitran_login(email=email, password=password)
                print(f"  Downloading {fname} ...")
                resp = session.get(url, stream=True)
                resp.raise_for_status()
                file_size = int(resp.headers.get('Content-Length'), 0)
                with tqdm(total=file_size or None, unit='B', unit_scale=True, unit_divisor=1024) as bar:
                    with open(zip_path, 'wb') as output_file:
                        for chunk in resp.iter_content(chunk_size=64*1024):
                            output_file.write(chunk)
                            bar.update(len(chunk))
                print(f"  Saved: {zip_path}")
        print("Finished downloading .zip archives for H2O from HITRANonline.")

        # Decompress relevant partitions & filter for relevant wavenumbers
        print("Decompressing database.")
        with open(data_path, 'w', newline='\n') as out:
            for zip_path in zip_paths:
                print(f"  Decompressing {os.path.basename(zip_path)} ...")
                with zipfile.ZipFile(zip_path) as zf:
                    par_name = next(n for n in zf.namelist() if n.endswith('.par'))
                    with zf.open(par_name) as par:
                        for line in par:
                            # local_iso = line[2]   # TODO: Add filtering by isotopologue IDs for the decompression

                            nu_val = float(line[3:15])
                            if nu_val > nu_max:     # Check where the upper bound wavenumber is exceeded
                                break
                            if nu_val >= nu_min:    # Check that line is above lower bound wavenumber
                                line_count += 1
                                text = line.decode('utf-8').rstrip('\r\n') + '\n'
                                out.write(text)
    
    # For literally everything else
    else:
        url = HITEMP_BASE_URL + meta['url']
        print(f"Fetching (HITEMP {mol_form}): {url}")

        bz2_fname = url.split('/')[-1]
        bz2_path = os.path.join(compressed_dir, bz2_fname)

        # Check for existing .bz2 archives on disk, save if not present
        if os.path.exists(bz2_path):
            print(f"  Skipping download, file already exists on disk: {bz2_path}")
        else:
            session = _hitran_login(email=email, password=password)
            print(f"  Downloading compressed archive to {bz2_path} ...")
            resp = session.get(url, stream=True)
            resp.raise_for_status()
            file_size = int(resp.headers.get('Content-Length'), 0)
            with tqdm(total=file_size or None, unit='B', unit_scale=True, unit_divisor=1024) as bar:
                with open(bz2_path, 'wb') as bz2_file:
                    for chunk in resp.iter_content(chunk_size=64*1024):
                        bz2_file.write(chunk)
                        bar.update(len(chunk))
            print("  Download complete.")

        # Decompress from saved file and filter by wavenumber
        print("  Decompressing .bz2 archive ...")
        decompressor = bz2.BZ2Decompressor()
        leftover = b''                          # bytes literal
        with open(bz2_path, 'rb') as bz2_file, open(data_path, 'w', newline='\n') as out:
            for compressed_chunk in iter(lambda: bz2_file.read(64*1024), b''):
                raw = leftover + decompressor.decompress(compressed_chunk)
                lines = raw.split(b'\n')
                leftover = lines.pop()          # carry over last fragment if incomplete

                for line in lines:
                    if not line:
                        continue
                    # local_iso = line[2]   # TODO: Add filtering by isotopologue IDs for the decompression
                    nu_val = float(line[3:15])
                    if nu_val > nu_max:
                        break
                    if nu_val >= nu_min:
                        line_count += 1
                        out.write(line.decode('utf-8').rstrip('\r\n') + '\n')
    
    # Create corresponding header file
    header_path = os.path.join(output_dir, table_name + '.header')
    with open(header_path, 'w') as header:
        json.dump({
            'table_name':   table_name,
            # 'iso_ids':      list(iso_ids), # TODO: Add isotopologue IDs in the header once filtering is implemented
            'nu_min':       nu_min,
            'nu_max':       nu_max,
            'source_url':   HITEMP_BASE_URL + meta['url'],
        }, header, indent=2)
    print(f"  Header -> {header_path}")
    print(f"  Lines written: {line_count}")

    return data_path, table_name


# Test example
if __name__ == '__main__':
    # # Fetch HITRAN file
    df_hitran = fetch_hitemp(table_name='CO2_iso1-2-3_4000-4100cm-1', mol_id=13, nu_min=1000, nu_max=1500)

    # Fetch HITEMP file
    df_hitemp = fetch_hitemp(mol_form='O3', mol_id=3, nu_min=1000, nu_max=1500)
