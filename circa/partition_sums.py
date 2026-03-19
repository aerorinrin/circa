# FUNCTION:   Retrieve total internal partition sums (TIPS) computed by Gamache et al. (2024)
# AUTHOR:     Gamache et al. (2025), JQSRT 345, 109568 (doi: 10.1016/j.jqsrt.2025.109568)
#             Data obtained from: https://zenodo.org/records/17191976 
#             Edited by I.C. Dumitrescu for adaptation to CIRCA

# Import external libraries
import os
import numpy as np
import pickle
from pathlib import Path

# Define name of TIPS data folder
data_folder = Path(__file__).parent.parent / 'tips_2024'

# Function for fetching TIPS data from .QTpy files
def iso_QT(mol_id, iso_id, T):
    """
    Fetches tabulated partition sums from the TIPS data folder.
    """
    filename = os.path.join(data_folder, '{:d}.QTpy'.format(mol_id))
    with open(filename, 'rb') as handle:
        QTdict = pickle.loads(handle.read())
  
    T1 = np.floor(T)
    T2 = np.ceil(T)

    Q1 = QTdict[str(iso_id)][T1]
    Q2 = QTdict[str(iso_id)][T2]
    
    QT = Q1 + (Q2-Q1)*(T-T1)
    
    return QT

# Test function
if __name__ == '__main__':
    print(iso_QT(2, 3, 203))
