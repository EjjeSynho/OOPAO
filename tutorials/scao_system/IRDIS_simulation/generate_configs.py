#%%
import sys
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '../../..')

from data_processing.SPHERE_data import LoadSPHEREsampleByID
from tools.config_manager import ConfigManager, GetSPHEREonsky
from tools.parameter_parser import ParameterParser

with open("../IRDIS_simulation/settings.json", "r") as f:
    file_data = json.load(f)
    DATA_PATH = file_data['path_data']
    CONFIGS_PATH = file_data['path_configs']
        
with open(DATA_PATH+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['invalid'] == False]
# psf_df = psf_df[psf_df['Wind speed (200 mbar)'].notna()]
# psf_df = psf_df[psf_df['Class A'] == True]
psf_df = psf_df[psf_df['λ left (nm)'] == 1625] # SELECT ONLY 1625 NM

selected_ids = psf_df.index.values.tolist()


def GenerateConfig(sample_id):
    data_sample = LoadSPHEREsampleByID(sample_id)

    path_ini = 'C:/Users/akuznets/Projects/TipToy/data/parameter_files/irdis.ini'
    config_file = ParameterParser(path_ini).params
    config_manager = ConfigManager(GetSPHEREonsky())
    config_file = config_manager.Modify(config_file, data_sample)
    # config_manager.Convert(merged_config, framework='numpy')

    root = 'C:/Users/akuznets/Projects/TipToy/'
    config_file['telescope']['PathPupil']     = root + 'data/calibrations/VLT_CALIBRATION/VLT_PUPIL/ALC2LyotStop_measured.fits'
    config_file['telescope']['PathApodizer']  = root + 'data/calibrations/VLT_CALIBRATION/VLT_PUPIL/APO1Apodizer_measured_All.fits'
    config_file['telescope']['PathStatModes'] = root + 'data/calibrations/VLT_CALIBRATION/VLT_STAT/LWEMODES_320.fits'
    config_file['atmosphere']['Cn2Weights']   = [0.95, 0.05]
    config_file['atmosphere']['Cn2Heights']   = [0, 10000]
    config_file['sensor_science']['DIT'] = data_sample['Integration']['DIT']
    config_file['sensor_science']['Num. DIT'] = data_sample['Integration']['Num. DITs']
    
    del config_file['sources_science']['Wavelength']['range L']
    del config_file['sources_science']['Wavelength']['range R']
    del config_file['sources_science']['Wavelength']['spectrum L']
    del config_file['sources_science']['Wavelength']['spectrum R']

    return config_file


def convert_numpy_dict(data):
    if isinstance(data, dict):
        return { key: convert_numpy_dict(value) for key, value in data.items() }
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

#%%
for id in selected_ids:
    config = convert_numpy_dict( GenerateConfig(id) )

    with open(CONFIGS_PATH + str(id) + '.json', 'w') as fp:
        json.dump(config, fp, indent=4)

