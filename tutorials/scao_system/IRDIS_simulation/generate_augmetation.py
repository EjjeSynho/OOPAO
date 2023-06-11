#%%
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
from scipy.stats import gamma
from sklearn.mixture import GaussianMixture
from OOPAO.tools.tools import deg2rad, seeing

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
# psf_df = psf_df[psf_df['λ left (nm)'] == 1625] # Select only 1625 nm
# selected_ids = psf_df.index.values.tolist()

#%%
def GenerateSamplesAugmentation(size):

    # Define the columns to select
    cols_to_select = [
        'Airmass',
        'r0 (SPARTA)',
        'Wind direction (200 mbar)',
        'Wind speed (header)',
        'Wind speed (200 mbar)',
        'Nph WFS']

    fitted_params_and_samples = {}

    def plot_distribution_and_fit(data, bins=50, gamma_color='red'):
        sns.histplot(data, bins=bins, kde=False, color='skyblue', stat='density')
        shape_param, loc, scale_param = gamma.fit(data)

        x = np.linspace(min(data), max(data), 1000)
        y = gamma.pdf(x, shape_param, loc=loc, scale=scale_param)

        # Plot the gamma PDF
        plt.plot(x, y, gamma_color, label='fitted gamma')
        plt.legend()

        plt.title("Histogram and Fitted Gamma Distribution")
        plt.show()

    # Loop over the columns to select
    for col in cols_to_select:
        # Select the column data
        data = psf_df[col]
        data = data.dropna()

        if col == 'Nph WFS':
            data = data[data < 150]

        # Fit a gamma distribution to the data
        shape_param, loc, scale_param = gamma.fit(data)

        # Store the fitted parameters in the dictionary
        fitted_params_and_samples[col] = {
            'shape_param': shape_param,
            'scale_param': scale_param,
            'loc': loc,
        }

        # Generate a vector of random values from the fitted distribution
        samples = gamma.rvs(shape_param, loc=loc, scale=scale_param, size=size)

        # Store the generated samples in the dictionary
        fitted_params_and_samples[col] = samples.reshape(-1)

        # Plot the histogram and fitted gamma distribution
        print(f"Plotting for column: {col}")
        plot_distribution_and_fit(data)

    # Now, fitted_params_and_samples dictionary contains the fitted parameters and generated samples for each column

    data = psf_df['Wind direction (header)']-180

    #Loop win directions
    data = data.append(data+360)
    data = data.append(data-360)
    data = data[data < 270]
    data = data[data > -270]

    data_reshaped = data.to_numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(data_reshaped)

    # Display the parameters of the mixture model
    print(f'weights_ = {gmm.weights_}')
    print(f'means_ = {gmm.means_}')
    print(f'covariances_ = {gmm.covariances_}')

    # Generate samples
    samples = gmm.sample(1000)[0]

    # Create a function for the Gaussian
    def gauss(x, mu, sigma, weight):
        coeff = weight / (np.sqrt(2.0 * np.pi) * sigma)
        return coeff * np.exp(-np.power((x - mu) / sigma, 2) / 2)

    data = data[data <  180]
    data = data[data > -180]

    # Plot the histogram
    plt.hist(data, bins=30, density=True, alpha=0.3)

    # Plot the PDFs for the Gaussians that make up the mixture
    x = np.linspace(min(data), max(data), 1000)

    for mu, sigma, weight in zip(gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten()), gmm.weights_):
        plt.plot(x, gauss(x, mu, sigma, weight))
        
    plt.show()

    # Generate samples
    samples, labels = gmm.sample(size)

    samples[samples < -180] += 180
    samples[samples >  180] -= 180

    fitted_params_and_samples['Wind direction (header)'] = samples.reshape(-1) + 180.

    #%
    rate = psf_df['Rate'].dropna().to_numpy()

    unique_elements, counts_elements = np.unique(rate, return_counts=True)
    probabilities = counts_elements / len(rate)

    # Print the calculated probabilities for check
    pprint(dict(zip(unique_elements, probabilities)))

    # Generate synthetic dataset
    synthetic_rate = np.random.choice(unique_elements, size=size, p=probabilities)

    fitted_params_and_samples['Rate'] = synthetic_rate.reshape(-1).astype(int)

    GL_power = np.random.uniform(low=0.9, high=1.2, size=size)
    GL_power[GL_power > 0.99] = 1.0

    fitted_params_and_samples['Wind direction (200 mbar)'][GL_power == 1.0] = np.nan
    fitted_params_and_samples['Wind speed (200 mbar)'][GL_power == 1.0] = np.nan

    max_ID = psf_df.index.max()
    IDs = np.arange(max_ID+1, max_ID+1+size)

    airmass = fitted_params_and_samples['Airmass']
    r0 = fitted_params_and_samples['r0 (SPARTA)']

    fitted_params_and_samples['Seeing'] = seeing(r0, 500e-9)
    fitted_params_and_samples['Zenith angle'] = np.arccos(1./airmass) / deg2rad
    fitted_params_and_samples['ID'] = IDs
    fitted_params_and_samples['GL power'] = GL_power.reshape(-1)
    fitted_params_and_samples['Δλ left (nm)'] = np.ones(size) * 1625.0
    fitted_params_and_samples['Δλ right (nm)'] = np.ones(size) * 1625.0
    fitted_params_and_samples = pd.DataFrame.from_dict(fitted_params_and_samples)
    fitted_params_and_samples = fitted_params_and_samples.set_index('ID')

    return fitted_params_and_samples


augmented_df = GenerateSamplesAugmentation(2000)
augmented_df.to_pickle(DATA_PATH + 'augmented_df.pickle')


#%%
def GetSPHEREaugmented():
    match_table = [

        (['atmosphere','Seeing'],        ['Seeing'],                  None),
        (['atmosphere','WindSpeed'],     ['Wind speed (header)'],     None),
        (['atmosphere','WindDirection'], ['Wind direction (header)'], None),
        (['telescope','Zenith'],         ['Zenith angle'],            None),
        (['telescope','ZenithAngle'],    ['Zenith angle'],            None),
        (['sensor_HO','NumberPhotons'],  ['Nph WFS'],                 None),
        (['RTC','SensorFrameRate_HO'],   ['Rate'],        lambda x: int(x)),
    ]
    return match_table


def GenerateConfigSynth(sample):
    path_ini = 'C:/Users/akuznets/Projects/TipToy/data/parameter_files/irdis.ini'
    config_file = ParameterParser(path_ini).params
    config_manager = ConfigManager(GetSPHEREaugmented())
    config_file = config_manager.Modify(config_file, sample.to_dict())
    # config_manager.Convert(merged_config, framework='numpy')

    root = 'C:/Users/akuznets/Projects/TipToy/'
    config_file['telescope']['PathPupil']     = root + 'data/calibrations/VLT_CALIBRATION/VLT_PUPIL/ALC2LyotStop_measured.fits'
    config_file['telescope']['PathApodizer']  = root + 'data/calibrations/VLT_CALIBRATION/VLT_PUPIL/APO1Apodizer_measured_All.fits'
    config_file['telescope']['PathStatModes'] = root + 'data/calibrations/VLT_CALIBRATION/VLT_STAT/LWEMODES_320.fits'
    config_file['sensor_science']['DIT']      = 1.0
    config_file['sensor_science']['Num. DIT'] = 1

    if not np.isnan(sample['Wind direction (200 mbar)']) and not np.isnan(sample['Wind speed (200 mbar)']):
        config_file['atmosphere']['WindDirection'] = np.append(config_file['atmosphere']['WindDirection'], sample['Wind direction (200 mbar)'])
        config_file['atmosphere']['WindSpeed']     = np.append(config_file['atmosphere']['WindSpeed'],     sample['Wind speed (200 mbar)'])
        config_file['atmosphere']['Cn2Heights']    = [0, 12400]
        config_file['atmosphere']['Cn2Weights']    = [sample['GL power'], 1.0-sample['GL power']]
    else:
        config_file['atmosphere']['Cn2Heights'] = [0]
        config_file['atmosphere']['Cn2Weights'] = [1]

    config_file['sources_science']['Wavelength'] = {\
        'central L': sample['Δλ left (nm)'],
        'central R': sample['Δλ right (nm)']
    }

    config_file['RTC']['LoopGain_HO'] = 0.5
    config_file['RTC']['LoopDelaySteps_HO'] = 2 if sample['Rate'] >= 600 else 0

    config_file['sources_HO']['Wavelength'] = 658e-9

    return config_file


def convert_numpy_dict(data):
    if isinstance(data, dict):
        return { key: convert_numpy_dict(value) for key, value in data.items() }
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

#%%
data_samples = pd.read_pickle(DATA_PATH + 'augmented_df.pickle')

for id in tqdm(data_samples.index.to_list()):
    sample = data_samples.loc[id]
    config = convert_numpy_dict( GenerateConfigSynth(sample) )

    with open(CONFIGS_PATH + str(id) + '.json', 'w') as fp:
        json.dump(config, fp, indent=4)

 