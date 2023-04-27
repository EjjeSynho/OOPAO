#%% -*- coding: utf-8 -*-
%reload_ext autoreload
%autoreload 2
%matplotlib inline

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import time
import os
import matplotlib.pyplot as plt
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Source import Source, Photometry
from OOPAO.Telescope import Telescope
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.tools.displayTools import displayMap
import pickle
from pprint import pprint

from parameter_files.parameterFile_VLT_SPHERE_SH_WFS import initializeParameterFile

# plt.ion()

#%%
with open('E:/ESO/Data/SPHERE/sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['invalid'] == False]
psf_df = psf_df[psf_df['Class A'] == True]
# psf_df = psf_df[np.isfinite(psf_df['λ left (nm)']) < 1700]
psf_df = psf_df[psf_df['Δλ left (nm)'] < 80]
psf_df = psf_df[psf_df['mag J'] < 9]
psf_df = psf_df[psf_df['Nph WFS'] > 100]

good_ids = psf_df.index.values.tolist()

#%%
# sys.path.append(os.path.abspath('C:/Users/akuznets/Projects/TipToy/'))
# sys.path.append(os.path.abspath('C:/Users/akuznets/Projects/TipToy/tools/'))
# sys.path.append(os.path.abspath('C:/Users/akuznets/Projects/TipToy/parameter_parser/'))

from data_processing.SPHERE_data import LoadSPHEREsampleByID
from tools.config_manager import ConfigManager, GetSPHEREonsky
from tools.parameter_parser import ParameterParser
from tools.utils import r0, deg2rad, seeing

# id_real = 465
# id_real = 576
# id_real = 719
id_real = 992

data_sample = LoadSPHEREsampleByID(id_real)

path_ini = 'C:/Users/akuznets/Projects/TipToy/data/parameter_files/irdis.ini'
config_file = ParameterParser(path_ini).params

config_manager = ConfigManager(GetSPHEREonsky())
config_file = config_manager.Modify(config_file, data_sample)
# config_manager.Convert(merged_config, framework='numpy')

root = 'C:/Users/akuznets/Projects/TipToy/'
config_file['telescope']['PathPupil']     = root + 'data/calibrations/VLT_CALIBRATION/VLT_PUPIL/ALC2LyotStop_measured.fits'
config_file['telescope']['PathApodizer']  = root + 'data/calibrations/VLT_CALIBRATION/VLT_PUPIL/APO1Apodizer_measured_All.fits'
config_file['telescope']['PathStatModes'] = root + 'data/calibrations/VLT_CALIBRATION/VLT_STAT/LWEMODES_320.fits'
config_file['atmosphere']['Cn2Weights'] = [0.95, 0.05]
config_file['atmosphere']['Cn2Heights'] = [0, 10000]

param = initializeParameterFile()

# IR_magnitude = 12.5 # Limiting is J ~ 12.5-13
# param['magnitude WFS'    ]     = target_magnitude
# param['magnitude science']     = IR_magnitude


param['pupil path']            = config_file['telescope']['PathPupil']
param['apodizer path']         = config_file['telescope']['PathApodizer']
param['diameter']              = config_file['telescope']['TelescopeDiameter']
param['centralObstruction']    = config_file['telescope']['ObscurationRatio']
param['windDirection']         = config_file['atmosphere']['WindDirection']
param['opticalBand WFS']       = config_file['sources_HO']['Wavelength']
param['windSpeed']             = config_file['atmosphere']['WindSpeed']
param['wavelength (left)']     = config_file['sources_science']['Wavelength']['central L'] * 1e-9
param['wavelength (right)']    = config_file['sources_science']['Wavelength']['central R'] * 1e-9
param['pixel scale']           = config_file['sensor_science']['PixelScale']
param['nSubaperture']          = config_file['sensor_HO']['NumberLenslets']
param['nPhotonPerSubaperture'] = config_file['sensor_HO']['NumberPhotons']
param['sizeLenslet']           = config_file['sensor_HO']['SizeLenslets']
param['readoutNoise']          = config_file['sensor_HO']['SigmaRON']
param['loopFrequency']         = config_file['RTC']['SensorFrameRate_HO']
param['gainCL']                = config_file['RTC']['LoopGain_HO']
param['mechanicalCoupling']    = config_file['DM']['InfCoupling']
param['detector gain']         = config_file['sensor_science']['Gain']

# This is for real SPHERE
param['WFS_pixelScale'] = config_file['sensor_HO']['PixelScale'] # in [mas]
param['WFS_pix_per_subap real'] = config_file['sensor_HO']['FieldOfView'] / config_file['sensor_HO']['NumberLenslets']
param['validSubap real'] = 1240

param['samplingTime'] = 1.0 / param['loopFrequency']

zenith_angle  = config_file['telescope']['ZenithAngle']
airmass = 1.0 / np.cos(zenith_angle * deg2rad)

w_dir   = config_file['atmosphere']['WindDirection']
w_speed = config_file['atmosphere']['WindSpeed']

if isinstance(w_dir, float):
    w_dir = (w_dir * np.ones(len(config_file['atmosphere']['Cn2Weights']))).tolist()

if isinstance(w_speed, float):
    w_speed = (w_speed * np.ones(len(config_file['atmosphere']['Cn2Weights']))).tolist()

param['r0'] = r0(config_file['atmosphere']['Seeing'], 500e-9)
param['L0'] = 25.0                                               # value of L0 in the visibile in [m]
param['fractionnalR0'] = config_file['atmosphere']['Cn2Weights'] # Cn2 profile
param['windSpeed'    ] = w_speed                                 # wind speed of the different layers in [m/s]
param['windDirection'] = w_dir                                   # wind direction of the different layers in [degrees]
param['altitude'     ] = (np.array(config_file['atmosphere']['Cn2Heights']) * airmass).tolist() # altitude of the different layers in [m]

flux_per_frame = param['nPhotonPerSubaperture'] * param['validSubap real']

#% Loading real SPHERE pupils and apodizer
from skimage.transform import resize
from astropy.io import fits

pupil    = fits.getdata(param['pupil path']).astype('float')
apodizer = fits.getdata(param['apodizer path']).astype('float')

# pupil_size = 240
pupil_size = 320 # real SPHERE case

pupil    = resize(pupil,    (pupil_size, pupil_size), anti_aliasing=False).astype(np.float32)
apodizer = resize(apodizer, (pupil_size, pupil_size), anti_aliasing=True).astype(np.float32)

pupil[np.where(pupil > 0.0)]  = 1.0
pupil[np.where(pupil == 0.0)] = 0.0

param['WFS_pix_per_subap'] = pupil_size // param['nSubaperture']

WFS_flux_correction = (pupil_size//param['nSubaperture'])**2 / param['WFS_pix_per_subap real']**2

#%% -----------------------     TELESCOPE   ----------------------------------
# create the Telescope object
tel_vis = Telescope(resolution          = pupil.shape[0],
                    pupilReflectivity   = param['vis_reflectivity'] * WFS_flux_correction, #realistic transmission to SAXO
                    diameter            = param['diameter'],
                    samplingTime        = param['samplingTime'],
                    centralObstruction  = param['centralObstruction'])

tel_IR_L = Telescope(resolution          = pupil.shape[0],
                     pupil               = pupil, 
                     pupilReflectivity   = apodizer*param['IR_reflectivity'],
                     diameter            = param['diameter'],
                     samplingTime        = param['samplingTime'],
                     centralObstruction  = param['centralObstruction'])

tel_IR_R = Telescope(resolution          = pupil.shape[0],
                     pupil               = pupil, 
                     pupilReflectivity   = apodizer*param['IR_reflectivity'],
                     diameter            = param['diameter'],
                     samplingTime        = param['samplingTime'],
                     centralObstruction  = param['centralObstruction'])


thickness_spider = 0.05                       # size in m
angle            = [45,135,225,315]           # in degrees
offset_X         = [-0.4,0.4,0.4,-0.4]        # shift offset of the spider
offset_Y         = None
tel_vis.apply_spiders(angle, thickness_spider, offset_X = offset_X, offset_Y= offset_Y)

plt.figure()
plt.imshow(tel_IR_L.pupilReflectivity)

# % -----------------------     NGS   ----------------------------------
# create the Source object
rad2mas  = 3600 * 180 * 1000 / np.pi

def magnitudeFromPhotons(tel, photons, band, sampling_time):
    zero_point = band[2]
    fluxMap = photons / tel.pupil.sum() * tel.pupil
    nPhoton = np.nansum(fluxMap / tel.pupilReflectivity) / (np.pi*(tel.D/2)**2) / sampling_time
    return -2.5 * np.log10(368 * nPhoton / zero_point )

photometry = Photometry()
vis_band = photometry(param['opticalBand WFS'])

SAXO_mag = magnitudeFromPhotons(tel_vis, flux_per_frame, vis_band, 1/param['loopFrequency'])

param['magnitude WFS'] = SAXO_mag
param['magnitude science'] = SAXO_mag #TODO: fix it

ngs_vis  = Source(optBand=param['opticalBand WFS'],    magnitude=param['magnitude WFS'])
ngs_IR_L = Source(optBand=param['wavelength (left)'],  magnitude=param['magnitude science'])
ngs_IR_R = Source(optBand=param['wavelength (right)'], magnitude=param['magnitude science'])

ngs_IR_R.nPhoton = ngs_IR_L.nPhoton

pixels_per_l_D_vis  = ngs_vis.wavelength*rad2mas  / param['pixel scale'] / param['diameter']
pixels_per_l_D_IR_L = ngs_IR_L.wavelength*rad2mas / param['pixel scale'] / param['diameter']
pixels_per_l_D_IR_R = ngs_IR_R.wavelength*rad2mas / param['pixel scale'] / param['diameter']

# combine the NGS to the telescope using '*' operator:
ngs_vis  * tel_vis
ngs_IR_L * tel_IR_L
ngs_IR_R * tel_IR_R

tel_vis.computePSF(zeroPaddingFactor=pixels_per_l_D_vis)
PSF_diff = tel_vis.PSF/tel_vis.PSF.max()
N = 50

fov_pix = tel_vis.xPSF_arcsec[1]/tel_vis.PSF.shape[0]
fov = N*fov_pix

plt.figure()
plt.imshow(np.log10(PSF_diff[N:-N,N:-N]), extent=(-fov,fov,-fov,fov))
plt.clim([-4.5,0])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')


#%% -----------------------     ATMOSPHERE   ----------------------------------
# create the Atmosphere object
atm = Atmosphere(telescope     = tel_vis,
                 r0            = param['r0'],
                 L0            = param['L0'],
                 windSpeed     = param['windSpeed'],
                 fractionalR0  = param['fractionnalR0'],
                 windDirection = param['windDirection'],
                 altitude      = param['altitude'],
                 param         = None) #param)
# initialize atmosphere
atm.initializeAtmosphere(tel_vis)
atm.update()

plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()

tel_vis+atm
tel_vis.computePSF(4)
plt.figure()
plt.imshow((np.log10(tel_vis.PSF)), extent=[tel_vis.xPSF_arcsec[0], tel_vis.xPSF_arcsec[1], tel_vis.xPSF_arcsec[0], tel_vis.xPSF_arcsec[1]])
plt.clim([-1,3])

plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object
misReg = MisRegistration(param)
# if no coordinates specified, create a cartesian dm
dm = DeformableMirror(telescope    = tel_vis,
                      nSubap       = param['nSubaperture'],
                      mechCoupling = param['mechanicalCoupling'],
                      misReg       = misReg)

plt.figure()
plt.plot(dm.coordinates[:,0], dm.coordinates[:,1],'x')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')

#%% -----------------------     SH WFS   ----------------------------------
# make sure tel and atm are separated to initialize the PWFS
tel_vis-atm

wfs = ShackHartmann(nSubap       = param['nSubaperture'],
                    telescope    = tel_vis,
                    lightRatio   = param['lightThreshold'],
                    is_geometric = param['is_geometric'])
tel_vis*wfs
plt.close('all')

plt.figure()
plt.imshow(wfs.valid_subapertures)
plt.title('WFS Valid Subapertures')
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame')


#%% -----------------------     Modal Basis   ----------------------------------
# compute the modal basis
foldername_M2C  = None  # name of the folder to save the M2C matrix, if None a default name is used 
filename_M2C    = None  # name of the filename, if None a default name is used 
# KL Modal basis
M2C = compute_M2C(telescope        = tel_vis, #M2C in theory for each r0, but who cares
                  atmosphere       = atm,
                  deformableMirror = dm,
                  param            = param,
                  HHtName          = 'SPHERE',
                  baseName         = 'basis',
                  nameFolder       = param['pathInput'],
                  mem_available    = 8.1e9,
                  nmo              = 1000,
                  nZer             = 3,
                  remove_piston    = True,
                #   recompute_cov    = False) # forces to recompute covariance matrix
                  recompute_cov    = True) # forces to recompute covariance matrix

tel_vis.resetOPD()
# project the mode on the DM
dm.coefs = M2C[:,:50]

tel_vis*dm
#
# show the modes projected on the dm, cropped by the pupil and normalized by their maximum value
displayMap(tel_vis.OPD, norma=True)
plt.title('Basis projected on the DM')

KL_dm = np.reshape(tel_vis.OPD, [tel_vis.resolution**2, tel_vis.OPD.shape[2]])

covMat = (KL_dm.T @ KL_dm) / tel_vis.resolution**2

plt.figure()
plt.imshow(covMat)
plt.title('Orthogonality')
plt.show()

plt.figure()
plt.plot(np.round(np.std(np.squeeze(KL_dm[tel_vis.pupilLogical,:]),axis = 0),5))
plt.title('KL mode normalization projected on the DM')
plt.show()

#%%
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
wfs.is_geometric = param['is_geometric']

# controlling 1000 modes
param['nModes'] = 1000
M2C_KL = np.asarray(M2C[:,:param['nModes']])
# Modal interaction matrix

import os

calib_mat_path = param['pathInput']+'calib_KL_'+str(param['nModes'])+'.pickle'

if not os.path.exists(calib_mat_path):
    print('Creating ', calib_mat_path, 'file...')
    stroke = 1e-9
    # calib_KL = InteractionMatrix(ngs           = ngs,
    #                              atm           = atm,
    #                              tel           = tel,
    #                              dm            = dm,
    #                              wfs           = wfs,
    #                              M2C           = M2C_KL,
    #                              stroke        = stroke,
    #                              nMeasurements = 200,
    #                              noise         = 'off')

    calib_KL_geo = InteractionMatrix(ngs          = ngs_vis,  #wavelength dependant --> different for different filter
                                    atm           = atm, #TODO: remove atm
                                    tel           = tel_vis,
                                    dm            = dm,
                                    wfs           = wfs,
                                    M2C           = M2C_KL,
                                    stroke        = stroke,
                                    nMeasurements = 200,
                                    noise         = 'off') #'on')

    with open(calib_mat_path, 'wb') as handle:
        pickle.dump(calib_KL_geo.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    print('Loading ', calib_mat_path, 'file...')

    from OOPAO.calibration.CalibrationVault import CalibrationVault
    calib_KL_geo = CalibrationVault(0, invert=False)

    with open(calib_mat_path, 'rb') as handle:
        calib_KL_geo.__dict__ = pickle.load(handle)


plt.figure()
plt.plot(np.std(calib_KL_geo.D, axis=0))

plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')


#%%
# These are the calibration data used to close the loop
wfs.is_geometric = param['is_geometric']

calib_CL = calib_KL_geo
M2C_CL   = M2C_KL.copy()

tel_vis.resetOPD()
# initialize DM commands
dm.coefs = 0
ngs_vis*tel_vis*dm*wfs
tel_vis+atm

# dm.coefs[100] = -1

tel_vis.computePSF(4)
plt.close('all')

# combine telescope with atmosphere
tel_vis+atm

# initialize DM commands
dm.coefs = 0
ngs_vis*tel_vis*dm*wfs
ngs_IR_R*tel_IR_R
ngs_IR_L*tel_IR_L

plt.show()

param['nLoop'] = 50
# allocate memory to save data
SR        = np.zeros(param['nLoop'])
total     = np.zeros(param['nLoop'])
residual  = np.zeros(param['nLoop'])
wfsSignal = np.arange(0, wfs.nSignal)*0
SE_PSF_L  = []
# SE_PSF_R  = []
LE_PSF_L  = np.log10(tel_vis.PSF_norma_zoom)
# LE_PSF_R  = np.log10(tel_vis.PSF_norma_zoom)

# loop parameters
gainCL = param['gainCL']
wfs.cam.readoutNoise = 0.0 #param['readoutNoise']
wfs.cam.photonNoise  = False #True

# Returns a gaussian function with the given parameters
def gaussian(N, height, center_x, center_y, width_x, width_y):
    gauss_2d = lambda x,y: height*np.exp( -(((center_x-x)/width_x)**2 + ((center_y-y)/width_y)**2)/2 )
    return gauss_2d(*(np.indices(([N,N])))-(N/2-0.5))

wfs.cog_weight = np.atleast_3d(gaussian(wfs.n_pix_subap, 1, 0, 0, 4, 4)).transpose([2,0,1]) #TODO: account for the reference slopes
wfs.threshold_cog = 3*wfs.cam.readoutNoise

reconstructor = M2C_CL @ calib_CL.M

#%%
%matplotlib qt
from OOPAO.tools.displayTools import cl_plot

OPD_residual = []
n_phs = []
WFS_signals = []

plot_obj = cl_plot(list_fig          = [atm.OPD, tel_vis.mean_removed_OPD, wfs.cam.frame, [dm.coordinates[:,0], np.flip(dm.coordinates[:,1]), dm.coefs], [[0,0],[0,0]], np.log10(tel_vis.PSF_norma_zoom), np.log10(tel_vis.PSF_norma_zoom)],
                   type_fig          = ['imshow','imshow','imshow','scatter','plot','imshow','imshow'],
                   list_title        = ['Turbulence OPD','Residual OPD','WFS Detector','DM Commands',None,None,None],
                   list_lim          = [None,None,None,None,None,[-4,0],[-4,0]],
                   list_label        = [None,None,None,None,['Time','WFE [nm]'],['Short Exposure PSF',''],['Long Exposure_PSF','']],
                   n_subplot         = [4,2],
                   list_display_axis = [None,None,None,None,True,None,None],
                   list_ratio        = [[0.95,0.95,0.1], [1,1,1,1]], s=5)

# fig = plt.figure(1)

for i in range(param['nLoop']):
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    t_start = time.perf_counter()
    atm.update()
    # save phase variance
    total[i] = np.std(tel_vis.OPD[np.where(tel_vis.pupil > 0)])*1e9
    # save turbulent phase
    turbPhase = tel_vis.src.phase
    # propagate to the WFS with the CL commands applied
    tel_vis*dm*wfs
    dm.coefs -= gainCL * (reconstructor @ wfsSignal)
    # store the slopes after computing the commands => 2 frames delay
    wfsSignal = wfs.signal
    t_end = time.perf_counter()
    print('Elapseed time:', str(np.round((t_end-t_start)*1e3).astype('int')), 'ms')
    
    #print(wfs.photon_per_subaperture)

    # update displays if required
    tel_IR_L.OPD = tel_vis.OPD_no_pupil * tel_IR_L.pupil
    # tel_IR_R.OPD = tel_vis.OPD_no_pupil * tel_IR_R.pupil
    t_start = time.perf_counter()
    tel_IR_L.computePSF(zeroPaddingFactor=pixels_per_l_D_IR_L)
    # tel_IR_R.computePSF(zeroPaddingFactor=pixels_per_l_D_IR_R)
    t_end = time.perf_counter()
    print('FFT time:', str(np.round((t_end-t_start)*1e3).astype('int')), 'ms')
    crop = slice(tel_IR_L.PSF_norma_zoom.shape[0]//2-64//2, tel_IR_L.PSF_norma_zoom.shape[1]//2+64//2)

    if i > 15:
        SE_PSF_L.append(np.log10(tel_IR_L.PSF_norma_zoom))
        # SE_PSF_R.append(np.log10(tel_IR_R.PSF_norma_zoom))
        LE_PSF_L = np.mean(SE_PSF_L, axis=0)
        # LE_PSF_R = np.mean(SE_PSF_R, axis=0)
    
    cl_plot(list_fig = [atm.OPD,
                        tel_vis.mean_removed_OPD,
                        wfs.cam.frame,dm.coefs,
                        [np.arange(i+1), residual[:i+1]],
                        np.log10(tel_IR_L.PSF_norma_zoom)[crop,crop],
                        LE_PSF_L],
            plt_obj = plot_obj)
    plt.pause(0.1)

    if plot_obj.keep_going is False: break
    
    SR[i] = np.exp(-np.var(tel_vis.src.phase[np.where(tel_vis.pupil==1)]))
    OPD   = tel_vis.OPD[np.where(tel_vis.pupil > 0)]
    residual[i] = np.std( tel_vis.OPD[np.where(tel_vis.pupil > 0)] )*1e9
    n_phs.append( np.median(wfs.photon_per_subaperture_2D[(wfs.validLenslets_x, wfs.validLenslets_y)]) )

    OPD_residual.append(np.copy(tel_vis.OPD))
    WFS_signals.append(np.copy(wfsSignal))
    print('Loop '+str(i)+'/'+str(param['nLoop'])+' -- turbulence: '+str(np.round(total[i],1))+', residual: ' +str(np.round(residual[i],1))+ '\n')


#%%
from tqdm import tqdm

param['nLoop'] = np.ceil(param['loopFrequency']).astype(np.uint16)
N_DITs = 10
param['nLoop'] = param['nLoop'] // N_DITs * N_DITs

WFS_signals   = np.zeros([wfs.signal.shape[0], param['nLoop']], dtype=np.float32)
OPD_turbulent = np.zeros([tel_vis.pupil.shape[0], tel_vis.pupil.shape[1], param['nLoop']], dtype=np.float32)
OPD_residual  = np.zeros([tel_vis.pupil.shape[0], tel_vis.pupil.shape[1], param['nLoop']], dtype=np.float32)
n_phs = np.zeros([param['nLoop']], dtype=np.float32)
OPDs_over_pupil = np.zeros(param['nLoop'])

t_start = time.perf_counter()
for i in tqdm(range(param['nLoop'])):
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save turbulent phase
    # OPD_turbulent[:,:,i] = np.copy(tel_vis.src.phase_no_pupil)
    OPD_turbulent[:,:,i] = np.copy(atm.OPD_no_pupil)
    # propagate to the WFS with the CL commands applied
    tel_vis * dm * wfs
    dm.coefs -= gainCL * (reconstructor @ wfs.signal)
    # store the slopes after computing the commands => 2 frames delay
    WFS_signals[:,i]    = np.copy(wfs.signal)
    OPD_residual[:,:,i] = np.copy(tel_vis.OPD_no_pupil)
    OPDs_over_pupil[i]  = np.std(OPD_residual[:,:,i][np.where(tel_vis.pupil > 0)])
    n_phs[i] = np.median(wfs.photon_per_subaperture_2D[(wfs.validLenslets_x, wfs.validLenslets_y)])

    #print('Loop '+str(i)+'/'+str(param['nLoop'])+' -- turbulence: '+str(np.round(total[i],1))+', residual: ' +str(np.round(residual[i],1))+ '\n')
t_end = time.perf_counter()
print('Elapsed time:', str(np.round((t_end-t_start)*1e3).astype('int')), 'ms')

# TODO: cut first 20 samples to reach the convergence!

#%%
# Binning is used to downsample phase screens to reduce the computation time
def binning(inp, N, regime='sum'):
    if N == 1: return inp
    out = np.stack(np.split(np.stack(np.split(np.atleast_3d(inp), inp.shape[0]//N, axis=0)), inp.shape[1]//N, axis=2))
    if   regime == 'max':  func = np.max
    elif regime == 'mean': func = np.mean
    else: func = np.sum 
    return np.squeeze( np.transpose( func(out, axis=(2,3), keepdims=True), axes=(1,0,2,3,4)) )


N_bin = 2
def ComputeLongExposure(OPD_residual, tel, pix_per_l_D):
    k = 2*np.pi/tel.src.wavelength

    pupil_smaller = binning(tel.pupil, N_bin, 'max').astype(np.float32)
    amplitude     = binning(tel.pupilReflectivity*np.sqrt(tel.src.fluxMap), N_bin, regime='max').astype(np.float32) * pupil_smaller
    phase_chunk   = binning(OPD_residual, N_bin, 'mean') * np.atleast_3d(pupil_smaller) * k
    EMF_chunk     = np.atleast_3d(amplitude)*np.exp(1j*phase_chunk)
    del phase_chunk

    t_start = time.perf_counter()
    PSFs_stack = tel.computePSFbatch(EMF_chunk, 256, pix_per_l_D).astype(np.float32)
    t_end = time.perf_counter()
    print('Elapsed time:', str(np.round((t_end-t_start)*1e3).astype('int')), 'ms')
    del EMF_chunk
    long_exposure = PSFs_stack.mean(axis=2)
    DITs = np.stack( np.split(PSFs_stack, N_DITs, axis=2) ).mean(axis=-1)
    variance = PSFs_stack.var(axis=2) #TODO: make noisy
    del PSFs_stack
    return long_exposure, variance, DITs

# Make it one second
# Make it multiple DITs
PSF_long_exposure_L, PSF_variance_L, DITs_L = ComputeLongExposure(OPD_residual, tel_IR_L, pixels_per_l_D_IR_L)
PSF_long_exposure_R, PSF_variance_R, DITs_R = ComputeLongExposure(OPD_residual, tel_IR_R, pixels_per_l_D_IR_R)

#%%
plt.figure()
crop = 256
plt.title(N_bin)
def el_croppo(crop):
    buf = slice(PSF_long_exposure_L.shape[0]//2-crop//2, PSF_long_exposure_L.shape[1]//2+crop//2)
    return (buf, buf,...)

test = np.hstack((
    PSF_long_exposure_L[el_croppo(crop)],
    PSF_long_exposure_R[el_croppo(crop)],
    np.abs(PSF_long_exposure_L[el_croppo(crop)] - PSF_long_exposure_R[el_croppo(crop)])
))


plt.imshow(np.log(test))
# plt.imshow(test)
# plt.imshow(np.log(test2))
# plt.colorbar()
plt.show()

#%%
PSF_0 = np.copy(data_sample['image'])[el_croppo(crop)]
PSF_2 = np.copy(long_exposure)[el_croppo(crop)]
PSF_0 /= PSF_0.sum()
PSF_2 /= PSF_2.sum()

plt.imshow(np.log(np.abs(np.hstack((PSF_0, PSF_2, PSF_2-PSF_0)))))
plt.show()

#%%
# simulation_result = np.save('C:/Users/akuznets/Data/SPHERE/PSF_OOPAO.npy', long_exposure[el_croppo(256)]/long_exposure[el_croppo(256)].sum())

#%%
def mask_circle(N, r, center=(0,0), centered=True):
    factor = 0.5 * (1-N%2)
    if centered:
        coord_range = np.linspace(-N//2+N%2+factor, N//2-factor, N)
    else:
        coord_range = np.linspace(0, N-1, N)
    xx, yy = np.meshgrid(coord_range-center[1], coord_range-center[0])
    pupil_round = np.zeros([N, N], dtype=np.int32)
    pupil_round[np.sqrt(yy**2+xx**2) < r] = 1
    return pupil_round


def ComputePSDfromScreens(screens):
    print('Computing FFTs of each phase screen and spectra...')
    spectra = np.zeros([screens.shape[0]*2, screens.shape[1]*2, screens.shape[2]], dtype=np.float32)
    screens_zp = np.zeros([screens.shape[0]*2, screens.shape[1]*2, screens.shape[2]], dtype=np.float32)

    insert_ids = slice(screens.shape[0]//2, screens.shape[0]//2+screens.shape[0])
    screens_zp[insert_ids, insert_ids, :] = screens #* np.atleast_3d(tel_vis.pupil) # zero padding
    spectra = np.abs(np.fft.fftshift( 1/spectra.shape[0] * np.fft.fft2(np.fft.fftshift(screens_zp), axes=(0,1)) ))
    del screens_zp
    print('Done!\nComputing PSD...')
    PSD = spectra.var(axis=2)
    del spectra
    return PSD

# PSD_residuals = ComputePSDfromScreens(OPD_residual)
circa = mask_circle(OPD_residual.shape[0], OPD_residual.shape[0]//2)

PSD_turbulent = binning(ComputePSDfromScreens(OPD_turbulent * circa[...,None]), 2, 'mean')
PSD_residuals = binning(ComputePSDfromScreens(OPD_residual  * circa[...,None]), 2, 'mean')

#%%
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt( (x-center[0])**2 + (y-center[1])**2 )
    r = r.astype('int')

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

N_PSD_bin = 1

PSD_binned_0 = binning(PSD_turbulent, N_PSD_bin, regime='max')
PSD_binned_1 = binning(PSD_residuals, N_PSD_bin, regime='max')

profi_0 = radial_profile(PSD_binned_0, (PSD_binned_0.shape[0]//2, PSD_binned_0.shape[1]//2))[:PSD_binned_0.shape[1]//2]
profi_1 = radial_profile(PSD_binned_1, (PSD_binned_1.shape[0]//2, PSD_binned_1.shape[1]//2))[:PSD_binned_1.shape[1]//2]


dk = 1.0/8.0 # PSD spatial frequency step [m^-1]
pitch = 0.2 # [m]
kc = 1/(2*dm.pitch)
k = np.arange(PSD_binned_1.shape[0]//2)*dk * N_PSD_bin

plt.figure(dpi=200)
plt.plot(k, profi_0, label='Open loop')
plt.plot(k, profi_1, label='Closed loop')
plt.axvline(x=kc, color='black', linestyle='-', label='Cut-off freq.')
plt.xlim([k[1],k.max()])
plt.title('Telemetry PSD')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Spatial frequency, $m^{-1}$')
plt.legend()
plt.grid()
plt.show()


#%%
# R mag
# H mag
# n_ph
# PSD AO-corrected
# PSD atmospheric
# wfs commands
# r0
# L0
# windspeed
# Cn2 profile


#%%
slopes = WFS_signals

from scipy.signal   import correlate
from scipy.optimize import curve_fit
from scipy.special  import factorial

N_steps  = WFS_signals.shape[0]
N_slopes = WFS_signals.shape[1]
N_modes_propag = reconstructor.shape[1]

autocorrs = []
for i in tqdm(range(N_slopes)):
    autocorrs.append( correlate(slopes[:,i], slopes[:,i], mode='same', method='fft') )
    # autocorrs.append( correlate(slopes[:,i], slopes[:,i], mode='same') )
autocorrs = np.array(autocorrs).mean(axis=0) / N_steps
autocorrs = autocorrs[np.argmax(autocorrs) : np.argmax(autocorrs)+20]

#%%
poisson  = lambda x, lmbd, height, b, a: height * lmbd**(x/a) / factorial(x/a) * np.exp(-lmbd) + b
dt = np.arange(1, len(autocorrs))

try:
    fit_params, _ = curve_fit(poisson, dt, autocorrs[1:], [4, 5e-9/N_steps, 0, 2])
    turbulence_var = poisson(0, *fit_params)
    slopes_var = autocorrs.max() - turbulence_var

except RuntimeError:
    slopes_var = autocorrs.max()

plt.plot(autocorrs)
plt.plot(poisson(dt, *fit_params))

# Propagate through the reconstructor matrix to get  the WFS variance
WFS_var = np.trace(reconstructor @ (np.eye(reconstructor.shape[1])*slopes_var) @ reconstructor.T) / N_modes_propag
WFS_err = np.sqrt(WFS_var)*1e9 # [nm OPD]



#%%

def GetWFSnoiseError(slopes, display=False):
    from scipy.signal   import correlate
    from scipy.optimize import curve_fit
    from scipy.special  import factorial

    N_steps  = WFS_signals.shape[0]
    N_slopes = WFS_signals.shape[1]
    N_modes_propag = reconstructor.shape[1]

    autocorrs = []
    for i in range(N_slopes):
        autocorrs.append( correlate(slopes[:,i], slopes[:,i], mode='same', method='fft') )
    autocorrs = np.array(autocorrs).mean(axis=0) / N_steps
    autocorrs = autocorrs[np.argmax(autocorrs) : np.argmax(autocorrs)+20]
    
    poisson  = lambda x, lmbd, height, b, a: height * lmbd**(x/a) / factorial(x/a) * np.exp(-lmbd) + b
    dt = np.arange(1, len(autocorrs))

    try:
        fit_params, _ = curve_fit(poisson, dt, autocorrs[1:], [4, 5e-9/N_steps, 0, 2])
        turbulence_var = poisson(0, *fit_params)
        slopes_var = autocorrs.max() - turbulence_var

    except RuntimeError:
        slopes_var = autocorrs.max()

    # Propagate through the reconstructor matrix to get  the WFS variance
    WFS_var = np.trace(reconstructor @ (np.eye(reconstructor.shape[1])*slopes_var) @ reconstructor.T) / N_modes_propag
    WFS_err = np.sqrt(WFS_var)*1e9 # [nm OPD]

    if display:
        dt_fine = np.arange(0, dt.max(), 0.01)
        plt.plot(autocorrs)
        plt.plot(dt_fine, poisson(dt_fine, *fit_params), '--')
        plt.grid()

    return WFS_err


print(GetWFSnoiseError(WFS_signals))

residual_err = np.zeros([OPD_residual.shape[2]])
SR = np.zeros([OPD_residual.shape[2]])

for i in range(OPD_residual.shape[2]):
    OPD = OPD_residual[:,:,i][np.where(tel_vis.pupil > 0)]
    # phase_vis = 2*np.pi/tel_vis.src.wavelength * OPD
    # SR[i] = np.exp(-np.var(phase_vis))
    phase_IR_L = 2*np.pi/tel_IR_L.src.wavelength * OPD #TODO: clarify from which tel SR should be computer
    phase_IR_R = 2*np.pi/tel_IR_R.src.wavelength * OPD
    SR[i] = np.exp(-np.var(phase_IR_L)) * 0.5 + np.exp(-np.var(phase_IR_R)) * 0.5
    residual_err[i] = np.std(OPD)*1e9

residual_err = residual_err.mean()
SR = SR.mean()

# wfsSignals
# OPDs
#

#%%
rad2mas  = 3600 * 180 * 1000 / np.pi
rad2arc  = rad2mas / 1000

wvl = tel_vis.src.wavelength
WFS_wvl = tel_vis.src.wavelength

WFS_d_sub = param['sizeLenslet']
WFS_pixelScale = param['WFS_pixelScale'] / 1e3 # [arcsec]
WFS_excessive_factor = 1.0
WFS_spot_FWHM = 0
r0 = param['r0']
WFS_nPix = param['WFS_pix_per_subap']
WFS_RON = 1
WFS_Nph = n_phs.mean()

# # Read-out noise calculation
# nD = np.maximum(1.0, rad2arc*wvl/WFS_d_sub/WFS_pixelScale]) #spot FWHM in pixels and without turbulence
# # Photon-noise calculation
# nT = np.maximum(1.0, np.hypot(WFS_spot_FWHM/1e3, rad2arc*WFS_wvl/r0) / WFS_pixelScale])


nD = rad2arc * wvl / WFS_d_sub / WFS_pixelScale #spot FWHM in pixels and without turbulence
nT = rad2arc * WFS_wvl / r0  / WFS_pixelScale


varShot = np.pi**2/(2*WFS_Nph) * (nT/nD)**2
varRON  = np.pi**2/3 * (WFS_RON**2/WFS_Nph**2) * (WFS_nPix**2/nD)**2

# Total noise variance calculation
varNoise = WFS_excessive_factor * (varRON+varShot)

WFE_std = wvl*np.sqrt(varNoise)/2/np.pi * 1e9

print(WFE_std)

#%%

magnitudes     = np.array([9, 10, 10.5, 10.75])
autocorr_noise = np.array([38, 63, 99, 128])
formula_noise  = np.array([25, 45, 58, 64])

plt.plot(magnitudes, autocorr_noise, label='Autocorrelation')
plt.plot(magnitudes, formula_noise, label='Analytical formula')
plt.xlabel('R magnitude')
plt.ylabel('Reconstruction WFE, [nm] RMS')
plt.legend()
plt.grid()



#%%

data_write = {
    'spectra': {
        'central L': ngs_IR_L.wavelength * 1e9,
        'central R': ngs_IR_R.wavelength * 1e9
    },

    'r0': param['r0'],
    'seeing': seeing(param['r0'], 500e-9),

    'Cn2': {
        'profile': param['fractionnalR0'],
        'heights': param['altitude']
    },

    'observation': {
        'magnitudes': {
            'V': SAXO_mag,
        }
    },

    'Strehl': SR,
    'Wind speed': param['windSpeed'],
    'Wind direction': param['windDirection'],

    'Detector': {
        'psInMas': param['pixel scale'],
        'ron': param['readoutNoise'],
        'gain': param['detector gain']
    },

    'WFS': {
        'Nph vis': n_phs.mean(),
        'rate': param['loopFrequency'],
        'commands': WFS_signals
    },

    'telescope': {
        'zenith': zenith_angle,
        'airmass': airmass
    },
    #TODO: loop gain
    # and WFS commands
    # exposure time
    'PSF L': DITs_L,
    'PSF R': DITs_R,

    'PSF L (variance)': PSF_variance_L,
    'PSF R (variance)': PSF_variance_R,

    'PSD residuals': PSD_residuals,
    'PSD atmospheric': PSD_turbulent
}

# Write to pickle file
with open('E:\ESO\Data\SPHERE\IRDIS_synthetic\{}_synth.pickle'.format(id_real), 'wb') as f:
    pickle.dump(data_write, f)

# %%
