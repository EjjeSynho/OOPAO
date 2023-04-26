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
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.tools.displayTools import cl_plot, displayMap
# %% -----------------------     read parameter file   ----------------------------------
from parameter_files.parameterFile_VLT_SPHERE_SH_WFS import initializeParameterFile

param = initializeParameterFile()
plt.ion()

param['resolution'] = 320

# % -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel_vis = Telescope(resolution=param['resolution'], \
                diameter=param['diameter'], \
                samplingTime=param['samplingTime'], \
                centralObstruction=param['centralObstruction'])

thickness_spider = 0.05  # size in m
angle = [45, 135, 225, 315]  # in degrees
offset_X = [-0.4, 0.4, 0.4, -0.4]  # shift offset of the spider
offset_Y = None

tel_vis.apply_spiders(angle, thickness_spider, offset_X=offset_X, offset_Y=offset_Y)

plt.figure()
plt.imshow(tel_vis.pupilReflectivity)
# %% -----------------------     NGS   ----------------------------------
# create the Source object
ngs_tel  = Source(optBand=param['opticalBand WFS'],    magnitude=param['magnitude WFS'])

# combine the NGS to the telescope using '*' operator:
ngs_tel * tel_vis

tel_vis.computePSF(zeroPaddingFactor=6)
PSF_diff = tel_vis.PSF / tel_vis.PSF.max()
N = 50

fov_pix = tel_vis.xPSF_arcsec[1] / tel_vis.PSF.shape[0]
fov = N * fov_pix
plt.figure()
plt.imshow(np.log10(PSF_diff[N:-N, N:-N]), extent=(-fov, fov, -fov, fov))
plt.clim([-4.5, 0])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')

# %% -----------------------     ATMOSPHERE   ----------------------------------
# create the Atmosphere object
atm = Atmosphere(telescope=tel_vis, \
                 r0=param['r0'], \
                 L0=param['L0'], \
                 windSpeed=param['windSpeed'], \
                 fractionalR0=param['fractionnalR0'], \
                 windDirection=param['windDirection'], \
                 altitude=param['altitude'], param=param)
# initialize atmosphere
atm.initializeAtmosphere(tel_vis)

atm.update()

plt.figure()
plt.imshow(atm.OPD * 1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()

tel_vis + atm
tel_vis.computePSF(8)
plt.figure()
plt.imshow((np.log10(tel_vis.PSF)), extent=[tel_vis.xPSF_arcsec[0], tel_vis.xPSF_arcsec[1], tel_vis.xPSF_arcsec[0], tel_vis.xPSF_arcsec[1]])
plt.clim([-1, 3])

plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

# %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object
misReg = MisRegistration(param)
# if no coordonates specified, create a cartesian dm
dm = DeformableMirror(telescope=tel_vis, \
                      nSubap=param['nSubaperture'], \
                      mechCoupling=param['mechanicalCoupling'], \
                      misReg=misReg)

plt.figure()
plt.plot(dm.coordinates[:, 0], dm.coordinates[:, 1], 'x')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')

# %% -----------------------     SH WFS   ----------------------------------
# make sure tel and atm are separated to initialize the PWFS
tel_vis - atm

wfs = ShackHartmann(nSubap=param['nSubaperture'], \
                    telescope=tel_vis, \
                    lightRatio=param['lightThreshold'], \
                    is_geometric=param['is_geometric'])

tel_vis * wfs
plt.close('all')

plt.figure()
plt.imshow(wfs.valid_subapertures)
plt.title('WFS Valid Subapertures')
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame')

# %% -----------------------     Modal Basis   ----------------------------------
# compute the modal basis
foldername_M2C = None  # name of the folder to save the M2C matrix, if None a default name is used
filename_M2C = None  # name of the filename, if None a default name is used
# KL Modal basis
M2C = compute_M2C(telescope=tel_vis, \
                  atmosphere=atm, \
                  deformableMirror=dm, \
                  param=param, \
                  mem_available=8.1e9, \
                  nmo=1000, \
                  nZer=3, \
                  remove_piston=True, \
                  recompute_cov=False)  # forces to recompute covariance matrix

tel_vis.resetOPD()
# project the mode on the DM
dm.coefs = M2C[:, :50]

tel_vis * dm
#
# show the modes projected on the dm, cropped by the pupil and normalized by their maximum value
displayMap(tel_vis.OPD, norma=True)
plt.title('Basis projected on the DM')

KL_dm = np.reshape(tel_vis.OPD, [tel_vis.resolution ** 2, tel_vis.OPD.shape[2]])

covMat = (KL_dm.T @ KL_dm) / tel_vis.resolution ** 2

plt.figure()
plt.imshow(covMat)
plt.title('Orthogonality')
plt.show()

plt.figure()
plt.plot(np.round(np.std(np.squeeze(KL_dm[tel_vis.pupilLogical, :]), axis=0), 5))
plt.title('KL mode normalization projected on the DM')
plt.show()

# %%
# wfs.is_geometric = False
stroke = 1e-9
# controlling 1000 modes
param['nModes'] = 1000
M2C_KL = np.asarray(M2C[:, :param['nModes']])
# Modal interaction matrix
# wfs.is_geometric = False
# calib_KL = InteractionMatrix(  ngs            = ngs,\
#                             atm            = atm,\
#                             tel            = tel,\
#                             dm             = dm,\
#                             wfs            = wfs,\
#                             M2C            = M2C_KL,\
#                             stroke         = stroke,\
#                             nMeasurements  = 200,\
#                             noise          = 'off')
wfs.is_geometric = True

calib_KL_geo = InteractionMatrix(ngs=ngs_tel, \
                                 atm=atm, \
                                 tel=tel_vis, \
                                 dm=dm, \
                                 wfs=wfs, \
                                 M2C=M2C_KL, \
                                 stroke=stroke, \
                                 nMeasurements=200, \
                                 noise='off')
plt.figure()
plt.plot(np.std(calib_KL_geo.D, axis=0))

plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')

# %%
%matplotlib qt
# These are the calibration data used to close the loop
wfs.is_geometric = False

calib_CL = calib_KL_geo
M2C_CL = M2C_KL.copy()

tel_vis.resetOPD()
# initialize DM commands
dm.coefs = 0
ngs_tel * tel_vis * dm * wfs
tel_vis + atm

# dm.coefs[100] = -1

tel_vis.computePSF(4)
plt.close('all')

# combine telescope with atmosphere
tel_vis + atm

# initialize DM commands
dm.coefs = 0
ngs_tel * tel_vis * dm * wfs

plt.show()

param['nLoop'] = 200
# allocate memory to save data
SR = np.zeros(param['nLoop'])
total = np.zeros(param['nLoop'])
residual = np.zeros(param['nLoop'])
wfsSignal = np.arange(0, wfs.nSignal) * 0
SE_PSF = []
LE_PSF = np.log10(tel_vis.PSF_norma_zoom)

plot_obj = cl_plot(list_fig=[atm.OPD, tel_vis.mean_removed_OPD, wfs.cam.frame,
                             [dm.coordinates[:, 0], np.flip(dm.coordinates[:, 1]), dm.coefs], [[0, 0], [0, 0]],
                             np.log10(tel_vis.PSF_norma_zoom), np.log10(tel_vis.PSF_norma_zoom)], \
                   type_fig=['imshow', 'imshow', 'imshow', 'scatter', 'plot', 'imshow', 'imshow'], \
                   list_title=['Turbulence OPD', 'Residual OPD', 'WFS Detector', 'DM Commands', None, None, None], \
                   list_lim=[None, None, None, None, None, [-4, 0], [-4, 0]], \
                   list_label=[None, None, None, None, ['Time', 'WFE [nm]'], ['Short Exposure PSF', ''],
                               ['Long Exposure_PSF', '']], \
                   n_subplot=[4, 2], \
                   list_display_axis=[None, None, None, None, True, None, None], \
                   list_ratio=[[0.95, 0.95, 0.1], [1, 1, 1, 1]], s=5)
plt.draw()

# loop parameters
gainCL = 0.4
wfs.cam.photonNoise = True
display = True

reconstructor = M2C_CL @ calib_CL.M

for i in range(param['nLoop']):
    a = time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save phase variance
    total[i] = np.std(tel_vis.OPD[np.where(tel_vis.pupil > 0)]) * 1e9
    # save turbulent phase
    turbPhase = tel_vis.src.phase
    # propagate to the WFS with the CL commands applied
    tel_vis * dm * wfs

    dm.coefs = dm.coefs - gainCL * np.matmul(reconstructor, wfsSignal)
    # store the slopes after computing the commands => 2 frames delay
    wfsSignal = wfs.signal
    b = time.time()
    print('Elapsed time: ' + str(b - a) + ' s')
    # update displays if required
    if display == True:
        tel_vis.computePSF(4)
        if i > 15:
            SE_PSF.append(np.log10(tel_vis.PSF_norma_zoom))
            LE_PSF = np.mean(SE_PSF, axis=0)

        cl_plot(list_fig=[atm.OPD, tel_vis.mean_removed_OPD, wfs.cam.frame, dm.coefs, [np.arange(i + 1), residual[:i + 1]],
                          np.log10(tel_vis.PSF_norma_zoom), LE_PSF],
                plt_obj=plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
            break

    SR[i] = np.exp(-np.var(tel_vis.src.phase[np.where(tel_vis.pupil == 1)]))
    residual[i] = np.std(tel_vis.OPD[np.where(tel_vis.pupil > 0)]) * 1e9
    OPD = tel_vis.OPD[np.where(tel_vis.pupil > 0)]

    print('Loop' + str(i) + '/' + str(param['nLoop']) + ' Turbulence: ' + str(total[i]) + ' -- Residual:' + str(
        residual[i]) + '\n')
# %%
