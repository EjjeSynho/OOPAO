# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:36:02 2020

@author: cheritie
"""
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append( os.path.normpath(os.path.join(script_dir, '..')) )

import json
from OOPAO.tools.tools import createFolder


def initializeParameterFile():
    # initialize the dictionaries
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['r0'                  ] = 0.15                                           # value of r0 in the visibile in [m]
    param['L0'                  ] = 30                                             # value of L0 in the visibile in [m]
    param['fractionnalR0'       ] = [0.45, 0.1, 0.1, 0.25, 0.1]                    # Cn2 profile
    param['windSpeed'           ] = [10, 12, 11, 15, 20]                           # wind speed of the different layers in [m/s]
    param['windDirection'       ] = [0, 72, 144, 216, 288]                         # wind direction of the different layers in [degrees]
    param['altitude'            ] = [0, 1000, 5000, 10000, 12000]                  # altitude of the different layers in [m]

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nLoop'               ] = 5000                                           # number of iteration
    param['loopFrequency'       ] = 1000                                           # frequency at which loop is running
    param['photonNoise'         ] = True                                           # Photon Noise enable  
    param['readoutNoise'        ] = 1                                              # Readout Noise value
    param['gainCL'              ] = 0.5                                            # integrator gain
    param['nModes'              ] = 600                                            # number of KL modes controlled 
    param['getProjector'        ] = True                                           # modal projector too get modal coefficients of the turbulence and residual phase
                                     
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['diameter'            ] = 8#.2                                           # diameter in [m]
    param['nSubaperture'        ] = 40                                             # number of PWFS subaperture along the telescope diameter
    param['nPixelPerSubap'      ] = 8                                              # sampling of the PWFS subapertures
    param['nPhotonPerSubaperture'] = 10
    param['resolution'          ] = param['nSubaperture']*param['nPixelPerSubap']  # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'     ] = param['diameter']/param['nSubaperture']        # size of a sub-aperture projected in the M1 space
    param['samplingTime'        ] = 1 / param['loopFrequency']                     # loop sampling time in [s]
    param['centralObstruction'  ] = 0.12                                           # central obstruction in percentage of the diameter
    param['nMissingSegments'    ] = 0                                              # number of missing segments on the M1 pupil
    param['IR_reflectivity'     ] = 1                                              # reflectivity of the science path
    param['vis_reflectivity'    ] = 0.3                                            # reflectivity of the SAXO path
    param['pixel scale'         ] = 12.3                                           # pixel scale in [mas] 

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['magnitude WFS'       ] = 8                                              # magnitude of the guide star
    param['magnitude science'   ] = 8                                              # magnitude of the guide star
    param['opticalBand WFS'     ] = 'R'                                            # optical band of the guide star
    param['opticalBand science' ] = 'H'                                            # optical band of the guide star
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'           ] = param['nSubaperture']+1                        # number of actuators 
    param['mechanicalCoupling'  ] = 0.4
    param['isM4'                ] = False                                          # tag for the deformable mirror class
    param['dm_coordinates'      ] = None                                           # tag for the eformable mirror class
    # mis-registrations                                                             
    param['shiftX'              ] = 0                                              # shift X of the DM in pixel size units (tel.D/tel.resolution) 
    param['shiftY'              ] = 0                                              # shift Y of the DM in pixel size units (tel.D/tel.resolution)
    param['rotationAngle'       ] = 0                                              # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'   ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['radialScaling'       ] = 0                                              # radial scaling in percentage of diameter
    param['tangentialScaling'   ] = 0                                              # tangential scaling in percentage of diameter
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['modulation'          ] = 3                                             # modulation radius in ratio of wavelength over telescope diameter
    param['lightThreshold'      ] = 0.5                                            # light threshold to select the valid pixels
    param['unitCalibration'     ] = False                                          # calibration of the PWFS units using a ramp of Tip/Tilt    
    param['is_geometric'        ] = False 
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # name of the system
    param['name'] = 'VLT_' +  param['opticalBand science'] +'_band_'+ str(param['nSubaperture'])+'x'+ str(param['nSubaperture'])  
       
    with open(os.path.normpath(os.path.join(script_dir, "../IRDIS_simulation/settings.json")), "r") as f:
        folder_data = json.load(f)
   
    # location of the calibration data
    param['pathCalib']  = folder_data["path_calib"]
    param['pathInput']  = param['pathCalib']
    # location of the output data
    param['pathOutput'] = folder_data["path_output"]
    # location of config files for simulations
    param['pathConfigs'] = folder_data["path_configs"]
    # location of the overall data folder
    param['pathData'] = folder_data["path_data"]
        
    param['pathPupils'] = folder_data["path_pupils"]
    
    print('Reading/Writting calibration data from ' + param['pathCalib'])
    print('Writting output data in ' + param['pathOutput'])

    createFolder(param['pathOutput'])
    
    return param