# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:32:15 2020

@author: cheritie
"""
import inspect
import numpy as np

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE PHOTOMETRY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
class Photometry:
    def __init__(self, inp=None):
        self.bands = {
            'U'   : [ 0.360e-6 , 0.070e-6 , 2.0e12 ],
            'B'   : [ 0.440e-6 , 0.100e-6 , 5.4e12 ],
            'V0'  : [ 0.500e-6 , 0.090e-6 , 3.3e12 ],
            'V'   : [ 0.550e-6 , 0.090e-6 , 3.3e12 ],
            'R'   : [ 0.640e-6 , 0.150e-6 , 4.0e12 ],
            'I'   : [ 0.790e-6 , 0.150e-6 , 2.7e12 ],
            'I1'  : [ 0.700e-6 , 0.033e-6 , 2.7e12 ],
            'I2'  : [ 0.750e-6 , 0.033e-6 , 2.7e12 ],
            'I3'  : [ 0.800e-6 , 0.033e-6 , 2.7e12 ],
            'I4'  : [ 0.700e-6 , 0.100e-6 , 2.7e12 ],
            'I5'  : [ 0.850e-6 , 0.100e-6 , 2.7e12 ],
            'I6'  : [ 1.000e-6 , 0.100e-6 , 2.7e12 ],
            'I7'  : [ 0.850e-6 , 0.300e-6 , 2.7e12 ],
            'R2'  : [ 0.650e-6 , 0.300e-6 , 7.92e12],
            'R3'  : [ 0.600e-6 , 0.300e-6 , 7.92e12],
            'R4'  : [ 0.670e-6 , 0.300e-6 , 7.92e12],
            'I8'  : [ 0.750e-6 , 0.100e-6 , 2.7e12 ],
            'I9'  : [ 0.850e-6 , 0.300e-6 , 7.36e12],
            'J'   : [ 1.215e-6 , 0.260e-6 , 1.9e12 ],
            'H'   : [ 1.654e-6 , 0.290e-6 , 1.1e12 ],
            'Kp'  : [ 2.1245e-6, 0.351e-6 , 6e11   ],
            'Ks'  : [ 2.157e-6 , 0.320e-6 , 5.5e11 ],
            'K'   : [ 2.179e-6 , 0.410e-6 , 7.0e11 ],
            'L'   : [ 3.547e-6 , 0.570e-6 , 2.5e11 ],
            'M'   : [ 4.769e-6 , 0.450e-6 , 8.4e10 ],
            'Na'  : [ 0.589e-6 , 0.0      , 3.3e12 ],
            'EOS' : [ 1.064e-6 , 0.0      , 3.3e12 ]
        }
        self.__wavelengths = np.array( [v[0] for _,v in self.bands.items()] )

        if inp is not None:
            return self.__call__(inp)


    def __call__(self, inp):
        if isinstance(inp, str):
            if inp not in self.bands.keys():
                print('Error: there is no band with the name "'+inp+'"')
                return None
            else:
                return self.bands[inp]

        elif isinstance(inp, float):    # perform interpolation of parameters for a current wavelength
            if inp < self.__wavelengths.min() or inp > self.__wavelengths.max():
                print('Error: specified value is outside the defined wavelength range!')
                return None

            difference = np.abs(self.__wavelengths - inp)
            dtype = [('number', int), ('value', float)]

            sorted = np.sort(np.array([(num, val) for num,val in enumerate(difference)], dtype=dtype), order='value')                        

            l_1 = self.__wavelengths[sorted[0][0]]
            l_2 = self.__wavelengths[sorted[1][0]]

            if l_1 > l_2: l_1, l_2 = l_2, l_1

            def find_params(input):
                for _,v in self.bands.items():
                    if input == v[0]:
                        return np.array(v)

            p_1 = find_params(l_1)
            p_2 = find_params(l_2)
            weight = ( (np.array([l_1, inp, l_2])-l_1)/(l_2-l_1) )[1]

            return weight*(p_2-p_1) + p_1

        else:
            print('Incorrect input: "'+inp+'"')
            return None


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE PHOTOMETRY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
class Source:    
    def __init__(self,
                 optBand,
                 magnitude,
                 coordinates = [0,0],
                 altitude = np.inf,
                 laser_coordinates = [0,0],
                 Na_profile = None,
                 FWHM_spot_up = None,
                 display_properties = True):
        """
        ************************** REQUIRED PARAMETERS **************************
        
        A Source object is characterised by two parameter:
        _ optBand               : the optical band of the source (see the method photometry)
        _ magnitude             : The magnitude of the star
                            
        ************************** COUPLING A SOURCE OBJECT **************************
        
        Once generated, a Source object "src" can be coupled to a Telescope "tel" that contains the OPD.
        _ This is achieved using the * operator     : src*tel
        _ It can be accessed using                  : tel.src       

    
        ************************** MAIN PROPERTIES **************************
        
        The main properties of a Source object are listed here: 
        _ src.phase     : 2D map of the phase scaled to the src wavelength corresponding to tel.OPD
        _ src.type      : Ngs or LGS  

        _ src.nPhoton   : number of photons per m2 per s. if this property is changed after the initialization, the magnitude is automatically updated to the right value. 
        _ src.fluxMap   : 2D map of the number of photons per pixel per frame (depends on the loop frequency defined by tel.samplingTime)  
        _ src.display_properties : display the properties of the src object
        
        The main properties of the object can be displayed using :
            src.print_properties()
            
        ************************** OPTIONAL PROPERTIES **************************
        _ altitude              : altitude of the source. Default is inf (NGS) 
        _ laser_coordinates     : The coordinates in [m] of the laser launch telescope
        _ Na_profile            : An array of 2 dimensions and n sampling points for the Sodium profile. The first dimension corresponds to the altitude and the second dimention to the sodium profile value.
        _ FWHM_spot_up          : FWHM of the LGS spot in [arcsec]
        ************************** EXEMPLE **************************

        Create a source object in H band with a magnitude 8 and combine it to the telescope
        src = Source(opticalBand = 'H', magnitude = 8) 
        src*tel

        
        """
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        self.is_initialized = False
        self.display_properties = display_properties
        photometry      = Photometry()
        tmp             = photometry(optBand)                   # get the photometry properties
        self.optBand    = optBand                               # optical band
        self.wavelength = tmp[0]                                # wavelength in [m]
        self.bandwidth  = tmp[1]                                # optical bandwidth
        self.zeroPoint  = tmp[2]/368                            # zero point
        self.magnitude  = magnitude                             # magnitude
        self.phase      = []                                    # phase of the source 
        self.phase_no_pupil      = []                           # phase of the source (no pupil)
        self.fluxMap    = []                                    # 2D flux map of the source
        self.nPhoton    = self.zeroPoint*10**(-0.4*magnitude)   # number of photon per m2 per s
        self.tag        = 'source'                              # tag of the object
        self.altitude = altitude                                # altitude of the source object in m    
        self.coordinates = coordinates                          # polar coordinates [r,theta] 
        self.laser_coordinates = laser_coordinates              # Laser Launch Telescope coordinates in [m] 

        if Na_profile is not None and FWHM_spot_up is not None:
            self.Na_profile = Na_profile
            self.FWHM_spot_up = FWHM_spot_up
            # consider the altitude weigthed by Na profile
            self.altitude = np.sum(Na_profile[0,:]*Na_profile[1,:])
            self.type = 'LGS'
        else:
            
            self.type     = 'NGS'

        if self.display_properties:
            self.print_properties()
            
        self.is_initialized = True

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE INTERACTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        
    def __mul__(self,telescope):
        telescope.src   = self
        if type(telescope.OPD) is list:
            telescope.resetOPD()
        
        if np.ndim(telescope.OPD) == 3:
            telescope.resetOPD()

        # update the phase of the source
        self.phase = telescope.OPD*2*np.pi/self.wavelength
        self.phase_no_pupil = telescope.OPD_no_pupil*2*np.pi/self.wavelength
        # compute the variance in the pupil
        self.var = np.var(self.phase[np.where(telescope.pupil==1)])
        # assign the source object to the telescope object

        self.fluxMap    = telescope.pupilReflectivity*self.nPhoton*telescope.samplingTime*(telescope.D/telescope.resolution)**2
        if telescope.optical_path is None:
            telescope.optical_path = []
            telescope.optical_path.append([self.type+'({})'.format(self.optBand), id(self)])
            telescope.optical_path.append([telescope.tag,id(telescope)])
        else:
            telescope.optical_path[0] =[self.type+'({})'.format(self.optBand), id(self)]
            
        return telescope
     
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    @property
    def nPhoton(self):
        return self._nPhoton
    
    @nPhoton.setter
    def nPhoton(self,val):
        self._nPhoton  = val
        self.magnitude = -2.5*np.log10(val/self.zeroPoint)
        if self.is_initialized:
            if self.display_properties:
                print('NGS flux updated!')
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('Wavelength \t'+str(round(self.wavelength*1e6,3)) + ' \t [microns]') 
                print('Optical Band \t'+str(self.optBand)) 
                print('Magnitude \t' + str(self.magnitude))
                print('Flux \t\t'+ str(np.round(self.nPhoton)) + str('\t [photons/m2/s]'))
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        
#    @property
#    def magnitude(self):
#        return self._magnitude
#    
#    @magnitude.setter
#    def magnitude(self,val):
#        self._magnitude  = val
#        self.nPhoton     = self.zeroPoint*10**(-0.4*val)
#            
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
    def show(self):
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        print(self.tag+':')
        for a in attributes:
            if not(a[0].startswith('__') and a[0].endswith('__')):
                if not(a[0].startswith('_')):
                    if not np.shape(a[1]):
                        tmp=a[1]
                        try:
                            print('          '+str(a[0])+': '+str(tmp.tag)+' object') 
                        except:
                            print('          '+str(a[0])+': '+str(a[1])) 
                    else:
                        if np.ndim(a[1])>1:
                            print('          '+str(a[0])+': '+str(np.shape(a[1])))   
            
    def print_properties(self):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOURCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%') 
        print('{: ^18s}'.format('Source') +'{: ^18s}'.format('Wavelength')+ '{: ^18s}'.format('Zenith [arcsec]')+ '{: ^18s}'.format('Azimuth [deg]')+ '{: ^18s}'.format('Altitude [m]')+ '{: ^18s}'.format('Magnitude') + '{: ^18s}'.format('Flux [phot/m2/s]') )
        print('------------------------------------------------------------------------------------------------------------------------------')        
        print('{: ^18s}'.format(self.type) +'{: ^18s}'.format(str(self.wavelength))+ '{: ^18s}'.format(str(self.coordinates[0]))+ '{: ^18s}'.format(str(self.coordinates[1]))+'{: ^18s}'.format(str(np.round(self.altitude,2)))+ '{: ^18s}'.format(str(self.magnitude))+'{: ^18s}'.format(str(np.round(self.nPhoton,1))) )
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%') 
