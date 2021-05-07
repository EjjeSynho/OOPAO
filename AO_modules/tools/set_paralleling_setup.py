# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:02:23 2021

@author: cheritie
"""
import ctypes
import socket 

def set_paralleling_setup(wfs):
    
    try : 
        mkl_rt = ctypes.CDLL('libmkl_rt.so')
        mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
    except:
        try:
            mkl_rt = ctypes.CDLL('./mkl_rt.dll')
            mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
        except:
            print('Could not optimize the parallelisation of the code ')
    
    name = socket.gethostname()
    count = 0 
    
    if name == 'HQL025432':
        mkl_set_num_threads(4)
        wfs.nJobs = 4 
        count = 1
        
    if name == 'mcao144.hq.eso.org':
        mkl_set_num_threads(8)
        wfs.nJobs = 12
        count = 1
        
    if name == 'mcao146':
        mkl_set_num_threads(6)
        wfs.nJobs = 12
        count = 1   
        
    if name == 'mcao145.hq.eso.org':
        mkl_set_num_threads(6)
        wfs.nJobs = 12
        count = 1    
    
    if name == 'mcao147.hq.eso.org':
        mkl_set_num_threads(6)
        wfs.nJobs = 10
        count = 1
        
    if name == 'mcao148':
        mkl_set_num_threads(6)
        wfs.nJobs = 12
        count = 1
        
    if name == 'mcao149':
        mkl_set_num_threads(8)
        wfs.nJobs = 12
        count = 1
    
    if name == 'mcao150.hq.eso.org':
        mkl_set_num_threads(10)
        wfs.nJobs = 16
        count = 1
        
    if name == 'mcao151.hq.eso.org':
        mkl_set_num_threads(10)
        wfs.nJobs = 10
        count = 1
        
# adding a new machine:
#    if name == 'nameOfTheMachine':
#        mkl_set_num_threads(6)
#        wfs.nJobs = 32
#        count = 1

    if count == 0 :
    
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(' WARNING THE PARALLELIZATION OF THE CODE IS NOT OPTIMIZED FOR THIS MACHINE!')
        import multiprocessing
        n_cpu = multiprocessing.cpu_count()
        if n_cpu > 16:
            print(' TAKING THE DEFAULT VALUES FOR JOBLIB : using 10% of the Threads .....')
            mkl_set_num_threads(n_cpu//10)
            wfs.nJobs = n_cpu//5
        else: 
            print(' TAKING THE DEFAULT VALUES FOR JOBLIB : using 10% of the Threads .....')
            mkl_set_num_threads(n_cpu//4)
            wfs.nJobs = n_cpu//2
        
        print(' YOU SHOULD CONSIDER ADDING A CASE IN AO_modules.tools.set_paralleling_setup.py! ')     
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        
