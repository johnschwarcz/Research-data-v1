# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os 
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
import math

"""Cont1 = Control - Template (150)"""
"""Cont2 = Control - Template (120)"""
"""Cont3 = Control - Template extra ingate (120)"""

"""To control output size"""
####################################################
Dropout = .1
####################################################

def Calculater (Source):
    i = 0
    for filename in os.listdir(Source):
        os.chdir(Source)
        if i == 0: 
            Condition = (genfromtxt(filename, delimiter=','))
        else:
            if i ==1: 
                nextline = genfromtxt(filename, delimiter=',')
                Condition = np.stack((Condition, nextline), axis = 0)
            else: 
                nextline = genfromtxt(filename, delimiter=',')
                Condition = np.vstack((Condition, nextline))
        i += 1
    return [Condition]

def Analyzer (Population, Name):
   Population = np.array(Population).squeeze()
   n_bins = 50
   fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
   val = Population[:,0]
   test = Population[:,1]
   time = Population[:,2]
   axs[0].hist(val, bins= n_bins)
   axs[0].set_title('val')
   axs[1].hist(test, bins=n_bins)
   axs[1].set_title('test')
   plt.suptitle(Name, fontsize=10)
   axs[0].set_xlim(xmin=.5)
   axs[1].set_xlim(xmin=.5)
   axs[0].set_xlim(xmax=.85)
   axs[1].set_xlim(xmax=.85)
   Mean_val = np.mean(val)
   Mean_test = np.mean(test)
   Mean_time = np.mean(time)
   Val_variance = np.var(val)
   Test_variance = np.var(test)
   return Mean_val, Mean_test, Mean_time, Val_variance, Test_variance

def welchTtest (mean1, mean2, var1, var2, n1, n2):
    t = (mean1-mean2)/(math.sqrt((var1/n1)+(var2/n2)))
    degrees_of_freedom = (((var1/n1)+(var2/n2))**2)/(((var1**2)/((n1**2)*(n1-1)))+((var2**2)/((n2**2)*(n2-1))))
    hedges_g = ((mean1 - mean2)/(math.sqrt((((n1-1)*var1) + (n2-1)*var2)/(n1+n2-2))))
    print ('T VALUE: ' + str(t) + ' degrees of freedom: ' + str(degrees_of_freedom) + ' Effect size (Hedges G) ' + str(hedges_g))

    return [t, degrees_of_freedom]

if Dropout == 0:
    drop = r'C:\Users\john\Desktop\Data\Dropout 0'
    """SAMPLE SIZES"""
    Exp_E100_N = 132
    C1_E100_N = 156
    C2_E100_N = 117
    C3_E100_N = 127
    Exp_E30_N = 384
    C1_E30_N = 388
    C2_E30_N = 333
    C3_E30_N = 344
    Exp_E15_N = 378
    C1_E15_N = 409
    C2_E15_N = 381
    C3_E15_N = 330
    Exp_E5_N = 401
    C1_E5_N = 369
    C2_E5_N = 365
    C3_E5_N = 359
    Exp_E4_N = 407
    C1_E4_N = 349
    C2_E4_N = 403
    C3_E4_N = 367
    Exp_E3_N = 407
    C1_E3_N = 405
    C2_E3_N = 357
    C3_E3_N = 360
    Exp_E2_N = 378
    C1_E2_N = 383
    C2_E2_N = 415
    C3_E2_N = 431
    Exp_E1_N = 409
    C1_E1_N = 413
    C2_E1_N = 330
    C3_E1_N = 427
    
if Dropout == .1:
    drop = r'C:\Users\john\Desktop\Data\Dropout .1'
    """SAMPLE SIZES"""
    Exp_E100_N = 142
    C1_E100_N = 132
    C2_E100_N = 156
    C3_E100_N = 156
    Exp_E30_N = 323
    C1_E30_N = 417
    C2_E30_N = 349
    C3_E30_N = 349
    Exp_E15_N = 338
    C1_E15_N = 403
    C2_E15_N = 400
    C3_E15_N = 395
    Exp_E5_N =  532
    C1_E5_N =  460
    C2_E5_N = 450
    C3_E5_N = 480
    Exp_E4_N = 404
    C1_E4_N = 480
    C2_E4_N = 535
    C3_E4_N = 515
    Exp_E3_N = 442
    C1_E3_N = 484
    C2_E3_N = 540
    C3_E3_N = 384
    Exp_E2_N = 660
    C1_E2_N =  570
    C2_E2_N =  449
    C3_E2_N = 476
    Exp_E1_N = 560
    C1_E1_N =  541
    C2_E1_N = 480
    C3_E1_N = 750
    
#if Dropout == .2:
#    drop = r'C:\Users\john\Desktop\Data\Dropout .2'
#    """SAMPLE SIZES"""
#    Exp_E100_N = 331
#    C1_E100_N = 
#    C2_E100_N = 
#    C3_E100_N = 
#    Exp_E30_N = 
#    C1_E30_N = 
#    C2_E30_N = 
#    C3_E30_N = 
#    Exp_E15_N = 
#    C1_E15_N = 
#    C2_E15_N = 
#    C3_E15_N = 
#    Exp_E5_N =  
#    C1_E5_N =  
#    C2_E5_N = 
#    C3_E5_N = 
#    Exp_E4_N = 
#    C1_E4_N = 
#    C2_E4_N = 
#    C3_E4_N = 
#    Exp_E3_N = 
#    C1_E3_N = 
#    C2_E3_N = 
#    C3_E3_N = 
#    Exp_E2_N = 
#    C1_E2_N =  
#    C2_E2_N =  
#    C3_E2_N = 
#    Exp_E1_N = 
#    C1_E1_N =  
#    C2_E1_N = 
#    C3_E1_N = 
    
if Dropout == .5:
    drop = r'C:\Users\john\Desktop\Data\Dropout .5'
    print ('DROPOUT: ' + str(Dropout))

####################################################################################################################################################
    
"""EPOCH 100"""
    
#Experimental 
Source =  drop + '\Epoch 100\Experimental (120)'
Exp_Epoch100 = Calculater(Source)
Exp_Epoch100_val_mean, Exp_Epoch100_test_mean, Exp_Epoch100_time, Exp_Epoch100_val_var, Exp_Epoch100_test_var = Analyzer(Exp_Epoch100,'Exp_Epoch100')

#Control 1 
Source = drop + '\Epoch 100\Control- Template (150)'
Cont1_Epoch100 = Calculater(Source)
Cont1_Epoch100_val_mean, Cont1_Epoch100_test_mean, Cont1_Epoch100_time, Cont1_Epoch100_val_var, Cont1_Epoch100_test_var =Analyzer(Cont1_Epoch100,'Cont1_Epoch100')

#Control 2 
Source = drop + '\Epoch 100\Control- Template (120)'
Cont2_Epoch100 = Calculater(Source)
Cont2_Epoch100_val_mean, Cont2_Epoch100_test_mean, Cont2_Epoch100_time, Cont2_Epoch100_val_var, Cont2_Epoch100_test_var = Analyzer(Cont2_Epoch100,'Cont2_Epoch100')

#Control 3
Source = drop + '\Epoch 100\Control- Extra in-gate (120)'
Cont3_Epoch100 = Calculater(Source)
Cont3_Epoch100_val_mean, Cont3_Epoch100_test_mean, Cont3_Epoch100_time, Cont3_Epoch100_val_var, Cont3_Epoch100_test_var = Analyzer(Cont3_Epoch100,'Cont3_Epoch100')

#T tests
print('Epoch 100')
print('                                                                                                                                                                    ' )
print('control 1:')
exp_cont1_T, exp_cont1_deg = welchTtest(Exp_Epoch100_test_mean, Cont1_Epoch100_test_mean, Exp_Epoch100_test_var, Cont1_Epoch100_test_var, n1=  Exp_E100_N, n2= C1_E100_N)
print('control 2:')
exp_cont2_T, exp_cont2_deg = welchTtest(Exp_Epoch100_test_mean, Cont2_Epoch100_test_mean, Exp_Epoch100_test_var, Cont2_Epoch100_test_var, n1=  Exp_E100_N, n2= C2_E100_N)
print('control 3:')
exp_cont3_T, exp_cont3_deg = welchTtest(Exp_Epoch100_test_mean, Cont3_Epoch100_test_mean, Exp_Epoch100_test_var, Cont3_Epoch100_test_var, n1=  Exp_E100_N, n2= C3_E100_N)
print('                                                                                                                                                                    ' )


"""EPOCH 30"""
    
#Experimental
Source =  drop + '\Epoch 30\Experimental (120)'
Exp_Epoch30 = Calculater(Source)
Exp_Epoch30_val_mean, Exp_Epoch30_test_mean, Exp_Epoch30_time, Exp_Epoch30_val_var, Exp_Epoch30_test_var = Analyzer(Exp_Epoch30,'Exp_Epoch30')

#Control 1
Source = drop + '\Epoch 30\Control- Template (150)'
Cont1_Epoch30 = Calculater(Source)
Cont1_Epoch30_val_mean, Cont1_Epoch30_test_mean, Cont1_Epoch30_time, Cont1_Epoch30_val_var, Cont1_Epoch30_test_var =Analyzer(Cont1_Epoch30,'Cont1_Epoch30')

#Control 2 
Source = drop + '\Epoch 30\Control- Template (120)'
Cont2_Epoch30 = Calculater(Source)
Cont2_Epoch30_val_mean, Cont2_Epoch30_test_mean, Cont2_Epoch30_time, Cont2_Epoch30_val_var, Cont2_Epoch30_test_var = Analyzer(Cont2_Epoch30,'Cont2_Epoch30')

#Control 3 
Source = drop + '\Epoch 30\Control- Extra in-gate (120)'
Cont3_Epoch30 = Calculater(Source)
Cont3_Epoch30_val_mean, Cont3_Epoch30_test_mean, Cont3_Epoch30_time, Cont3_Epoch30_val_var, Cont3_Epoch30_test_var = Analyzer(Cont3_Epoch30,'Cont3_Epoch30')

#T tests
print('Epoch 30')
print('                                                                                                                                                                    ' )
print('control 1:')
exp_cont1_T, exp_cont1_deg = welchTtest(Exp_Epoch30_test_mean, Cont1_Epoch30_test_mean, Exp_Epoch30_test_var, Cont1_Epoch30_test_var, n1= Exp_E30_N, n2= C1_E30_N )
print('control 2:')
exp_cont2_T, exp_cont2_deg = welchTtest(Exp_Epoch30_test_mean, Cont2_Epoch30_test_mean, Exp_Epoch30_test_var, Cont2_Epoch30_test_var, n1= Exp_E30_N, n2= C2_E30_N )
print('control 3:')
exp_cont3_T, exp_cont3_deg = welchTtest(Exp_Epoch30_test_mean, Cont3_Epoch30_test_mean, Exp_Epoch30_test_var, Cont3_Epoch30_test_var, n1= Exp_E30_N, n2= C3_E30_N )
print('                                                                                                                                                                    ' )


"""EPOCH 15"""

#Experimental 
Source =  drop + '\Epoch 15\Experimental (120)'
Exp_Epoch15 = Calculater(Source)
Exp_Epoch15_val_mean, Exp_Epoch15_test_mean, Exp_Epoch15_time, Exp_Epoch15_val_var, Exp_Epoch15_test_var = Analyzer(Exp_Epoch15,'Exp_Epoch15')

#Control 1 
Source = drop + '\Epoch 15\Control- Template (150)'
Cont1_Epoch15 = Calculater(Source)
Cont1_Epoch15_val_mean, Cont1_Epoch15_test_mean, Cont1_Epoch15_time, Cont1_Epoch15_val_var, Cont1_Epoch15_test_var =Analyzer(Cont1_Epoch15,'Cont1_Epoch15')

#Control 2
Source = drop + '\Epoch 15\Control- Template (120)'
Cont2_Epoch15 = Calculater(Source)
Cont2_Epoch15_val_mean, Cont2_Epoch15_test_mean, Cont2_Epoch15_time, Cont2_Epoch15_val_var, Cont2_Epoch15_test_var = Analyzer(Cont2_Epoch15,'Cont2_Epoch15')

#Control 3 
Source = drop + '\Epoch 15\Control- Extra in-gate (120)'
Cont3_Epoch15 = Calculater(Source)
Cont3_Epoch15_val_mean, Cont3_Epoch15_test_mean, Cont3_Epoch15_time, Cont3_Epoch15_val_var, Cont3_Epoch15_test_var = Analyzer(Cont3_Epoch15,'Cont3_Epoch15')

#T tests
print('Epoch 15')
print('                                                                                                                                                                    ' )
print('control 1:')
exp_cont1_T, exp_cont1_deg = welchTtest(Exp_Epoch15_test_mean, Cont1_Epoch15_test_mean, Exp_Epoch15_test_var, Cont1_Epoch15_test_var, n1= Exp_E15_N, n2= C1_E15_N )
print('control 2:')
exp_cont2_T, exp_cont2_deg = welchTtest(Exp_Epoch15_test_mean, Cont2_Epoch15_test_mean, Exp_Epoch15_test_var, Cont2_Epoch15_test_var, n1= Exp_E15_N, n2= C2_E15_N )
print('control 3:')
exp_cont3_T, exp_cont3_deg = welchTtest(Exp_Epoch15_test_mean, Cont3_Epoch15_test_mean, Exp_Epoch15_test_var, Cont3_Epoch15_test_var, n1= Exp_E15_N, n2= C3_E15_N )
print('                                                                                                                                                                    ' )


"""EPOCH 5"""

#Experimental 
Source =  drop + '\Epoch 5\Experimental (120)'
Exp_Epoch5 = Calculater(Source)
Exp_Epoch5_val_mean, Exp_Epoch5_test_mean, Exp_Epoch5_time, Exp_Epoch5_val_var, Exp_Epoch5_test_var = Analyzer(Exp_Epoch5,'Exp_Epoch5')

#Control 1 
Source = drop + '\Epoch 5\Control- Template (150)'
Cont1_Epoch5 = Calculater(Source)
Cont1_Epoch5_val_mean, Cont1_Epoch5_test_mean, Cont1_Epoch5_time, Cont1_Epoch5_val_var, Cont1_Epoch5_test_var =Analyzer(Cont1_Epoch5,'Cont1_Epoch5')

#Control 2 
Source = drop + '\Epoch 5\Control- Template (120)'
Cont2_Epoch5 = Calculater(Source)
Cont2_Epoch5_val_mean, Cont2_Epoch5_test_mean, Cont2_Epoch5_time, Cont2_Epoch5_val_var, Cont2_Epoch5_test_var = Analyzer(Cont2_Epoch5,'Cont2_Epoch5')

#Control 3
Source = drop + '\Epoch 5\Control- Extra in-gate (120)'
Cont3_Epoch5 = Calculater(Source)
Cont3_Epoch5_val_mean, Cont3_Epoch5_test_mean, Cont3_Epoch5_time, Cont3_Epoch5_val_var, Cont3_Epoch5_test_var = Analyzer(Cont3_Epoch5,'Cont3_Epoch5')

#T tests
print('Epoch 5')
print('                                                                                                                                                                    ' )
print('control 1:')
exp_cont1_T, exp_cont1_deg = welchTtest(Exp_Epoch5_test_mean, Cont1_Epoch5_test_mean, Exp_Epoch5_test_var, Cont1_Epoch5_test_var, n1= Exp_E5_N, n2= C1_E5_N )
print('control 2:')
exp_cont2_T, exp_cont2_deg = welchTtest(Exp_Epoch5_test_mean, Cont2_Epoch5_test_mean, Exp_Epoch5_test_var, Cont2_Epoch5_test_var, n1= Exp_E5_N, n2= C2_E5_N )
print('control 3:')
exp_cont3_T, exp_cont3_deg = welchTtest(Exp_Epoch5_test_mean, Cont3_Epoch5_test_mean, Exp_Epoch5_test_var, Cont3_Epoch5_test_var, n1= Exp_E5_N, n2= C3_E5_N )
print('                                                                                                                                                                    ' )


"""EPOCH 4"""

#Experimental 
Source =  drop + '\Epoch 4\Experimental (120)'
Exp_Epoch4 = Calculater(Source)
Exp_Epoch4_val_mean, Exp_Epoch4_test_mean, Exp_Epoch4_time, Exp_Epoch4_val_var, Exp_Epoch4_test_var = Analyzer(Exp_Epoch4,'Exp_Epoch4')

#Control 1
Source = drop + '\Epoch 4\Control- Template (150)'
Cont1_Epoch4 = Calculater(Source)
Cont1_Epoch4_val_mean, Cont1_Epoch4_test_mean, Cont1_Epoch4_time, Cont1_Epoch4_val_var, Cont1_Epoch4_test_var =Analyzer(Cont1_Epoch4,'Cont1_Epoch4')

#Control 2 
Source = drop + '\Epoch 4\Control- Template (120)'
Cont2_Epoch4 = Calculater(Source)
Cont2_Epoch4_val_mean, Cont2_Epoch4_test_mean, Cont2_Epoch4_time, Cont2_Epoch4_val_var, Cont2_Epoch4_test_var = Analyzer(Cont2_Epoch4,'Cont2_Epoch4')

#Control 3 
Source = drop + '\Epoch 4\Control- Extra in-gate (120)'
Cont3_Epoch4 = Calculater(Source)
Cont3_Epoch4_val_mean, Cont3_Epoch4_test_mean, Cont3_Epoch4_time, Cont3_Epoch4_val_var, Cont3_Epoch4_test_var = Analyzer(Cont3_Epoch4,'Cont3_Epoch4')

#T tests
print('Epoch 4')
print('                                                                                                                                                                    ' )
print('control 1:')
exp_cont1_T, exp_cont1_deg = welchTtest(Exp_Epoch4_test_mean, Cont1_Epoch4_test_mean, Exp_Epoch4_test_var, Cont1_Epoch4_test_var, n1= Exp_E4_N, n2= C1_E4_N )
print('control 2:')
exp_cont2_T, exp_cont2_deg = welchTtest(Exp_Epoch4_test_mean, Cont2_Epoch4_test_mean, Exp_Epoch4_test_var, Cont2_Epoch4_test_var, n1= Exp_E4_N, n2= C2_E4_N )
print('control 3:')
exp_cont3_T, exp_cont3_deg = welchTtest(Exp_Epoch4_test_mean, Cont3_Epoch4_test_mean, Exp_Epoch4_test_var, Cont3_Epoch4_test_var, n1= Exp_E4_N, n2= C3_E4_N )
print('                                                                                                                                                                    ' )


"""EPOCH 3"""

#Experimental 
Source =  drop + '\Epoch 3\Experimental (120)'
Exp_Epoch3 = Calculater(Source)
Exp_Epoch3_val_mean, Exp_Epoch3_test_mean, Exp_Epoch3_time, Exp_Epoch3_val_var, Exp_Epoch3_test_var = Analyzer(Exp_Epoch3,'Exp_Epoch3')

#Control 1
Source = drop + '\Epoch 3\Control- Template (150)'
Cont1_Epoch3 = Calculater(Source)
Cont1_Epoch3_val_mean, Cont1_Epoch3_test_mean, Cont1_Epoch3_time, Cont1_Epoch3_val_var, Cont1_Epoch3_test_var =Analyzer(Cont1_Epoch3,'Cont1_Epoch3')

#Control 2 
Source = drop + '\Epoch 3\Control- Template (120)'
Cont2_Epoch3 = Calculater(Source)
Cont2_Epoch3_val_mean, Cont2_Epoch3_test_mean, Cont2_Epoch3_time, Cont2_Epoch3_val_var, Cont2_Epoch3_test_var = Analyzer(Cont2_Epoch3,'Cont2_Epoch3')

#Control 3 
Source = drop + '\Epoch 3\Control- Extra in-gate (120)'
Cont3_Epoch3 = Calculater(Source)
Cont3_Epoch3_val_mean, Cont3_Epoch3_test_mean, Cont3_Epoch3_time, Cont3_Epoch3_val_var, Cont3_Epoch3_test_var = Analyzer(Cont3_Epoch3,'Cont3_Epoch3')

#T tests
print('Epoch 3')
print('                                                                                                                                                                    ' )
print('control 1:')
exp_cont1_T, exp_cont1_deg = welchTtest(Exp_Epoch3_test_mean, Cont1_Epoch3_test_mean, Exp_Epoch3_test_var, Cont1_Epoch3_test_var, n1= Exp_E3_N, n2= C1_E3_N )
print('control 2:')
exp_cont2_T, exp_cont2_deg = welchTtest(Exp_Epoch3_test_mean, Cont2_Epoch3_test_mean, Exp_Epoch3_test_var, Cont3_Epoch3_test_var, n1= Exp_E3_N, n2= C2_E3_N )
print('control 3:')
exp_cont3_T, exp_cont3_deg = welchTtest(Exp_Epoch3_test_mean, Cont3_Epoch3_test_mean, Exp_Epoch3_test_var, Cont3_Epoch3_test_var, n1= Exp_E3_N, n2= C3_E3_N )
print('                                                                                                                                                                    ' )


"""EPOCH 2"""

#Experimental
Source =  drop + '\Epoch 2\Experimental (120)'
Exp_Epoch2 = Calculater(Source)
Exp_Epoch2_val_mean, Exp_Epoch2_test_mean, Exp_Epoch2_time, Exp_Epoch2_val_var, Exp_Epoch2_test_var = Analyzer(Exp_Epoch2,'Exp_Epoch2')

#Control 1 
Source = drop + '\Epoch 2\Control- Template (150)'
Cont1_Epoch2 = Calculater(Source)
Cont1_Epoch2_val_mean, Cont1_Epoch2_test_mean, Cont1_Epoch2_time, Cont1_Epoch2_val_var, Cont1_Epoch2_test_var =Analyzer(Cont1_Epoch2,'Cont1_Epoch2')

#Control 2 
Source = drop + '\Epoch 2\Control- Template (120)'
Cont2_Epoch2 = Calculater(Source)
Cont2_Epoch2_val_mean, Cont2_Epoch2_test_mean, Cont2_Epoch2_time, Cont2_Epoch2_val_var, Cont2_Epoch2_test_var = Analyzer(Cont2_Epoch2,'Cont2_Epoch2')

#Control 3 
Source = drop + '\Epoch 2\Control- Extra in-gate (120)'
Cont3_Epoch2 = Calculater(Source)
Cont3_Epoch2_val_mean, Cont3_Epoch2_test_mean, Cont3_Epoch2_time, Cont3_Epoch2_val_var, Cont3_Epoch2_test_var = Analyzer(Cont3_Epoch2,'Cont3_Epoch2')

#T tests
print('Epoch 2')
print('                                                                                                                                                                    ' )
print('control 1:')
exp_cont1_T, exp_cont1_deg = welchTtest(Exp_Epoch2_test_mean, Cont1_Epoch2_test_mean, Exp_Epoch2_test_var, Cont1_Epoch2_test_var, n1= Exp_E2_N, n2= C1_E2_N )
print('control 2:')
exp_cont2_T, exp_cont2_deg = welchTtest(Exp_Epoch2_test_mean, Cont2_Epoch2_test_mean, Exp_Epoch2_test_var, Cont3_Epoch2_test_var, n1= Exp_E2_N, n2= C2_E2_N )
print('control 3:')
exp_cont3_T, exp_cont3_deg = welchTtest(Exp_Epoch2_test_mean, Cont3_Epoch2_test_mean, Exp_Epoch2_test_var, Cont3_Epoch2_test_var, n1= Exp_E2_N, n2= C3_E2_N )
print('                                                                                                                                                                    ' )


"""EPOCH 1"""

#Experimental 
Source =  drop + '\Epoch 1\Experimental (120)'
Exp_Epoch1 = Calculater(Source)
Exp_Epoch1_val_mean, Exp_Epoch1_test_mean, Exp_Epoch1_time, Exp_Epoch1_val_var, Exp_Epoch1_test_var = Analyzer(Exp_Epoch1,'Exp_Epoch1')

#Control 1 
Source = drop + '\Epoch 1\Control- Template (150)'
Cont1_Epoch1 = Calculater(Source)
Cont1_Epoch1_val_mean, Cont1_Epoch1_test_mean, Cont1_Epoch1_time, Cont1_Epoch1_val_var, Cont1_Epoch1_test_var =Analyzer(Cont1_Epoch1,'Cont1_Epoch1')

#Control 2 
Source = drop + '\Epoch 1\Control- Template (120)'
Cont2_Epoch1 = Calculater(Source)
Cont2_Epoch1_val_mean, Cont2_Epoch1_test_mean, Cont2_Epoch1_time, Cont2_Epoch1_val_var, Cont2_Epoch1_test_var = Analyzer(Cont2_Epoch1,'Cont2_Epoch1')

#Control 3 
Source = drop + '\Epoch 1\Control- Extra in-gate (120)'
Cont3_Epoch1 = Calculater(Source)
Cont3_Epoch1_val_mean, Cont3_Epoch1_test_mean, Cont3_Epoch1_time, Cont3_Epoch1_val_var, Cont3_Epoch1_test_var = Analyzer(Cont3_Epoch1,'Cont3_Epoch1')

#T tests
print('Epoch 1')
print('                                                                                                                                                                    ' )
print('control 1:')
exp_cont1_T, exp_cont1_deg = welchTtest(Exp_Epoch1_test_mean, Cont1_Epoch1_test_mean, Exp_Epoch1_test_var, Cont1_Epoch1_test_var, n1= Exp_E1_N, n2= C1_E1_N )
print('control 2:')
exp_cont2_T, exp_cont2_deg = welchTtest(Exp_Epoch1_test_mean, Cont2_Epoch1_test_mean, Exp_Epoch1_test_var, Cont3_Epoch1_test_var, n1= Exp_E1_N, n2= C2_E1_N )
print('control 3:')
exp_cont3_T, exp_cont3_deg = welchTtest(Exp_Epoch1_test_mean, Cont3_Epoch1_test_mean, Exp_Epoch1_test_var, Cont3_Epoch1_test_var, n1= Exp_E1_N, n2= C3_E1_N )
print('                                                                                                                                                                    ' )

########################################################################################################################################################################################

"""plotting"""


#    Val_over_Epoch

Progress_exp = (Exp_Epoch1_test_mean, Exp_Epoch2_test_mean, Exp_Epoch3_test_mean, Exp_Epoch4_test_mean, Exp_Epoch5_test_mean, Exp_Epoch15_test_mean, Exp_Epoch30_test_mean, Exp_Epoch100_test_mean)
Progress_cont1 = (Cont1_Epoch1_test_mean, Cont1_Epoch2_test_mean, Cont1_Epoch3_test_mean, Cont1_Epoch4_test_mean, Cont1_Epoch5_test_mean, Cont1_Epoch15_test_mean, Cont1_Epoch30_test_mean, Cont1_Epoch100_test_mean)
Progress_cont2 = (Cont2_Epoch1_test_mean, Cont2_Epoch2_test_mean, Cont2_Epoch3_test_mean, Cont2_Epoch4_test_mean, Cont2_Epoch5_test_mean, Cont2_Epoch15_test_mean, Cont2_Epoch30_test_mean, Cont2_Epoch100_test_mean)
Progress_cont3 = (Cont3_Epoch1_test_mean, Cont3_Epoch2_test_mean, Cont3_Epoch3_test_mean, Cont3_Epoch4_test_mean, Cont3_Epoch5_test_mean, Cont3_Epoch15_test_mean, Cont3_Epoch30_test_mean, Cont3_Epoch100_test_mean)

Progress_exp_val = (Exp_Epoch1_val_mean, Exp_Epoch2_val_mean, Exp_Epoch3_val_mean, Exp_Epoch4_val_mean, Exp_Epoch5_val_mean, Exp_Epoch15_val_mean, Exp_Epoch30_val_mean, Exp_Epoch100_val_mean)
Progress_cont1_val = (Cont1_Epoch1_val_mean, Cont1_Epoch2_val_mean, Cont1_Epoch3_val_mean, Cont1_Epoch4_val_mean, Cont1_Epoch5_val_mean, Cont1_Epoch15_val_mean, Cont1_Epoch30_val_mean, Cont1_Epoch100_val_mean)
Progress_cont2_val = (Cont2_Epoch1_val_mean, Cont2_Epoch2_val_mean, Cont2_Epoch3_val_mean, Cont2_Epoch4_val_mean, Cont2_Epoch5_val_mean, Cont2_Epoch15_val_mean, Cont2_Epoch30_val_mean, Cont2_Epoch100_val_mean)
Progress_cont3_val = (Cont3_Epoch1_val_mean, Cont3_Epoch2_val_mean, Cont3_Epoch3_val_mean, Cont3_Epoch4_val_mean, Cont3_Epoch5_val_mean, Cont3_Epoch15_val_mean, Cont3_Epoch30_val_mean, Cont3_Epoch100_val_mean)
x_ax = [1,2,3,4,5,15,30,100]

fig = plt.figure()
ax = plt.axes()
plt.plot(Progress_exp,'-b', label = 'exp test')
plt.plot(Progress_cont1,'-r', label = 'cont1 test')
plt.plot(Progress_cont2,'-g', label = 'cont2 test')
plt.plot(Progress_cont3,'-y', label = 'cont3 test')
plt.plot(Progress_exp_val,':b', label = 'exp val')
plt.plot(Progress_cont1_val,':r', label = 'cont1 val')
plt.plot(Progress_cont2_val,':g', label = 'cont2 val')
plt.plot(Progress_cont3_val,':y', label = 'cont3 val')
plt.legend()
plt.title('network accuracy over recorded epochs')
ax.set(xlabel='Epoch:    1            2            3            4            5    ...    15    ...   30    ...   100          ')

fig = plt.figure()
ax = plt.axes()
plt.plot(x_ax, Progress_exp,'-b', label = 'exp test')
plt.plot(x_ax, Progress_cont1,'-r', label = 'cont1 test')
plt.plot(x_ax, Progress_cont2,'-g', label = 'cont2 test')
plt.plot(x_ax, Progress_cont3,'-y', label = 'cont3 test')
plt.plot(x_ax, Progress_exp_val,':b', label = 'exp val')
plt.plot(x_ax, Progress_cont1_val,':r', label = 'cont1 val')
plt.plot(x_ax, Progress_cont2_val,':g', label = 'cont2 val')
plt.plot(x_ax, Progress_cont3_val,':y', label = 'cont3 val')
plt.legend()
plt.title('network accuracy estimations over all epochs')