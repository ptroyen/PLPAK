"""
This file shows how to read a mechanism file with reaction rates specified as expressions in the cantera like format.
The mechanism file is used to solve the chemical kinetics.
Bolsig+ interpolated expressions can be used.
Can also specify your own expressions for rates that depend on T, Tv and Te.

"""

import sys
import os

# dir_plpak = os.path.dirname("E:\POKHAREL_SAGAR\gits\plasmaKinetics\plpak")
dir_plpak_data = os.path.dirname("../data")
dir_plpak_src = os.path.dirname("../src/plpak")
sys.path.append(dir_plpak_src)
import plpak as pl

import cantera as ct
import numpy as np
import scipy as sc

# import functions from test.py file all of them
from test import plotTemporalScale

pi = np.pi
log = np.log
exp = np.exp

bolsigFit = pl.System.bolsigFit # needs E in eV
coul_log = pl.System.coul_log # needs Te in eV and n_e in m^-3
coul_freq = pl.System.coul_freq # needs Te in eV and n_e in m^-3

## Other data required:
# Air Electron neutral momentum transfer collision frequency
bolsigFit_air_nu_en = np.array([-29.95, 0.3370, -0.3595, 0.2342E-01, -0.3446E-03])  # 1/s/NTOT





## ODE system with laser absorption
def dYdt_all_laser(t, Y, plasmaSys, laser):
    '''
    Solve the species and temperatures at the same time in a single ODE system
    Also include the laser absorption
    update the system based on the state
    last three are bulk temerature, vibrational temperature, and electronic temperature
    '''

    plasmaSys.Ysp = Y[0:-3]

    plasmaSys.Temp[0] = Y[-3]
    plasmaSys.Temp[1] = Y[-2]
    plasmaSys.Temp[2] = Y[-1]

    # update the system
    plasmaSys.update()
    # show nu_en and nu_ei
    print("nu_en: ", plasmaSys.nu_en, "nu_ei: ", plasmaSys.nu_ei)

    # zeros dydt
    dydt = np.zeros(len(Y))

    # update the species rates
    dydt[0:-3] = plasmaSys.dYdt
    # update the temperature rates
    dydt[-3:] = plasmaSys.dTdt


    # alsso include the photo-detachment rates from O2-
    addPhotoDetachments(dydt,plasmaSys,laser) # this will change the RHS of O2- and O2 and electron in dydt


    # print("laser Switch: ", laser.switch)

    if laser.switch == 1:
        # Update the electron temperature equation due to the laser absorption
        ## update the laser model to get the current intensity
        # laser.updateIn(t) # also updates the laser intensity in the class os that laser absorption can be calculated
        ## shape is defined while creating the laser object, options : shape='gauss', shape='shaped'
        Inc = laser.updateIn(t)
        # get nu_en and nu_ei
        nueff = plasmaSys.nu_ei + plasmaSys.nu_en
        neTot = plasmaSys.Ysp[-1]
        # find absorption: J/m3-s
        Qlaser = laser.laserAbsorption(neTot, nueff)
        # update the electron temperature equation
        dydt[-1] = dydt[-1] + Qlaser/(1.5*pl.CO_kB*neTot)


    return dydt






## define a function that provides the rate of photo-detachment of O2- given In and energy of the photon
def photoDetachmentRate(laser):
    '''
    :param In: laser intensity in W/m2
    :param E: energy of the photon in J
    :return: rate of photo-detachment in 1/s , the rrate of production = rate of photo-detachment * [O2-]
    '''
    # get the cross section in cm2
    sigma = lambda E: E*(E-0.15)**1.5*(0.370e-17 + 0.071e-17*(E-0.15)) # cm2, E0 and E in eVs, E0 is the onset value = 0.15 eV


    # E from laser
    EeV = pl.CO_h*pl.CO_c/(laser.lam*pl.CO_eC) # eV
    In_ =laser.switch*laser.In

    # convert to m2
    sig = sigma(EeV)*1.0e-4

    print("Cross section: ", sig, "m2")
    detach_rate = sig*(In_/(EeV*pl.CO_eC)) # 1/s
    
    return detach_rate
    
## define a function that corrects the RHS after including the photo-detachment rate
def addPhotoDetachments(dydts,plasmaSys,laser):
    '''
    :param dydts: RHS of the ODE system
    :param plasmaSys: plasmaSystem object
    :param laser: laserModel object
    :return: corrected RHS by changing dydts
    '''

    Ysp = plasmaSys.Ysp
    

    idO2m = 20
    ide = 21
    idO2 = 7

    # # check from gas object if the ids are correct
    # if plasmaSys.gas.species_index('O2m') != idO2m:
    #     print("O2- id is not correct")
    # if plasmaSys.gas.species_index('ele') != ide:
    #     print("electron id is not correct")
    # if plasmaSys.gas.species_index('O2') != idO2:
    #     print("O2 id is not correct")


    dndt = photoDetachmentRate(laser)*Ysp[idO2m]

    # correct the RHS
    dydts[idO2m] = dydts[idO2m] - dndt
    dydts[ide] = dydts[ide] + dndt
    dydts[idO2] = dydts[idO2] + dndt

    # priint the rate of photo-detachment
    print("Photo-detachment rate: ", dndt, "#/m3-s","[O2-]: ", Ysp[idO2m], " [O2]: ", Ysp[idO2], " [ele]: ", Ysp[ide])





## Solve the species and temperatures at the same time in a single ODE system
def dYdt_all_base(t, Y, plasmaSys):
    # update the system based on the state
    # last three are bulk temerature, vibrational temperature, and electronic temperature
    plasmaSys.Ysp = Y[0:-3]

    plasmaSys.Temp[0] = Y[-3]
    plasmaSys.Temp[1] = Y[-2]
    plasmaSys.Temp[2] = Y[-1]

    # update the system
    plasmaSys.update()

    # zeros dydt
    dydt = np.zeros(len(Y))

    # update the species rates
    dydt[0:-3] = plasmaSys.dYdt
    # update the temperature rates
    dydt[-3:] = plasmaSys.dTdt

    return dydt


## Solve the species and temperatures at the same time in a single ODE system
def dYdt_all_base_TvO2(t, Y, plasmaSys):
    # update the system based on the state
    # last three are bulk temerature, vibrational temperature, and electronic temperature
    plasmaSys.Ysp = Y[0:-4]

    plasmaSys.Temp[0] = Y[-4]
    plasmaSys.Temp[1] = Y[-3]
    plasmaSys.Temp[2] = Y[-2]

    TvO2 = Y[-1]
    theta_v_O2 = 2256.0 # K
    Tg = plasmaSys.Temp[0]

    # update the system
    plasmaSys.update()

    # zeros dydt
    dydt = np.zeros(len(Y))

    # update the species rates
    dydt[0:-4] = plasmaSys.dYdt
    # update the temperature rates
    dydt[-4:-1] = plasmaSys.dTdt



    # update the vibrational temperature of O2 --------------------------------
    dEv_dTv_o2 = plasmaSys.dEv_dTv_(TvO2, theta_v_O2)
    nO2 = plasmaSys.Ysp[plasmaSys.gas.species_index('O2')]
        
    Ev_Tv = plasmaSys.vibEnergy(TvO2, theta_v_O2)
    Ev_T = plasmaSys.vibEnergy(Tg, theta_v_O2)
    QVT_O2 = (Ev_Tv - Ev_T)*nO2/plasmaSys.tauVT_O2 # J/m3-s

    dydt[-1] = (-QVT_O2/nO2)/dEv_dTv_o2
    # update gas temperature equation with QVT_O2
    dydt[-4] += QVT_O2/(plasmaSys.cp_mix*plasmaSys.rho)
    ## ---------------------------------------------------------------------------
    # show TvO2
    print("TvO2: ", TvO2)



    return dydt




def directSolve(fname='solnPlasma',Te0=11600.0,dt=1.0e-6,mechFile="airPlasma/combinedAirMech.yaml",sp='N2',**kwargs):
    

    import matplotlib.pyplot as plt

    # select cylcer from 30 colors in cmap
    import matplotlib as mpl
    # from tab20 import tab20_data
    colors = plt.cm.tab20(np.linspace(0,1,20))
    # update the cycler
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

    # Direct Solve with give mechanism file
    # plasmaSystem = pl.PlasmaSolver
    # plasmaSystem = newPlasmaSolver
    plasmaSystem = pl.PlasmaSolver3T

    TvibO2 = kwargs.get('TvO2',None)


    ## which ODEfunction to use : dydt_all = [dYdt_all_base, dYdt_all_laser]
    dYdt_all = dYdt_all_base
    # dYdt_all = dYdt_all_laser


    if TvibO2 is not None:
        dYdt_all = dYdt_all_base_TvO2





    # make the plasma system
    # plasmaSys = plasmaSystem('tutorial0.yaml',verbose=False) # initialize the plasma system through the mechanism file
    # mechFile = "airPlasma/N2PlasmaMech.yaml"
    # mechFile = "airPlasma/combinedAirMech.yaml"
    # mechFile = "airPlasma/AirPlasmaMechanism.yaml"
    plasmaSys = plasmaSystem(mechFile,verbose=False) 

    nSp = plasmaSys.nsp

    # from kwargs
    fracO2Read = kwargs.get('fracO2',0.22)


    # refmech = "plasmaN2.yaml"
    # refPlasmaSys = plasmaSystem(refmech,verbose=False)
    
    # # print all the reactions from ref and current mechanism
    # # number of reactions in both the mechanisms
    # nRxn = len(plasmaSys.gas.reactions())
    # nRxnRef = len(refPlasmaSys.gas.reactions())

    # for i in range(nRxn):
    #     print("Reaction ", i, " : ", refPlasmaSys.gas.reaction(i), " : ", plasmaSys.gas.reaction(i))

    # wait for input
    # input("Press Enter to continue...")

    if sp == 'N2':
    # # Initial conditions For N2 mechanism  -----------------------------------------------------------------------
        # Species: [N, N2, N2_A, N2_B, N2_C, Np, N2p, N3p, N4p, ele]
        # [N, N2, N2_A, N2_B, N2_C, Np, N2p, N3p, N4p, ele]
       
        n0 = 2.45e25
        n0 = n0*300.0/760.0
        # nI = 1.00e23
        nI = 1.0e21 # papeer n2
        # nI = 5.00e22

        # for In when at 1 atm ne = 1.0e23
        # nI = 8.27070513237335e+22   # 300 torr


        # find Id of species for initial conditions:
        idN2 = plasmaSys.gas.species_index('N2')
        idN2p = plasmaSys.gas.species_index('N2p')
        idele = plasmaSys.gas.species_index('ele')
        # idO2 = plasmaSys.gas.species_index('O2')
        # idO2p = plasmaSys.gas.species_index('O2p')
        # idNO = plasmaSys.gas.species_index('NO')

        Ysp0 = np.zeros(nSp)


        ne = nI
        n0 = n0 - nI
        Ysp0 = np.array([0.0,n0,0,0,0,0,nI,0,0,ne])
    ##-----------------------------------------------------------------------------------------------------------------
    elif sp == 'Air':
    ##  For air mechanism -----------------------------------------------------------------------------------------------
        '''
          species: [N, N2, N2_A, N2_B, N2_C, N2_ap
                O, O2, O_D, O3, NO,
                Np, N2p, N3p, N4p,
                O2p, Op, O4p, NOp,
                Om, O2m, ele]
        '''
        # find Id of species for initial conditions:
        idN2 = plasmaSys.gas.species_index('N2')
        idN2p = plasmaSys.gas.species_index('N2p')
        idele = plasmaSys.gas.species_index('ele')
        idO2 = plasmaSys.gas.species_index('O2')
        idO2p = plasmaSys.gas.species_index('O2p')
        idNO = plasmaSys.gas.species_index('NO')

        Ysp0 = np.zeros(nSp)
        # Mixture , N2^+, O2^+ and E only
        # ipotN2 = 15.5810
        # ipotO2 = 12.0687
        # fac = ipotN2*ipotO2/(ipotN2+ipotO2)
        # ne = 1.0e23
        # nN2p = fac*ne/ipotN2
        # # nN2p = ne
        # # nN2p = 0.0
        # nO2p = fac*ne/ipotO2
        # N2+ =  1.5967858672000928e+22
        # O2+ =  1.2472092965932226e+23
        # e =  1.4068878833132318e+23
        nNeutral = 2.4475e25
        perO2 = 0.22

    # #    ## For 1.0e23; Aleksandrov
    #     nN2p = 1.5967858672000928e+22
    #     nO2p = 1.2472092965932226e+23

        # # # for 4.0e23 ; Chizhov
        # ne = 4.0e23
        # nO2p = 3.2795623949898156e+23
        # nN2p = ne - nO2p
        

        # For ne = 1.4e22 ; Papeer
        ne = 1.4e22
        nN2p = 1.565427079273731e+21
        nO2p = ne - nN2p

        # ## Gerardo Thomson tighut focusing : 8.0e18 W/m2 , ne = 5.0e23
        # nN2p = 1.8e23
        # nO2p = 5.2e23

        nN2p = 2.0e23
        nO2p = 3.0e23

        nN2p = 2.5e23
        nO2p = 3.5e23

        # n2p = 2.5e23
        # nO2p = 4.5e23

        # nN2p = 1.0e23
        # nO2p = 4.0e23

        ne = nN2p + nO2p

        # # For ne = 1.0e22 ; laux
        # nO2p = 7.2e21*0.001
        # nN2p = 1.754e21*0.001

        # # N2 for Aleks with 2.75 percent O2
        # perO2 = 0.0275
        # ne = 1.0e23
        # nN2p = 7.128615137544911e+22
        # # nN2p = 4.128615137544911e+22
        # nO2p = ne - nN2p


        ne = nN2p + nO2p
        rho_c = nN2p + nO2p - ne
        NN2 = nNeutral*(1-perO2) - nN2p
        NO2 = nNeutral*(perO2) - nO2p
        # NO2 = 1000

        # Ysp0 = np.array([0,NN2,0,0,0,0,0,NO2,0,0,0,0,nN2p,0,0,nO2p,0,0,0,0,0,ne])

                # update the initial conditions
        Ysp0[idN2] = NN2
        Ysp0[idN2p] = nN2p
        Ysp0[idele] = ne
        Ysp0[idO2] = NO2
        Ysp0[idO2p] = nO2p
        # Ysp0[idNO] = NO

    elif sp == 'N2combined':
    ##  For air mechanism -----------------------------------------------------------------------------------------------
        '''
          species: [N, N2, N2_A, N2_B, N2_C, N2_ap
                O, O2, O_D, O3, NO,
                Np, N2p, N3p, N4p,
                O2p, Op, O4p, NOp,
                Om, O2m, ele]
        '''

        # find Id of species for initial conditions:
        idN2 = plasmaSys.gas.species_index('N2')
        idN2p = plasmaSys.gas.species_index('N2p')
        idele = plasmaSys.gas.species_index('ele')
        idO2 = plasmaSys.gas.species_index('O2')
        idO2p = plasmaSys.gas.species_index('O2p')
        idNO = plasmaSys.gas.species_index('NO')

        Ysp0 = np.zeros(nSp)



        perO2 = 0.0
        perNO = 0.000
        # Mixture , N2^+, O2^+ and E only
        # ipotN2 = 15.5810
        # ipotO2 = 12.0687
        # fac = ipotN2*ipotO2/(ipotN2+ipotO2)
        # ne = 1.0e23
        # nN2p = fac*ne/ipotN2
        # # nN2p = ne
        # # nN2p = 0.0
        # nO2p = fac*ne/ipotO2
        # N2+ =  1.5967858672000928e+22
        # O2+ =  1.2472092965932226e+23
        # e =  1.4068878833132318e+23
        nNeutral = 2.45e25

       ## For 1.0e23; Aleksandrov
        nN2p = 1.0e23
        nO2p = 0.0
    # #     # nN2p = 9.0e22
    # #     # nO2p = 1.0e22

        # # # for 4.0e23 ; Chizov
        # nN2p = 5.0e23
        # nO2p = 0.0

        # # # For ne = 1.4e22 ; Papeer
        # nO2p = 0.0
        # nN2p = 1.0e21   # papeer n2

        # # For ne = 1.0e22 ; laux
        # nO2p = 7.2e21*0.001
        # nN2p = 1.754e21*0.001

        # # arbitrary
        # nO2p = 0.0
        # nN2p = 1.0e21


        ne = nN2p + nO2p
        # perO2 = 0.0
        rho_c = nN2p + nO2p - ne
        NN2 = nNeutral*(1.0 - perO2 - perNO) - nN2p
        NO2 = nNeutral*(perO2) - nO2p
        NO = nNeutral*(perNO)
        # NO2 = 1000

        # Ysp0 = np.array([0,NN2,0,0,0,0,0,NO2,0,0,NO,0,nN2p,0,0,nO2p,0,0,0,0,0,ne])

        # update the initial conditions
        Ysp0[idN2] = NN2
        Ysp0[idN2p] = nN2p
        Ysp0[idele] = ne
        Ysp0[idO2] = NO2
        Ysp0[idO2p] = nO2p
        Ysp0[idNO] = NO


    ##-----------------------------------------------------------------------------------------------------------------
    elif sp == 'AirAleks':
    ##  For air mechanism -----------------------------------------------------------------------------------------------
        '''
          species: [N, N2, N2_A, N2_B, N2_C, N2_ap
                O, O2, O_D, O3, NO,
                Np, N2p, N3p, N4p,
                O2p, Op, O4p, NOp,
                Om, O2m, N2O2p, H2O , H2OO2p,
                ele]
        '''
        # Mixture , N2^+, O2^+ and E only
        # ipotN2 = 15.5810
        # ipotO2 = 12.0687
        # fac = ipotN2*ipotO2/(ipotN2+ipotO2)
        # ne = 1.0e23
        # nN2p = fac*ne/ipotN2
        # # nN2p = ne
        # # nN2p = 0.0
        # nO2p = fac*ne/ipotO2
        # N2+ =  1.5967858672000928e+22
        # O2+ =  1.2472092965932226e+23
        # e =  1.4068878833132318e+23
        nNeutral = 2.5e25
        perH2O = 0.00
        perN2 = 0.78
        perO2 = 0.22

        # correct because of perH2O
        perN2 = perN2/(1-perH2O)
        perO2 = perO2/(1-perH2O)

        # For 1.0e23
        # nN2p = 1.5967858672000928e+22
        # nO2p = 1.2472092965932226e+23

        # # for 4.0e23; ChiZov
        # nN2p = 7.3410e+22
        # nO2p = 3.176e+23

        # nN2p = 9.5410e+22
        # nO2p = 2.576e+23

        # For ne = 1.4e22 ; Papeer
        nO2p = 1.232e22
        nN2p = 3.0e21

        # # For ne = 1.0e22 ; laux
        # nO2p = 7.2e21
        # nN2p = 1.754e21

        ne = nN2p + nO2p
        rho_c = nN2p + nO2p - ne
        NN2 = nNeutral*(perN2) - nN2p
        NO2 = nNeutral*(perO2) - nO2p
        NH2O = nNeutral*(perH2O)
        # NO2 = 1000

        Ysp0 = np.array([0,NN2,0,0,0,0,0,NO2,0,0,0,0,nN2p,0,0,nO2p,0,0,0,0,0,0,NH2O,0,ne])
    ##-----------------------------------------------------------------------------------------------------------------

    elif sp == 'N2Aleks':
        """
        [N, N2, N2_A, N2_B, N2_C,
            O, O2, 
            Np, N2p, N3p, N4p,
            O2p, O4p, N2O2p,
            ele]
        """
        n0 = 2.5e25
        nI = 1.00e23
        ne = nI

        no2 = n0*0.0275
        nn2 = n0*0.9725

        # no2 = n0*0.22
        # nn2 = n0*0.78
        no2p = 0.5e22
        nn2p = 9.5e22

        # no2 = n0*0
        # nn2 = n0*1
        # no2p = 0
        # nn2p = 1.0e23

        no2 = no2 - no2p
        nn2 = nn2 - nn2p
        ne = no2p + nn2p
        Ysp0 = np.array([0,nn2,0,0,0,0,no2,0,nn2p,0,0,no2p,0,0,0,0,ne])
    ##-----------------------------------------------------------------------------------------------------------------

   

    # # Initial conditions for tutorial0
    # n0 = 2.5e25
    # nI = 1.00e23
    # ne = nI
    # n0 = n0 - 2*nI
    # # [N, N2, N2p, ele]
    # Ysp0 = np.array([0.0, n0, nI, ne])

    
# # # Initial conditions For N2 mechanism  -----------------------------------------------------------------------
#     # Species: [N, N2, N2_A, N2_B, N2_C, Np, N2p, N3p, N4p, ele]
#     n0 = 2.5e25
#     nI = 1.00e23
#     ne = nI
#     n0 = n0 - 2*nI
#     Ysp0 = np.array([0.0,n0,0,0,0,0,nI,0,0,ne])
# ##-----------------------------------------------------------------------------------------------------------------

# ##  For air mechanism -----------------------------------------------------------------------------------------------
#     '''
#       species: [N, N2, N2_A, N2_B, N2_C, N2_ap
#             O, O2, O_D, O3, NO,
#             Np, N2p, N3p, N4p,
#             O2p, Op, O4p, NOp,
#             Om, O2m, ele]
#     '''
#     # Mixture , N2^+, O2^+ and E only
#     ipotN2 = 15.5810
#     ipotO2 = 12.0687
#     fac = ipotN2*ipotO2/(ipotN2+ipotO2)
#     ne = 1.0e23
#     nNeutral = 2.5e25
#     nN2p = fac*ne/ipotN2
#     nO2p = fac*ne/ipotO2
#     rho_c = nN2p + nO2p - ne
#     NN2 = nNeutral*(0.78) - nN2p
#     NO2 = nNeutral*(0.22) - nO2p

#     Ysp0 = np.array([0,NN2,0,0,0,0,0,NO2,0,0,0,0,nN2p,0,0,nO2p,0,0,0,0,0,ne])
# ##-----------------------------------------------------------------------------------------------------------------

# ##  For air mechanism But only N2 present -----------------------------------------------------------------------------
#     '''
#       species: [N, N2, N2_A, N2_B, N2_C, N2_ap
#             O, O2, O_D, O3, NO,
#             Np, N2p, N3p, N4p,
#             O2p, Op, O4p, NOp,
#             Om, O2m, ele]
#     '''
#     n0 = 2.5e25
#     nI = 1.00e23
#     ne = nI
#     n0 = n0 - 2*nI

#     Ysp0 = np.array([0,n0,0,0,0,0,0,0,0,0,0,0,nI,0,0,0,0,0,0,0,0,ne])
# ##---------------------------------------------------------------------------

    
    # T0s = np.array([300.0, 300.0, 11600.0])
    T0s = np.array([300.0,350.0,Te0])


    # Initialize with initial conditions
    plasmaSys.Ysp0 = Ysp0
    plasmaSys.Temp0 = T0s 
    plasmaSys.initialize()


    # press to continue

    eId = plasmaSys.gas.species_index('ele')

    # print the initial state of the system
    print("Initial state of the system")
    print("Tg = ", plasmaSys.Temp[0], "Tv = ", plasmaSys.Temp[1], "Te = ", plasmaSys.Temp[2])
    print("Ysp = ", plasmaSys.Ysp)
    print("Ug = ", plasmaSys.Ug)
    print("Hg = ", plasmaSys.Hg)

    # input("Press Enter to continue...")
    # setup the array for initial conditions all solved at once
    # need to combine Ysp and Temp
    yt0 = np.concatenate((plasmaSys.Ysp, plasmaSys.Temp))



    if TvibO2 is not None:
        yt0 = np.concatenate((plasmaSys.Ysp, plasmaSys.Temp, [TvibO2]))




    # time details
    t0 = 0
    # tf = 5.0e-4
    tf = dt
    dt = 1e-14
    max_dt = 1e-9
    relaxdt = 1.0e-5
    t = t0
    int_t = (t0, tf)


    # All Solution
    # Make an object to hold current solution
    soln = pl.Solution(t0, plasmaSys)
    # # append initial state : the initial state is already appended during instantiation
    # soln.solnPush(t_array[0], plasmaSys)

    solMethods=["RK45","RK23","DOP853","Radau","BDF","LSODA"]

    # print what is being solved
    print("Solving the system using ", solMethods[3], " method")
    # initial conditions
    print("Initial conditions: ", yt0)
    # input("Press Enter to continue...")

    # Now integrate the system
    solY = sc.integrate.solve_ivp(dYdt_all, int_t, yt0, method=solMethods[5], args=(plasmaSys,), rtol=1e-4, atol=1e-6)


    # print the message from the solver
    print(solY.message)


    # Now for post-processing, calculate all the properties based on the solved state
    for i in range(len(solY.t)):
        nsp = plasmaSys.nsp
        # set the state
        plasmaSys.Ysp = solY.y[0:nsp,i]
        plasmaSys.Temp[1] = solY.y[nsp+1,i]
        plasmaSys.Temp[2] = solY.y[nsp+2,i]
        # plasmaSys.Hg = solY.y[-3,i]
        plasmaSys.Temp[0] = solY.y[nsp+0,i]

        # # find temperature and set it
        # X = plasmaSys.numbDensity2X(plasmaSys.Ysp) # mole fraction
        # # plasmaSys.gas.X = X
        # X[eId] = 0.0
        # p = plasmaSys.pressure()
        # ## If no energy exchagne happens in the bulk gas then the 
        # # enthalpy of the gas remains unchanged and not the internal energy
        # plasmaSys.gas.HPX = plasmaSys.Hg, p, X
        # plasmaSys.Temp[0] = plasmaSys.gas.T

        # update the system
        plasmaSys.update()

        # push the solution
        soln.solnPush(solY.t[i], plasmaSys)
        # print the time and temperature
        print(solY.t[i], plasmaSys.Temp[0], plasmaSys.Temp[1], plasmaSys.Temp[2])

    # save the solution
    soln.solnSave(fname+'.npz')


    # save # time,N,N2,N2_A,N2_B,N2_C,Np,N2p,N3p,N4p,ele in .txt file
    # np.savetxt(fname+'.txt', np.transpose(np.concatenate((solY.t.reshape(1,-1), solY.y[0:-3,:]))), delimiter='\t')
    ## put the header names in the file as well : fname+'.txt'
    header_all = "time\t"
    for i in range(nsp):
        header_all += plasmaSys.gas.species_name(i) + "\t"
    # last headers are the temperatures
    header_all += "Tg\tTv\tTe"
    np.savetxt(fname+'.txt', np.transpose(np.concatenate((solY.t.reshape(1,-1), solY.y[:,:]))), delimiter='\t', header=header_all)
    
    # get electron density
    Nes = solY.y[-4,:]
    # save time and electron density
    header_Ne = "time\tNe"
    np.savetxt(fname+'_Ne.txt', np.transpose(np.concatenate((solY.t.reshape(1,-1), Nes.reshape(1,-1)))), delimiter='\t', header=header_Ne)

    # make a figure to plot during the integration
    fig, ax = plt.subplots(2,1, sharex=True)
    # size
    fig.set_size_inches(10, 8)


    ax[0].set_ylabel('N (m-3)')
    ax[1].set_ylabel('T (K)')
    ax[0].set_xlabel('Time (s)')
    ax[1].set_xlabel('Time (s)')
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')


    nspecies = plasmaSys.nsp
    res_ysps = solY.y[0:nspecies]
    res_Ts = solY.y[nspecies:nspecies+3]

    # get results for electron density only
    resNe = res_ysps[:,-1]

    # name of species
    spNames = plasmaSys.gas.species_names

    # Get the absolute path for ../data
    dir_plpak_data = os.path.abspath("../data")

    # Set dir_plpak to the path ../data/dataExp
    dir_plpak = os.path.join(dir_plpak_data, "dataExp")

    print("directory for data: ", dir_plpak_data)
    print("directory for Expdata: ", dir_plpak)
        
    # Species to plot
    # plt_sps = ['N2','N','ele','N2p','Np']
    plt_sps = spNames
    # plot the results
    for i in range(len(spNames)):
        if spNames[i] in plt_sps:
            if spNames[i] == 'ele':
                ax[0].plot(solY.t, res_ysps[i,:], label=spNames[i], color='k',lw=2,
                    alpha=0.8,marker='o',markersize=4,markerfacecolor="None",markevery=0.1)
            else:
                ax[0].plot(solY.t, res_ysps[i,:], label=spNames[i])


    if sp == 'N2Aleks' or sp == "N2combined" or sp == "N2":
        # plot reference solution from Aleksandrov
        print("Plotting reference solution from Aleksandrov")
        # refdata = np.loadtxt("Aleksandrov_1atm_N2femtosecondDecay.dat", skiprows=1)
        fload = os.path.join(dir_plpak,"Aleksandrov_1atm_N2femtosecondDecay.dat")
        refdata = np.loadtxt(fload, skiprows=1)
        reft = refdata[:,0]*1.0e-9
        refNe = refdata[:,1]
        ax[0].plot(reft, refNe, 'o', label='Aleksandrov', markersize=7.5,alpha=0.7,color='b',markerfacecolor="None")

        # also plot papeer N2
        # refdata = np.loadtxt("airPlasma/papeerN214e21normalized.csv")
        fload = os.path.join(dir_plpak,"papeerN214e21normalized.csv")
        refdata = np.loadtxt(fload)
        Npeak = 1.0e21 # reference values are normalized with this value
        reft = refdata[:,0]
        refNe = refdata[:,1]*Npeak
        ax[0].plot(reft, refNe, 's', label='Papeer N2', markersize=7.5,alpha=0.7,color='k',markerfacecolor="None")

    # if solving for air plot ref
    if sp == 'Air' or sp == 'AirAleks':
        print("Plotting reference")
        # refdata = np.loadtxt("airData.csv")
        fload = os.path.join(dir_plpak,"airData.csv")
        refdata = np.loadtxt(fload)
        reft = refdata[:,0]*1.0e-9
        refNe = refdata[:,1]
        ax[0].plot(reft, refNe, 'o', label='Chizhov', markersize=7.5,alpha=0.7,color='b',markerfacecolor="None")

        # another reference : Papeer
        # refdata = np.loadtxt("airPlasma/papeerAir14e21normalized.csv")
        fload = os.path.join(dir_plpak,"papeerAir14e21normalized.csv")
        refdata = np.loadtxt(fload)
        Npeak = 1.4e22 # reference values are normalized with this value
        reft = refdata[:,0]
        refNe = refdata[:,1]*Npeak
        ax[0].plot(reft, refNe, 'o', label='Papeer', markersize=7.5,alpha=0.7,color='r',markerfacecolor="None")

        # # also plot papeer N2
        # refdata = np.loadtxt("airPlasma/papeerN214e21normalized.csv")
        # Npeak = 1.4e22 # reference values are normalized with this value
        # reft = refdata[:,0]
        # refNe = refdata[:,1]*Npeak
        # ax[0].plot(reft, refNe, 'o', label='Papeer N2', markersize=7.5,alpha=0.7,color='y',markerfacecolor="None")

        # # Reference laux : nanosecond pulsed discharges in air
        # refdata = np.loadtxt("airPlasma/AirLaux.csv")
        # reft = refdata[:,0]
        # refNe = refdata[:,1]
        # ax[0].plot(reft, refNe, 'o', label='Laux', markersize=7.5,alpha=0.7,color='g',markerfacecolor="None")

        # plot aleksandrovs N2 with 2.75 percent O2
        print("Plotting reference solution from Aleksandrov")
        # refdata = np.loadtxt("Aleksandrov_1atm_N2femtosecondDecay.dat", skiprows=0)
        fload = os.path.join(dir_plpak,"Aleksandrov_1atm_N2femtosecondDecay.dat")
        refdata = np.loadtxt(fload, skiprows=0)
        reft = refdata[:,0]*1.0e-9
        refNe = refdata[:,1]
        ax[0].plot(reft, refNe, 's', label=r'Aleksandrov: N2 + 2.75 % O2', markersize=7.5,alpha=0.7,color='c',markerfacecolor="None")


    # # for N2 do similar
    # if sp == "N2" or sp == "N2Aleks":
    #     print("Plotting reference")
    #     refdata = np.loadtxt("dataExp/N2temporalDecay.csv")
    #     reft = refdata[:,0]*1.0e-9
    #     refNe = refdata[:,1]
    #     ax[0].plot(reft, refNe, 'o', label='Chizhov', markersize=7.5,alpha=0.7,color='b',markerfacecolor="None")

    #     # another reference : Papeer
    #     refdata = np.loadtxt("airPlasma/papeerN214e21normalized.csv")
    #     Npeak = 1.4e22 # reference values are normalized with this value
    #     reft = refdata[:,0]
    #     refNe = refdata[:,1]*Npeak
    #     ax[0].plot(reft, refNe, 'o', label='Papeer', markersize=7.5,alpha=0.7,color='r',markerfacecolor="None")


    # temperature
    ax[1].plot(solY.t, res_Ts[0,:], label='Tg')
    ax[1].plot(solY.t, res_Ts[1,:], label='Tv')
    ax[1].plot(solY.t, res_Ts[2,:], label='Te')

    ## Tv if present
    if TvibO2 is not None:
        ax[1].plot(solY.t, solY.y[-1,:], label='TvibO2')


    # ax[0].legend()
    # legend placement horizontal wit ncols = nspecies / 3
    ax[0].legend(ncol=int(len(plt_sps)/3),loc='upper center', bbox_to_anchor=(0.5, 1.29), fancybox=False, shadow=False)
    ax[1].legend()


    # label ticks in ax[0] at top
    ax[0].tick_params(axis='x', which='both', top=True)


    # grid
    ax[0].grid()
    ax[1].grid()

    # # limit x = 1.0e-14
    # ax[0].set_xlim(1.0e-14, 2*tf)
    ax[0].set_xlim(1.0e-20, 1.1*tf)

    # limit y
    # ax[0].set_ylim(1.0e17, 5.0e25)
    # ax[0].set_ylim(1.0e17, 5.0e23)
    ax[0].set_ylim(1.0e12, 5.0e25)


    plt.tight_layout()
    # show the plot
    plt.savefig('plasmaSys_debug.png')
    plt.show()
                
# ## plotting Routine
# def plotSoln(fname='solnPlasma',pvars):
#     pl.Solution.plotSoln(fname,pvars)


## test building ode from mech
def testBuildODE(mechFile = "./airPlasma/N2PlasmaMech.yaml",lang='CXX'):
    # create a plasma fluid object
    pf = pl.PlasmaMechanism(mechFile)

    # change the language to CXX


    
    # now build the ODEs
    pf.buildODE_fromMech(lang=lang)




if __name__ == "__main__":

    #### Example To Build ODEs  ####
    # mechFile = "./airPlasma/combinedAirMechEne.yaml"
    # testBuildODE(mechFile,lang='python')


    # Run the test
    # fname = "fracO2Change22p0IN_N2_test"
    # fname = "fracO2Change0p1IN_N2_test"
    # fname = "fracO2Change2p0IN_N2_test"
    # fname = "fracO2Change0p0IN_N2_test"
    # fname = "airPlasmaSolnTemporal"
    # fname = "N2SolnsCheckN"
    # fname = "TestlowP"
    fname="testRun"

    # mechFile="mecFiles/AirPlasmaMech.yaml"
    # mechFile="mecFiles/AirPlasmaMech_HighT.yaml"
    # mechFile="mecFiles/AirPlasmaMech_HighT_Associon.yaml"
    mechFile="../data/AirPlasmaMech_v2.yaml"
    sp='Air'
    # sp='N2combined'

    # mechFile="mecFiles/N2PlasmaMech.yaml"
    # sp='N2'

    # # For Aleksandrovs
    # fname = "TestplasmaSolnAleks"
    # mechFile="airPlasma/N2PlasmaMechAleks.yaml"
    # sp='N2Aleks'

    directSolve(fname=fname,Te0=3.0*11600.0,dt=1.0e-6,mechFile=mechFile,sp=sp,ne=1.0e23)



    #### Post Processing ####
    #  # plot the solution object
    pvars = ['Ysp','Temp','Wrxn','Qmodes','dhrxn','Qdot']
    # pl.Solution.plotSoln(fname,pvars=pvars,prx=14,fmax=True,grp=11) # prx = # of max reactions to check, for fmax = True, if false provide array of reactions to check in prx
    
    prx = [222,107,194,6,7,168,165,224,98,215] # keep , 168,222,107,21,98
    prx = [171,170,52,35,108,173,167,187,184,52] # keep: 167, 52
    prx = [183,172,102,94,56,60,150,182,36,149] # keep: 172, 94 , 182
    prx = [103,101,74,80,99,71,26,75,17,114] # keep, 103, 80 , 99
    prx = [26,75,17,114] # keep: 75
    # include only keep now
    prx = [168,222,107,21,98,167,52,172,94,182,103,80,99,75] # remove 75,21,99,107
    prx = [168,222,98,167,52,172,94,182,103,80] # remove : 222,98,52,94
    prx = [127,168,167,172,182,103,80]
    # pl.Solution.plotSoln(fname,pvars=pvars,prx=prx,fmax=False,grp=11) # prx = # of max reactions to check, for fmax = True, if false provide array of reactions to check in prx
    # grp = 0 shows first set of max prx reactions if fmax = True
    # grp = 1 shows second set of max prx reactions if fmax = True which comes after grp = 0

    # saveTo = 'airImpRxnsAll_'
    # plotTemporalScale(fname,pids=pids,saveTo=saveTo)
    # # show which reactions are these
    # pl.Solution.printReactions(mechFile=mechFile,pids=pids)   # pids='All' or provide array of pids to check

    # # # # # # plot plotProduction rate of electrons, use data at atTime to find max or mins 
    # pl.Solution.plotProduction(fname,mechFile=mechFile,prx=14,plotBoth=True,atTime=1.0e-9)
    pl.Solution.plotProduction(fname,mechFile=mechFile,prx=14,plotBoth=True,atTime=None)

    ### plot production rates of Te
    # pl.Solution.plotProduction_Qrxn_I(fname,mechFile=mechFile,prx=6,plotBoth=True,id_col=2)
    
    # # # # # plot the solution object
    # pvars = ['Ysp','Temp','Wrxn','Qmodes','dhrxn','Qdot']
    # pl.Solution.plotSoln(fname,pvars=pvars,prx=12,fmax=True) # prx = # of max reactions to check, for fmax = True, if false provide array of reactions to check in prx
    # # # # # pl.Solution.plotSoln(fname,pvars=pvars,prx=[],fmax=False) # prx = # of max reactions to check, for fmax = True, if false provide array of reactions to check in prx
    
    # # plot number densities of species:
    # pvars = ['N2','O2','N',"N_D",'N_P','O',"NO","NOp","O2p","N2p",'O2m','N4p','ele']
    # plabels = [r'$N_2$',r'$O_2$',r'$N$',r'$N(D)$',r'$N(P)$',r'$O$',r'$NO$',r'$NO^+$',r'$O_2^+$',r'$N_2^+$',r'$O_2^-$',r'$N_4^+$',r'$e^-$']
    # pl.Solution.plotNumberDensities(fname,mechFile=mechFile,
    #                                 plotVars=pvars,plotLabels=plabels)
    