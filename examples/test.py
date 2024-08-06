import sys
import os

dir_plpak = os.path.dirname("../src/plpak")
sys.path.append(dir_plpak)
import plpak as pl

import cantera as ct
import numpy as np
import scipy as sc

import matplotlib.pyplot as plt

# import a .py file from the same directory
import funcRxns as fns


# erro settings for numpy
# np.seterr(divide='raise', invalid='raise')

# dYsp_dt for ODE solve
def dYsp_dt(t, Ysp, n2Plasma):
    # update the system based on the state
    n2Plasma.Ysp = Ysp
    # now update the system
    pl.updateFromYsp(n2Plasma)
    # return the dYsp_dt
    dydt = n2Plasma.dYdt


    if n2Plasma.verbose:
        # show the molefraction here
        print('X = ', n2Plasma.X)
        print('p = ', n2Plasma.p, 'rho = ', n2Plasma.rho)

    return dydt

## Solve the species and temperatures at the same time in a single ODE system
def dYdt_all(t, Y, n2Plasma):
    # update the system based on the state
    # last three are bulk internal energy, vibrational temperature, and electronic temperature
    n2Plasma.Ysp = Y[0:-3]
    # n2Plasma.Hg = Y[-3]  # try with a constant bulk temparature first - so dTdt = 0 for Tg
    n2Plasma.Temp[1] = Y[-2]
    n2Plasma.Temp[2] = Y[-1]
    n2Plasma.Temp[0] = Y[-3]

    # # find temperature and set it
    # X = n2Plasma.numbDensity2X(n2Plasma.Ysp) # mole fraction
    # # set electrons to zero
    # X[-1] = 0.0
    # # n2Plasma.gas.X = X
    # # print the gas object
    # print("Got this",n2Plasma.gas())
    # print("Set U = ",n2Plasma.Ug)
    # print("H = ",n2Plasma.Hg)
    # p = n2Plasma.pressure()
    # ## If no energy exchagne happens in the bulk gas then the 
    # # enthalpy of the gas remains unchanged and not the internal energy
    # n2Plasma.gas.HPX = n2Plasma.Hg, p, X
    # n2Plasma.Temp[0] = n2Plasma.gas.T
    # # # self.p = self.pressure()        # pressure in Pa
    # # # n2Plasma.rho = n2Plasma.density()       # density in kg/m3
    # # n2Plasma.gas.UV = n2Plasma.Ug , 1.0/n2Plasma.rho

    # update the system
    n2Plasma.update()

    

    # print("Set UV",n2Plasma.gas())

    # n2Plasma.Temp[0] = n2Plasma.gas.T

    # # now update the system from plasmaSystems
    # # systems updates with temperature so needed to update Temp[0]
    # n2Plasma.update()

    print("System Update",n2Plasma.gas())
        

    # zeros dydt
    dydt = np.zeros(len(Y))

    # update the species
    dydt[0:-3] = n2Plasma.dYdt
    dydt[-3:] = n2Plasma.dTdt

    # print 
    print("Time = ", t, "Y = ", Y, "dydt = ", dydt)

    return dydt
    



def dT_dt(t, T, n2Plasma):

     # return the dT_dt
    dTdt = np.zeros(3)

    # # if negative temperature is found, return dT_dt without updating the system
    # if T[1] < 0.0 or T[2] < 0.0:
    #     dTdt = n2Plasma.dTdt
    #     return dTdt

        




    # show T got here
    print('T got in ddT = ', T, 'time = ', t)

    # update the system based on the state
    Ug = T[0]; Tv = T[1]; Te = T[2]
    
    # rho = n2Plasma.rho
    # # print rho
    # print('Ug = ', Ug,'p = ', n2Plasma.p, 'rho = ', n2Plasma.rho)
    # n2Plasma.gas.UV = Ug, (1.0/n2Plasma.rho) # set the internal energy - J/kg and density - kg/m3

    n2Plasma.X = n2Plasma.numbDensity2X(n2Plasma.Ysp) # mole fraction
    # # put electrons zero
    # n2Plasma.X[-1] = 0.0
    # n2Plasma.gas.TPX = n2Plasma.Temp[0], n2Plasma.p, n2Plasma.X
    n2Plasma.gas.UV = Ug, (1.0/n2Plasma.rho)
    Tg = n2Plasma.gas.T

    # T after update from cantera
    print('T after update from cantera = ', Tg)
    # Tg = n2Plasma.Temp[0]

    # n2Plasma.Temp = np.array([Tg, Tv, Te])
    n2Plasma.Temp[0] = Tg
    n2Plasma.Temp[1] = Tv
    n2Plasma.Temp[2] = Te
    # # now update the system
    updateFromTemp(n2Plasma)

    # T after update from cantera
    print('T after update to system = ', n2Plasma.Temp[0])


   
    dTdt[0] = n2Plasma.dTdt[0]
    dTdt[1] = n2Plasma.dTdt[1]
    dTdt[2] = n2Plasma.dTdt[2]


    # # show Qmodes
    # print('Qmodes = ', n2Plasma.Qmodes)
    # #Qdot
    # print('Qdot = ', n2Plasma.Qdot)

    # show dTdt
    # print('dTdt = ', dTdt)

    # # Warn if ddt[0] is more than 0.1
    # if dTdt[0] > 0.1:
    #     print('Warning: dTdt[0] is more than 0.1')
    #     # terminate
    #     sys.exit()


    

    if n2Plasma.verbose:
        # show all the variables here
        print("TIME = ", t, "s")
        print('Tg = ', Tg, 'Tv = ', Tv, 'Te = ', Te,'rho = ', rho, 'Ug = ', Ug, 'dTdt = ', dTdt,'p = ', n2Plasma.p)

    return dTdt

def updateFromTemp(n2Plasma):
    '''
    Update the system object to new state based on temperature change only.
    This should be used to update the sate while integrating the temperature equation as species are not changing.
    '''
    # update from base class
    n2Plasma.p = n2Plasma.pressure()        # pressure in Pa
    # n2Plasma.rho = n2Plasma.density()       # density in kg/m3 doesnt change then constant volume
    
    n2Plasma.X = n2Plasma.numbDensity2X(n2Plasma.Ysp) # mole fraction also doenst change in temperature
    # # put electrons zero
    # n2Plasma.X[-1] = 0.0
    # n2Plasma.gas.TPX = n2Plasma.Temp[0], n2Plasma.p, n2Plasma.X
    # n2Plasma.cp_mix = n2Plasma.cp_mix_()     # specific heat of the mixture in J/kmol/K
    
    n2Plasma.hsp = n2Plasma.hsp_()           # enthalpy of formation of the species in J/kmol
    n2Plasma.Krxn = n2Plasma.Krxn_()
    n2Plasma.Wrxn = n2Plasma.Wrxn_()
    n2Plasma.dhrxn = n2Plasma.dhrxn_()       # enthalpy of reaction in J/kmol
    n2Plasma.Qrxn, n2Plasma.Qdot  = n2Plasma.Qrxn_()         # reating rate in J/m3-s
    
    n2Plasma.Qmodes = n2Plasma.Qmodes_()     # heating rate in J/m3-s for each mode
    
    # n2Plasma.dYdt = n2Plasma.dYdt_()         # concentration units / s
    n2Plasma.dTdt = n2Plasma.dTdt_()         # temperature units / s

    # print values after update
    print("After update")
    print('p = ', n2Plasma.p, 'rho = ', n2Plasma.rho,'dTdt = ', n2Plasma.dTdt)

# test onlyKinetics
def test_onlyKinetics():

    # Which plasmaSystem to solve
    plasmaSys = pl.N2Plasma_54Rxn

    # Initial conditions
    n0 = 2.5e25
    nI = 1.0e23
    ne = 1.0e23
    n0 = n0 - 2*nI
    Ysp = np.array([0,n0,0,0,0,0,nI,0,0,ne])
    T = np.array([300,300,11600*0.6])


    # make an object of the plasma kinetics - The system to solve
    n2Plasma = plasmaSys(Ysp, T)


    # Once update the system based on the initial conditions
    n2Plasma.update()
    eId = n2Plasma.gas.species_index('ele')

    # time details
    t0 = 0
    tf = 1e-5
    dt = 1e-16
    max_dt = 1e-10
    relaxdt = 1.0e-5
    t_array = np.arange(t0, tf, dt)
    t = t0




    # All Solution
    # Make an object to hold current solution
    soln = pl.Solution(t0, n2Plasma)
    # # append initial state : the initial state is already appended during instantiation
    # soln.solnPush(t_array[0], n2Plasma)


    solMethods=["RK45","RK23","DOP853","Radau","BDF","LSODA"]
    # Now integrate the system
    # for t in t_array[1:]:
    while t < tf:
        # update the time domain
        t = t + dt
        ct0 = t-dt
        ctf = t
        int_t = (ct0, ctf)


        # integrate the system
        # previous solution is used as initial condition
        n2Plasma.Ysp0 = n2Plasma.Ysp
        solTemp = sc.integrate.solve_ivp(dYsp_dt, int_t, n2Plasma.Ysp0, method=solMethods[3], args=(n2Plasma,))

        
        # update Ysp
        n2Plasma.Ysp = solTemp.y[:,-1]
        # push the solution
        soln.solnPush(t, n2Plasma)

        # control the time step
        tau_chem = 0.5*(n0+ne) / np.max(np.abs(n2Plasma.Wrxn))
        print("Time = ", t, "tau_chem = ", tau_chem, "dt = ", dt)
        dt = dt + relaxdt*(tau_chem - dt)
        dt = min(dt, max_dt)


    # convert soln to numpy array
    soln.t = np.array(soln.t)
    soln.Ysp = np.array(soln.Ysp)
    soln.Temp = np.array(soln.Temp)
    soln.Wrxn = np.array(soln.Wrxn)


    # Plot the results
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    name_sp = n2Plasma.gas.species_names
    N2ID = n2Plasma.gas.species_index('N2')
    for i in range(len(name_sp)):
        # dont plot N2
        if i != N2ID:
            ax.plot(soln.t, soln.Ysp[:,i], label=name_sp[i])

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Concentration (m-3)')

    # log scale in y and x
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.show()


## Test with energy equation as well
def test_withEnergy(fname='solnPlasma'):

    # Plot the results
    import matplotlib.pyplot as plt
    
    # Which plasmaSystem to solve
    plasmaSys = pl.N2Plasma_54Rxn # recomb heat goes to electron
    # plasmaSys = pl.N2Plasma_54Rxn_2 # recomb heat goes to neutrals

    # if a heating laser is used
    laser = pl.LaserModel(In=2.0e16,switch=0) # switch off the laser# Default is 1064 nm ns-laser



    # Initial conditions
    n0 = 2.5e25
    nI = 1.00e23
    ne = nI
    n0 = n0 - 2*nI
    Ysp = np.array([0.0,n0,0,0,0,0,nI,0,0,ne])

    Te0 = 1.0*11600.0
    T = np.array([300.0,300.0,Te0])

    # Ug = T[0]; Tv = T[1]; Te = T[2]
    # rho = n2Plasma.rho
    # n2Plasma.gas.UV = Ug, 1.0/rho # set the internal energy - J/kg and density - kg/m3
    # Tg = n2Plasma.gas.T

##------------------------------------------------------
    # #1  make an object of the plasma kinetics - The system to solve
    # n2Plasma = plasmaSys(Ysp, T,verbose=False)  # just plasma 
    n2Plasma = plasmaSys(Ysp, T,laser,verbose=False)  # just plasma 

    # #2 if using combined model--------------------------------------
    # # if using laser modify the plasma system to include these new effects
    # # Done my instantiating CombinedModels
    # n2Plasma0 = plasmaSys(Ysp, T,verbose=False)
    # n2Plasma = pl.CombinedModels(n2Plasma0, laser)
##------------------------------------------------------

    # Once update the system based on the initial conditions
    n2Plasma.update()

    ## print the temperatue of the system
    print("Tg = ", n2Plasma.Temp[0], "Tv = ", n2Plasma.Temp[1], "Te = ", n2Plasma.Temp[2])

    eId = n2Plasma.gas.species_index('ele')
    Ug0 = n2Plasma.gas.int_energy_mass # initial internal energy - J/kg
    Ug = Ug0

    # time details
    t0 = 0
    tf = 100.0e-9
    tf = 10.0e-6
    dt = 1e-11
    max_dt = 0.5e-9
    relaxdt = 1.0e-5
    t_array = np.arange(t0, tf, dt)
    t = t0

    # All Solution
    # Make an object to hold current solution
    soln = pl.Solution(t0, n2Plasma)
    # # append initial state : the initial state is already appended during instantiation
    # soln.solnPush(t_array[0], n2Plasma)

    solMethods=["RK45","RK23","DOP853","Radau","BDF","LSODA"]
    # Now integrate the system
    # for t in t_array[1:]:
    # make a figure to plot during the integration
    fig, ax = plt.subplots()

    while t < tf:
        # update the time domain
        t = t + dt
        ct0 = t-dt
        ctf = t
        int_t = (ct0, ctf)

        # integrate the system----------------
        # previous solution is used as initial condition
        n2Plasma.Ysp0 = n2Plasma.Ysp
        solY = sc.integrate.solve_ivp(dYsp_dt, int_t, n2Plasma.Ysp0, method=solMethods[3], args=(n2Plasma,))
        # update Ysp
        n2Plasma.Ysp = solY.y[:,-1]
        # now update the state of gas
        Ug = n2Plasma.gas.int_energy_mass
        # update temperature
        # n2Plasma.Temp0 = n2Plasma.gas
        pl.updateFromYsp(n2Plasma)
        #internal energy has changed from chemical reactions
        
        #------------------------------------

        # # integrate energy equation --------
        # initial condition for the system
        n2Plasma.Temp0 = n2Plasma.Temp
        # show the tempereature
        print("Tg = ", n2Plasma.Temp0[0], "Tv = ", n2Plasma.Temp0[1], "Te = ", n2Plasma.Temp0[2])
        yt0 = np.array([Ug, n2Plasma.Temp0[1], n2Plasma.Temp0[2]])
        # print supplied initial condition
        print("Ug0 = ", yt0[0], "Tv0 = ", yt0[1], "Te0 = ", yt0[2])
        solT = sc.integrate.solve_ivp(dT_dt, int_t, yt0, method=solMethods[5], args=(n2Plasma,))
        Ug = solT.y[0,-1]

        # # plot the solution for temperature
        # ax.clear()
        # ax.plot(solT.t, solT.y[0,:], label='Ug')
        # ax.plot(solT.t, solT.y[1,:], label='Tv')
        # ax.plot(solT.t, solT.y[2,:], label='Te')
        # ax.set_xlabel('Time (s)')
        # # legend
        # ax.legend()
        # # show the plot
        # plt.pause(0.01)
        # plt.show(block=False)

        # verify the temperature are positive
        if np.any(solT.y[1:2,:] < 0):
            print("Negative temperature")
            break
        # update temperature
        rho = n2Plasma.rho
        n2Plasma.gas.UV = Ug, 1.0/rho
        Tg = n2Plasma.gas.T
        n2Plasma.Temp = np.array([Tg, solT.y[1,-1], solT.y[2,-1]])

        


        # now update the state of gas
        updateFromTemp(n2Plasma)


        
        # print the solution: Temp
        print("Time = ", t, 'dt = ', dt, 'Temp = ', n2Plasma.Temp)
        

        # push the solution
        soln.solnPush(t, n2Plasma)

        # control the time step
        tau_chem = 0.5*(n0+ne) / np.max(np.abs(n2Plasma.Wrxn))
        # print("Time = ", t, "tau_chem = ", tau_chem, "dt = ", dt)
        dt = dt + relaxdt*(tau_chem - dt)
        dt = min(dt, max_dt)

    # convert soln to numpy array
    # soln.t = np.array(soln.t)
    # soln.Ysp = np.array(soln.Ysp)
    # soln.Temp = np.array(soln.Temp)
    # soln.Wrxn = np.array(soln.Wrxn)
    # soln.Qrxn = np.array(soln.Qrxn)
    soln.soln2np()

    # save the electron density
    np.savetxt('N2Plasma_Density.txt', np.c_[soln.t, soln.Ysp[:,eId]], fmt='%1.5e', delimiter='\t', header='Time (s)\tElectron Density (m-3)')


    # Save the data to a file for later use
    # format: time , N, N2, N2_A, N2_B, N2_C, Np, N2p, N3p, N4p, ele
    # Put header on the data
    header = "time,N,N2,N2_A,N2_B,N2_C,Np,N2p,N3p,N4p,ele"
    np.savetxt('N2PlasmaResult_Ysp.txt', np.c_[soln.t, soln.Ysp], fmt='%1.5e', delimiter='\t', header=header)


    # save the entire solution to numpy file
    # write the data to a numpy format
    saveTo = fname + '.npz'
    np.savez(saveTo, t=soln.t, Ysp=soln.Ysp, Temp=soln.Temp, Wrxn=soln.Wrxn, Qrxn=soln.Qrxn, dhrxn=soln.dhrxn,Qdot=soln.Qdot,Qmodes=soln.Qmodes)

    

    fig, ax = plt.subplots()

    name_sp = n2Plasma.gas.species_names
    N2ID = n2Plasma.gas.species_index('N2')
    for i in range(len(name_sp)):
        # dont plot N2
        if i != N2ID:
            ax.plot(soln.t, soln.Ysp[:,i], label=name_sp[i])

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Concentration (m-3)')

    # log scale in y and x
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.savefig('N2Plasma_Density.png')
    plt.show()
    plt.close()
    

    # plot temperature
    fig, ax = plt.subplots()
    ax.plot(soln.t, soln.Temp[:,0], label='Tg')
    ax.plot(soln.t, soln.Temp[:,1], label='Tv')
    ax.plot(soln.t, soln.Temp[:,2], label='Te')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (K)')
    # log in x
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig('N2Plasma_Temp.png')
    plt.show()
    plt.close()
    

    # make another plot for Qrxn
    fig, ax = plt.subplots()

    maxQ = np.max(np.abs(soln.Qrxn))
    # plot the heating rate for each reaction
    for i in range(n2Plasma.nrxn):
        toplt = np.abs(soln.Qrxn[:,0,i])/maxQ
        # plot only if > 0.1
        if np.max(toplt) > 1.0e-6:
            ax.plot(soln.t, soln.Qrxn[:,0,i], label=str(i+1))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Qrxn (J/m3-s)')
    # log in x
    ax.set_xscale('log')
    ax.legend()
    plt.savefig('N2Plasma_Qrxn.png')
    plt.show()
    plt.close()
    


    # plot Wrxn
    fig, ax = plt.subplots()
    maxW = np.max(np.abs(soln.Wrxn))
    # plot the reaction rate for each reaction
    for i in range(n2Plasma.nrxn):
        toplt = np.abs(soln.Wrxn[:,i])/maxW
        # plot only if > 0.1
        if np.max(toplt) > 1.0e-6:
            ax.plot(soln.t, soln.Wrxn[:,i], label=str(i+1))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Wrxn (m3/s)')
    # log in x
    ax.set_xscale('log')
    ax.legend()
    plt.savefig('N2Plasma_Wrxn.png')
    plt.show()
    plt.close()
    

    # plot dhrxn with time
    fig, ax = plt.subplots()
    maxdhrxn = np.max(np.abs(soln.dhrxn))
    for i in range(n2Plasma.nrxn):
        res = soln.dhrxn[:,0,i] # J/kmol
        if np.max(np.abs(res)/maxdhrxn) > 5.0e-1:
            # change to eV/particle
            res = pl.ODEBuilder.J_kmol2eV(res)
            # scatter plot
            ax.scatter(soln.t, res, label=str(i+1))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('dhrxn (eV)')
    # log in x
    ax.set_xscale('log')
    ax.legend()
    plt.savefig('N2Plasma_dhrxn.png')
    plt.show()
    plt.close()



## Multi-tempeatrue model test and debug
# ## Test with energy equation as well
def all_solve(fname='solnPlasma',Te0=11600.0,dt=1.0e-6,laser=None,ne=1.0e23,plasmaChem=pl.N2Plasma_54Rxn,sp='N2',n0=2.5e25):

    '''
    Default plasma kinetics used is : pl.N2Plasma_54Rxn
    '''
    # Plot the results
    import matplotlib.pyplot as plt
    
    # # Which plasmaSystem to solve
    # plasmaSys = pl.N2Plasma_54Rxn # recomb heat goes to electron
    # # plasmaSys = pl.N2Plasma_54Rxn_2 # recomb heat goes to neutrals

    plasmaSys = plasmaChem
    


    # # Initial conditions
    # n0 = 2.5e25
    # # nI = 6.50e22
    # # ne = nI
    # nI = ne
    # n0 = n0 - 2*nI
    # Ysp = np.array([0.0,n0,0,0,0,0,nI,0,0,ne])

    if sp == 'N2':
    # # Initial conditions For N2 mechanism  -----------------------------------------------------------------------
        # Species: [N, N2, N2_A, N2_B, N2_C, Np, N2p, N3p, N4p, ele]
        # n0 = 2.5e25
        # nI = 1.00e23
        # nI = 5.00e22
        # ne = nI
        nI = ne
        n0 = n0 - 2*nI
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
        nNeutral = 2.4475e25
        perO2 = 0.22

        # #    ## For 1.0e23; Aleksandrov
        #     nN2p = 1.5967858672000928e+22
        #     nO2p = 1.2472092965932226e+23

        # # for 4.0e23 ; Chizhov
        ne = 4.0e23
        nO2p = 3.2795623949898156e+23
        nN2p = ne - nO2p
        

        # # For ne = 1.4e22 ; Papeer
        # ne = 1.4e22
        # nN2p = 1.565427079273731e+21
        # nO2p = ne - nN2p

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

        Ysp0 = np.array([0,NN2,0,0,0,0,0,NO2,0,0,0,0,nN2p,0,0,nO2p,0,0,0,0,0,ne])
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
        Ysp0 = np.array([0,nn2,0,0,0,0,no2,0,nn2p,0,0,no2p,0,0,ne])
    ##-----------------------------------------------------------------------------------------------------------------
    Ysp = Ysp0

    # Te0 = 5.0*11600.0
    T0s = np.array([300.0,400.0,Te0])


    # if a heating laser is used
    if laser is None:
        laser = pl.LaserModel(switch=0) # switch off the laser# Default is 1064 nm ns-laser



    # make an object of the plasma kinetics - The system to solve
    n2Plasma = plasmaSys(Ysp, T0s[:],verbose=False)
    ##------------------------------------------------------
    # #1  make an object of the plasma kinetics - The system to solve
    # n2Plasma = plasmaSys(Ysp, T,verbose=False)  # just plasma 
    n2Plasma = plasmaSys(Ysp, T0s[:],laser,verbose=False)  # just plasma 

    # #2 if using combined model--------------------------------------
    # # if using laser modify the plasma system to include these new effects
    # # Done my instantiating CombinedModels
    # n2Plasma0 = plasmaSys(Ysp, T0s[:],verbose=False)
    # n2Plasma = pl.CombinedModels(n2Plasma0, laser)
##------------------------------------------------------




    # Once update the system based on the initial conditions
    n2Plasma.update()
    eId = n2Plasma.gas.species_index('ele')

    # print the initial state of the system
    print("Initial state of the system")
    print("Tg = ", n2Plasma.Temp[0], "Tv = ", n2Plasma.Temp[1], "Te = ", n2Plasma.Temp[2])
    print("Ug = ", n2Plasma.Ug)
    print("Hg = ", n2Plasma.Hg)

    # setup the array for initial conditions all solved at once
    # need to combine Ysp and Temp
    yt0 = np.concatenate((n2Plasma.Ysp, n2Plasma.Temp))
    # change the Tg to Ug
    # yt0[-3] = n2Plasma.Ug
    # yt0[-3] = n2Plasma.Hg

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
    soln = pl.Solution(t0, n2Plasma)
    # # append initial state : the initial state is already appended during instantiation
    # soln.solnPush(t_array[0], n2Plasma)

    solMethods=["RK45","RK23","DOP853","Radau","BDF","LSODA"]


    # Now integrate the system
    solY = sc.integrate.solve_ivp(dYdt_all, int_t, yt0, method=solMethods[5], args=(n2Plasma,), rtol=1e-4, atol=1e-6)


    # print the message from the solver
    print(solY.message)


    # Now for post-processing, calculate all the properties based on the solved state
    for i in range(len(solY.t)):
        # set the state
        n2Plasma.Ysp = solY.y[0:-3,i]
        n2Plasma.Temp[1] = solY.y[-2,i]
        n2Plasma.Temp[2] = solY.y[-1,i]
        # n2Plasma.Hg = solY.y[-3,i]
        n2Plasma.Temp[0] = solY.y[-3,i]

        # # find temperature and set it
        # X = n2Plasma.numbDensity2X(n2Plasma.Ysp) # mole fraction
        # # n2Plasma.gas.X = X
        # X[eId] = 0.0
        # p = n2Plasma.pressure()
        # ## If no energy exchagne happens in the bulk gas then the 
        # # enthalpy of the gas remains unchanged and not the internal energy
        # n2Plasma.gas.HPX = n2Plasma.Hg, p, X
        # n2Plasma.Temp[0] = n2Plasma.gas.T

        # update the system
        n2Plasma.update()

        # push the solution
        soln.solnPush(solY.t[i], n2Plasma)
        # print the time and temperature
        print(solY.t[i], n2Plasma.Temp[0], n2Plasma.Temp[1], n2Plasma.Temp[2])

    # save the solution
    soln.solnSave(fname+'.npz')

    # save # time,N,N2,N2_A,N2_B,N2_C,Np,N2p,N3p,N4p,ele in .txt file
    # np.savetxt(fname+'.txt', np.transpose(np.concatenate((solY.t.reshape(1,-1), solY.y[0:-3,:]))), delimiter='\t')
    np.savetxt(fname+'.txt', np.transpose(np.concatenate((solY.t.reshape(1,-1), solY.y[:,:]))), delimiter='\t')
    
    # make a figure to plot during the integration
    fig, ax = plt.subplots(2,1, sharex=True)
    # size
    fig.set_size_inches(10, 8)


    ax[0].set_ylabel('N (m-3)')
    ax[1].set_ylabel('T (K)')
    ax[1].set_xlabel('Time (s)')
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    res_ysps = solY.y[0:-3]
    res_Ts = solY.y[-3:]

    # name of species
    spNames = n2Plasma.gas.species_names

    # Species to plot
    # plt_sps = ['N2','N','ele','N2p','Np']
    # if want to plot all
    plt_sps = spNames
    # plot the results
    for i in range(len(spNames)):
        if spNames[i] in plt_sps:
            if spNames[i] == 'ele':
                ax[0].plot(solY.t, res_ysps[i,:], label=spNames[i], color='k',lw=2,alpha=0.7)
            else:
                ax[0].plot(solY.t, res_ysps[i,:], label=spNames[i])

    # temperature
    ax[1].plot(solY.t, res_Ts[0,:], label='Tg')
    ax[1].plot(solY.t, res_Ts[1,:], label='Tv')
    ax[1].plot(solY.t, res_Ts[2,:], label='Te')


    ax[0].legend()
    ax[1].legend()

    # grid
    ax[0].grid()
    ax[1].grid()

    # limit x = 1.0e-14
    ax[0].set_xlim(1.0e-14, 2*tf)

    # limit y
    ax[0].set_ylim(1.0e14, 5.0e25)

    # show the plot
    plt.show()
    plt.savefig('N2Plasma_debug.png')


    
# # make plasmaKinetics for n2plasma_54 reactions
# def buildODE_fromMech(mechfile):
#     import sys
#     import datetime

#     # make a file name to log the output
#     now = datetime.datetime.now()
#     logName = 'outputLog_'+now.strftime("%Y-%m-%d_%H-%M")+'.txt'
#     sys.stdout = open(logName, 'wt')

#     # print the runtime details
#     print('Python version ' + sys.version)
#     print('Date and time: ' + now.strftime("%Y-%m-%d %H:%M"))
#     print('------------------------------------')
#     print('\n')

#     # make an object of the class with a mechanism file
#     system = pl.ODEBuilder('plasmaN2.yaml')
#     system.mode = 'number_density'
#     # system.mode = 'mass_fraction'
#     # system.language = 'python'
#     system.language = 'CXX'
#     system.subHname = 'N2A'

#     # # check the heat of reaction using system.updateHrxnExpressions(self,ODEBuilder.extraFuncHrxnN2_54)
#     # hRxn_Sym, h_sp, h_sp_call = system.updateHrxnExpressions(ODEBuilder.extraFuncHrxnN2_54)

#     # update ODE System -- Recombination heating goes to Te
#     # system.updateODESystem(reactionRateConstant=pl.ODEBuilder.getN2PlasmaKrxn,extraFuncHrxn=pl.ODEBuilder.extraFuncHrxnN2_54)
#     # Recombination heating goes to Tg
#     system.updateODESystem(reactionRateConstant=pl.ODEBuilder.getN2PlasmaKrxn,extraFuncHrxn=pl.ODEBuilder.extraFuncHrxnN2_54_2)

#     # get the expressions
#     system.getSystemExpression()


# make plasmaKinetics for n2plasma_54 reactions
def buildODE_n2plasma():
    import sys
    import datetime

    # make a file name to log the output
    now = datetime.datetime.now()
    logName = 'outputLog_'+now.strftime("%Y-%m-%d_%H-%M")+'.txt'
    sys.stdout = open(logName, 'wt')

    # print the runtime details
    print('Python version ' + sys.version)
    print('Date and time: ' + now.strftime("%Y-%m-%d %H:%M"))
    print('------------------------------------')
    print('\n')

    # make an object of the class with a mechanism file
    system = pl.ODEBuilder('plasmaN2.yaml')
    system.mode = 'number_density'
    # system.mode = 'mass_fraction'
    # system.language = 'python'
    system.language = 'CXX'
    system.subHname = 'N2A'

    # # check the heat of reaction using system.updateHrxnExpressions(self,ODEBuilder.extraFuncHrxnN2_54)
    # hRxn_Sym, h_sp, h_sp_call = system.updateHrxnExpressions(ODEBuilder.extraFuncHrxnN2_54)

    # update ODE System -- Recombination heating goes to Te
    # system.updateODESystem(reactionRateConstant=pl.ODEBuilder.getN2PlasmaKrxn,extraFuncHrxn=pl.ODEBuilder.extraFuncHrxnN2_54)
    # Recombination heating goes to Tg
    system.updateODESystem(reactionRateConstant=pl.ODEBuilder.getN2PlasmaKrxn,extraFuncHrxn=pl.ODEBuilder.extraFuncHrxnN2_54_2)

    # get the expressions
    system.getSystemExpression()

# make plasmaKinetics for n2plasma_54 reactions
def buildODE_n2plasmaAleks():
    import sys
    import datetime

    # make a file name to log the output
    now = datetime.datetime.now()
    logName = 'outputLog_'+now.strftime("%Y-%m-%d_%H-%M")+'.txt'
    sys.stdout = open(logName, 'wt')

    # print the runtime details
    print('Python version ' + sys.version)
    print('Date and time: ' + now.strftime("%Y-%m-%d %H:%M"))
    print('------------------------------------')
    print('\n')

    # make an object of the class with a mechanism file
    # system = pl.ODEBuilder('plasmaN2.yaml')
    # system = pl.ODEBuilder("airPlasma/N2PlasmaMechAleks.yaml")
    system = pl.ODEBuilder("N2PlasmaMechAleks_build.yaml")
    system.mode = 'number_density'
    # system.mode = 'mass_fraction'
    system.language = 'python'
    # system.language = 'CXX'
    system.subHname = 'N2A'

    # # check the heat of reaction using system.updateHrxnExpressions(self,ODEBuilder.extraFuncHrxnN2_54)
    # hRxn_Sym, h_sp, h_sp_call = system.updateHrxnExpressions(ODEBuilder.extraFuncHrxnN2_54)

    # update ODE System -- Recombination heating goes to Te
    system.updateODESystem(reactionRateConstant=fns.getN2AleksPlasmaKrxn,extraFuncHrxn=fns.extraFuncHrxnN2Aleks)
    # Recombination heating goes to Tg
    # system.updateODESystem(reactionRateConstant=pl.ODEBuilder.getN2PlasmaKrxn,extraFuncHrxn=pl.ODEBuilder.extraFuncHrxnN2_54_2)

    # get the expressions
    system.getSystemExpression()

# # define a function to plot from saved data
# def plotCompare(fname='solnPlasma'):

#     import os
#     import sys
#     # add this path "E:\POKHAREL_SAGAR\gits\pyblish\plots" to the system path explicitly
#     pathName = "E:\POKHAREL_SAGAR\gits\pyblish\plots"
#     sys.path.append(pathName)
#     import publish    



#     import matplotlib.pyplot as plt
#     # load the data
#     data = np.load(fname+'.npz',allow_pickle=True)
#     # np.savez(saveTo, t=soln.t, Ysp=soln.Ysp, Temp=soln.Temp, Wrxn=soln.Wrxn, Qrxn=soln.Qrxn, dhrxn=soln.dhrxn,Qdot=soln.Qdot,Qmodes=soln.Qmodes)
#     solt = data['t']
#     solnYsp = data['Ysp']
#     solnTemp = data['Temp']
#     solnWrxn = data['Wrxn']
#     solnQrxn = data['Qrxn']
#     solndhrxn = data['dhrxn']
#     solnQdot = data['Qdot']
#     solnQmodes = data['Qmodes']



# test multipl times setting UV in cantera
def test_UV_set(ntime=1000):
     # Plot the results
    import matplotlib.pyplot as plt
    
    # Which plasmaSystem to solve
    # plasmaSys = pl.N2Plasma_54Rxn # recomb heat goes to electron
    plasmaSys = pl.N2Plasma_54Rxn_2 # recomb heat goes to neutrals


    # Initial conditions
    n0 = 2.5e25
    nI = 7.50e20
    ne = nI
    n0 = n0 - 2*nI
    Ysp = np.array([0.0,n0,0,0,0,0,nI,0,0,ne])

    Te0 = 0.4*11600.0
    T = np.array([400.0,400.0,Te0])

    # Ug = T[0]; Tv = T[1]; Te = T[2]
    # rho = n2Plasma.rho
    # n2Plasma.gas.UV = Ug, 1.0/rho # set the internal energy - J/kg and density - kg/m3
    # Tg = n2Plasma.gas.T


    # make an object of the plasma kinetics - The system to solve
    n2Plasma = plasmaSys(Ysp, T,verbose=False)
    # Once update the system based on the initial conditions
    n2Plasma.update()

    ## print the temperatue of the system
    print("Tg = ", n2Plasma.Temp[0], "Tv = ", n2Plasma.Temp[1], "Te = ", n2Plasma.Temp[2])

    eId = n2Plasma.gas.species_index('ele')
    Ug0 = n2Plasma.gas.int_energy_mass # initial internal energy - J/kg
    Ug = Ug0

    # time details
    t0 = 0
    tf = 100.0e-9
    dt = 1e-13
    max_dt = 1e-11
    relaxdt = 1.0e-5
    t_array = np.arange(t0, tf, dt)
    t = t0

    # All Solution
    # Make an object to hold current solution
    soln = pl.Solution(t0, n2Plasma)
    # # append initial state : the initial state is already appended during instantiation
    # soln.solnPush(t_array[0], n2Plasma)

    solMethods=["RK45","RK23","DOP853","Radau","BDF","LSODA"]
    # Now integrate the system
    # for t in t_array[1:]:
    # make a figure to plot during the integration
    fig, ax = plt.subplots()

    # update the system based on the state
    Ug = -11046.78983358496
    rho =  1.1633371918604651
    t = 0.0
    dt = 1e-13
    for i in range(ntime):
        

        # print("Ug = ", Ug, "rho = ", rho)
        
        # # rho = n2Plasma.rho
        # # print rho
        # # print('Ug = ', Ug,'p = ', n2Plasma.p, 'rho = ', n2Plasma.rho)
        # n2Plasma.gas.UV = Ug, (1.0/n2Plasma.rho) # set the internal energy - J/kg and density - kg/m3
        # Tg = n2Plasma.gas.T
        # # Tg = n2Plasma.Temp[0]

        # n2Plasma.Temp = np.array([Tg, 300, 11600.0])
        # # now update the system
        # updateFromTemp(n2Plasma)


        # Ug = Ug - 1.0e-16*Ug
        # rho = rho - 1.0e-16*rho
        T0 = np.array([Ug,400.0,11600.0])

        # call dT_dt
        dT_dt(t, T0, n2Plasma)
        t = t + dt




# Test running a simulation directly from the read mechanism file - Just kinetics
## Multi-tempeatrue model test and debug
# ## Test with energy equation as well
def test_directSolve(fname='solnPlasma',Te0=11600.0,dt=1.0e-6,laser=None):

    # Plot the results
    import matplotlib.pyplot as plt
    
    # Which plasmaSystem to solve
    plasmaSys = pl.N2Plasma_54Rxn # recomb heat goes to electron
    # plasmaSys = pl.N2Plasma_54Rxn_2 # recomb heat goes to neutrals
    
    # Direct Solve with give n mechanism file
    plasmaSys = pl.PlasmaSolver

    # make the plasma system
    n2Plasma = plasmaSys('tutorial0.yaml',verbose=False) # initialize the plasma system through the mechanism file
    
    

    # # Initial conditions
    # n0 = 2.5e25
    # nI = 1.00e23
    # ne = nI
    # n0 = n0 - 2*nI
    # Ysp = np.array([0.0,n0,0,0,0,0,nI,0,0,ne])

    # Initial conditions
    n0 = 2.5e25
    nI = 1.00e23
    ne = nI
    n0 = n0 - 2*nI
    # [N, N2, N2p, ele]
    Ysp0 = np.array([0.0, n0, nI, ne])
    # T0s = np.array([300.0, 300.0, 11600.0])

    # Te0 = 5.0*11600.0
    T0s = np.array([300.0,300.0,Te0])


    # # if a heating laser is used
    # if laser is None:
    #     laser = pl.LaserModel(switch=0) # switch off the laser# Default is 1064 nm ns-laser


    # # make an object of the plasma kinetics - The system to solve
    # n2Plasma = plasmaSys(Ysp, T0s[:],verbose=False)
    # ##------------------------------------------------------
    # # #1  make an object of the plasma kinetics - The system to solve
    # # n2Plasma = plasmaSys(Ysp, T,verbose=False)  # just plasma 
    # n2Plasma = plasmaSys(Ysp, T0s[:],laser,verbose=False)  # just plasma 


    # Initialize with initial conditions
    n2Plasma.Ysp0 = Ysp0
    n2Plasma.Temp0 = T0s
    n2Plasma.initialize()


    # #2 if using combined model--------------------------------------
    # # if using laser modify the plasma system to include these new effects
    # # Done my instantiating CombinedModels
    # n2Plasma0 = plasmaSys(Ysp, T0s[:],verbose=False)
    # n2Plasma = pl.CombinedModels(n2Plasma0, laser)
##------------------------------------------------------




    # Once update the system based on the initial conditions
    # n2Plasma.update()
    eId = n2Plasma.gas.species_index('ele')

    # print the initial state of the system
    print("Initial state of the system")
    print("Tg = ", n2Plasma.Temp[0], "Tv = ", n2Plasma.Temp[1], "Te = ", n2Plasma.Temp[2])
    print("Ug = ", n2Plasma.Ug)
    print("Hg = ", n2Plasma.Hg)

    # setup the array for initial conditions all solved at once
    # need to combine Ysp and Temp
    yt0 = np.concatenate((n2Plasma.Ysp, n2Plasma.Temp))
    # change the Tg to Ug
    # yt0[-3] = n2Plasma.Ug
    # yt0[-3] = n2Plasma.Hg

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
    soln = pl.Solution(t0, n2Plasma)
    # # append initial state : the initial state is already appended during instantiation
    # soln.solnPush(t_array[0], n2Plasma)

    solMethods=["RK45","RK23","DOP853","Radau","BDF","LSODA"]


    # Now integrate the system
    solY = sc.integrate.solve_ivp(dYdt_all, int_t, yt0, method=solMethods[3], args=(n2Plasma,), rtol=1e-3, atol=1e-6)


    # print the message from the solver
    print(solY.message)


    # Now for post-processing, calculate all the properties based on the solved state
    for i in range(len(solY.t)):
        # set the state
        n2Plasma.Ysp = solY.y[0:-3,i]
        n2Plasma.Temp[1] = solY.y[-2,i]
        n2Plasma.Temp[2] = solY.y[-1,i]
        # n2Plasma.Hg = solY.y[-3,i]
        n2Plasma.Temp[0] = solY.y[-3,i]

        # # find temperature and set it
        # X = n2Plasma.numbDensity2X(n2Plasma.Ysp) # mole fraction
        # # n2Plasma.gas.X = X
        # X[eId] = 0.0
        # p = n2Plasma.pressure()
        # ## If no energy exchagne happens in the bulk gas then the 
        # # enthalpy of the gas remains unchanged and not the internal energy
        # n2Plasma.gas.HPX = n2Plasma.Hg, p, X
        # n2Plasma.Temp[0] = n2Plasma.gas.T

        # update the system
        n2Plasma.update()

        # push the solution
        soln.solnPush(solY.t[i], n2Plasma)
        # print the time and temperature
        print(solY.t[i], n2Plasma.Temp[0], n2Plasma.Temp[1], n2Plasma.Temp[2])

    # save the solution
    soln.solnSave(fname+'.npz')

    # save # time,N,N2,N2_A,N2_B,N2_C,Np,N2p,N3p,N4p,ele in .txt file
    # np.savetxt(fname+'.txt', np.transpose(np.concatenate((solY.t.reshape(1,-1), solY.y[0:-3,:]))), delimiter='\t')
    np.savetxt(fname+'.txt', np.transpose(np.concatenate((solY.t.reshape(1,-1), solY.y[:,:]))), delimiter='\t')
    
    # make a figure to plot during the integration
    fig, ax = plt.subplots(2,1, sharex=True)
    # size
    fig.set_size_inches(10, 8)


    ax[0].set_ylabel('N (m-3)')
    ax[1].set_ylabel('T (K)')
    ax[1].set_xlabel('Time (s)')
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')

    res_ysps = solY.y[0:-3]
    res_Ts = solY.y[-3:]

    # name of species
    spNames = n2Plasma.gas.species_names

    # Species to plot
    # plt_sps = ['N2','N','ele','N2p','Np']
    plt_sps = spNames
    # plot the results
    for i in range(len(spNames)):
        if spNames[i] in plt_sps:
            if spNames[i] == 'ele':
                ax[0].plot(solY.t, res_ysps[i,:], label=spNames[i], color='k',lw=2,alpha=0.7)
            else:
                ax[0].plot(solY.t, res_ysps[i,:], label=spNames[i])

    # temperature
    ax[1].plot(solY.t, res_Ts[0,:], label='Tg')
    ax[1].plot(solY.t, res_Ts[1,:], label='Tv')
    ax[1].plot(solY.t, res_Ts[2,:], label='Te')


    ax[0].legend()
    ax[1].legend()

    # grid
    ax[0].grid()
    ax[1].grid()

    # limit x = 1.0e-14
    ax[0].set_xlim(1.0e-14, 2*tf)

    # limit y
    ax[0].set_ylim(1.0e14, 5.0e25)

    # show the plot
    plt.show()
    plt.savefig('N2Plasma_debug.png')

   
# test the mec wrapper - reading new mechanism from yaml file
def testWrapper(mech_):

    # mecfile = "./mecWrapper/testYaml.yaml"
    mecfile = mech_
    # create a plasma fluid object
    pf = pl.PlasmaMechanism(mecfile)

    # evaluate rates at a given state
    # Ts = np.array([300.0, 300.0, 11600.0])
    rates = pf.getRateConstants(300,300,11600.0)
    print(rates)


# define a functio nwhich takes soln object and plot the temporal scale from Wrxn
def plotTemporalScale(fname='solnPlasma',pids=None,**kwargs):

        import matplotlib as mpl
        import matplotlib.ticker
        import scienceplots
        plt.style.use('science')
        # plt.style.use(['science','ieee'])
        plt.rcParams.update({'figure.dpi': '100'})
        ## the cycler repeats by default with just 4 line types. Improve this to 10
        from cycler import cycler
        # 4 colors from cmap
        colors = ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c']
        # line styles 
        linestyles = ['-', '--', '-.', ':']
        symbs = ['o', 'v', 's']
        # make cycler from combination of styles
        c0 = cycler(marker=symbs)
        # use linestyles to get more combinations
        c1 = c0 * cycler(linestyle=linestyles,color=colors,markevery=[0.1,0.1,0.1,0.1])
        # markerfacecolor='none', markeredgecolor='k', markeredgewidth=1.5, markersize=6
        plt.rcParams['lines.markersize'] = 6
        plt.rcParams['lines.markerfacecolor'] = 'None'
        # update
        mpl.rcParams['axes.prop_cycle'] = c1
        # plt.rcParams['lines.linewidth'] = 1.5

        # savename from kwargs
        saveTo = kwargs.get('saveTo','temporalScales')




        # load the data
        data = np.load(fname+'.npz',allow_pickle=True)
        # np.savez(saveTo, t=soln.t, Ysp=soln.Ysp, Temp=soln.Temp, Wrxn=soln.Wrxn, Qrxn=soln.Qrxn, dhrxn=soln.dhrxn,Qdot=soln.Qdot,Qmodes=soln.Qmodes)
        solt = data['t']
        solnYsp = data['Ysp']
        solnTemp = data['Temp']
        solnWrxn = data['Wrxn']
        solnQrxn = data['Qrxn']
        solndhrxn = data['dhrxn']
        solnQdot = data['Qdot']
        solnQmodes = data['Qmodes']
        solncp_mix = data['cp_mix']

        solne = solnYsp[:,-1]

        n0 = 2.5e25

        fmtSaves = ['png','pdf']
        ww = 4.5
        hh = 4.5

        # make a figure to plot
        fig, ax = plt.subplots(2,1, sharex=True)

        # size
        fig.set_size_inches(10, 10)

        # calculate temporal scale
        # taus = 1.0/(solnWrxn/n0)
        tausden = (solnWrxn/solne.reshape(-1,1))
        taus = 1.0/(tausden + 1.0e-40)


        # plot all the temporal scales in time
        for i in range(solnWrxn.shape[1]):
            ax[0].plot(solt, taus[:,i], label=str(i+1))



        if pids is None:
            # find the smallest 10 taus
            top10 = np.argsort(taus[0,:])[:15]
            # top10 = np.argsort(taus[20,:])[25:35]

            # sample in both dimensions to sort and find smallest
            # first sample in time
            # tids = np.arange(0,taus.shape[0],100)

        else:
            top10 = pids

        # plot the top 10 temporal scales
        for i in top10:
            ax[1].plot(solt, taus[:,i], label=str(i+1))

        # set the labels
        ax[0].set_ylabel('tau (s)')
        ax[1].set_ylabel('tau (s)')
        ax[1].set_xlabel('Time (s)')
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')

        # legend
        ax[0].legend()
        ax[1].legend()

        # grid
        ax[0].grid()
        ax[1].grid()

        # x limit 
        ax[0].set_xlim(1.0e-14, 2*solt[-1])
        ax[1].set_xlim(1.0e-14, 2*solt[-1])

        # ylimit
        ax[1].set_ylim(1.0e-14, 1.0e-2)
        ax[0].set_ylim(1.0e-14, 1.0)

        # show the plot
        plt.show()

        # make a separate plot same as ax[1]
        fig, ax = plt.subplots(1,1, sharex=True)
        # size
        fig.set_size_inches(ww,hh)

        # plot the top 10 temporal scales
        for i in top10:
            ax.plot(solt, taus[:,i], label='R'+str(i+1))

        # set the labels
        ax.set_ylabel(r'$\tau$ (s)')
        ax.set_xlabel('Time (s)')
        ax.set_xscale('log')
        ax.set_yscale('log')

        import matplotlib.ticker as ticker

        # locmaj = ticker.LogLocator(base=10.0,subs=(2.0, ),numticks=50)
        # ax.xaxis.set_major_locator(locmaj)
        locmin = ticker.LogLocator(base=10.0,subs=np.arange(2,10)*.1,numticks=100)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())


        # legend
        ax.legend(ncol=2)
        

        # limits
        ax.set_xlim(1.0e-14, 2*solt[-1])
        ax.set_ylim(1.0e-12, 1.0e-2)



        # # grid with major and minor ticks
        # ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.2,axis='both')
        # ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.1,axis='both')






        # fig tight
        fig.tight_layout()

        # grid
        # ax.grid()
        for fmt in fmtSaves:
            # plt.savefig('temporalScales.'+fmt, bbox_inches='tight',dpi=300)
            plt.savefig(saveTo+'.'+fmt, bbox_inches='tight',dpi=300)
        plt.show()
        




    



if __name__ == '__main__':
    # test()
    # test_withEnergy()
    # buildODE_n2plasma()
    # buildODE_n2plasmaAleks()

    # test_UV_set(100000)

    # # To plasma decay only-------------------------------------------------
    # # Uses the plasmaSystems ( rates in function)
    # # Te0 = 3.43*11600.0
    # # ne = 6.50e22
    # Te0 = 3.0*11600.0
    # pres = 760.0 # torr
    # n0 = 2.4e25*pres/760.0
    # # ne = 9.5e22*pres/760.0
    # ne = 1.0e23*pres/760.0
    # # Te0 = 1*11600.0
    # laser = pl.LaserModel(In=2.0e16,switch=0) # switch off the laser# Default is 1064 nm ns-laser
    # all_solve("solnPlasma",Te0,10.0e-6,laser,ne=ne,n0=n0)


    # # Air plasma decay only-------------------------------------------------
    # sp='Air'
    # plasmaChem = pl.AirPlasmaRxn
    # Te0 = 3.0*11600.0
    # # pres = 760.0 # torr
    # # n0 = 2.4e25*pres/760.0
    # # # ne = 9.5e22*pres/760.0
    # # ne = 1.0e23*pres/760.0
    # # Te0 = 1*11600.0
    # # laser = pl.LaserModel(In=2.0e16,switch=0) # switch off the laser# Default is 1064 nm ns-laser
    # all_solve("solnPlasma",Te0,10.0e-6,plasmaChem=plasmaChem,sp=sp)


    # # #  # plot the solution object
    # # # pvars = ['Ysp','Temp','Wrxn','Qrxn','dhrxn','Qdot']
    # # pvars = ['Wrxn']
    # # # prx=[2,3,4,13,17]
    # # prx=[3,4]
    # # pl.Solution.plotSoln("solnPlasma3",pvars=pvars,prx=prx,fmax=False) # prx = # of max reactions to check, for fmax = True, if false provide array of reactions to check in prx

    #  # plot the solution object
    # pvars = ['Ysp','Temp','Wrxn','Qmodes','dhrxn','Qdot']
    # pl.Solution.plotSoln("solnPlasma",pvars=pvars,prx=12,fmax=True) # prx = # of max reactions to check, for fmax = True, if false provide array of reactions to check in prx

    ##--------------------------------------------------------------------

    # # To run a laser absorption till breakdown case ----------------------
    # # Uses the plasmaSystems ( rates in function)
    # Te0 = 1.0*11600.0
    # laser = pl.LaserModel(In=2.0e16,switch=1) # switch off the laser# Default is 1064 nm ns-laser
    # all_solve("solnPlasma",Te0,0.58e-8,laser)
    # ##--------------------------------------------------------------------


    # mech = "N2NewMech.yaml" ----- 
    # # Test custom mechanism file 
    # testWrapper(mech)
    ##--------------------------------------------------------------------

    # test direct solve from custom mechanism file
    # test_directSolve("directSolve",Te0=11600.0,dt=1.0e-6,laser=None)


    #  # plot the solution object
    # pvars = ['Ysp','Temp','Wrxn','Qmodes','dhrxn','Qdot']
    # pl.Solution.plotSoln("solnPlasma3",pvars=pvars,prx=12,fmax=True) # prx = # of max reactions to check, for fmax = True, if false provide array of reactions to check in prx


    # to plot temporal scale, provide which reactions to plot in pids - 16 electron impact ionization , huge when Te is high
    pids = np.array([40,35,52,12,32,45,39])-1
    # pids = None
    plotTemporalScale("solnPlasma",pids=pids)


    ## To solve n2plasmaAleks ( some O2 percent)
    # N2PlasmaAleks
    # To plasma decay only-------------------------------------------------
    # Uses the plasmaSystems ( rates in function)
    # Te0 = 3.43*11600.0
    # ne = 6.50e22
    Te0 = 1.0*11600.0
    ne = 1.0e23
    # Te0 = 1*11600.0
    laser = pl.LaserModel(In=2.0e16,switch=0) # switch off the laser# Default is 1064 nm ns-laser
    all_solve("solnPlasma",Te0,1.0e-7,laser,ne=ne,plasmaChem=pl.N2PlasmaAleks,sp='N2Aleks')
    ##--------------------------------------------------------------------