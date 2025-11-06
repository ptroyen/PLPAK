"""
---------------------------------------------------------------------------------------------
Integrator for the plasma kinetics and energy equations.
    Time Loop
        1. Solve species conservation equations.
        2. Solve energy conservation equations for different energy modes. Tg, Tv , and Te.
    Time Loop End
To be used with results/output from makePlasmaKinetics_II.py library.
---------------------------------------------------------------------------------------------
Date : 2023-03-20
Author : Sagar Pokharel ; pokharel_sagar@tamu.edu // https://github.com/ptroyen

"""

import numpy as np
import cantera as ct
from abc import ABC, abstractmethod

# add current directory to the path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Import the plasma kinetics library
from makePlasmaKinetics import *
# This also imports the constants defined in the library
# # Defined the constants
# R = 8.314 # J/mol/K
# e = 1.60e-19 # C
# kB = 1.38e-23 # J/K
# m_e = 9.11e-31 # kg
# Na = 6.02e23 # mol^-1

# Alias for methods in the library
NASA7Enthalpy = ODEBuilder.NASA7Enthalpy



# Define the system class
class System(ABC):
    '''
    Abstract Base Class: system
    Class to define the system object. This holds all the information from the makePlasmaKinetics_II.py library.
    The system would be defined once the species concentration and temperature are known.
    '''

    # def __init__(self, Ysp, T, mech, nrxn, verbose=False):
    #     tempGas = ct.Solution(mech)
    #     self.__init__(self, Ysp, T, tempGas , nrxn, verbose)
    #     self.mech = mech
    #     self.gas = tempGas

    def __init__(self, Ysp, T,  mech , nrxn, verbose=False):
        '''
        Initialize the system object.
        '''
        # Not changing during the simulation besides Ysp and Temp
        self.Ysp = Ysp.copy() # species concentration
        self.Temp = T.copy() # temperature
        self.mech = mech
        self.nrxn = nrxn
        self.gas = ct.Solution(mech)
        # self.gas = gas
        self.spMw = self.gas.molecular_weights*1.0e-3 # kg/mol

        reactants = self.gas.reactant_stoich_coeffs # stoichiometric coefficients for reactants
        products = self.gas.product_stoich_coeffs # stoichiometric coefficients for products

        # change reactants and products to integer values
        reactants = reactants.astype(int)
        products = products.astype(int)

        self.effStoic = products - reactants # effective stoichiometric coefficients
        self.effStoic = self.effStoic.T # transpose to get the shape as (nrxn, nsp)

        # inital conditions
        self.Ysp0 = Ysp
        self.Temp0 = T

        self.nsp = len(Ysp) # number of species

        ## Needs Update
        self.p = 0.0
        self.rho = 0.0
        # check the update method
        self.rho = self.density() # assumed constant density
        self.p = self.pressure() # assumed constant pressure

        self.X = np.zeros_like(Ysp) # mole fraction

        # reaction rate constants 
        self.Krxn = np.zeros(nrxn) # SI units

        # reaction rates or rate of progression of the reaction
        self.Wrxn = np.zeros(nrxn) # concentration units / s

        self.dYdt = np.zeros_like(Ysp) # concentration units / s
        self.dTdt = np.zeros_like(T) # temperature units / s

        # bulk internal energy of the mixture per unit mass
        self.Ug = 0.0 # J/kg
        self.Hg = 0.0 # J/kg

        self.cp_mix = None # specific heat of the mixture in J/kg/K
        self.hsp = np.zeros_like(Ysp) # enthalpy of formation of the species in J/kmol
        self.dhrxn = np.zeros((3,nrxn)) # enthalpy of reaction in J/kmol for each mode
        self.Qrxn = np.zeros((3,nrxn)) # reating rate in J/m3-s
        self.Qdot = np.zeros_like(T) # heating rate in J/m3-s summed from all reactions

        # self.Qmodes = np.zeros(4)   # heating rate in J/m3-s for each mode
        self.Qmode_names = ['EV', 'ET', 'VE','VT'] # names of the modes
        # make a dictionary for the modes where key is Qmode_names and values are float values
        self.Qmodes = dict(zip(self.Qmode_names, np.zeros(len(self.Qmode_names))))


        ## Electron neutral collision frequency and electron ion collision frequency
        ## Should be updated by Qmodes_() method
        self.nu_en = 0.0 # electron neutral collision frequency
        self.nu_ei = 0.0 # electron ion collision frequency
        # make an array to store the bolsig parameters for nu_en
        self.bolsig_en = np.array([-29.97, 0.3164, -0.4094, 0.4047E-01, -0.1363E-02]) # m3/s , for N2
        # self.bolsig_en = np.array([-29.95, 0.3370, -0.3595, 0.2342E-01, -0.3446E-03]) # m3/s , for Air , should be overwritten by actual values

        self.bgID = 0
        try:
            self.bgID = self.gas.species_index('N2')
        except ValueError:
            pass

        # flags
        self.verbose = verbose

        # print number of species and reactions
        if self.verbose:
            print('Number of species = ', self.nsp)
            print('Number of reactions = ', self.nrxn)

            # From gas object
            print('Number of species in the gas object = ', len(self.gas.species()))
            print('Number of reactions in the gas object = ', len(self.gas.reactions()))


        # ## Before Update check if everything is consistent
        # # check number of species
        # if self.nsp != len(self.gas.species()):
        #     raise ValueError('Number of species in the gas object and Ysp are not consistent')
        # # check number of reactions
        # if self.nrxn != len(self.gas.reactions()):
        #     raise ValueError('Number of reactions in the gas object and nrxn are not consistent')


        # update the system using self.update() which should be defned later
        self.update()
    
    def initialize(self):
        self.Ysp = self.Ysp0.copy()
        self.Temp = self.Temp0.copy()
        # As constant density simulation for cell, explicitly update the density initially as update() will not update density
        self.rho = self.density()
        self.update()

    def update(self):
        '''
        Update the system object to new state based on Ysp and Temp.
        '''
        # update from base class
        self.p = self.pressure()        # pressure in Pa
        # self.rho = self.density()       # density in kg/m3
        self.X = System.numbDensity2X(self.Ysp) # mole fraction
        # remove electrons in bulk gas
        self.X[-1] = 0.0
        self.gas.TPX = self.Temp[0], self.p, self.X
        self.cp_mix = self.cp_mix_()     # specific heat of the mixture in J/kmol/K
        # self.cp_mass = 
        self.hsp = self.hsp_()           # enthalpy of formation of the species in J/kmol

        # update from abstract class
        self.Krxn = self.Krxn_()         # SI units
        self.Wrxn = self.Wrxn_()         # concentration units / s
        self.dhrxn = self.dhrxn_()       # enthalpy of reaction in J/kmol
        self.Qrxn, self.Qdot  = self.Qrxn_()         # reating rate in J/m3-s
        self.Qmodes = self.Qmodes_()     # heating rate in J/m3-s for each mode
        self.dYdt = self.dYdt_()         # concentration units / s
        self.dTdt = self.dTdt_()         # temperature units / s
        self.Ug = self.gas.int_energy_mass # bulk internal energy of the mixture per unit mass - J/kg
        self.Hg = self.gas.enthalpy_mass # bulk enthalpy of the mixture per unit mass - J/kg

        # # check if the update is consistent
        # # check number of species
        if self.nsp != len(self.Ysp):
            raise ValueError('Number of species in the gas object and Ysp are not consistent')


    # update with same energy exchange
    def update_constH(self):
        '''
        Update the system object to new state based on Ysp and Temp.
        '''
        # update from base class
        self.p = self.pressure()        # pressure in Pa
        self.rho = self.density()       # density in kg/m3
        self.X = System.numbDensity2X(self.Ysp) # mole fraction
        self.gas.TPX = self.Temp[0], self.p, self.X
        self.cp_mix = self.cp_mix_()     # specific heat of the mixture in J/kmol/K
        self.hsp = self.hsp_()           # enthalpy of formation of the species in J/kmol

        # update from abstract class
        self.Krxn = self.Krxn_()         # SI units
        self.Wrxn = self.Wrxn_()         # concentration units / s
        # self.dhrxn = self.dhrxn_()       # enthalpy of reaction in J/kmol
        self.Qrxn, self.Qdot  = self.Qrxn_()         # reating rate in J/m3-s
        self.Qmodes = self.Qmodes_()     # heating rate in J/m3-s for each mode
        self.dYdt = self.dYdt_()         # concentration units / s
        self.dTdt = self.dTdt_()         # temperature units / s
        self.Ug = self.gas.int_energy_mass # bulk internal energy of the mixture per unit mass - J/kg



    # new method for limited update from inside dYsp_dt
    # # this is better to include with integrator rather than here
    # def updateFromYsp(self):
    #     '''
    #     Update the system object to new state based on Ysp change only.
    #     This should be used to update the sate while integrating the species equation as temeprature is not changing.
    #     '''
    #     # update from base class
    #     self.p = self.pressure()        # pressure in Pa
    #     self.rho = self.density()       # density in kg/m3
    #     self.Krxn = self.Krxn_()
    #     self.Wrxn = self.Wrxn_()
    #     self.dYdt = self.dYdt_()

        

    # Methods Required for the system object
    ## base methods: cp_mix(self), hsp(self), pressure(self), density(self)
    ## abstract methods: Krxn(self), Wrxn(self), dYdt(self), dTdt(self), dhrxn(self), Qrxn(self), Qdot(self)
    def get_electron_index(self):
        """Helper method to find electron species index regardless of naming convention.
        
        Checks common electron species names: 'E', 'e-', 'E-', 'ele'
        Returns the first match found or raises ValueError if no electron species found.
        """
        gas = self.gas
        electron_names = ['E', 'e-', 'E-', 'ele']
        for e_name in electron_names:
            try:
                return gas.species_index(e_name)
            except ValueError:
                continue
        raise ValueError("No electron species found in mechanism. Expected one of: " + str(electron_names))

    def pressure(self):
        '''
        Calculate the pressure in Pa for plasma.
        Uses the Ysp and Temp to calculate the pressure.
        Ysp is in number density and Temp is in K.
        Electron pressure is calculated using the electron temperature.
        '''
        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]
        eleID = self.get_electron_index()

        p = 0.0
        for i in range(len(self.Ysp)):
            if i == eleID:
                # p = p + self.Ysp[i]*kB*Te
                # p = p + self.Ysp[i]*kB*Tg
                continue
            else:
                p = p + self.Ysp[i]*kB*Tg
        return p

    def density(self):
        '''
        Calculate the density in kg/m3 for plasma.
        Uses the Ysp and Temp to calculate the density.
        Ysp is in number density and Temp is in K.
        '''
        # eleID = self.gas.species_index('ele')

        rhos = (self.Ysp*self.spMw/Na)
        rho = np.sum(rhos)

        return rho

    def hsp_(self):
        '''
        Calculate the enthalpy of each species in J/kmol.
        '''
        gas = self.gas
        eleID = self.get_electron_index()
        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]
        hsp = [gas.species()[i].thermo.h(Tg) for i in range(len(gas.species()))]
        hsp[eleID] = gas.species()[eleID].thermo.h(Te)
        hsp = np.array(hsp) # J/kmol

        return hsp

        
    def cp_mix_(self):
        '''
        Calculate the specific heat of the mixture in J/kg/K.
        '''
        # moleFraction = self.numberDensity2moleFraction(self.Ysp)
        # self.gas.TPX = self.Temp, self.p, moleFraction
        cp_mix = self.gas.cp_mass # J/kg/K
        return cp_mix

    # # update the heat exchange
    # def updateEnergyExchange(self):
    #     '''
    #     Updates all terms related to energy exhagne based on Ysp and Temp.
    #     Terms updated: hsp, 
    #     '''
    #     # update the heat exchange
    #     self.Qrxn, self.Qdot  = self.Qrxn_()
    #     self.Qmodes = self.Qmodes_()



    ## Abstract Methods: Qrxn, dhrxn, Krxn, Wrxn, dYdt, dTdt, JacY, JacT

    # @abstractmethod
    # def Qrxn_(self):
    #     '''
    #     Calculate the heating rate in J/m3-s for each reaction.
    #     Also provides Qdot which is the total heating rate in J/m3-s for each energy equation.
    #     '''
    #     pass

    # Implement the abstract methods
    def Qrxn_(self):
        '''
        Calculate the heating rate for each reaction in J/m3-s.
        dhrxn is in J/kmol and Ysp is in number density.
        Wrxn is in numDensity/s and Temp is in K.
        -- can be put in the base class --
        '''
        gas = self.gas

        # Qrxn = -(self.Wrxn/Na)*self.dhrxn*1.0e-3 # J/m3-s

        QrxnTg = -self.Wrxn/Na*self.dhrxn[0,:]*1.0e-3 # J/m3-s
        QrxnTv = -self.Wrxn/Na*self.dhrxn[1,:]*1.0e-3 # J/m3-s
        QrxnTe = -self.Wrxn/Na*self.dhrxn[2,:]*1.0e-3 # J/m3-s

        Qrxn = np.array([QrxnTg, QrxnTv, QrxnTe])


        ## Qrxn changes here so should be abstract method
        # sum all the reactions, so sum over the rows
        # J/m3-s
        # Qdot = np.sum(Qrxn, axis = 1)

        Qdotg = np.sum(QrxnTg)
        Qdotv = np.sum(QrxnTv)
        Qdote = np.sum(QrxnTe)

        Qdot = np.array([Qdotg, Qdotv, Qdote])
        
        # validate shape of Qdot should have 3 elements
        if Qdot.shape[0] != 3:
            raise ValueError('Qdot should have 3 elements')

        # if self.verbose:
        #     print('Qdot = ', Qdot)
        #     print('Shape of Qdot = ', Qdot.shape)
        #     print ('Qrxn = ', Qrxn)
        #     print('Shape of Qrxn = ', Qrxn.shape)
        #     print('dhrxn = ', self.dhrxn)
        #     print('Wrxn = ', self.Wrxn)


        return Qrxn, Qdot

    # @abstractmethod
    def Qmodes_(self):
        '''
        Calculate the heating rate in J/m3-s for modes in Qmode_names.
        The corresponding Qrxn or the dhrxn should be zero if the mode is included here.
        '''
        Qmode_names = ['ET','EV','VE','VT']
        self.Qmode_names = Qmode_names

        # zeros 
        Qmodes = np.zeros(len(Qmode_names))
        Qmodes_dict = dict(zip(Qmode_names,Qmodes))

        # update collision frequencies
        # use self.bolesig_en for electron neutral collision frequency
        # update self.nu_en and self.nu_ei
        return Qmodes_dict



    @abstractmethod
    def dhrxn_(self):
        '''
        Calculate the enthalpy of reaction in J/kmol for each reaction.
        '''
        pass

    @abstractmethod
    def Krxn_(self):
        '''
        Calculate the reaction rate constants in SI units.
        '''
        pass




    @abstractmethod
    def Wrxn_(self):
        '''
        Calculate the rate of progress in concentration units / s.
        '''
        pass

    @abstractmethod
    def dYdt_(self):
        '''
        Calculate the change in species concentration in concentration units / s -- mostly numdensity/s.
        '''
        pass

    # @abstractmethod
    # def dTdt_(self):
    #     '''
    #     Calculate the change in temperature in temperature units / s.
    #     '''
    #     pass

    # @abstractmethod
    def dTdt_(self):
        '''
        Calculate the change in temperature in temperature units / s.

        The bulk gas energy equation is better solved as internal energy rather than temperature.
        Other two energy equations can be solved as temperatures.

        Test: Formulation with singe energy equation with internal energy for the bulk gas.

        '''
        nrxn = self.nrxn
        nsp = self.nsp
        Qdotg = self.Qdot[0]
        Qdotv = self.Qdot[1]
        Qdote = self.Qdot[2]
        rho = self.rho
        cp = self.cp_mix

        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]

        nN2 = self.Ysp[self.bgID] # number density of N2
        ne = self.Ysp[self.get_electron_index()]

        # eID = self.gas.species_index('ele')
        # Tv = self.Temp[1]
        dEv_dTv = self.dEv_dTv_(Tv,3905.0) # vibrational energy per particle derivative with respect to temperature

        QVT = self.Qmodes['VT']
        QET = self.Qmodes['ET']
        QEV = self.Qmodes['EV']
        QVE = self.Qmodes['VE']

        if self.verbose:
            # show these values
            print('Qdotg = ', Qdotg, 'Qdotv = ', Qdotv, 'Qdote = ', Qdote)
            print('QVT = ', QVT, 'QET = ', QET, 'QEV = ', QEV, 'QVE = ', QVE)

        # d()dt:
        dydt = np.zeros(np.shape(self.Temp))

        # dydt[0] = Qdotg/(cp*rho) # + QET + QVT
        # dydt[0] = 0.0
        dydt[0] = (Qdotg + QET + QVT)/(cp*rho)
        dydt[1] = ((Qdotv + QEV - QVE - QVT)/nN2)/dEv_dTv

        Ce = 1.5*kB*ne
        # dydt[2] = (Qdote + QVE - QET - 0.0*QEV)/Ce - self.dYdt[eID]*Te/ne
        dydt[2] = (Qdote + QVE - QET - QEV)/Ce # - self.dYdt[eID]*Te/ne
        # The dne/dt term is required if solving both Ysp and Temp simultaneously.
        # When the electron density is already modified then it is not required ?

        # dydt[2] = (0.0*Qdote + QVE - 0.0*QET - QEV)/Ce - self.dYdt[eID]*Te/ne

        # dydt[2] = - self.dYdt[eID]*Te/ne + (QVE - QEV + Qdote - 0.0*QET)/Ce

        return dydt
    
    # @abstractmethod
    def JacY_(self):
        '''
        Calculate the Jacobian of the species concentration.
        '''
        pass

    # @abstractmethod
    def JacT_(self):
        '''
        Calculate the Jacobian of the temperature.
        '''
        pass


    # # static methods : BolsigFit, moleFraction 
    # NASA7 already in the makePlasmaKinetics_II.py library outputs enthalpy in J/mol and needed in J/kmol

    @staticmethod
    def numbDensity2X(Ysp):
        '''
        Convert number density to mole fraction.
        '''
        X = Ysp/np.sum(Ysp)
        return X

    # Bolsig fit Fit coefficients y=exp(A+B*ln(x)+C/x+D/x^2+E/x^3)
    #    54.42      0.7364     -0.2075      0.3281E-01 -0.1439E-02
    @staticmethod
    def bolsigFit(x,Coef):
        '''
        Calculate the Bolsig Fit with average energy ?? in eV.
        Coef = [A,B,C,D,E]
        For units need to check the coeff provided.
        The fit is for mean energy per particle.
        x should be in eV
        '''
        # print('Bolsig Fit: x = ',x)
        # check if 0 or negative

        # if x is less than 0.1 then assume 0.1
        # show
        # print('In Bolsig Fit, x = ',x, ' and Coef = ',Coef)

        
        y = np.exp(Coef[0] + Coef[1]*np.log(x) + Coef[2]/x + Coef[3]/x**2 + Coef[4]/x**3)
        # if np.min(x) <= 0.1:
        # # print('In Bolsig Fit, x = ',x)
        #     tx = np.ones(np.shape(x))*0.1
        #     y = np.exp(Coef[0] + Coef[1]*np.log(tx) + Coef[2]/tx + Coef[3]/tx**2 + Coef[4]/tx**3)
        # print("Fit = ",y)
        # if nan or inf return 0
        
        if np.isnan(y) or np.isinf(y):
            print ('In Bolsig Fit, x = ',x, ' Calcualted y = ',y)
            
            # tx = np.ones(np.shape(x))*0.1
            # y = np.exp(Coef[0] + Coef[1]*np.log(tx) + Coef[2]/tx + Coef[3]/tx**2 + Coef[4]/tx**3)
            y = 0.0

        # print("Result from Bolsig Fit = ",y)
        return y

    # vibrational energy per particle
    @staticmethod
    def vibEnergy(Tv,Theta_c):
        '''
        Calculate the vibrational energy per particle - J/particle
        Multiply by number density to get the vibrational energy density in J/m3
        Tv = vibrational temperature in K
        Theta_c = characteristic vibrational temperature in K
        '''
        Ev = kB*Theta_c/(np.exp(Theta_c/Tv)-1.0)
        return Ev

    # dEv/dTv
    @staticmethod
    def dEv_dTv_(Tv,Theta_c):
        '''
        Calculate the derivative of the vibrational energy per particle with respect to Tv - J/particle/K
        Tv = vibrational temperature in K
        Theta_c = characteristic vibrational temperature in K
        '''
        dEv_dTv = Theta_c**2*kB*np.exp(Theta_c/Tv)/(Tv**2.0*(np.exp(Theta_c/Tv) - 1.0)**2.0)
        return dEv_dTv

    # p*tau_VT
    @staticmethod
    def ptauVT(Tv,A,mu):
        '''
        Calculate the p*tau_VT in [atm-s]
        Tv = vibrational temperature in K
        Parameters as defined in the paper:
        A : 
        mu :
        ----
        Systematics of Vibrational Relaxation  # Cite as: J. Chem. Phys. 39, 3209 (1963); https://doi.org/10.1063/1.1734182 
        --Roger C. Millikan and Donald R. White
        '''
        # p[atm]*tau[s] = exp(A(T**(-1/3) - 0.015 mu**(1/4)) - 18.42) [atm*s]
        ptauVT = np.exp(A*(Tv**(-1.0/3.0) - 0.015*mu**(1.0/4.0)) - 18.42)
        return ptauVT

    @staticmethod
    def ptauVT_Park(Tv,A,B):
        '''
        Calculate the p*tau_VT in [atm-s]
        Tv = vibrational temperature in K
        
        Parameters as defined in the paper of Millikan and White needs to be modified so directly supply A and B
        A:
        B:
        ----
        https://link.springer.com/content/pdf/10.1007/s00193-012-0401-z.pdf
        1. Gehre RM, Wheatley V, Boyce RR. Revised model coefficients for vibrational relaxation in a nitrogen–oxygen gas mixture. Shock Waves. 2012;22(6):647-651. doi:10.1007/s00193-012-0401-z
        '''
        # p[atm]*tau[s] = exp(A(T**(-1/3) - 0.015 mu**(1/4)) - 18.42) [atm*s]
        ptauVT = np.exp(A*(Tv**(-1.0/3.0) - B) - 18.42)
        return ptauVT
    
    # static method to calculate the coulomb logarithm and electron ion collision frequency
    # '''
    # # Lambda and one line Functions
    # # Te in K, n_e in m^-3
    # # coulomb log
    # coul_log = lambda Te, n_e: 0.5*log(1.0 + 0.25*CO_eps0*CO_kB*Te/(pi*CO_Qe**2*n_e*(CO_Qe**4/(16*pi**2*CO_eps0**2*CO_kB**2*Te**2) + CO_h**2/(2*pi*CO_kB*CO_me*Te))))
    # # coulomb collision frequency
    # coul_freq = lambda Te, n_e: 3.633e-6*coul_log(Te, n_e)*n_e*Te**(-1.5) # 1/s , 
    # '''

    @staticmethod
    def coul_log(Te, n_e):
        '''
        Input:
        Te: electron temperature in K
        n_e: electron number density in m^-3
        ---
        Calculate the coulomb logarithm.
        - Hyperbolic trajectories considered.
        - Should be positive at all times.
        '''
        coul_log = 0.5*np.log(1.0 + 0.25*CO_eps0*CO_kB*Te/(np.pi*CO_Qe**2*n_e*(CO_Qe**4/(16*np.pi**2*CO_eps0**2*CO_kB**2*Te**2) + CO_h**2/(2*np.pi*CO_kB*CO_me*Te))))
        return coul_log

    @staticmethod
    def coul_freq(Te, n_e):
        '''
        Input:
        Te: electron temperature in K
        n_e: electron number density in m^-3
        ---
        Calculate the coulomb collision frequency. [1/s]
        - Chen Plasma Book
        - Zeldovich and Raizer
        '''
        coul_freq = 3.633e-6*System.coul_log(Te, n_e)*n_e*Te**(-1.5) # 1/s
        return coul_freq

