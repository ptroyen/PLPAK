"""
This code has functions to make an ODE from plasma Kinetics.
    date : 2022-12-3
    author: Sagar Pokharel ; pokharel_sagar@tamu.edu // https://github.com/ptroyen

In the codes import this file and use as follows:
    from makePlasmaKinetics import *
    # Read reaction mechanism
    gas = ct.Solution('plasmaN2.cti')

    # Test the symbolic function to get the reaction rates for all reactions
    # Define the names of the variables
    spName = 'y'
    rateName = 'K_rxn'
    mwName = 'spMW'
    rhoName = 'rho_'
    NaName = 'Na'

    # get the symbolic expression for production of each species
    wsp_ys, sp_ys, rates_list, spMW, rho, Na = getProductionRatesSym(gas,spName,rateName,mwName,rhoName,NaName)

    # substitute the values to the variables : Na = 6.023e23, rho = 1.0
    wsp_ysn = [i.subs({Na:6.023e23,rho:1.0}) for i in wsp_ys]
    # print one of wsp_ys to C++
    printExpression(wsp_ysn[0:4],'CXX','dydx')

#---------------------------UPDATES---------------------------------#
# 2022-12-3 : Initial version for mass fraction system
# 2023-01-14: Include a mode argument to get ODE for mass fraction, molar density, or mass density, etc.
# 2023-03  : Heat of Reaction for multi temperature exchange
# 2023-03-20: Include an abstract base class called "System" which can be built from the outo=put of "ODEBuilder"


"""
from abc import ABC, abstractmethod
import cantera as ct
import sympy as sp
import numpy as np


import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from mecWrapper import PlasmaMechanism


R = 8.314 # J/mol/K
e = 1.60e-19 # C
kB = 1.38e-23 # J/K
m_e = 9.11e-31 # kg
Na = 6.02e23 # mol^-1
h = 6.626e-34 # Planck's constant
c =  2.99792458e8 # speed of light
m_p = 1.6726219e-27 # proton mass
pi = np.pi # pi
eps0 = 8.854187817e-12 # vacuum permittivity

# SOME GLOBAL CONSTANTS
CO_kB = 1.38e-23 # J/K : Boltzmann constant
CO_eC = 1.6e-19 # C : electron charge
CO_me = 9.1e-31 # kg : electron mass
CO_h = 6.626e-34 # J s : Planck constant
CO_hc = CO_h/(2*3.14159265) # J m : Planck constant
CO_eps0 = 8.854e-12 # F/m : permittivity of free space
CO_c = 2.998e8 # m/s : speed of light
CO_Qe = 1.60217662e-19 # C : electron charge
CO_amu = 1.660539040e-27 # kg : atomic mass unit
CO_pi = 3.14159265



class ODEBuilder:
    
    # some data members

    # NASA7 coefficients for N2 - first row is low temperature, second row is high temperature
    N2Avg1NASA7 = np.array([[3.07390924e+00,1.99120676e-03,-2.08768566e-06,2.62439456e-09,-1.34639011e-12,7.05970674e+04,6.42271362e+00],
                        [3.50673966e+00,1.30229712e-03,-6.69945124e-07,1.24320721e-10,-7.97758691e-15,7.03934476e+04,3.91162118e+00]])
    N2Avg2NASA7 = np.array([[2.73767346e+00,4.27761848e-03,-6.82914221e-06,6.93754842e-09,-2.81752532e-12,8.05427151e+04,7.90284209e+00],
                        [3.77161049e+00,1.00425875e-03,-5.74156450e-07,1.11907972e-10,-7.44792998e-15,8.02048573e+04,2.47921910e+00]])
    N2Avg3NASA7 = np.array([[2.64595791e+00,4.87994632e-03,-7.18000185e-06,6.35412823e-09,-2.33438726e-12,8.94758420e+04,8.34579376e+00],
                        [4.10955639e+00,2.57529645e-04,8.95120840e-09,-1.09217976e-11,5.27900491e-16,8.90514137e+04,8.01508324e-01]])


    # Excitation energy for ground state nitrogen vibrationally from 0 to v whenre v = [1-8] in eV per molecule
    N2_Uvs_eV = np.array([0.2888726, 0.57419078, 0.85595285, 1.13415714, 1.40880197, 1.67988565, 1.9474065, 2.21136285])

    def __init__(self, mechanism_file, mode='mass_fraction'):
        self.gas = ct.Solution(mechanism_file)

        # # validate the mechanism file using cantera
        # ct.kinetics.checkBalance(self.gas.kinetics)

        # Symbols and defaults - BASIC
        self.spName = 'Ysp'
        self.rateName = 'K_rxn'
        self.mwName = 'SpMW'
        self.rhoName = 'rho'
        self.NaName = 'Na'
        self.NumName = 'N_tot'
        self.mode = mode
        self.language = 'CXX'
        self.TgName = 'Tg'
        self.TvName = 'Tv'
        self.TeName = 'Te'

        # symbols required for heat of reactions
        self.hspName = 'hsp'
        # self.hspExtraName = 'hspExtra'  # for extra species, e.g N2(A,v=0-4) in energy exchange
        self.subHname = ''
        self.omegaRxnExtraName = 'omegaRxnExtra' # for extra species to find rate of heat exchange, eg K for N2(X,v=1) is different
        
        # symbols used for calling functions to calculate required properties
        self.hspFuncNamePre = 'specieThermo_['
        self.hspFuncNamePost = '].ha(p,T)'

        # Check if mode is valid
        valid_modes = ['mass_density', 'molar_density', 'number_density', 'mass_fraction', 'mole_fraction']
        if self.mode not in valid_modes:
            raise ValueError(f'Invalid mode "{self.mode}"')

        # for output
        self.assignDdt = 'dYdt'
        self.assignWrxn = 'W_rxn'
        self.assignHRxn = 'HRxn'
        self.assignEnthalpy = 'H'
        self.assignHeatRate = 'QRxn'

        # Initialize empty list to hold ODE system
        self.ode_system = []
        self.ode_systemFromOmega = []
        self.eachReactionRateConstant = []
        self.eachReactionRate = []
        self.enthalpyCall = []
        self.subHnames = []
        self.heatOfReaction = []
        self.rateOfHeating = [] # all 3 modes


        # Other operation flags
        self.verbose = False
        self.outFnamePrefix = 'OUT'
        self.outFnameSuffix = 'txt'


        



    def updateODESystem(self, reactionRateConstant=None, extraFuncHrxn=None,energyExchangeDict=None):
        # for kineics
        if reactionRateConstant is not None:
            Krxn, Tg, Tv, Te = reactionRateConstant(self.TgName, self.TvName, self.TeName)
        else:
            print('Trying to use reactions rates from the mechanism file')

        # return wsp_ys, wspFromOmega_ys, reac_stoic_sym, omegaIRxn, sp_ys, rates_list, spMW, rho, Na, Num
        wsp_ys, wspFromOmega_ys, reac_stoic_sym, omegaIRxn, sp_ys, rates_list, spMW, rho, Na, Num = self.getProductionRatesSym()
        
        # For heat of reaction
        hRxn_Sym, h_sp, h_sp_call, subHNames = self.updateHrxnExpressions(extraFuncHrxn,energyExchangeDict)

        
        self.eachReactionRateConstant = Krxn
        self.ode_system = wsp_ys
        self.ode_systemFromOmega = wspFromOmega_ys
        self.eachReactionRate = reac_stoic_sym
        self.enthalpyCall = h_sp_call
        self.subHnames = subHNames
        self.heatOfReaction = np.array(hRxn_Sym)


        # Rate of heating - J/m^3/s
        # # reac_stoic_sym reaction rate for each reaction - is always number density/s
        ## The mode only changes the dy/dt term
        # make symbol for heat of reaction for each reaction, use self.assignHRxn
        hRxnNamesTg = [sp.Symbol(self.assignHRxn + '[0]' + '['+str(i)+']') for i in range(self.gas.n_reactions)]
        hRxnNamesTv = [sp.Symbol(self.assignHRxn + '[1]' + '['+str(i)+']') for i in range(self.gas.n_reactions)]
        hRxnNamesTe = [sp.Symbol(self.assignHRxn + '[2]' + '['+str(i)+']') for i in range(self.gas.n_reactions)]

        hRxnNamesAll = np.array([[hRxnNamesTg[i], hRxnNamesTv[i], hRxnNamesTe[i]] for i in range(self.gas.n_reactions)])
        heatRate = np.array([(hRxnNamesAll[i,:] * omegaIRxn[i]  / 6.022e26) for i in range(self.gas.n_reactions)]) # J/m^3/s as hRxn_Sym is in J/kmol/s and reac_stoic_sym is in number/m^3/s

        # loop through heatOfReaction array and change heatRate to zero if heatOfReaction is zero
        for i in range(self.gas.n_reactions):
            if self.heatOfReaction[i][0] == 0:
                heatRate[i][0] = 0
            if self.heatOfReaction[i][1] == 0:
                heatRate[i][1] = 0
            if self.heatOfReaction[i][2] == 0:
                heatRate[i][2] = 0


        # In rateOfHeating positive means heating, negative means cooling so need to flip the sign from heat of reaction
        self.rateOfHeating = -heatRate
        

        print('-----------------------------------------------------')
        print('-----------------------------------------------------')
        # print the settings used
        print('ODEBuilder settings:')
        print('ODE mode - Output In: ', self.mode)
        print('ODE system language : ', self.language)
        print('Verbose : ', self.verbose)


        # Start of Input Print section
        print('-----------------------------------------------------')

        # Show the symbols used in the ODE system
        print('ODE system symbols:')
        print('Species  : ', self.spName)
        print('Rate     : ', self.rateName)
        print('Individal Reaction Rate : ', self.assignWrxn)
        print('MW       : ', self.mwName)
        print('Density  : ', self.rhoName)
        print('Avogadro : ', self.NaName)
        print('Total Number Density      : ', self.NumName)
        print('Enthalpy function call : ', self.hspFuncNamePre, self.hspFuncNamePost)
        print('Enthalpy Subgroup name : ', self.subHname)
        print('Individual extra reaction extra name : ', self.omegaRxnExtraName)
        print('ODE system assign to : ', self.assignDdt)
        print('Heat of reaction assign to : ', self.assignHRxn)
        print('Enthalpy assign to : ', self.assignEnthalpy)

        print('-----------------------------------------------------')

        # print all species names
        print('Species names:')
        print(self.gas.species_names)

        # first show all the reactions read from the mechanism
        print('Reactions read from the mechanism:')
        for i in range(self.gas.n_reactions):
            print('RXN', i, ' : ', self.gas.reaction(i))

        # A line to separate the output
        print('-----------------------------------------------------')

        # reaction rate constant
        print('Reaction rate constant:')
        for i in range(self.gas.n_reactions):
            print(self.eachReactionRateConstant[i])



        # print eachReactionRate - rate of progress for each reaction
        print('-----------------------------------------------------')
        # Kinetics
        print('--------*** Kinetics ***--------')
        print('\nReaction rates: ' + self.assignWrxn + ' Units of K in SI so units = concentration_used/s')
        for i in range(self.gas.n_reactions):
            print((self.eachReactionRate[i]))

        # print ode system from omega
        print('\nODE system where ' + self.assignWrxn + ' is the rate of individual reaction kmol/m3/s  or numberdensity/s \n')
        for i in range(self.gas.n_total_species):
            print((self.ode_systemFromOmega[i]))

               
        print('\nDetailed ODE System:')
        # print the ODE system with sympy pretty print
        for i in range(self.gas.n_total_species):
            # print(sp.latex(self.ode_system[i]))
            print((self.ode_system[i]))

        print('-----------------------------------------------------')

        # Heat of reaction
        print('--------*** Heat of reaction ***--------')

        # enthalpy 
        print('\nEnthalpy: ' + self.assignEnthalpy)

        # enthalpy call for "i"
        print('\nEnthalpy call for "i" - Units: J/kmol :' + self.hspFuncNamePre + 'i' + self.hspFuncNamePost)

        # subGroup enthalpy
        print('\nSubGroup enthalpy: ' + self.subHname)
        print('Units: J/kmol')
        print(subHNames)

        # print enthalpy for each species
        print('\nEnthalpy for each species:')
        for i in range(self.gas.n_total_species):
            print((self.enthalpyCall[i]))

        print('-----------------------------------------------------')

        print('\nHeat of reaction: ' + self.assignHRxn)
        print('Units: J/kmol')
        for i in range(self.gas.n_reactions):
            print((self.heatOfReaction[i][:]))

        print('-----------------------------------------------------')

        # print heatRate
        print('Heat rate: '+ self.assignHeatRate + ' Units: J/m^3/s')
        for i in range(self.gas.n_reactions):
            print(heatRate[i,:])


    # def printExpression(expr,language,asignToName,gap=None):
    # Getting the expressions in various languages
    def getSystemExpression(self,gap=None):
        # call printExpression with appropriate arguments
        print('\n\n-----------------------------------------------------')
        print('SYSTEM EXPRESSIONS')
        print('-----------------------------------------------------\n')


        
        language = self.language

        # reacction rate constant for each reaction
        expr = self.eachReactionRateConstant
        assignTo = self.rateName
        print('Reaction rate constant:')
        self.printExpression(expr,language,assignTo,gap)
        print('-----------------------------------------------------')

        

        # show the rates for each reaction - rate of progress for each reaction
        assignTo = self.assignWrxn
        expr = self.eachReactionRate
        print('Reaction rates:')
        self.printExpression(expr,language,assignTo,gap)
        print('-----------------------------------------------------')


        # show the ODE system from omega
        assignTo = self.assignDdt
        expr = self.ode_systemFromOmega
        print('ODE system from omega:')
        self.printExpression(expr,language,assignTo,gap)
        print('-----------------------------------------------------')

        # show the Complete ODE system
        expr = self.ode_system
        assignTo = self.assignDdt+"_"
        print('Complete ODE system:')
        # for species rate
        self.printExpression(expr,language,assignTo,gap)
        print('-----------------------------------------------------')

        # show enthalpy for each species
        expr = self.enthalpyCall
        assignTo = self.assignEnthalpy
        print('Enthalpy for each species:')
        self.printExpression(expr,language,assignTo,gap)

        # show the enthalpy subgroup
        print('-----------------------------------------------------')
        print('!!!! Enthalpy subgroup - Update the enthalpy Implementation. This is for species',self.subHname)
        print(self.subHnames)
        print('-----------------------------------------------------')



        # show the heat of reaction
        expr = self.heatOfReaction[:,0]
        assignTo = self.assignHRxn+'[0]'
        print('Heat of reaction Tg:')
        self.printExpression(expr,language,assignTo,gap)
        print('-----------------------------------------------------')
        expr = self.heatOfReaction[:,1]
        assignTo = self.assignHRxn+'[1]'
        print('Heat of reaction Tv:')
        self.printExpression(expr,language,assignTo,gap)
        print('-----------------------------------------------------')
        expr = self.heatOfReaction[:,2]
        assignTo = self.assignHRxn+'[2]'
        print('Heat of reaction Te:')
        self.printExpression(expr,language,assignTo,gap)

        print('-----------------------------------------------------')
        # self.rateOfHeating
        expr = self.rateOfHeating[:,0]
        assignTo = self.assignHeatRate+'[0]'
        print('Rate of heating: J/m^3/s')
        self.printExpression(expr,language,assignTo,gap)
        print('-----------------------------------------------------')
        expr = self.rateOfHeating[:,1]
        assignTo = self.assignHeatRate+'[1]'
        print('Rate of heating: J/m^3/s')
        self.printExpression(expr,language,assignTo,gap)
        print('-----------------------------------------------------')
        expr = self.rateOfHeating[:,2]
        assignTo = self.assignHeatRate+'[2]'
        print('Rate of heating: J/m^3/s')
        self.printExpression(expr,language,assignTo,gap)
        print('-----------------------------------------------------')




    def getReactionRatesSym(self):
        
        # spName = Generic name of species : Mass fraction
        # rateName = Generic name of rate constants : K_rxn
        # mwName = Generic name of molecular weight : spMW
        # rhoName = Generic name of density : rho
        # NaName = Generic name of Avogadro's number : Na
        # NumName = Generic name of total number density : Num
        # mode = Mass density, molar density, number density, mass fraction , mole fraction, etc.

        # Returns : reac_stoic_sym: which is the rate of progression of individual reactions and units shold be number_denisty/s


        gas = self.gas
        spName = self.spName
        rateName = self.rateName
        mwName = self.mwName
        rhoName = self.rhoName
        NaName = self.NaName
        NumName = self.NumName
        mode = self.mode

        nrxn = gas.n_reactions
        # reactants = gas.reactant_stoich_coeffs() # stoichiometric coefficients for reactants
        # products = gas.product_stoich_coeffs() # stoichiometric coefficients for products

        reactants = gas.reactant_stoich_coeffs3 # stoichiometric coefficients for reactants
        products = gas.product_stoich_coeffs3 # stoichiometric coefficients for products

        # make integers for easier simplification
        reactants = reactants.astype(int)
        products = products.astype(int)

        sp_ys = [sp.Symbol(spName+'['+str(i)+']', positive=True) for i in range(gas.n_total_species)] 
        rates_list = [sp.Symbol(rateName+'['+str(i)+ ']') for i in range(nrxn)]
        spMW = [sp.Symbol(mwName+'['+str(i)+ ']') for i in range(gas.n_total_species)]
        rho, Na, Num = sp.symbols(rhoName+' '+NaName+' '+NumName, positive=True)

        # get the number denisty from the mode variable
        if mode == 'mass_density':
            # rho = sp.symbols(rhoName, positive=True)
            sp_list = [sp_ys[i] * Na / spMW[i] for i in range(gas.n_total_species)] # from mass density
        elif mode == 'molar_density':
            sp_list = [sp_ys[i] * Na  for i in range(gas.n_total_species)] # from molar density
        elif mode == 'number_density':
            sp_list = [sp_ys[i] for i in range(gas.n_total_species)] # from number density so same
        elif mode == 'mass_fraction':
            sp_list = [sp_ys[i] * rho * Na / spMW[i] for i in range(gas.n_total_species)] # from mass fraction
        elif mode == 'mole_fraction':
            sp_list = [sp_ys[i] * Num  for i in range(gas.n_total_species)] # from mole fraction
        else:
            raise ValueError(f'Invalid mode "{mode}"')

        # Use mass action symbollic, here reactants is a matrix of stoichiometric coefficients ( not symbolic)
        reac_stoic_sym = [sp_list ** reactants[:,i] for i in range(nrxn)]
        # product the elements row wise for reac_stoic
        reac_stoic_sym = rates_list * np.prod(reac_stoic_sym, axis=1)
        # Get rid of reactions that doesn't change the species/simplify the expressions
        reac_stoic_sym = [sp.simplify(i) for i in reac_stoic_sym]
        return reac_stoic_sym, sp_ys, rates_list, spMW, rho, Na, Num

    def getProductionRatesSym(self):
        # spName = Generic name of species : Mass fraction
        # rateName = Generic name of rate constants : K_rxn
        # mwName = Generic name of molecular weight : spMW
        # rhoName = Generic name of density : rho
        # NaName = Generic name of Avogadro's number : Na

        gas = self.gas
        mode = self.mode
        # products = gas.product_stoich_coeffs()
        # reactants = gas.reactant_stoich_coeffs()

        products = gas.product_stoich_coeffs3
        reactants = gas.reactant_stoich_coeffs3

        # make integers for easier simplification
        reactants = reactants.astype(int)
        products = products.astype(int)

        nrxn = gas.n_reactions

        eff_stoicN = products - reactants # normal
        eff_stoic = eff_stoicN.T # transpose
        

        # print(eff_stoicN)
        # print(eff_stoic)

        # print("Reactants")
        # print(reactants)
        # print("Products")
        # print(products)

        # # print eff_stoic
        if self.verbose == True:
            for i in range(nrxn):
                # reaction
                print('Reaction ',i,gas.reaction(i))
                # print('In Latex',i,sp.latex(gas.reaction(i)))
                # reactants , products and eff_stoic
                print('reactants: ',reactants[:,i])
                print('products: ',products[:,i])
                print('eff_stoic: ',eff_stoicN[:,i])


        # eff_stoic = eff_stoic.T
        # get the reaction rates
        # reac_stoic_sym = getReactionRatesSym(gas,spName,rateName,mwName,rhoName,NaName)
        reac_stoic_sym, sp_ys, rates_list, spMW, rho, Na, Num = self.getReactionRatesSym()


        # use self.assignWrxn to make symbols to hold reaction rates for individual reactions units: kmol/m^3/s
        # The expressions for omegaIRxn are in reac_stoic_sym
        omegaIRxn = [sp.Symbol(self.assignWrxn+'['+str(i)+']') for i in range(nrxn)]
        
        # find the rate for each species by adding contributions from all the reactions
        # matrix multiply reac_stoic_sym and eff_stoic
        # The results will be in number density rate of change
        wsp = np.matmul(reac_stoic_sym, eff_stoic)

        # # print reac_stoic_sym and eff_stoic
        # print("reac_stoic_sym")
        # print(reac_stoic_sym)
        # print("eff_stoic")
        # print(eff_stoic)
        # print("Reactants")
        # print(reactants)
        # print("Products")
        # print(products)

        # simplify the expressions
        wsp = [sp.simplify(i) for i in wsp]

        # new wsp with omegaIRxn as a symbol
        wspFromOmega = np.matmul(omegaIRxn, eff_stoic)

        # simplify the expressions
        wspFromOmega = [sp.simplify(i) for i in wspFromOmega]

        # use getModesFromNumDensity to get the rate in different modes for both wsp and wspFromOmega
        # getModesFromNumDensity(self,wsp,sp_ys,spMW,rho,Na,Num)
        wsp_ys = self.getModesFromNumDensity(wsp,sp_ys,spMW,rho,Na,Num)
        wspFromOmega_ys = self.getModesFromNumDensity(wspFromOmega,sp_ys,spMW,rho,Na,Num)

        


        # # given the number density rate change, get the rate in different modes
        # if mode == 'mass_density':
        #     # rho = sp.symbols(rhoName, positive=True)
        #         wsp_ys = [wsp[i]*spMW[i]/Na for i in range(gas.n_total_species)]
        # elif mode == 'molar_density':
        #         wsp_ys = [wsp[i]/Na for i in range(gas.n_total_species)]
        # elif mode == 'number_density':
        #         wsp_ys = [wsp[i] for i in range(gas.n_total_species)]
        # elif mode == 'mass_fraction':
        #     # now get the number density rate change to mass fraction rate change :  sp_n = sp_list*rho*6.023e23/spMW
        #     # wsp_ys = wsp_n/(rho*6.023e23/spMW)
        #     wsp_ys = [wsp[i]/(rho*Na/spMW[i]) for i in range(gas.n_total_species)]
        # elif mode == 'mole_fraction':
        #     # wsp_ys = wsp_n/(N_tot)
        #     wsp_ys = [wsp[i]/Num for i in range(gas.n_total_species)]
        # else:
        #     raise ValueError(f'Invalid mode "{mode}"')

        # simplify the expressions
        wsp_ys = [sp.simplify(i) for i in wsp_ys]
        wspFromOmega_ys = [sp.simplify(i) for i in wspFromOmega_ys]

        return wsp_ys, wspFromOmega_ys, reac_stoic_sym, omegaIRxn, sp_ys, rates_list, spMW, rho, Na, Num
        # return wsp_ys


    def getModesFromNumDensity(self,wsp,sp_ys,spMW,rho,Na,Num):
        gas = self.gas
        mode = self.mode

        # given the number density rate change, get the rate in different modes
        if mode == 'mass_density':
            # rho = sp.symbols(rhoName, positive=True)
                wsp_ys = [wsp[i]*spMW[i]/Na for i in range(gas.n_total_species)]
        elif mode == 'molar_density':
                wsp_ys = [wsp[i]/Na for i in range(gas.n_total_species)]
        elif mode == 'number_density':
                wsp_ys = [wsp[i] for i in range(gas.n_total_species)]
        elif mode == 'mass_fraction':
            # now get the number density rate change to mass fraction rate change :  sp_n = sp_list*rho*6.023e23/spMW
            # wsp_ys = wsp_n/(rho*6.023e23/spMW)
            wsp_ys = [wsp[i]/(rho*Na/spMW[i]) for i in range(gas.n_total_species)]
        elif mode == 'mole_fraction':
            # wsp_ys = wsp_n/(N_tot)
            wsp_ys = [wsp[i]/Num for i in range(gas.n_total_species)]
        else:
            raise ValueError(f'Invalid mode "{mode}"')

        return wsp_ys


    # for heat of reaction 
    ## - WITH FLOAT VALUES --------------------------------------------------------------------------------------------
    def getSpeciesEnthalpy(self, T, Te):
        """
        Returns the enthalpy of each species at the specified temperature and pressure.

        Parameters:
            - T (float): Temperature (in K)
            - Te (float): Electron temperature (in K)

        Returns:
            - h_species (list): List of enthalpies (in J/kmol) of each species at the specified temperature and pressure
        """
        # make an array to store the enthalpy of each species at the specified temperature and pressure
        h_species = [self.gas.species()[i].thermo.h(T) for i in range(self.gas.n_species)]
        # change the enthalpy of 'ele' from temperature of Te
        e_id = self.gas.species_index('ele')
        h_species[e_id] = self.gas.species()[e_id].thermo.h(Te)
        return h_species


    def getHeatRxn(self, T, Te):
        '''
        The deltaH for each reaction : units J/kmol
        '''

        gas = self.gas
        # calculate enthalpy of species
        h_species = self.getSpeciesEnthalpy(T, Te)
        # calculate the enthalpy of each reaction
        nrxn = gas.n_reactions
        h_rxns = np.zeros(nrxn)
        reactants = gas.reactant_stoich_coeffs() # stoichiometric coefficients for reactants
        products = gas.product_stoich_coeffs() # stoichiometric coefficients for products
        # change reactants and products to integer values
        reactants = reactants.astype(int)
        products = products.astype(int)
        for i in range(nrxn):
            h_reactants = [reactants[j,i]*h_species[j] for j in range(gas.n_species)]
            h_products = [products[j,i]*h_species[j] for j in range(gas.n_species)]
            h_rxns[i] = sum(h_products) - sum(h_reactants)
        return h_rxns



    def findEleRxns(self) -> list:
        """
        Find all the reactions with electron participation either in reactants or products.

        Args:
        - gas: Cantera Solution object.

        Returns:
        - ele_rxns: List of reaction IDs with electron participation.
        """
        gas = self.gas
        ele_rxns = []
        elID = gas.species_index('ele')
        for i in range(gas.n_reactions):
            if (gas.reactant_stoich_coeffs()[elID, i] != 0) or (gas.product_stoich_coeffs()[elID, i] != 0):
                ele_rxns.append(i)
        return ele_rxns



    def intermediateHrxn_eV(gas, hNet_eV, hRxn_eV, rxnID, actualDH_eVDict):
        """
        Updates hNet_eV and hRxn_eV for reactions with intermediately excited states.
    
        In these cases the amount of energy exchanged to a mode should be known, say DeltaH_g = -1 eV
        Then populate deltaH_g = -1 eV and deltaH_net = deltaH_net{previous} - deltaH_g
        
        Parameters:
        -----------
        gas: ct.Solution
            Cantera Solution object
        hNet_eV: np.ndarray
            Array of net reaction enthalpies in eV
        hRxn_eV: np.ndarray
            Array of individual reaction enthalpies in eV
        rxnID: np.ndarray
            Array of reaction IDs which has intermediately excited states or heat of reaction for a mode is known
        actualDH_eVDict: dict
            Dictionary of actual heat exchange for [Tg,Tv,Te] for reactions with intermediately excited states
        
        Returns:
        --------
        None
        """
        rxIDInterSps = rxnID
        for i in range(np.size(rxIDInterSps)):
            hNet_eV[rxIDInterSps[i]] -= np.sum(actualDH_eVDict[rxIDInterSps[i]])
            hRxn_eV[rxIDInterSps[i], :] = actualDH_eVDict[rxIDInterSps[i]]



    def remainingHrxn_eV(gas,ele_rxns, hNet_eV, hRxn_eV, rxIDSuperElastic, rxIDNetModes, ratioNetModesDict={}):
        """
        Updates the heat of reaction in electron volts (eV) for all modes using default algorithm.
        Any changes to default algorithm should already be made in hRxn_eV and corresponding arguments like rxIDSuperElastic
        should be provided accordingly. Input corresponding heat of reactions (hNet_eV) calcualted from [entahlpy sum(products) - sum(reactants)]

        Args:
        ele_rxns (list): List of electron reactions.
        hNet_eV (dict): Dictionary with the heat of reactions (in eV) .
        rxIDNetModes (dict): Dictionary containing the IDs of the electron reactions for which modes of energy transfer 
                            are already defined and their corresponding mode ratios. Defaults to an empty dictionary.
        rxIDSuperElastic (set): Set containing the IDs of the electron reactions that are superelastic. Defaults to 
                                an empty set.
        ratioNetModesDict (dict): Dictionary containing the mode ratios for the electron reactions for which 
                                rxIDNetModes is defined. Defaults to an empty dictionary.

        Returns:
        dict: Dictionary with the updated heat of reaction in eV for all energy modes.

        """

        nrxn = gas.n_reactions
        noEle_rxns = [i for i in range(nrxn) if i not in ele_rxns]

        # go through ele_rxns and update the heat of reaction
        for i in ele_rxns:
            if hNet_eV[i] > 0:
                # energy required ( endothermic reaction ) --> get from electron energy
                if i in rxIDNetModes:
                    # modes of transfer defined already, eg for e->v exchange, energy goes to v but gets subtracted from E
                    hRxn_eV[i,:] = [hNet_eV[i] * ratioNetModesDict[i][0], 
                                hNet_eV[i] * ratioNetModesDict[i][1], 
                                hNet_eV[i] * ratioNetModesDict[i][2]]
                else:
                    ## Check if electron is present in reactants. If it is then energy taken from electrons else from bulk gas
                    if gas.reactant_stoich_coeffs()[gas.species_index('ele'), i] == 0:
                        # energy taken from bulk gas
                        hRxn_eV[i,0] = hNet_eV[i] + hRxn_eV[i,0]
                    else:
                        # energy taken from electrons
                        hRxn_eV[i,2] = hNet_eV[i] + hRxn_eV[i,2]
            else:
                if i in rxIDNetModes:
                    # modes of transfer defined already, eg for v->e exchange , energy goes to E but gets subtracted from V
                    hRxn_eV[i,:] = [hNet_eV[i] * ratioNetModesDict[i][0], 
                                hNet_eV[i] * ratioNetModesDict[i][1], 
                                hNet_eV[i] * ratioNetModesDict[i][2]]
                # if super-elastic collision then go to E
                elif i in rxIDSuperElastic:
                    hRxn_eV[i,2] = hNet_eV[i] + hRxn_eV[i,2]
                else:
                    # energy produced ( exothermic reaction ) --> goes to bulk gas temperature
                    hRxn_eV[i,0] = hNet_eV[i] + hRxn_eV[i,0]

        # go through noEle_rxns and update the heat of reaction
        for i in noEle_rxns:
            # if modes transfer is not Tg but the ratio is known
            if i in rxIDNetModes:
                hRxn_eV[i,:] = [hNet_eV[i] * ratioNetModesDict[i][0], 
                            hNet_eV[i] * ratioNetModesDict[i][1], 
                            hNet_eV[i] * ratioNetModesDict[i][2]]
            else:
                hRxn_eV[i,0] = hNet_eV[i] + hRxn_eV[i,0]

        # return hRxn_eV

    # Now SYMBOLIC METHODS
    def intermediateHrxn_eV_Sym(self, hNet_eV, hRxn_eV, rxnID, actualDH_eVDict):
        """
        Updates hNet_eV and hRxn_eV for reactions with intermediately excited states.
    
        In these cases the amount of energy exchanged to a mode should be known, say DeltaH_g = -1 eV
        Then populate deltaH_g = -1 eV and deltaH_net = deltaH_net{previous} - deltaH_g
        
        Parameters:
        -----------
        gas: ct.Solution
            Cantera Solution object
        hNet_eV: np.ndarray
            Array of net reaction enthalpies in eV
        hRxn_eV: np.ndarray
            Array of individual reaction enthalpies in eV
        rxnID: np.ndarray
            Array of reaction IDs which has intermediately excited states or heat of reaction for a mode is known
        actualDH_eVDict: dict
            Dictionary of actual heat exchange for [Tg,Tv,Te] for reactions with intermediately excited states
        
        Returns:
        --------
        None
        """
        gas = self.gas
        for i in range(np.size(rxnID)):
            hNet_eV[rxnID[i]] -= np.sum(actualDH_eVDict[rxnID[i]])
            hRxn_eV[rxnID[i]] = [actualDH_eVDict[rxnID[i]][0],actualDH_eVDict[rxnID[i]][1],actualDH_eVDict[rxnID[i]][2]]



    def remaningHrxn_eV_Sym(self, hNet_eV_Sym, hRxn_eV_Sym,
                        hNet_eV, rxIDSuperElastic, rxIDNetModes, ratioNetModesDict={}):
        """
        Updates the heat of reaction in electron volts (eV) for all modes using default algorithm.
        Any changes to default algorithm should already be made in hRxn_eV and corresponding arguments like rxIDSuperElastic
        should be provided accordingly. Input corresponding heat of reactions (hNet_eV) calcualted from [entahlpy sum(products) - sum(reactants)]

        Args:
        ele_rxns (list): List of electron reactions.
        hNet_eV (dict): Dictionary with the heat of reactions (in eV) .
        rxIDNetModes (dict): Dictionary containing the IDs of the electron reactions for which modes of energy transfer 
                            are already defined and their corresponding mode ratios. Defaults to an empty dictionary.
        rxIDSuperElastic (set): Set containing the IDs of the electron reactions that are superelastic. Defaults to 
                                an empty set.
        ratioNetModesDict (dict): Dictionary containing the mode ratios for the electron reactions for which 
                                rxIDNetModes is defined. Defaults to an empty dictionary.

        Returns:
        dict: Dictionary with the updated heat of reaction in eV for all energy modes.

        """


        # # Initialize dictionary to store updated heat of reaction values
        # hRxn_eV = {}
        gas = self.gas
        nrxn = gas.n_reactions
        ele_rxns = self.findEleRxns()
        noEle_rxns = [i for i in range(nrxn) if i not in ele_rxns]

        # go through ele_rxns and update the heat of reaction
        for i in ele_rxns:
            if hNet_eV[i] > 0:
                # energy required ( endothermic reaction ) --> get from electron energy
                if i in rxIDNetModes:
                    # modes of transfer defined already, eg for e->v exchange, energy goes to v but gets subtracted from E
                    # hRxn_eV[i,:] = [hNet_eV[i] * ratioNetModesDict[i][0], 
                    #               hNet_eV[i] * ratioNetModesDict[i][1], 
                    #               hNet_eV[i] * ratioNetModesDict[i][2]]
                    
                    # symbolic
                    hRxn_eV_Sym[i] = [hNet_eV_Sym[i] * ratioNetModesDict[i][0],
                                        hNet_eV_Sym[i] * ratioNetModesDict[i][1],
                                        hNet_eV_Sym[i] * ratioNetModesDict[i][2]]

                else:
                    # hRxn_eV[i,2] = hNet_eV[i] + hRxn_eV[i,2]
                    if gas.reactant_stoich_coeffs()[gas.species_index('ele'), i] == 0:
                        # energy taken from bulk gas
                        hRxn_eV_Sym[i][0] = hNet_eV_Sym[i] + hRxn_eV_Sym[i][0]
                    else:
                        # energy taken from electrons
                        hRxn_eV_Sym[i][2] = hNet_eV_Sym[i] + hRxn_eV_Sym[i][2]                
                    # symbolic
                    # hRxn_eV_Sym[i][2] = hNet_eV_Sym[i] + hRxn_eV_Sym[i][2]
            else:
                if i in rxIDNetModes:
                    # modes of transfer defined already, eg for v->e exchange , energy goes to E but gets subtracted from V
                    # hRxn_eV[i,:] = [hNet_eV[i] * ratioNetModesDict[i][0], 
                    #               hNet_eV[i] * ratioNetModesDict[i][1], 
                    #               hNet_eV[i] * ratioNetModesDict[i][2]]

                    # symbolic
                    hRxn_eV_Sym[i] = [hNet_eV_Sym[i] * ratioNetModesDict[i][0],
                                        hNet_eV_Sym[i] * ratioNetModesDict[i][1],
                                        hNet_eV_Sym[i] * ratioNetModesDict[i][2]]
                # if super-elastic collision then go to E
                elif i in rxIDSuperElastic:
                    # hRxn_eV[i,2] = hNet_eV[i] + hRxn_eV[i,2]

                    # symbolic
                    hRxn_eV_Sym[i][2] = hNet_eV_Sym[i] + hRxn_eV_Sym[i][2]
                else:
                    # energy produced ( exothermic reaction ) --> goes to bulk gas temperature
                    # hRxn_eV[i,0] = hNet_eV[i] + hRxn_eV[i,0]

                    # symbolic
                    hRxn_eV_Sym[i][0] = hNet_eV_Sym[i] + hRxn_eV_Sym[i][0]

        # go through noEle_rxns and update the heat of reaction
        for i in noEle_rxns:
            # if modes transfer is not Tg but the ratio is known
            if i in rxIDNetModes:
                # hRxn_eV[i,:] = [hNet_eV[i] * ratioNetModesDict[i][0], 
                #               hNet_eV[i] * ratioNetModesDict[i][1], 
                #               hNet_eV[i] * ratioNetModesDict[i][2]]

                # symbolic
                hRxn_eV_Sym[i] = [hNet_eV_Sym[i] * ratioNetModesDict[i][0],
                                    hNet_eV_Sym[i] * ratioNetModesDict[i][1],
                                    hNet_eV_Sym[i] * ratioNetModesDict[i][2]]
            else:
                # hRxn_eV[i,0] = hNet_eV[i] + hRxn_eV[i,0]

                # symbolic
                hRxn_eV_Sym[i][0] = hNet_eV_Sym[i] + hRxn_eV_Sym[i][0]

        # return hRxn_eV


    ### Heat of reaction, symbolic, from provided energyExchange Dictionary
    ## Dict: type: zeros, ratios, valueseV, defaults
    def energyExchangeDict_Hrxn_eV_Sym(self, energyExchangeDict, hNet_eV, hNet_eV_Sym, hRxn_eV_Sym):
        """
        Updates the heat of reaction in electron volts (eV) for all modes based on different types of energy exchanges:
        1. zeros: No energy exchange
        2. ratios: Energy exchange ratios
        3. valueseV: Energy exchange values in eV
        4. defaults: Default energy exchange values in eV

        Input:
        energyExchangeDict: Dictionary with the energy exchange ids and corresponding energy exchange values
        hNet_eV: Float values of net heat of reaction for all reactions at STP for reference.
        hNet_eVSym: Dictionary with the net heat of reaction including all modes. This is a scalar for one reaction. eV per molecule
        hRxn_eVSym: Dictionary with the heat of reaction for all modes. This is a list of three elements for one reaction. eV per molecule
        """

        # extract the ids and vals
        rxnID = energyExchangeDict['ids']
        vals = energyExchangeDict['vals']
        nrxn = self.gas.n_reactions
        reactants = self.gas.reactant_stoich_coeffs().astype(int)
        products = self.gas.product_stoich_coeffs().astype(int)
        idReacEle = [i for i in range(nrxn) if reactants[self.gas.species_index('ele'), i] > 0]

        ## Now for all three modes float
        hRxn_modes = np.zeros((3, nrxn))

        # for zeros
        for i in rxnID['zeros']:
            hNet_eV_Sym[i] = 0.0
            hRxn_eV_Sym[i] = [0,0,0]

        # for valueseV
        for i in range(len(rxnID['valueseV'])):
            rxid = rxnID['valueseV'][i]
            # hNet_eV_Sym[rxid] = vals['valueseV'][i][0] + vals['valueseV'][i][1] + vals['valueseV'][i][2]
            hRxn_eV_Sym[rxid] = vals['valueseV'][i]

            # also update the floats
            hRxn_modes[0,rxid] = vals['valueseV'][i][0]
            hRxn_modes[1,rxid] = vals['valueseV'][i][1]
            hRxn_modes[2,rxid] = vals['valueseV'][i][2]


        # for ratios
        for i in range(len(rxnID['ratios'])):
            rxid = rxnID['ratios'][i]
            hRxn_eV_Sym[rxid] = [hNet_eV_Sym[rxid] * vals['ratios'][i][0],
                                    hNet_eV_Sym[rxid] * vals['ratios'][i][1],
                                    hNet_eV_Sym[rxid] * vals['ratios'][i][2]]
            
            # also update the floats
            hRxn_modes[0,rxid] = hNet_eV[rxid] * vals['ratios'][i][0]
            hRxn_modes[1,rxid] = hNet_eV[rxid] * vals['ratios'][i][1]
            hRxn_modes[2,rxid] = hNet_eV[rxid] * vals['ratios'][i][2]
            
        # now for defaults
        for i in rxnID['defaults']:
            # check if the reaction has electron in reactants
            diff = hNet_eV[i] - hRxn_modes[0,i] - hRxn_modes[1,i] - hRxn_modes[2,i]
            # symbolic diff
            diffSym = hNet_eV_Sym[i] - hRxn_eV_Sym[i][0] - hRxn_eV_Sym[i][1] - hRxn_eV_Sym[i][2]
            if diff > 0.0 and i in idReacEle:
                hRxn_modes[2,i] += diff
                hRxn_eV_Sym[i][2] += diffSym
            else:
                hRxn_modes[0,i] += diff
                hRxn_eV_Sym[i][0] += diffSym


    # returened : hRxn_eV_Sym in eV per molecule

    # Get the heat of reaction expressions
    def updateHrxnExpressions(self, extraFuncHrxn=None, energyExchangeDict=None):
        '''
        Update the heat of reaction expressions for all energy modes.

        Args:
        extraFuncHrxn (function): Function that provides necessary details like super elastic reactions, 
                                exchagne mode ratios, etc. for the heat of reaction expressions. Also specific reactions' net
                                heat of reaction can be modified if needed. Defaults to None.

        Returns:
        hRxn_Sym : list of sympy expressions for the heat of reaction for all energy modes in units J/kmol
        h_sp :  symbols used for enthalpy of species - J/kmol
        h_sp_call: how to call the function/member to get the enthalpy of species - J/kmol
        '''

        

        gas = self.gas
        nrxn = gas.n_reactions
        reactants = gas.reactant_stoich_coeffs().astype(int)
        products = gas.product_stoich_coeffs().astype(int)

        # h_sp = [sp.Symbol('h_sp['+str(i)+']') for i in range(gas.n_total_species)]

        # hspName = self.hspName
        # h_sp = [sp.Symbol(self.hspName[i]) for i in range(gas.n_total_species)]
        h_sp = [sp.Symbol(self.hspName+'['+str(i)+']') for i in range(gas.n_total_species)]
        h_sp_call = [sp.Symbol(self.hspFuncNamePre+str(i)+self.hspFuncNamePost) for i in range(gas.n_total_species)]
        elec_id = gas.species_index('ele')
        h_sp_call[elec_id] = sp.Symbol('(1.3806e-23*2.5)*(Te - 298.15)')

        
        # make a list to hold expression for heat of reaction for all modes and all reactions
        hNet_Sym = [] # J/kmol
        hNet_eVSym = [] # eV per molecule

        # make a list to hold expression for heat of reaction for all three modes and all reactions
        hRxn_eVTgSym = []
        hRxn_eVTvSym = []
        hRxn_eVTeSym = []

        

        # Now symbolic manipulations
        for i in range(nrxn):
            h_reactants = [reactants[j,i]*h_sp[j] for j in range(gas.n_total_species)]
            h_products = [products[j,i]*h_sp[j] for j in range(gas.n_total_species)]
            h_rxnI = np.sum(h_products) - np.sum(h_reactants)
            hNet_Sym.append(h_rxnI)

        # in eV
        for i in range(nrxn):
            hNet_eVSym.append(hNet_Sym[i]/6.022e26/1.602e-19)

        # Make all zeros first for heat of reaction for all three modes and all reactions
        for i in range(nrxn):
            hRxn_eVTgSym.append(0*hNet_eVSym[i])
            hRxn_eVTvSym.append(0*hNet_eVSym[i])
            hRxn_eVTeSym.append(0*hNet_eVSym[i])

        # make a list of list with all modes such that hRxn_eVSym[i] = [hRxn_eVTgSym[i], hRxn_eVTvSym[i], hRxn_eVTeSym[i]]
        # hRxn_eVSym = [i for i in zip(hRxn_eVTgSym,hRxn_eVTvSym,hRxn_eVTeSym)]
        hRxn_eVSym = [[hRxn_eVTgSym[i], hRxn_eVTvSym[i], hRxn_eVTeSym[i]] for i in range(nrxn)]

        # For Organizing reactions find the actual value at 298.15 K
        # set the temperature for the gas phase
        Tg = 300.0
        Tv = Tg
        Te = 300.0

        hNet = self.getHeatRxn(Tg,Te) # J/kmol
        hNet_eV = ODEBuilder.J_kmol2eV(hNet) # eV/molecule


        # Now the user defined function should modify the hNet_Sym if needed and provide the extra data needed
        # check if extraFuncHrxn is provided
        if extraFuncHrxn is not None:
            # call the user function to get the necessary details
            # This function takes J/kmol and outputs eV/molecule as well as J/kmol and all the numbers are for ev/molecule
            # rxIDSuperElastic, rxIDNetModes, ratioNetModesDict,rxIDInterSps, aDh_eVDict, subH_SymName
            rxIDSuperElastic, rxIDNetModes, ratioNetModesDict, rxIDInterSps, aDh_eVDict, subH_SymName =  extraFuncHrxn(gas, hNet_Sym, hNet_eVSym, h_sp, subHname=self.subHname)
            
             # hNet_Sym and hNet_eVSym both are updated in the extraFunction
            # Now categorize the reactions into different modes
            self.intermediateHrxn_eV_Sym(hNet_eVSym, hRxn_eVSym, rxIDInterSps, aDh_eVDict) # reaction with intermediates

            self.remaningHrxn_eV_Sym(hNet_eVSym, hRxn_eVSym, hNet_eV,
                                rxIDSuperElastic, rxIDNetModes, ratioNetModesDict)    # remaining reactions : super elastic, ratio between modes are known
        # else:
        #     # assume necessary details are empty
        #     rxIDSuperElastic = []
        #     rxIDNetModes = []
        #     ratioNetModesDict = {}
        #     rxIDInterSps = []
        #     aDh_eVDict = {}
        #     subH_SymName = []

        
       

        subH_SymName = []
        ## if energy exchangeDict is provided then update based on that
        if energyExchangeDict is not None:
            # use the dictionary to update the heat of reaction expressions
            self.energyExchangeDict_Hrxn_eV_Sym(energyExchangeDict, hNet_eV, hNet_eVSym, hRxn_eVSym)


        # the expressions for J/kmol now
        # hRxn_eVSym is already in eV/molecule
        hRxn_Sym = []
        # finally change the expressions back to J/kmol
        for i in range(nrxn):
            hRxn_Sym.append([hRxn_eVSym[i][j]*1.602e-19*6.022e26 for j in range(3)])
            # simplify the expressions
            hRxn_Sym[i] = [sp.simplify(hRxn_Sym[i][j],rational=True) for j in range(3)]
            # use float for constant numbers in the expressions
            hRxn_Sym[i] = [sp.N(hRxn_Sym[i][j],n=10) for j in range(3)] # correct to 10 decimal places

        # Now should return the hRxn_Sym and all the symbols either defined here or defined in functions called here
        # symbols from function call might include new enthalpy symbols for sub-gropu of species
        return hRxn_Sym, h_sp, h_sp_call,subH_SymName

    # # Calculate rate of heating for each reaction: J/m3/s
    # def getProductionHeatRate(self, Tg, Te, Tv,



    # START OF STATIC METHODS -- > these methods are not dependent on the class object
    
    @staticmethod
    def extraFuncHrxnN2_54(gas, hNet_Sym, hNet_eVSym, h_sp, subHname):
        """
        Update the heat of reaction for vibrationally distinguished species only for heat calculation, e.g. N2(A,v>9).
        Also provide extra parameters needed for other reactions.
        This modifies the hNet_Sym and hNet_eVSym and returns the extra parameters needed for other reactions along with 
            the symbols defined here which are the enthalpy of sub-group of species, e.g. N2(A,v>9).
        
        Parameters:
        gas: Cantera Gas object
        hNet_Sym: list of sympy symbols for the heat of reaction
        hNet_eVSym: list of sympy symbols for the heat of reaction in eV
        h_sp: list of enthalpies of the species in J/kmol
        subHname: optional string specifying the substring of the species names for which the heat of reaction needs to be updated
        
        Returns:
        rxIDSuperElastic, rxIDNetModes, ratioNetModesDict,rxIDInterSps, aDh_eVDict, subH_SymName
        
        """
        nrxn = gas.n_reactions
        # Find separate enethalpy for different groups [J/kmol] for N2(A)
        # id for N2(A)
        id_N2A = gas.species_index('N2_A')
        # make symbols for enthalpy of N2(A) for different vibrational groups: 0-4, 5-9, and greater than 9
        # hN2AG1 = sp.Symbol('h_N2A_vG1') # v = 0-4
        # hN2AG2 = sp.Symbol('h_N2A_vG2') # v = 5-9
        # hN2AG3 = sp.Symbol('h_N2A_vG3') # v > 9
        # use subHname to make the symbols
        hN2AG1 = sp.Symbol('h'+subHname+'_vG1') # v = 0-4
        hN2AG2 = sp.Symbol('h'+subHname+'_vG2') # v = 5-9
        hN2AG3 = sp.Symbol('h'+subHname+'_vG3') # v > 9

        # list with all extra symbols introduced
        subH_SymName = [hN2AG1, hN2AG2, hN2AG3]

        # energy at different vibrational levels in eV
        N2VibE_ev = np.array([0.43464128,0.71995946,1.00172153,1.27992582,1.55457065,1.82565433,2.09317519,2.35713153])

        # update heat of reaction for R44,45,46,47,50,51, id starts at 0 in python so subtract 1
        hNet_Sym[44-1] = hN2AG1 - h_sp[gas.species_index('N2')]
        hNet_Sym[45-1] = -hNet_Sym[44-1]
        hNet_Sym[46-1] = hN2AG2 - h_sp[gas.species_index('N2')]
        hNet_Sym[47-1] = -hNet_Sym[46-1]
        hNet_Sym[50-1] = hN2AG3 - h_sp[gas.species_index('N2')]
        hNet_Sym[51-1] = -hNet_Sym[50-1]

        # excitation energy per molecule from 0-v where v=[1,2,3,4,5,6,7,8]
        N2VibExE_ev = N2VibE_ev[1:] - N2VibE_ev[0] # subtract the ground state v= 0 energy
        N2vExcite = N2VibExE_ev*1.602e-19*6.022e26 # convert to J/kmol ( same unit as cantera thermo.h() function)

        # ##------------------------UPDATE THIS IN ENERGY RELEASE RATE LATER ------------------------------------
        # # update the heat of reaction for N2(v) excitation and deexcitation,
        # # Reactions: 42, 43 ;   id starts at 0 in python so subtract 1
        # # for rate of heat release these reactions need individual rate constants for different vibrational levels and also different energy of excitation
        # hNet_Sym[42-1] = N2vExcite[0] # needs energy
        # hNet_Sym[43-1] = -N2vExcite[0] # gives energy
        # ##------------------------UPDATE THIS IN ENERGY RELEASE RATE LATER ------------------------------------


        ### For reactions giving of photons, make hNet = 0.0 -----------------------------------------------
        # Such reactions are: 2,3,4,13,17 -- id starts at 0 in python so subtract 1
        ## Reactions which are handeled by energy exchange modes explicitly, eg, QEV,QET,QVT,QVE are also set to 0.0
        ## Such reactions are : 41,42,43,54 -- 
        rxIDPhotons = np.array([2,3,4,13,17,41,42,43,54]) - 1 # id starts at 0 in python so subtract 1
        for i in rxIDPhotons:
            hNet_Sym[i] = 0.0
        ## ------------------------------------------------------------------------------------


        # change to eV
        for i in range(nrxn):
            hNet_eVSym[i] = hNet_Sym[i]/6.022e26/1.602e-19

        # Now other reactions that need to be updated
        # 1. Super-elastic collisions , energy goes to E ---------------------------------------
        # Rxns : 45,47,49, 51, 53 
        rxIDSuperElastic = np.array([45,47,49,51,53]) - 1 # id starts at 0 in python so subtract 1
        # Some recombination reactions also increase electron energy, update them here
        rxIDRecomb = np.array([10,15,19]) - 1 # id starts at 0 in python so subtract 1
        rxIDSuperElastic = np.concatenate((rxIDSuperElastic,rxIDRecomb))
        ##------------------------------------------------------------------------------------

        # 2. Reactions where other than translational energy is exchanged ---------------------
        # ratio of exchange in different modes is known
        # Rxns: 5,6,25,26,42,43
        rxIDNetModes = np.array([5,6,25,26,42,43]) - 1 # id starts at 0 in python so subtract 1

        # define the ratios here [ T, V, E]
        ratioNetModes = np.zeros((len(rxIDNetModes),3))
        ratioNetModes[0,:] = [3.5/5.0, 1.5/5.0, 0] #5
        ratioNetModes[1,:] = [0, 1.0, 0] #6
        ratioNetModes[2,:] = [0, 1.0, 0] #25
        ratioNetModes[3,:] = [0, 1.0, 0] #26

        # for N2(v) excitation and deexcitation
        ratioNetModes[4,:] = [0, -1.0, 1.0] #Rxn 42, E -> V :: needs energy as net is positive so put 1 where energy loss happens
        ratioNetModes[5,:] = [0, -1.0, 1.0] #Rxn 43, V -> E :: net is negative here.
        # make a dictionary for ratioNetModes such that key is reaction id and value is the ratio
        ratioNetModesDict = dict(zip(rxIDNetModes,ratioNetModes))
        ##------------------------------------------------------------------------------------


        # 3. Reactions where intermediately excited states are involved ----------------------
        # In these cases the amount of energy exchanged to a mode should be known, say DeltaH_g = -1 eV
        # Then populate deltaH_g = -1 eV and deltaH_net = deltaH_net{previous} - deltaH_g
        rxIDInterSps = np.array([12,40]) - 1
        actualDH_eV = np.zeros((np.size(rxIDInterSps),3))
        actualDH_eV[0,:] = np.array([-3.5,0,0]) # for R12, gives actual heat exchange in Tg,Tv,Te, negative sign means energy is produced
        actualDH_eV[1,:] = np.array([-1.0,0,0]) # for R40, gives actual heat exchange in Tg,Tv,Te, negative sign means energy is produced
        # make dictionary for actualDH_eV such that key is reaction id and value is the actual heat exchange
        aDh_eVDict = dict(zip(rxIDInterSps,actualDH_eV))

        
        return rxIDSuperElastic, rxIDNetModes, ratioNetModesDict,rxIDInterSps, aDh_eVDict, subH_SymName
  
    @staticmethod
    def extraFuncHrxnN2_54_2(gas, hNet_Sym, hNet_eVSym, h_sp, subHname):
        """
        Recombination heating goes to the bulk gas - translational energy.
        Update the heat of reaction for vibrationally distinguished species only for heat calculation, e.g. N2(A,v>9).
        Also provide extra parameters needed for other reactions.
        This modifies the hNet_Sym and hNet_eVSym and returns the extra parameters needed for other reactions along with 
            the symbols defined here which are the enthalpy of sub-group of species, e.g. N2(A,v>9).
        
        Parameters:
        gas: Cantera Gas object
        hNet_Sym: list of sympy symbols for the heat of reaction
        hNet_eVSym: list of sympy symbols for the heat of reaction in eV
        h_sp: list of enthalpies of the species in J/kmol
        subHname: optional string specifying the substring of the species names for which the heat of reaction needs to be updated
        
        Returns:
        rxIDSuperElastic, rxIDNetModes, ratioNetModesDict,rxIDInterSps, aDh_eVDict, subH_SymName
        
        """
        nrxn = gas.n_reactions
        # Find separate enethalpy for different groups [J/kmol] for N2(A)
        # id for N2(A)
        id_N2A = gas.species_index('N2_A')
        # make symbols for enthalpy of N2(A) for different vibrational groups: 0-4, 5-9, and greater than 9
        # hN2AG1 = sp.Symbol('h_N2A_vG1') # v = 0-4
        # hN2AG2 = sp.Symbol('h_N2A_vG2') # v = 5-9
        # hN2AG3 = sp.Symbol('h_N2A_vG3') # v > 9
        # use subHname to make the symbols
        hN2AG1 = sp.Symbol('h'+subHname+'_vG1') # v = 0-4
        hN2AG2 = sp.Symbol('h'+subHname+'_vG2') # v = 5-9
        hN2AG3 = sp.Symbol('h'+subHname+'_vG3') # v > 9

        # list with all extra symbols introduced
        subH_SymName = [hN2AG1, hN2AG2, hN2AG3]

        # energy at different vibrational levels in eV
        N2VibE_ev = np.array([0.43464128,0.71995946,1.00172153,1.27992582,1.55457065,1.82565433,2.09317519,2.35713153])

        # update heat of reaction for R44,45,46,47,50,51, id starts at 0 in python so subtract 1
        hNet_Sym[44-1] = hN2AG1 - h_sp[gas.species_index('N2')]
        hNet_Sym[45-1] = -hNet_Sym[44-1]
        hNet_Sym[46-1] = hN2AG2 - h_sp[gas.species_index('N2')]
        hNet_Sym[47-1] = -hNet_Sym[46-1]
        hNet_Sym[50-1] = hN2AG3 - h_sp[gas.species_index('N2')]
        hNet_Sym[51-1] = -hNet_Sym[50-1]

        # excitation energy per molecule from 0-v where v=[1,2,3,4,5,6,7,8]
        N2VibExE_ev = N2VibE_ev[1:] - N2VibE_ev[0] # subtract the ground state v= 0 energy
        N2vExcite = N2VibExE_ev*1.602e-19*6.022e26 # convert to J/kmol ( same unit as cantera thermo.h() function)

        # ##------------------------UPDATE THIS IN ENERGY RELEASE RATE LATER ------------------------------------
        # # update the heat of reaction for N2(v) excitation and deexcitation,
        # # Reactions: 42, 43 ;   id starts at 0 in python so subtract 1
        # # for rate of heat release these reactions need individual rate constants for different vibrational levels and also different energy of excitation
        # hNet_Sym[42-1] = N2vExcite[0] # needs energy
        # hNet_Sym[43-1] = -N2vExcite[0] # gives energy
        # ##------------------------UPDATE THIS IN ENERGY RELEASE RATE LATER ------------------------------------


        ### For reactions giving of photons, make hNet = 0.0 -----------------------------------------------
        # Such reactions are: 2,3,4,13,17 -- id starts at 0 in python so subtract 1
        ## Reactions which are handeled by energy exchange modes explicitly, eg, QEV,QET,QVT,QVE are also set to 0.0
        ## Such reactions are : 41,42,43,54 -- 
        rxIDPhotons = np.array([2,3,4,13,17,41,42,43,54]) - 1 # id starts at 0 in python so subtract 1
        for i in rxIDPhotons:
            hNet_Sym[i] = 0.0
        ## ------------------------------------------------------------------------------------


        # change to eV
        for i in range(nrxn):
            hNet_eVSym[i] = hNet_Sym[i]/6.022e26/1.602e-19

        # Now other reactions that need to be updated
        # 1. Super-elastic collisions , energy goes to E ---------------------------------------
        # Rxns : 45,47,49, 51, 53 
        rxIDSuperElastic = np.array([45,47,49,51,53]) - 1 # id starts at 0 in python so subtract 1
        # Some recombination reactions also increase electron energy, update them here
        # rxIDRecomb = np.array([10,15,19]) - 1 # id starts at 0 in python so subtract 1
        # rxIDSuperElastic = np.concatenate((rxIDSuperElastic,rxIDRecomb))
        ##------------------------------------------------------------------------------------

        # 2. Reactions where other than translational energy is exchanged ---------------------
        # ratio of exchange in different modes is known
        # Rxns: 5,6,25,26,42,43
        rxIDNetModes = np.array([5,6,25,26,42,43]) - 1 # id starts at 0 in python so subtract 1

        # define the ratios here [ T, V, E]
        ratioNetModes = np.zeros((len(rxIDNetModes),3))
        ratioNetModes[0,:] = [3.5/5.0, 1.5/5.0, 0] #5
        ratioNetModes[1,:] = [0, 1.0, 0] #6
        ratioNetModes[2,:] = [0, 1.0, 0] #25
        ratioNetModes[3,:] = [0, 1.0, 0] #26

        # for N2(v) excitation and deexcitation
        ratioNetModes[4,:] = [0, -1.0, 1.0] #Rxn 42, E -> V :: needs energy as net is positive so put 1 where energy loss happens
        ratioNetModes[5,:] = [0, -1.0, 1.0] #Rxn 43, V -> E :: net is negative here.
        # make a dictionary for ratioNetModes such that key is reaction id and value is the ratio
        ratioNetModesDict = dict(zip(rxIDNetModes,ratioNetModes))
        ##------------------------------------------------------------------------------------


        # 3. Reactions where intermediately excited states are involved ----------------------
        # In these cases the amount of energy exchanged to a mode should be known, say DeltaH_g = -1 eV
        # Then populate deltaH_g = -1 eV and deltaH_net = deltaH_net{previous} - deltaH_g
        rxIDInterSps = np.array([12,40]) - 1
        actualDH_eV = np.zeros((np.size(rxIDInterSps),3))
        actualDH_eV[0,:] = np.array([-3.5,0,0]) # for R12, gives actual heat exchange in Tg,Tv,Te, negative sign means energy is produced
        actualDH_eV[1,:] = np.array([-1.0,0,0]) # for R40, gives actual heat exchange in Tg,Tv,Te, negative sign means energy is produced
        # make dictionary for actualDH_eV such that key is reaction id and value is the actual heat exchange
        aDh_eVDict = dict(zip(rxIDInterSps,actualDH_eV))

        
        return rxIDSuperElastic, rxIDNetModes, ratioNetModesDict,rxIDInterSps, aDh_eVDict, subH_SymName
  




    @staticmethod
    def NASA7Enthalpy(x,T,Tswitch=None):
        '''
        x has two rows and 7 columns
        First row is for T < Tswitch
        Second row is for T > Tswitch
        Enthalpy in units of J/mol
        '''
        Ru = 8.31446261815324
        if Tswitch is not None:
            dH = np.zeros(len(T))
            for i in range(len(dH)):
                if T[i] < Tswitch:
                    dH[i] = x[0,0] + x[0,1]/2*T[i] + x[0,2]/3*T[i]**2 + x[0,3]/4*T[i]**3 + x[0,4]/5*T[i]**4 + x[0,5]/T[i]
                else:
                    dH[i] = x[1,0] + x[1,1]/2*T[i] + x[1,2]/3*T[i]**2 + x[1,3]/4*T[i]**3 + x[1,4]/5*T[i]**4 + x[1,5]/T[i]
        else:
            dH = x[0] + x[1]/2*T + x[2]/3*T**2 + x[3]/4*T**3 + x[4]/5*T**4 + x[5]/T
        dH = dH*T*Ru
        return dH

    @staticmethod
    def J_kmol2eV(h_rxns):
        '''
        Convert the heat of reaction from J/kmol to eV/molecule

        Note: ct.avogadro = 6.022140857e+26 so h_rxns should be is in J/kmol
        '''
        h_ev = h_rxns/ct.avogadro/ct.electron_charge
        return h_ev


    @staticmethod
    # create a method which takes input Tg, Tv and Te and gives the RateConstants in SI units
    def getN2PlasmaKrxn(TgName,TvName,TeName):
        '''
        This function takes input Tg, Tv and Te and gives the RateConstants in SI units
        e.g: for $ e^- + N_2^+ \rightarrow 2 N $ reaction : Krxn = 2.8e-7*(300.0/Tg)**0.5

        This is only for the N2 plasma kinetics and the inputs are symbolic variables with units in Kelvin
        '''

        # make sympy symbols for Tg, Tv and Te
        Tg = sp.symbols(TgName, positive=True)
        Tv = sp.symbols(TvName, positive=True)
        Te = sp.symbols(TeName, positive=True)


        # make the rateConstant variables with 54 reactions
        Krxn = []
        rxnTot = 54


        # Reference data were in units of cm , s and per particle and need conversion to m, s, particles
        fac1 = 1.0 # for one reactant system - 1/s
        fac2 = 1.0e-6 # for two reactant system - cm3/s
        fac3 = 1.0e-12 # for three reactant system - cm6/s
        Na = 6.022e23 # Avogadro's number


        # # To check the expressions
        # fac1 = 1
        # fac2 = 1
        # fac3 = 1

        # make Na symbol
        # Na = sp.symbols('Na', positive=True)


        with sp.evaluate(False):

            # - equation: 2 N + N2 => N2_B + N2  # Reaction 1
            #   8:27 × 10−34 exp  500 Tg  - cm6/s
            K1 = fac3*8.27e-34 * sp.exp(500/Tg)

            # - equation: N2_A => N2  # Reaction 2
            #  0:5 s-1
            K2 = fac1*0.5

            # - equation: N2_B => N2_A  # Reaction 3
            #  1:52 × 105 s-1
            K3 = fac1*1.52e5


            # - equation: N2_C => N2_B  # Reaction 4
            #   2:69 × 107 s-1
            K4 = fac1*2.69e7


            # - equation: 2 N2_A => N2_B + N2  # Reaction 5
            #   2:9 × 10−9q 300 Tg cm3 s−1
            K5 = fac2*2.9e-9*(Tg/300)**0.5


            # - equation: 2 N2_A => N2_C + N2  # Reaction 6
            #  2:6 × 10−10q 300 Tg cm3 s−1
            K6 = fac2*2.6e-10*(Tg/300)**0.5


            # - equation: ele + N4p => N2_A + N2  # Reaction 7
            #   2:5 % × 2:4 × 10−6q 300 cm3 s−1
            K7 = fac2*2.5e-2*2.4e-6*(300/Te)**0.5


            # - equation: ele + N4p => N2_B + N2  # Reaction 8
            #   87 % × 2:4 × 10−6q 300 cm3 s−1
            K8 = fac2*87e-2*2.4e-6*(300/Te)**0.5


            # - equation: ele + N4p => N2_C + N2  # Reaction 9
            #   11 % × 2:4 × 10−6q 300 cm3 s−1
            K9 = fac2*11e-2*2.4e-6*(300/Te)**0.5


            # - equation: 2 ele + N4p => ele + 2 N2  # Reaction 10
            #   7 × 10−20  300 Te 4:5 cm3 s−1
            K10 = fac3*7e-20*(300/Te)**4.5


            # - equation: ele + N3p => N + N2  # Reaction 11
            #   2:0 × 10−7q 300 cm3 s−1
            K11 = fac2*2.0e-7*(300/Te)**0.5


            # - equation: ele + N2p => 2 N  # Reaction 12
            #   2:8 × 10−7q 300 cm3 s−1
            K12 = fac2*2.8e-7*(300/Te)**0.5


            # - equation: ele + N2p => N2  # Reaction 13
            #   4 × 10−12  300 Te 0:7 cm3 s−1
            K13 = fac2*4e-12*(300/Te)**0.7

            # - equation: ele + N2p + N2 => 2 N2  # Reaction 14
            #   6 × 10−27  300 Te 1:5 cm6 s−1
            K14 = fac3*6e-27*(300/Te)**1.5

            # - equation: 2 ele + N2p => ele + N2  # Reaction 15
            #   1 × 10−19  300 Te 4:5 cm6 s−1
            K15 = fac3*1e-19*(300/Te)**4.5

            # - equation: ele + N2 => 2 ele + N2p  # Reaction 16
            #   5:05 × 10−11 pTe+1:10 × 10−5Te1:5 / exp  1:82T×e 105  cm3 s−1
            K16 = fac2*5.05e-11*(Te**0.5+1.10e-5*Te**1.5)/sp.exp(1.82e5/Te)


            # - equation: ele + Np => N  # Reaction 17
            #   3:5 × 10−12  300 Te 0:7 cm3 s−1
            K17 = fac2*3.5e-12*(300/Te)**0.7

            # - equation: ele + Np + N2 => N + N2  # Reaction 18
            #   6 × 10−27  300 Te 1:5 cm6 s−1
            K18 = fac3*6e-27*(300/Te)**1.5

            # - equation: 2 ele + Np => ele + N  # Reaction 19
            #  1 × 10−19  300 Te 4:5 cm6 s−1
            K19 = fac3*1e-19*(300/Te)**4.5

            # - equation: N3p + N2_A => Np + 2 N2  # Reaction 20
            #  6 × 10−10 cm3 s−1
            K20 = fac2*6e-10

            # - equation: N2p + N2_A => N3p + N  # Reaction 21
            #   3 × 10−10 cm3 s−1
            K21 = fac2*3e-10

            # - equation: N2p + N2_A => Np + N + N2  # Reaction 22
            #   4 × 10−10 cm3 s−1
            K22 = fac2*4e-10

            # - equation: N2_A + N2 => 2 N2  # Reaction 23
            #   2:0 × 10−17 cm3 s−1
            K23 = fac2*2.0e-17

            # - equation: N + N2_A => N + N2  # Reaction 24
            #   6:2 × 10−11  300 Tg 2=3 cm3 s−1
            K24 = fac2*6.2e-11*(300/Tg)**(2/3)

            # - equation: N2_B + N2 => N2_A + N2  # Reaction 25
            #   1:2 × 10−11 cm3 s−1
            K25 = fac2*1.2e-11

            # - equation: N2_C + N2 => N2_B + N2  # Reaction 26
            #  1:2 × 10−11  300 Tg 0:33 cm3 s−1
            K26 = fac2*1.2e-11*(300/Tg)**0.33

            # - equation: N4p + N2 => N2p + 2 N2  # Reaction 27
            #   2:1 × 10−16 exp  121 Tg  cm3 s−1
            K27 = fac2*2.1e-16*sp.exp(Tg/121)

            # - equation: N4p + N => Np + 2 N2  # Reaction 28
            #  1 × 10−11 cm3 s−1
            K28 = fac2*1e-11

            # - equation: N4p + N => N3p + N2  # Reaction 29
            #   1 × 10−9 cm3 s−1
            K29 = fac2*1e-9

            # - equation: N3p + N => N2p + N2  # Reaction 30
            #  6:6 × 10−11 cm3 s−1
            K30 = fac2*6.6e-11

            # - equation: N3p + N2 => Np + 2 N2  # Reaction 31
            #  6 × 10−10 cm3 s−1
            K31 = fac2*6e-10

            # - equation: N2p + N2 => Np + N + N2  # Reaction 32
            #  1:2 × 10−11 cm3 s−1
            K32 = fac2*1.2e-11

            # - equation: N2p + N2 => N3p + N  # Reaction 33
            #   5:5 × 10−12 cm3 s−1
            K33 = fac2*5.5e-12

            # - equation: N2p + N => Np + N2  # Reaction 34
            #   7:2 × 10−13 exp  300 Tg  cm3 s−1
            K34 = fac2*7.2e-13*sp.exp(300/Tg)

            # - equation: N2p + 2 N2 => N4p + N2  # Reaction 35
            #  6:8 × 10−29 300 Tg 1:64 cm6 s−1
            K35 = fac3*6.8e-29*(300/Tg)**1.64

            # - equation: N2p + N + N2 => N3p + N2  # Reaction 36
            #   0:9 × 10−29 exp  400 Tg  cm6 s−1
            K36 = fac3*0.9e-29*sp.exp(400/Tg)

            # - equation: Np + N2 => N2p + N  # Reaction 37
            #  1 × 10−13 cm3 s−1
            K37 = fac2*1e-13

            # - equation: Np + 2 N2 => N3p + N2  # Reaction 38
            #  2:0 × 10−29 300 Tg 2:0 cm6 s−1
            K38 = fac3*2.0e-29*(300/Tg)**2.0

            # - equation: Np + N + N2 => N2p + N2  # Reaction 39
            #  1 × 10−29 300 Tg  cm6 s−1
            K39 = fac3*1e-29*(300/Tg)


            # - equation: ele + N2 => ele + 2 N  # Reaction 40
            #   1:2 × 1025 / NA Te1:6 exp − 113 200 Te  cm3 s−1 : Na = avogadro constant
            K40 = fac2*1.2e25/(Na*Te**1.6)*sp.exp(-113200/Te)

            # - equation: ele + N2 => ele + N2  # Reaction 41
            #   8:25 × 10−8 cm3 s−1
            K41 = fac2*8.25e-8

        
            # - equation: ele + N2 => ele + N2  # Reaction 42
            #   8:45 × 10−9 cm3 s−1
            K42 = fac2*8.45e-9

            # - equation: ele + N2 => ele + N2  # Reaction 43
            #   2:69 × 10−11 cm3 s−1
            K43 = fac2*2.69e-11

            # - equation: ele + N2 => ele + N2_A  # Reaction 44
            #   1:59 × 10−12 cm3 s−1
            K44 = fac2*1.59e-12


            # - equation: ele + N2_A => ele + N2  # Reaction 45
            #   2:67 × 10−11 cm3 s−1
            K45 = fac2*2.67e-11

            # - equation: ele + N2 => ele + N2_A  # Reaction 46
            #   5:63 × 10−12 cm3 s−1
            K46 = fac2*5.63e-12

            # - equation: ele + N2_A => ele + N2  # Reaction 47
            #   1:80 × 10−10 cm3 s−1
            K47 = fac2*1.80e-10


            # - equation: ele + N2 => ele + N2_B  # Reaction 48
            #   2:35 × 10−11 cm3 s−1
            K48 = fac2*2.35e-11

            # - equation: ele + N2_B => ele + N2  # Reaction 49
            #    9:85 × 10−10 cm3 s−1
            K49 = fac2*9.85e-10

            # - equation: ele + N2 => ele + N2_A  # Reaction 50
            #   4:06 × 10−12 cm3 s−1
            K50 = fac2*4.06e-12

            # - equation: ele + N2_A => ele + N2  # Reaction 51
            #   2:15 × 10−10 cm3 s−1
            K51 = fac2*2.15e-10

            # - equation: ele + N2 => ele + N2_C  # Reaction 52
            #    6:18 × 10−12 cm3 s−1
            K52 = fac2*6.18e-12

            # - equation: ele + N2_C => ele + N2  # Reaction 53
            #    4:06 × 10−9 cm3 s−1
            K53 = fac2*4.06e-9

            # - equation: N2 + N2 => 2 N2  # Reaction 54
            #   3:05 × 10−21 cm3 s−1
            K54 = fac2*3.05e-21

        # Put all in the list
        for i in range(rxnTot):
            Krxn.append(locals()["K"+str(i+1)])
        # print("Krxn: ", Krxn)

        # return Krxn along with the symbols for Temperature used in the rate constants
        # The symbols are required for substution of variables later on
        return Krxn, Tg, Tv, Te

    @staticmethod
    # Define a method to do substution of temperatures for reaction rate constants
    def getKrxnFloat(Krxn, Tg, Tv, Te, TgVal, TvVal, TeVal):
        KrxnFloat = []
        for i in range(len(Krxn)):
            # if sympy expression is not float, then substitute the values of Tg, Tv, Te
            if not isinstance(Krxn[i], float):
                # print("Krxn[i]: ", Krxn[i])
                KrxnFloat.append(Krxn[i].subs([(Tg, TgVal), (Tv, TvVal), (Te, TeVal)]))
            else:
                KrxnFloat.append(Krxn[i])
        return KrxnFloat

    
            
    @staticmethod
    # Define a symbolic function that prints the given expression in the chosen programming language with a name to assign to the expression
    # The function takes input: symbolic expression, language, asignToName
    def printExpression(expr,language,asignToName,gap=None,fname=None):
        # the languages to choose are: CXX, julia, python, matlab and fortran
        # reac_stoic_sym[0:1] to call for just one element array
        

        # expr = self.ode_system
        # find the number of elements in the expression
        n = len(expr)

        # if gap then a newline after each print
        if gap != None:
            nline = '\n'
        else:
            nline = ''

        Fnline = '\n'

        # make a folder named 'OUT' to store the output files if not already present
        import os
        if not os.path.exists('OUT'):
            os.makedirs('OUT')

        # default file name
        if fname == None:
            # fname = 'OUT_'+asignToName+language[0:3]+'.txt'
            fname = 'OUT/'+asignToName+language[0:3]+'.txt'

        # open the file
        with open(fname, 'w') as f:
            # CXX
            if language == 'CXX':
                from sympy.printing.cxx import cxxcode
                for i in range(n):

                    # find if the expression is just a float
                    if (isinstance(expr[i], sp.Float) or isinstance(expr[i], float) or isinstance(expr[i], int)):
                        # print the expression as a float
                        to_print = asignToName+'['+str(i)+'] = '+str(expr[i])+';'
                    else:
                        to_print = (cxxcode(expr[i], assign_to=asignToName+'['+str(i)+']', standard='C++11'))
                        
                    to_print = to_print.replace('std::','Foam::')
                    print(to_print+nline)
                    # now write to file
                    f.write(to_print+Fnline)
                    
            # Julia
            elif language == 'julia':
                from sympy.printing.julia import julia_code
                for i in range(n):
                    print(julia_code(expr[i], assign_to=asignToName+'['+str(i)+']')+nline)
                    # print(nline)
                    f.write(julia_code(expr[i], assign_to=asignToName+'['+str(i)+']')+Fnline)
            # Python
            elif language == 'python':
                from sympy.printing.pycode import pycode
                from sympy.codegen.ast import Assignment
                for i in range(n):
                    # to assign to a variable
                    symi = sp.Symbol(asignToName+'['+str(i)+']')
                    # assignment operation
                    oper = Assignment(symi, expr[i])
                    # print the expression
                    print(pycode(oper)+nline)
                    # print(pycode(expr[i], assign_to=asignToName+'['+str(i)+']'))
                    # print(nline)
                    f.write(pycode(oper)+Fnline)
            # Octave/Matlab
            elif language == 'matlab':
                from sympy.printing.octave import octave_code
                for i in range(n):
                    print(octave_code(expr[i], assign_to=asignToName+'['+str(i)+']')+nline)
                    # print(nline)
                    f.write(octave_code(expr[i], assign_to=asignToName+'['+str(i)+']')+Fnline)
            # Fortran
            elif language == 'fortran':
                from sympy.printing.fcode import fcode
                for i in range(n):
                    print(fcode(expr[i], assign_to=asignToName+'['+str(i)+']', standard=95)+nline)
                    # print(nline)
                    f.write(fcode(expr[i], assign_to=asignToName+'['+str(i)+']', standard=95)+Fnline)
            else:
                print('Language not supported')
                print('Choose from: CXX, julia, python, matlab and fortran')





def J_kmol2eV(h_rxns):
    '''
    Convert the heat of reaction from J/kmol to eV/molecule

    Note: ct.avogadro = 6.022140857e+26 so h_rxns should be is in J/kmol
    '''
    h_ev = h_rxns/ct.avogadro/ct.electron_charge
    return h_ev


# Define a test function to run : plasmaN2.cti should be present in the same folder
def test_chem():

    # make an object of the class with a mechanism file
    system = ODEBuilder('plasmaN2_Te_Test.yaml') ## gas = ct.Solution('plasmaN2 plasmaN2_Te_Test.yaml') ## ## plasmaN2_Te_Test.yaml plasmaN2.yaml
    system.mode = 'number_density'
    system.language = 'python'

    # update ODE System
    system.updateODESystem()

    # get the expressions
    system.getSystemExpression()

    # look at allPlasmaN2Solve.py on how to use this directly in python with substution

# Define a test function to run the heat of reaction calculation
def test_Hrxn():

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
    system = ODEBuilder('plasmaN2.yaml')
    system.mode = 'number_density'
    # system.mode = 'mass_fraction'
    system.language = 'python'
    system.subHname = 'N2A'

    # # check the heat of reaction using system.updateHrxnExpressions(self,ODEBuilder.extraFuncHrxnN2_54)
    # hRxn_Sym, h_sp, h_sp_call = system.updateHrxnExpressions(ODEBuilder.extraFuncHrxnN2_54)

    # update ODE System
    system.updateODESystem(reactionRateConstant=ODEBuilder.getN2PlasmaKrxn,extraFuncHrxn=ODEBuilder.extraFuncHrxnN2_54)

    # get the expressions
    system.getSystemExpression()


    # # hRxn_Sym, h_sp, h_sp_call = system.updateHrxnExpressions() # none default

    # # separate the three modes
    # # hRxn_Sym_Tg = [hRxn_Sym[i][0] for i in range(len(hRxn_Sym))]
    # # hRxn_Sym_Tv = [hRxn_Sym[i][1] for i in range(len(hRxn_Sym))]
    # # hRxn_Sym_Te = [hRxn_Sym[i][2] for i in range(len(hRxn_Sym))]

    # # change list to array, hRxn_Sym
    # hRxn_Sym = np.array(hRxn_Sym)
    # hRxn_Sym_Tg = hRxn_Sym[:,0]
    # hRxn_Sym_Tv = hRxn_Sym[:,1]
    # hRxn_Sym_Te = hRxn_Sym[:,2]

    # # print the expressions
    # system.printExpression(hRxn_Sym_Tg,system.language,'hRxn_SymTg')
    # system.printExpression(hRxn_Sym_Tv,system.language,'hRxn_SymTv')
    # system.printExpression(hRxn_Sym_Te,system.language,'hRxn_SymTe')


if __name__ == '__main__':
    # test_chem()
    test_Hrxn()