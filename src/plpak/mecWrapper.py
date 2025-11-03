"""
---------------------------------------------------------------------------------------------
Wrapper for Reading and Handling Cantera Mechanism Files with Extra Attributes

Description:
    This module provides a wrapper to read and handle Cantera mechanism files containing extra attributes that are not natively supported by Cantera.
    The wrapper uses a YAML reader, which can work with either ruamel.yaml or yaml library.

Functionality:
- Read and load Cantera mechanism files using the YAML reader.
- Create a Cantera gas object that combines native Cantera attributes and the extra attributes from the mechanism file.
- Implement units conversion for rates to ensure system consistency.
- Offer both compiled and symbolic expressions for rates.
- Ensure variable consistency between the mechanism file and the wrapper for symbolic expressions, with optional variable name changes in the mechanism file if required.
- Provide a method to calculate/evaluate float values of rates based on the system state (e.g., temperatures, pressure, etc.).

Requirements:
- Cantera library for handling mechanism files and gas objects.
- ruamel.yaml or yaml library for YAML parsing and loading.

Author : Sagar Pokharel ; pokharel_sagar@tamu.edu // https://github.com/ptroyen
Date  : 2023-07-24
---------------------------------------------------------------------------------------------
"""


import cantera as ct
import numpy as np
import sympy as sp
from abc import ABC, abstractmethod



# add current directory to the path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the plasma kinetics library
from makePlasmaKinetics import *
# from makePlasmaKinetics import J_kmol2eV
# from math import *
import math
# import the base class
from plasmaKineticSystem import System, NASA7Enthalpy


try:
    from ruamel.yaml import YAML
except ImportError:
    import yaml as YAML

# define alias for bolsigFit
bolsigFit = System.bolsigFit # input (energy in eV,parms) get rate in m3/s

# make custom object like gas ( solution object) in cantera based on the new mechanism file - called PlasmaFluid   
class PlasmaMechanism:
    '''
    Description:
        Cantera mechanism built solution object +  additional details for the plasma model implementation.
        The file includes reaction expressions that might depend on temperatures (Tg, Tv, Te) and other variables as required.
        The class provides functionality to read and handle the new attributes, 'newAttributeSet' and 'newAttribute', present in the mechanism file.

    Attributes:
    - newAttributeSet: Specifies the default units for new attributes (e.g., unit: cm, m).
    - reactions: Contains a list of reactions with their corresponding rate constants and type.

    Examples of New Attribute Usage:
    - The 'newAttribute' property can include expressions with different temperatures.
    - If units are omitted, default units defined in 'newAttributeSet' are assumed.
    - Different units and expressions can be specified, overriding default units.

    Backward Compatibility:
    The mechanism file is backward compatible, allowing Cantera to read it while skipping the new attributes.
    The 'rate-constant' attribute, although not used, must be provided with arbitrary values as required by Cantera.

    Example mechanism file:
    #-------------------------------------------------------------

    newAttributeSet:
        unit: cm
        vars: {T: 'T',Tv: 'Tv', Te: 'Te'}

    reactions:
    - equation: N2 + O2 => 2 NO
        rate-constant: {A: 1.35E+16, b: -0.8, Ea: 104300}
        type: elementary
        newAttribute: (1.0e-14*Te**0.5)
    - equation: 2 NO => N2 + O2
        rate-constant: {A: 3.15E+14, b: 0.5, Ea: 0}
        type: elementary
        newAttribute: 1.0e-6
    - equation: 2 NO + O2 => N2 + 2 O2
        rate-constant: {A: 3.15E+14, b: 0.5, Ea: 0}
        type: elementary
        newAttribute: {unit: cm, K: (1.0e-12)}

    #-------------------------------------------------------------
    '''

    def __init__(self,mech,verbose=False):
        '''
        Initialize the PlasmaFluid object.
        '''

        # member variables
        self.mechFile = mech
        self.verbose = verbose
        self.gas_ct = ct.Solution(self.mechFile)
        self.nR = self.gas_ct.n_reactions
        self.reactants = []
        self.products = []

        # Find IDs with Bolsig expressions
        self.bolsigIDs = []
        self.bolsigExpressions = []




        # load the custom mechanism
        self.com_expressions, self.sym_expressions, self.syms, self.energyExchangeIDs, self.energyExchangeVals = self.load_custom_mechanism()


    def load_custom_mechanism(self):

        mech_filename = self.mechFile
        yaml = YAML(typ='safe')
        with open(mech_filename, "r") as file:
            mechanism_data = yaml.load(file)

        # gas = ct.Solution(mech_filename)
        gas = self.gas_ct

        # find the reactants and products stoichiometrc coefficients from gas

        # reactants = gas.reactant_stoich_coeffs() # stoichiometric coefficients for reactants
        # products = gas.product_stoich_coeffs() # stoichiometric coefficients for products

        reactants = gas.reactant_stoich_coeffs3 # stoichiometric coefficients for reactants
        products = gas.product_stoich_coeffs3 # stoichiometric coefficients for products

        # change reactants and products to integer values
        reactants = reactants.astype(int)
        products = products.astype(int)

        self.reactants = reactants
        self.products = products

        # # show reactants and products
        # print("Reactants: ", reactants)
        # print("Products: ", products)

        # print the reactions and the stoichiometric coefficients
        for i in range(gas.n_reactions):
            reaction = gas.reaction(i)
            print(f"Reaction {i+1}: {reaction.equation}")
            print(f"Reactants: {reactants[:,i]}")
            print(f"Products: {products[:,i]}")
            
        # store expressions here
        new_attributes = [] 
        kevals = []

        cm2m = 1e-2 # conversion factor from cm to m
        m2cm = 1e2 # conversion factor from m to cm

        """
        newAttributeSet:
            unit: cm
            vars: {Tg: 'T',Tv: 'Tv', Te: 'Te'}
        """
        ## settings for newAttribute
        # print(mechanism_data['newAttributeSet']) 
        # settings = mechanism_data['newAttributeSet']
        # if newAttributeSet is not specified, then use default values
        if 'newAttributeSet' in mechanism_data:
            print("newAttributeSet found in mechanism file.")
            settings = mechanism_data['newAttributeSet']
        else:
            print("newAttributeSet not found in mechanism file. Using default values.")
            settings = {'unit': 'm', 'vars': {'T': 'T', 'Tv': 'Tv', 'Te': 'Te'}}
        
        if 'unit' in settings:
            newUnits = settings['unit']
        else:
            newUnits = 'm'
            # message
            print("No units specified for newAttribute. Default units assumed: m")

        if 'vars' in settings:
            newVars = settings['vars']
        else:
            newVars = {'T': 'T', 'Tv': 'Tv', 'Te': 'Te'}
            # message
            print("No variables specified for newAttribute. Default variables assumed: T, Tv, Te")


        # loop over the reactions and get the newAttribute for each reaction
        for reaction_data in mechanism_data['reactions']:

            # if newAttribute is not specified or empty, then use default value of 0
            if 'newAttribute' not in reaction_data or reaction_data['newAttribute'] == None or reaction_data['newAttribute'] == '':
                # new_attribute = '0.0'
                # update to include arrheneius expression from Cantera - generic reactions
                # if new attribute is not present then assume it is an Arrhenius expression
                ar_A  = reaction_data['rate-constant']['A'] # as written in the mechanism file, does not know the units
                ar_E  = reaction_data['rate-constant']['Ea'] # as written in the mechanism file, does not know the units
                ar_n  = reaction_data['rate-constant']['b']

                print(f"A={ar_A}, E={ar_E}, n={ar_n}")

                # now make the newAttribute, need to change to SI units
                # new_attribute['unit'] = 'm'
                # make an expressioon using float values of A, E, n
                # units are not checked and not corrected. correct it here.
                express_ = str(ar_A) + '*' + 'T**' + str(ar_n) + '*' + 'exp(-' + str(ar_E) + '/(' + '8.31446*T' + '))'
                new_attribute = {'unit': 'm', 'K': express_}

            else:
                new_attribute = reaction_data['newAttribute']
                
            # check for empty dictionary
            if new_attribute == {}:
                new_attribute = '0.0'
            
            new_attributes.append(new_attribute)

        # Check if the reactions are specified in predefined format, eg Bolsig, Arrhenius, etc.
        # Only bolsig is implemented.
        # If using bolsig, use unit = m as bolsig uses m as default units but the code reads unit and changes the rates 

        # change units if required in newAttributes
        for i, new_attribute in enumerate(new_attributes):

            # if new_attribute is float change to string
            if isinstance(new_attribute, float):
                new_attribute = str(new_attribute)
                new_attributes[i] = new_attribute

            # if bolsig format, make appropriate expression
            if 'Bolsig' in new_attribute:
                # make the expression - new_attribute['Bolsig'] should be a list
                val = "bolsigFit(1.5*Te/11600.0, " + str(new_attribute['Bolsig']) + ")"
                # add attribute['K'] as a new key value pair
                new_attribute['K'] = val
                
            
            # now even bolsig has a 'K' key. So units can be changed


            # check the keys in the newAttribute
            if 'unit' in new_attribute:
                # check if the unit is cm
                if new_attribute['unit'] == 'cm':
                    # convert to m
                    order = reactants[:,i].sum()
                    # val = new_attribute['K']*cm2m**order
                    # change the expression of the newAttribute, not float yet
                    val = new_attribute['K'] + '*' + str(cm2m**(3.0*(order-1))) 
                    # replace the dictionary with just the value
                    kevals.append(val)
                else:
                    # if not cm, then assume it is m
                    # new_attribute = new_attribute['K']
                    kevals.append(new_attribute['K'])

            # if only K is specified, then assume units are in default units defined in settings
            else:
                # check the default units
                if newUnits == 'cm':
                    # convert to m
                    order = reactants[:,i].sum()
                    # val = new_attribute*cm2m**order
                    # change the expression of the newAttribute, not float yet

                    print("new_attribute: for reaction ", i+1, " is ", new_attribute)
                    # check if 'K' is a key in the dictionary
                    if 'K' in new_attribute:
                        val = new_attribute['K'] + '*' + str(cm2m**(3.0*(order-1)))
                    else:
                        val = new_attribute + '*' + str(cm2m**(3.0*(order-1)))
                    
                    # replace the dictionary with just the value
                    # new_attribute = val
                    kevals.append(val)
                else:
                    # if not cm, then assume it is m
                    if 'K' in new_attribute:
                        # new_attribute = new_attribute['K']
                        kevals.append(new_attribute['K'])
                    else:
                        # new_attribute = new_attribute
                        kevals.append(new_attribute)


        # Find IDs with Bolsig expressions
        self.bolsigIDs = []
        self.bolsigExpressions = []
        for i, new_attribute in enumerate(new_attributes):
            if 'Bolsig' in new_attribute:
                self.bolsigIDs.append(i)
                self.bolsigExpressions.append(kevals[i])

        # print the bolsig expressions
        print("Bolsig expressions: ")
        for i, bolsigExpression in enumerate(self.bolsigExpressions):
            print("parBolsig_{} = {}".format(self.bolsigIDs[i], new_attributes[self.bolsigIDs[i]]['Bolsig']))
        for i, bolsigExpression in enumerate(self.bolsigExpressions):
            print("K_rxn[{}] = {}".format(self.bolsigIDs[i], bolsigExpression))

        print("********************* K-s READ *********************\n")
        # All expressions are now in Kevals list


        ## Now reading the energyExchange attributes for each reaction
        # energyExchange is a dictionary
        # keys: type [zero, ratios, valueseV, default]
        # for ratios, provide a list of ratios for t, v, e and should sum to 1
        # for valueseV, provide a list of values in eV for t, v, e and also provide fix: True or False
        # if fix is True, then the values are fixed and not changed within the solver
        # if fix is False, then the values are once passed with default algorithm to balance the energy and then fixed
        # if fix is not provided assume fix: False
        # if type is not provided assume type: default which balances the energy with the genral convention

        energyExchanges = []
        for reaction_data in mechanism_data['reactions']:
            # if energyExchange is not specified or empty, then use default value of 0
            if 'energyExchange' not in reaction_data or reaction_data['energyExchange'] == None or reaction_data['energyExchange'] == '' or reaction_data['energyExchange'] == {}:
                energyExchange = {'type': 'default'}
            else:
                energyExchange = reaction_data['energyExchange']
            
            energyExchanges.append(energyExchange)

        # In each energy exchange if the type is valueseV, then check if it has fix: True or False
        # if fix is not provided assume fix: False
        for i, energyExchange in enumerate(energyExchanges):

            # check if the type provided are valid and if so check the input for the valid types are also valid
            # valid types are zero, ratios, valueseV, default
            if energyExchange['type'] not in ['zero', 'ratios', 'valueseV', 'default']:
                print("Invalid type for energyExchange. Valid type =  zero, ratios, valueseV, default.")
                print(" Check Rxn: ", mechanism_data['reactions'][i])
                sys.exit()
            
            # ratios should have R as a key
            if energyExchange['type'] == 'ratios':
                if 'R' not in energyExchange:
                    print("Provide 'R' for energyExchange type = ratios. Check Rxn: ", mechanism_data['reactions'][i])
                    sys.exit()
            
            # valueseV should have val as a key
            if energyExchange['type'] == 'valueseV':
                if 'val' not in energyExchange:
                    print("Provide 'val' in eV per particle for energyExchange type = valueseV. Check Rxn: ", mechanism_data['reactions'][i])
                    sys.exit()
                if 'fix' not in energyExchange:
                    energyExchange['fix'] = False


        # if 'energyExchange' in mechanism_data:
        #     print("energyExchange found in mechanism file.")
        #     energyExchange = mechanism_data['energyExchange']
        # else:
        #     print("energyExchange not found in mechanism file. Using default values.")
        #     energyExchange = {'type': 'default'}

        
        # for each reaction now print the energyExchange
        for i, energyExchange in enumerate(energyExchanges):
            print("Reaction: ", i+1)
            print("energyExchange: ", energyExchange)

        
        idZeros = []
        idRatios = []
        idValueseV = []
        idDefaults = []

        energyValueseV = []
        energyRatios = []

        for i, energyExchange in enumerate(energyExchanges):
            if energyExchange['type'] == 'zero':
                idZeros.append(i)
            elif energyExchange['type'] == 'ratios':
                idRatios.append(i)
                energyRatios.append(energyExchange['R'])
            elif energyExchange['type'] == 'valueseV':
                idValueseV.append(i)
                energyValueseV.append(energyExchange['val'])
                # also check if fix is True or False
                if energyExchange['fix'] == False:
                    # add to defaults as well
                    idDefaults.append(i)
            elif energyExchange['type'] == 'default':
                idDefaults.append(i)

        # make a dictionary of the energyExchange
        energyExchangeIDs = {'zeros': idZeros, 'ratios': idRatios, 'valueseV': idValueseV, 'defaults': idDefaults}
        energyExchangeVals = {'ratios': energyRatios, 'valueseV': energyValueseV}


        # once show the reaction ids of each type
        print("Reaction ids for energyExchange: ")
        print("Zeros: ", idZeros)
        print("Ratios: ", idRatios)
        print("ValueseV: ", idValueseV)
        print("Defaults: ", idDefaults)
        

        ## Get expression in symbols and in compiled form
        # make symbols based on newVars. newVars keys are variable names and values are the symbols
        Tg = sp.symbols(newVars['T'])
        Tv = sp.symbols(newVars['Tv'])
        Te = sp.symbols(newVars['Te'])

        TSymbols = [Tg, Tv, Te]
        symbolic_expressions = []

        # simpify only one without Bolsig
        for i, attribute in enumerate(new_attributes):
            if 'Bolsig' not in attribute:
                # symbolic_expressions.append(sp.sympify(attribute))
                # expression in m
                exprSI = kevals[i]
                symbolic_expressions.append((exprSI))  ## symbolize when needed later
            else:
                # symbolic_expressions.append(('404-NotSym'))
                symbolic_expressions.append(('Bolsig'))
                print("Bolsig expression found. Not symbolizing , overwrite 404. Reactions: ", i+1)

        # symbolic_expressions = sp.sympify(kevals)

        # compiled_expressions = compile_expressions(new_attributes)
        compiled_expressions = self.__compile_expressions(kevals)


        # show all new attributes
        print("Expressions:")
        for new_attribute in kevals:
            print(new_attribute)

        print("Compiled Expressions:")
        for compiled_expression in compiled_expressions:
            print(compiled_expression)

        print("Symbolic Expressions:")
        for symbolic_expression in symbolic_expressions:
            print(symbolic_expression)
            
        print("********************* K-s Compiled and Symbolized ********************* \n")

        return compiled_expressions , symbolic_expressions, TSymbols, energyExchangeIDs, energyExchangeVals
    
    # define a method that prints the bolsig expressions
    def printBolsigExpressions(self):
        '''
        Print the bolsig expressions for each reaction.
        '''
        print("Bolsig expressions: ")
        # for i, bolsigExpression in enumerate(self.bolsigExpressions):
        #     print("parBolsig_{} = {}".format(self.bolsigIDs[i], new_attributes[self.bolsigIDs[i]]['Bolsig']))
        for i, bolsigExpression in enumerate(self.bolsigExpressions):
            print("K_rxn[{}] = {}".format(self.bolsigIDs[i], bolsigExpression))

    def __compile_expressions(self, new_attributes):
        '''
        Compile the expressions in new_attributes.
        '''
               
        return [compile(new_attribute, '', 'eval') for new_attribute in new_attributes]
    

    def __evaluate_custom_attributes(self, variable_values):
        compiled_expressions = self.com_expressions
        namespace = vars(math)
        namespace.update(variable_values)
        # bolsigFit is a function add this as well
        namespace['bolsigFit'] = bolsigFit
        return [eval(compiled_expr, namespace) for compiled_expr in compiled_expressions]

    def getRateConstants(self, Tg, Tv, Te):
        '''
        Get the rate constants at given temperature for all the reactions in float format.
        '''
        # import math
        # get the number of reactions
        # nR = len(self.com_expressions)
        nR = self.gas_ct.n_reactions

        # create a list to store the rates
        rates = np.zeros(nR)
        print("Number of reactions: ", nR)

        var_names = [str(symbol) for symbol in self.syms]
        var_values = [Tg, Tv, Te]
            
        variable_values = dict(zip(var_names, var_values)) 

        print("Variable values: ", variable_values)

        # # Directly use __evaluate_custom_attributes
        rates = self.__evaluate_custom_attributes(variable_values)

        # namespace = vars(math)
        # namespace.update(variable_values)
        # # loop over the expressions
        # for i in range(nR):

        #     # get the expression
        #     expr = self.com_expressions[i]

        #     print("Expression: ", expr)
        #     # evaluate the expression
        #     rates[i] = eval(expr, namespace)

        #     print("Rate: of reaction ", i+1, " is ", rates[i])


        


        # return the rates
        return rates
    
    ## Method to provide symbolic rates-- Needed for ODEBUilder
    def getSymbolicRates(self, TgName, TvName, TeName):
        '''
        This function takes input Tg, Tv and Te and gives the RateConstants in SI units
        e.g: for $ e^- + N_2^+ \rightarrow 2 N $ reaction : Krxn = 2.8e-7*(300.0/Tg)**0.5

        inputs are symbolic variables names (strings) with units in Kelvin
        '''

        # Make the symbols. Shoudl be consistent with the names used in the mechanism file
        T = sp.symbols(TgName)
        Tv = sp.symbols(TvName)
        Te = sp.symbols(TeName)

        # while building the ODE, the builder's setting will determine the name of the variables, symbols.
        # depending on what TgName, TvName, TeName are passed, they should be equivalent to the symbols used in the mechanism file

        # simpify with these new symbols
        symbolic_expressions = self.sym_expressions
        # simpify each expression
        symbolic_expressions = [sp.sympify(expr,locals={'T':T,'Tv':Tv,'Te':Te}) for expr in symbolic_expressions]

    
        return symbolic_expressions, T , Tv, Te
    
    ## Method to provide modifications for heat of reactions: Needed for ODEBuilder
    def getExtraFuncHrxn(self, gas, hNet_Sym, hNet_eVSym, h_sp, subHname):
        '''
        Parameters:
        gas: Cantera Gas object
        hNet_Sym: list of sympy symbols for the heat of reaction
        hNet_eVSym: list of sympy symbols for the heat of reaction in eV
        h_sp: list of enthalpies of the species in J/kmol
        subHname: optional string specifying the substring of the species names for which the heat of reaction needs to be updated
        
        Returns:
        rxIDSuperElastic, rxIDNetModes, ratioNetModesDict,rxIDInterSps, aDh_eVDict, subH_SymName
        
        '''

        subH_SymName = []
        rxIDSuperElastic = []

        ## if id in zero then make net zero
        for i in self.energyExchangeIDs['zeros']:
            hNet_Sym[i] = 0.0
        
        nrxn = self.nR
        # change everything to eV
        hNet_eVSym = [hNet_Sym[i]/(1.602e-19*6.022e26) for i in range(nrxn)]

        ## Super Elastic are can be defined as type: ratios here
        ## Include all with rations known
        rxIDNetModes = self.energyExchangeIDs['ratios']
        ratioNetModes = self.energyExchangeVals['ratios']
        ratioNetModesDict = dict(zip(rxIDNetModes,ratioNetModes))

        ## Include all with valueseV known
        # values known are two types: fix: True or False
        # pass only fix = False to build and for fix = True change the values directly here
        evalids=[]      # with defaults
        valueseV = []
        for i in np.arange(len(self.energyExchangeIDs['valueseV'])):
            rxid = self.energyExchangeIDs['valueseV'][i]
            if rxid not in self.energyExchangeIDs['defaults']:
                # change the value
                hNet_eVSym[rxid] = (self.energyExchangeVals['valueseV'][i][0] + 
                                self.energyExchangeVals['valueseV'][i][1] +
                                self.energyExchangeVals['valueseV'][i][2])
            else:
                evalids.append(rxid)
                valueseV.append(self.energyExchangeVals['valueseV'][i])


        # change to J/kmol
        hNet_Sym = [hNet_eVSym[i]*(1.602e-19*6.022e26) for i in range(nrxn)]

        # # find ids in valueseV with fix = True
        # evalids = [i for i in self.energyExchangeIDs['valueseV'] if i not in self.energyExchangeIDs['defaults']]
        # valueseV = [self.energyExchangeVals['valueseV'][i] for i in evalids]
        # rxIDInterSps = self.energyExchangeIDs['valueseV']
        # actualDH_eV = self.energyExchangeVals['valueseV']
        # aDh_eVDict = dict(zip(rxIDInterSps,actualDH_eV))

        aDh_eVDict = dict(zip(evalids,valueseV))
        rxIDInterSps = evalids

        return rxIDSuperElastic, rxIDNetModes, ratioNetModesDict,rxIDInterSps, aDh_eVDict, subH_SymName





    # Build ODE from just the provided custom mechanism file
    ## A static method to build the ODE system entirely from a mechanism file:
    def buildODE_fromMech(self,lang='python'):
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
        system = ODEBuilder(self.mechFile)
        # system = self
        system.mode = 'number_density'
        # system.mode = 'mass_fraction'
        system.language = lang
        # system.language = 'CXX'
        system.subHname = 'N2A'

        # using the same mech file make a new plasmaMechanism object
        # plMech = PlasmaMechanism(self.mechFile)

        # # check the heat of reaction using system.updateHrxnExpressions(self,ODEBuilder.extraFuncHrxnN2_54)
        # hRxn_Sym, h_sp, h_sp_call = system.updateHrxnExpressions(ODEBuilder.extraFuncHrxnN2_54)

        # update ODE System -- Recombination heating goes to Te
        # system.updateODESystem(reactionRateConstant=pl.ODEBuilder.getN2PlasmaKrxn,extraFuncHrxn=pl.ODEBuilder.extraFuncHrxnN2_54)
        # Recombination heating goes to Tg
        # system.updateODESystem(reactionRateConstant=pl.ODEBuilder.getN2PlasmaKrxn,extraFuncHrxn=pl.ODEBuilder.extraFuncHrxnN2_54_2)
        
        # make a new dict ; energyExchangeIDs and energyExchangeVals
        # dict
        energyExchange = {'ids': self.energyExchangeIDs, 'vals': self.energyExchangeVals}
        
        # system.updateODESystem(reactionRateConstant=self.getSymbolicRates,extraFuncHrxn=self.getExtraFuncHrxn)
        system.updateODESystem(reactionRateConstant=self.getSymbolicRates, energyExchangeDict=energyExchange)

        # get the expressions
        system.getSystemExpression()

        # print bolisg expressions
        self.printBolsigExpressions()


        
# Another class which solves directly from the mechanism file
# this class is derived from another abstract base class : plasmaKineticSystem
class PlasmaSolver(System):
    '''
    Description:
        PlasmaSolver is a derived class from the abstract base class System.
        The class provides functionality to solve the plasma kinetics system using the Cantera mechanism file with added attributes which hold the expressions for the reactions.
    
    plasmaKineticSystem <-- PlasmaSolver [ This instantiates the PlasmaMechanism object and uses it to solve the system ]


    - Make a PlasmaMechanism object
    - Instantitate a System object with added attributes
    - Solve the system within PlasmaSolver
    
    '''

    def __init__(self,mech_,verbose=False):
        '''
           Initialize the PlasmaSolver object.
        '''
        
        # PlasmaMechanism object
        self.plMech = PlasmaMechanism(mech_,verbose=verbose)
        # self.gas = self.plMech.gas_ct
        eff_stoicN = self.plMech.products - self.plMech.reactants
        self.effStoic = eff_stoicN.T # transpose

        # number of species
        self.nsp = self.plMech.gas_ct.n_species
        self.nrxn = self.plMech.gas_ct.n_reactions

        # Initial consitions are not provided initially, make it 
        Y0 = np.ones(self.nsp)/self.nsp # uniform distribution
        # Y0[0] = 0.5
        # Y0[-1] = 0.5
        T0 = [300.0, 300.0, 300.0] # standard for heat of reactions and all



        System.__init__(self, Y0, T0, mech_ ,self.nrxn, verbose=verbose)

        # store the initial enthalpy at standard conditions of T0 and also calculate the net heat of reaction for each reaction at stp
        self.hsp0 = self.hsp
        self.hNet0 = self.getHeatRxn()
        # self.idReacEle = []
        self.idEndo0 = []
        self.idExo0 = []

        # get initial heat of reaction for all modes at stp
        self.hRxn_modes0 = self.getHeatRxnModes()

        # find the reactions with electrons as reactants
        # gas.reactant_stoich_coeffs()[gas.species_index('ele'), i] == 0
        self.idReacEle = [i for i in range(self.nrxn) if self.plMech.reactants[self.get_electron_index(), i] > 0]



    def getHeatRxn(self):
        '''
        The deltaH for each reaction ( scalar value for each reaciton ) : units J/kmol
        '''

        gas = self.gas
        # calculate enthalpy of species
        # h_species = self.getSpeciesEnthalpy(T, Te)

        h_species = self.hsp

        # calculate the enthalpy of each reaction
        nrxn = self.nrxn
        h_rxns = np.zeros(nrxn)

        reactants = self.plMech.reactants
        products = self.plMech.products

        for i in range(nrxn):
            h_reactants = [reactants[j,i]*h_species[j] for j in range(gas.n_species)]
            h_products = [products[j,i]*h_species[j] for j in range(gas.n_species)]
            h_rxns[i] = sum(h_products) - sum(h_reactants)
        return h_rxns



    # now heat of reaction for all three energy modes from hNet
    def getHeatRxnModes(self):
        '''
        For all three modes the heat of reaction: units J/kmol
        '''
        reactants = self.plMech.reactants
        products = self.plMech.products

        # hNet = self.hNet0
        hNet = self.getHeatRxn()
        # hNet_eV = ODEBuilder.J_kmol2eV(hNet)
        hNet_eV = J_kmol2eV(hNet)

        # find the reactions with electrons as reactants
        # gas.reactant_stoich_coeffs()[gas.species_index('ele'), i] == 0
        self.idReacEle = [i for i in range(self.nrxn) if reactants[self.get_electron_index(), i] > 0]

        # # endothermic
        # self.idEndo0 = [i for i in range(self.nrxn) if hNet[i] > 0]
        # # exothermic
        # self.idExo0 = [i for i in range(self.nrxn) if hNet[i] < 0]

        idZeros = self.plMech.energyExchangeIDs['zeros']
        idRatios = self.plMech.energyExchangeIDs['ratios']
        idValueseV = self.plMech.energyExchangeIDs['valueseV']
        idDefaults = self.plMech.energyExchangeIDs['defaults']

        ratioValues = self.plMech.energyExchangeVals['ratios']
        valueseVValues = self.plMech.energyExchangeVals['valueseV']
        


        # Changes from energyExchange: type = zero
        for i in idZeros:
            hNet[i] = 0.0

        ## Now for all three modes
        hRxn_modes = np.zeros((3, self.nrxn))

        # Changes from energyExchange: type = valueseV
        # Use this for N2(A,v>0) as the excitation energy will be known directly; and force the fix to ture as you dont want to overwrite the values
        
        # loop len(idValueseV)
        for i in np.arange(len(idValueseV)):
            rxid = idValueseV[i]
            hRxn_modes[0,rxid] = valueseVValues[i][0]*1.602e-19*6.022e26
            hRxn_modes[1,rxid] = valueseVValues[i][1]*1.602e-19*6.022e26
            hRxn_modes[2,rxid] = valueseVValues[i][2]*1.602e-19*6.022e26
            
        # Changes from energyExchange: type = ratios
        # The superelastic collisions can be included here
        for i in np.arange(len(idRatios)):
            rxid = idRatios[i]
            hRxn_modes[0,rxid] = hNet[rxid]*ratioValues[i][0]
            hRxn_modes[1,rxid] = hNet[rxid]*ratioValues[i][1]
            hRxn_modes[2,rxid] = hNet[rxid]*ratioValues[i][2]
            
        # Changes from energyExchange: type = default
        # Need to run the default algorithm for idValueseV and idDefaults
        # if fix is false for idValueseV, then idDefaults includes idValueseV as well
        for i in idDefaults:
            # # if also in idValueseV
            # if i in idValueseV: 
            #     # find difference 
                diff = hNet[i] - hRxn_modes[0,i] - hRxn_modes[1,i] - hRxn_modes[2,i]
                if diff > 0.0 and i in self.idReacEle:
                    # extra energy goes to electron temperature
                    hRxn_modes[2,i] = diff + hRxn_modes[2,i]
                else:
                    hRxn_modes[0,i] = diff + hRxn_modes[0,i]

        return hRxn_modes



    # start implementation of abstract methods
    def dhrxn_(self):
        '''
        Calculate the enthalpy of reaction in J/kmol for each reaction.
        '''
        nrxn = self.nrxn
        hsp = self.hsp
    
        # HRxn = np.zeros((3, nrxn))


        # HRxn = self.hRxn_modes0         # not updating, using values at stp

        # # if you want to use updated values get it from getHeatRxnModes()
        HRxn = self.getHeatRxnModes()

        return HRxn
    
        

    def Krxn_(self):
        '''
        Calculate the reaction rate constants in SI units.
        '''
        nrxn = self.nrxn
        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]

        EeV = 1.5*Te/11600.0 # mean electron energy in eV

        K_rxn = self.plMech.getRateConstants(Tg, Tv, Te)

        return K_rxn



    def Wrxn_(self):
        '''
        Calculate the rate of progress in concentration units / s.
        '''
        nrxn = self.nrxn
        K_rxn = self.Krxn
        Ysp = self.Ysp

        reactants = self.plMech.reactants
        products = self.plMech.products
        rho = self.rho

        reac_stoic_sym = [Ysp ** reactants[:,i] for i in range(nrxn)]
        reac_stoic_sym = K_rxn * np.prod(reac_stoic_sym, axis=1)
            
        return reac_stoic_sym
        

    def dYdt_(self):
        '''
        Calculate the change in species concentration in concentration units / s -- mostly numdensity/s.
        '''
        nrxn = self.nrxn
        nsp = self.nsp
        W_rxn = self.Wrxn

        eff_stoicN = self.plMech.products - self.plMech.reactants
        eff_stoic = eff_stoicN.T # transpose

        # reac

        # wsp = np.matmul(reac_stoic_sym, eff_stoic)
        wsp = np.matmul(W_rxn, eff_stoic)

        return wsp
    
