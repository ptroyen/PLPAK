import sympy as sp
import numpy as np
        
def getN2AleksPlasmaKrxn(TgName,TvName,TeName):
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
        rxnTot = 67 # total number of reactions in the mechanism - count from 1

        # check if same read from gas object


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

## -----
            K55 = (1.95e-13*(300.0/Te)**0.7)

            K56 = (4.2e-12*(300.0/Te)**0.5)
        
            K57 = (3.1e-23*(Te)**(-1.5))*fac3

            K58 = (3.1e-23*(Te)**(-1.5))*fac3

            K59 = (3.1e-23*(Te)**(-1.5))*fac3

            K60 = (3.3e-6*(300.0/Tg)**(4.0)*sp.exp(-5030.0/Tg))*fac2

            K61 = (1.3e-6*(300.0/Te)**(0.5))*fac2
        
            K62 = (1.1e-6*(300.0/Tg)**(5.3)*sp.exp(-2357.0/Tg))*fac2
        
            K63 = (1.0e-9)*fac2

       
            K64 = (2.4e-30*(300.0/Tg)**(3.2))*fac3
            K65 = (0.9e-30*(300.0/Tg)**(2.0))*fac3
            K66 = (2.0e-19*(Te/300.0)**(-4.5))*fac3
        
            K67 = sp.exp(-7.9-13.4/(43.87*1.0e-17))*fac2




        
             

        # Put all in the list
        for i in range(rxnTot):
            Krxn.append(locals()["K"+str(i+1)])
        # print("Krxn: ", Krxn)

        # return Krxn along with the symbols for Temperature used in the rate constants
        # The symbols are required for substution of variables later on
        return Krxn, Tg, Tv, Te


def extraFuncHrxnN2Aleks(gas, hNet_Sym, hNet_eVSym, h_sp, subHname):
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

        # if not 68 then raise error
        if nrxn != 67:
            raise ValueError("Number of reactions is not 68, check the gas object, it has {} reactions".format(nrxn))

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
        # rxIDInterSps = np.array([12,40]) - 1
        rxIDInterSps = np.array([12,40,55,67]) - 1
        actualDH_eV = np.zeros((np.size(rxIDInterSps),3))
        actualDH_eV[0,:] = np.array([-3.5,0,0]) # for R12, gives actual heat exchange in Tg,Tv,Te, negative sign means energy is produced
        actualDH_eV[1,:] = np.array([-1.0,0,0]) # for R40, gives actual heat exchange in Tg,Tv,Te, negative sign means energy is produced
        actualDH_eV[2,:] = np.array([-5.0,0,0])
        actualDH_eV[3,:] = np.array([-2.04,0,0])
        # make dictionary for actualDH_eV such that key is reaction id and value is the actual heat exchange
        aDh_eVDict = dict(zip(rxIDInterSps,actualDH_eV))

        
        return rxIDSuperElastic, rxIDNetModes, ratioNetModesDict,rxIDInterSps, aDh_eVDict, subH_SymName