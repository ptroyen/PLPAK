
import numpy as np

from abc import ABC, abstractmethod

# add current directory to the path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the plasma kinetics library
from makePlasmaKinetics import *
# import the base class
from plasmaKineticSystem import System, NASA7Enthalpy
from extraModels import LaserModel
from mecWrapper import PlasmaSolver

# get the values
bolsigFit = System.bolsigFit
coul_log = System.coul_log
coul_freq = System.coul_freq
ptauVT_Park = System.ptauVT_Park


class PlasmaSolver3T(PlasmaSolver):
    '''
    Class for solving the plasma kinetics with three temperatures using the mech file provided.
    Needs the reaction mechanism as the input with the specification of the rate expressions ( can depend on all temperatures), and the energy exchange modes also need to be specified.
    Check the Qmodes method in the class for the energy exchange between various modes like QET,QVE,QEV,QVT, etc. These energy exchange are dependent on mixture composition so a single universal way might not be suitable. If needed make a custom class derived from this class and override the Qmodes method.
    '''

    # bolsigFit = System.bolsigFit
    # Vibrational energy of N2 in eV for different vibrational level v= 0-8 - values in eV
    N2v_UeV = np.array([0.14576868,0.43464128,0.71995946,1.00172153,1.27992582,1.55457065,1.82565433,2.09317519,2.35713153])

    n2Tvib_c = 3393.456 # K is the characteristic temperature for vibrational excitation of N2 - data from matlab code and calculated with : omega_e*100*h*c/kB
    Na = 6.02214076e23 # Avogadro's number
    m_e = 9.10938356e-31 # electron mass in kg
    kB = 1.38064852e-23 # Boltzmann constant in J/K


    # N2+M: VT relaxation from modified Park, A and B coefficients : p*tau = exp(A(T**(-1/3) - B) - 18.42) [atm*s]
    # https://link.springer.com/content/pdf/10.1007/s00193-012-0401-z.pdf
    VT_N2_ABs = {  'N2': [221.5, 0.0290],
                    'O2': [228.7, 0.0295],
                    'NO': [225.3, 0.0293],
                    'N': [180.8, 0.0262],
                    'O': [72.4, 0.0150] }

    # For O2+M also needed the relaxation times, not always used so store in a new variable
    # tauVT_O2 = None # if not None then use it
    # partners : N2, O2, NO, N, O
    # for O2 + O this is wrong, use adrienko-boyd results, the nature of the fit is different, high T high p-tau, for this only
    VT_O2_ABs = {  'N2': [131.3, 0.0295],
                    'O2': [135.9, 0.0300],
                    'NO': [133.7, 0.0298],
                    'N': [72.4, 0.0150],
                    'O': [47.7, 0.0590] }       ## for O, at low temperature rapid relaxation happens, so use different formulation


    def __init__(self, mechFile, verbose=False):
        super().__init__(mechFile, verbose=False)
        self.tauVT_O2 = None # if not None then use it
        self.QVT_O2 = None # if not None then use it

    def Qmodes_(self):
        '''
        Calculate the heating rate for specific reaction which involve only energy exchange and no kinetics in J/m3-s.
        These are categorized as, ET, EV, VE, VT. Here reactions correspond to : 41, 42, 43, 54
        Positive means heating and negative means cooling. And the naming shows the direction of energy flow. e.g. ET means energy from electron to thermal.
        Note: the corresponding HRxn should be made zero. --> Check dhrxn_()
        '''
        gas = self.gas
        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]
        p_atm = self.p*(1.0/101325.0) # atm

        meanEe = 1.5*Te/11600.0 # mean energy of electron in eV   

        eID = gas.species_index('ele')
        n2ID = gas.species_index('N2')
        if 'O2' in gas.species_names:
            o2ID = gas.species_index('O2')
        else:
            o2ID = None

        m_n2 = gas.molecular_weights[n2ID]/(self.Na*1000.0) # kg per particle

        ne = self.Ysp[eID]
        nN2 = self.Ysp[n2ID]
        nO2 = self.Ysp[o2ID]

        # total neutral number density
        nTot = self.p/(CO_kB*Tg)

        # # # get mole fraction for group of species : N2, all excited
        # XN2s = (self.Ysp[gas.species_index('N2')] + self.Ysp[gas.species_index('N2_A')] +
        #              self.Ysp[gas.species_index('N2_B')] + self.Ysp[gas.species_index('N2_C')] +
        #                 self.Ysp[gas.species_index('N2_ap')])/nTot

        # XNs = (self.Ysp[gas.species_index('N')] + self.Ysp[gas.species_index('N_D')] +
        #                 self.Ysp[gas.species_index('N_P')] + self.Ysp[gas.species_index('Np')])/nTot

        XNs = (self.Ysp[gas.species_index('N')] + self.Ysp[gas.species_index('Np')])/nTot
        XO2s = 0.0
        XNOs = 0.0
        XOs = 0.0
        # check if O2 is present, if yes then add it to the mole fraction
        if 'O2' in gas.species_names:
            XO2s = (self.Ysp[gas.species_index('O2')] + self.Ysp[gas.species_index('O3')] + 
                        self.Ysp[gas.species_index('O2P')] + self.Ysp[gas.species_index('O2m')] )/nTot
            XOs = (self.Ysp[gas.species_index('O')] + self.Ysp[gas.species_index('O_D')] +
                        self.Ysp[gas.species_index('Op')] + self.Ysp[gas.species_index('Om')] )/nTot
            XNOs = (self.Ysp[gas.species_index('NO')] + self.Ysp[gas.species_index('NOp')] )/nTot
        XN2s = 1.0 - XNs - XO2s - XNOs - XOs - ne/nTot

        XSums = 1.0 - ne/nTot



        # average mass of species
        mAvg = np.sum(self.Ysp*gas.molecular_weights)/(self.Na*1000.0) # kg per m3
        mAvg = mAvg/nTot # kg per particle

        # print("m_n2 : ", m_n2, "mAvg : ", mAvg)

        # energy for vibrational excitation from ground state of N2
        dU_eV = self.N2v_UeV[1:] - self.N2v_UeV[0]

        # change to J per molecule
        dU_J = dU_eV*1.60217662e-19

        # population ratio of vibrational levels : look Boyd - nonequiiibrium gas dynamics - page 119
        vs = np.arange(1,9)
        n2v_n2 = np.exp(-vs*self.n2Tvib_c/Tv)*(1.0 - np.exp(-self.n2Tvib_c/Tv))

        # For VE excitation
        # group v = 0 and 1 but v=0 cannot be exchanged
        # n2v_n2[0] = n2v_n2[0] + 1.0*(1.0 - np.exp(-self.n2Tvib_c/Tv))

        # # ET : rxn 41 : old variant
        # # Rxn 41 is elastic collsion : Here it is rxn 40 :1.0e22 per m3 BolsigFit ( results in SI ): -29.87      0.2346     -0.6179      0.9290E-01 -0.4900E-02
        # r40Bolsig = np.array([-29.87, 0.2346, -0.6179, 0.9290E-01, -0.4900E-02]) # m3/s
        # # Qet with maxwellean distribution 0 - 5ev: -29.97      0.3164     -0.4094      0.4047E-01 -0.1363E-02
        # r40Bolsig_m = np.array([-29.97, 0.3164, -0.4094, 0.4047E-01, -0.1363E-02]) # m3/s
        # Kr40 = self.bolsigFit(meanEe, r40Bolsig)
        # # Kr40 = self.bolsigFit(meanEe, r40Bolsig_m)
        # neu_el = Kr40*nN2 # only elastic collisions
        # del_n2 = 2.0*self.m_e/m_n2
        # QET = 1.5*self.kB*ne*neu_el*(Te-Tg)*del_n2
        # # show the values of QET
        # print("QET_0 : ", QET, "neu_el : ", neu_el, "del_n2 : ", del_n2)

        # ET using electron neutral and electron ion collisions -- updated variant
        # self.bolsig_en has the fit for electron neutral collisions
        neu_en = self.bolsigFit(meanEe, self.bolsig_en)*nTot
        neu_ei = self.coul_freq(Te,ne)
        self.nu_en = neu_en
        self.nu_ei = neu_ei
        mass_ratio = 2.0*self.m_e/mAvg
        QET = 1.5*self.kB*ne*(neu_en + neu_ei)*(Te-Tg)*mass_ratio

        # show the values of QET
        # print("QET_1 : ", QET,"nu_eff : ", neu_en + neu_ei,"mass ratio : ", mass_ratio)


        





        # EV : Has 8 reaction rate constants : Rxn : 42
        # non maxwellian : unstable at low temperatures
        ev1bfit = np.array([-29.19, -1.851, -6.737, 1.286, -0.7418E-01])
        ev1_2bfit = np.array([-36.24, 0.5479, -0.6106, -0.5111E-01, 0.6544E-02])
        ev2bfit = np.array([-25.40, -3.553, -12.10, -0.7369E-01, 0.1797])
        ev3bfit = np.array([-32.80, -1.082, 4.705, -13.03, 2.778])
        ev4bfit = np.array([-28.85, -2.729, -3.144, -12.11, 2.901])
        ev5bfit = np.array([-46.02, 3.238, 38.00, -45.25, 11.16])
        ev6bfit = np.array([-69.11, 11.00, 99.19, -104.3, 28.15])
        ev7bfit = np.array([-80.44, 14.60, 128.8, -135.5, 38.17])
        ev8bfit = np.array([-22.70, -4.335, -49.60, 87.94, -65.61])
        
        # for low n_e, Bolsig:
        # v1_1 :  -35.46      0.1119      -1.535      0.1146     -0.3288E-02
        # v1_2 : -22.11      -5.144      -15.38       1.584     -0.6550E-01
        # v2 :  -20.56      -6.158      -17.49      0.1047      0.2667

        ev1bfit = np.array([-35.46, 0.1119, -1.535, 0.1146, -0.3288E-02])
        ev1_2bfit = np.array([-22.11, -5.144, -15.38, 1.584, -0.6550E-01])
        ev2bfit = np.array([-20.56, -6.158, -17.49, 0.1047, 0.2667])


        # using maxwellian
        ev1bfit =  np.array([-36.57, 0.7034, -0.5930, 0.1285E-01, -0.1910E-03])
        ev1_2bfit= np.array([-30.05, -1.462, -3.650, 0.1433, -0.8351E-02])
        ev2bfit = np.array([-30.52, -1.481, -3.680, 0.1211, -0.7475E-02])
        ev3bfit = np.array([-30.92, -1.487, -3.625, 0.8931E-01, -0.5492E-02])
        ev4bfit = np.array([-31.26, -1.493, -3.706, 0.6115E-01, -0.3563E-02])
        ev5bfit = np.array([-31.42, -1.492, -3.813, 0.7003E-01, -0.4394E-02])
        ev6bfit = np.array([-31.51, -1.515, -4.017, 0.8081E-01, -0.8393E-02])
        ev7bfit = np.array([-31.96, -1.562, -4.479, 0.2102, -0.2704E-01])
        ev8bfit = np.array([-32.83, -1.499, -4.361, 0.3886E-01, -0.2111E-02])





        # maxwellian distribution more stable at low temperatures
        # v1 -1 :  -36.39      0.5623     -0.7357      0.2483E-01 -0.5307E-03
        #v1 - 2 :   -30.11      -1.434      -3.586      0.1274     -0.6987E-02
        # if meanEe < 0.5:
        # ev1bfit = np.array([-36.39, 0.5623, -0.7357, 0.2483E-01, -0.5307E-03]) # problematic one - this one is better
        # ev1_2bfit = np.array([-30.11, -1.434, -3.586, 0.1274, -0.6987E-02])

        # # The fits for eV exchange are notorious at low temperatures or low average energy of electron
        # # So include one more parameter below wich the minimum energy of electron is limited to the ones
        # # for which the fits are valid
        # # mean energy limits
        # eVlimts = np.array([0.1,0.2,0.2,0.3,0.3,0.4,0.5,0.6,0.8])
        # # if minimum than the limits
        # # notMin = (meanEe > eVlimts )*1.0
        # notMin = (meanEe > eVlimts*0.01 )*1.0
        # # make notMin an array
        # notMin = np.array(notMin)
        # Kevs = np.zeros(8)
        # Kevs[0] = self.bolsigFit(meanEe, ev1bfit)+self.bolsigFit(meanEe, ev1_2bfit)
        # Kevs[1] = self.bolsigFit(meanEe, ev2bfit)*notMin[2]
        # Kevs[2] = self.bolsigFit(meanEe, ev3bfit)*notMin[3]
        # Kevs[3] = self.bolsigFit(meanEe, ev4bfit)*notMin[4]
        # Kevs[4] = self.bolsigFit(meanEe, ev5bfit)*notMin[5]
        # Kevs[5] = self.bolsigFit(meanEe, ev6bfit)*notMin[6]
        # Kevs[6] = self.bolsigFit(meanEe, ev7bfit)*notMin[7]
        # Kevs[7] = self.bolsigFit(meanEe, ev8bfit)*notMin[8]

        Kevs = np.zeros(8)
        Kevs[0] = self.bolsigFit(meanEe, ev1bfit)+self.bolsigFit(meanEe, ev1_2bfit)
        Kevs[1] = self.bolsigFit(meanEe, ev2bfit)
        Kevs[2] = self.bolsigFit(meanEe, ev3bfit)
        Kevs[3] = self.bolsigFit(meanEe, ev4bfit)
        Kevs[4] = self.bolsigFit(meanEe, ev5bfit)
        Kevs[5] = self.bolsigFit(meanEe, ev6bfit)
        Kevs[6] = self.bolsigFit(meanEe, ev7bfit)
        Kevs[7] = self.bolsigFit(meanEe, ev8bfit)




        # Kevs = np.ones(8)*self.Krxn[42-1]       # should have different rate constants for each vibrational level
  
        
        QEV = np.sum(Kevs*dU_J)*ne*nN2

        # VE : rxn 43 , Also has 8 reaction rate constants
        # 8 bolsig fits for inverse of rate constants
        # v1 has two : -37.08      0.8902      0.9820     -0.3074      0.2145E-01
        # v1, 0.29 eV : -27.09      -2.702      -7.682      0.9072     -0.4804E-01
        # v2, 0.59 eV : -28.30      -2.443      -5.490      0.5449     -0.2777E-01
        # v3, 0.88 eV :  -29.24      -2.231      -3.871      0.3577     -0.1886E-01
        # v4, 1.17 eV :  -29.83      -2.139      -3.121      0.2757     -0.1466E-01
        # v5, 1.47 eV :  -30.25      -2.029      -2.586      0.2638     -0.1429E-01
        # v6, 1.76 eV :  -30.55      -1.963      -2.219      0.2365     -0.1313E-01
        # v7, 2.06 eV :  -31.24      -1.914      -2.049      0.2576     -0.1440E-01
        # v8, 2.35 eV :  -32.05      -1.868      -1.885      0.2699     -0.1544E-01
        evi1bfit = np.array([-37.08, 0.8902, 0.9820, -0.3074, 0.2145E-01])
        evi1_2bfit = np.array([-27.09, -2.702, -7.682, 0.9072, -0.4804E-01])
        evi2bfit = np.array([-28.30, -2.443, -5.490, 0.5449, -0.2777E-01])
        evi3bfit = np.array([-29.24, -2.231, -3.871, 0.3577, -0.1886E-01])
        evi4bfit = np.array([-29.83, -2.139, -3.121, 0.2757, -0.1466E-01])
        evi5bfit = np.array([-30.25, -2.029, -2.586, 0.2638, -0.1429E-01])
        evi6bfit = np.array([-30.55, -1.963, -2.219, 0.2365, -0.1313E-01])
        evi7bfit = np.array([-31.24, -1.914, -2.049, 0.2576, -0.1440E-01])
        evi8bfit = np.array([-32.05, -1.868, -1.885, 0.2699, -0.1544E-01])


        # at low ne:
        # v1 1 :  -35.79      0.2432     -0.6925      0.6016E-01 -0.1544E-02
        # v1 2 :  -25.34      -3.678      -9.555       1.016     -0.5366E-01
        # v2 :   -27.49      -2.916      -6.225      0.4342     -0.1674E-01
        evi1bfit = np.array([-35.79, 0.2432, -0.6925, 0.6016E-01, -0.1544E-02])
        evi1_2bfit = np.array([-25.34, -3.678, -9.555, 1.016, -0.5366E-01])
        evi2bfit = np.array([-27.49, -2.916, -6.225, 0.4342, -0.1674E-01])


        # using maxwellian
        evi1bfit =  np.array([-36.57, 0.7034, -0.1578, 0.1280E-01, -0.1881E-03])
        evi1_2bfit=np.array([-30.11, -1.442, -3.125, 0.1103, -0.4922E-02])
        evi2bfit = np.array([-30.58, -1.460, -2.699, 0.8537E-01, -0.3753E-02])
        evi3bfit = np.array([-30.96, -1.469, -2.225, 0.5941E-01, -0.2368E-02])
        evi4bfit = np.array([-31.29, -1.482, -1.901, 0.4216E-01, -0.1532E-02])
        evi5bfit = np.array([-31.46, -1.479, -1.547, 0.4624E-01, -0.1714E-02])
        evi6bfit = np.array([-31.57, -1.493, -1.267, 0.2814E-01, -0.9272E-03])
        evi7bfit = np.array([-32.18, -1.476, -0.9732, 0.1897E-01, -0.2112E-03])
        evi8bfit = np.array([-32.88, -1.479, -0.7599, 0.1671E-01, -0.1877E-03])


        Kves = np.zeros(8)
        Kves[0] = self.bolsigFit(meanEe, evi1bfit) + self.bolsigFit(meanEe, evi1_2bfit)
        Kves[1] = self.bolsigFit(meanEe, evi2bfit)
        Kves[2] = self.bolsigFit(meanEe, evi3bfit)
        Kves[3] = self.bolsigFit(meanEe, evi4bfit)
        Kves[4] = self.bolsigFit(meanEe, evi5bfit)
        Kves[5] = self.bolsigFit(meanEe, evi6bfit)
        Kves[6] = self.bolsigFit(meanEe, evi7bfit)
        Kves[7] = self.bolsigFit(meanEe, evi8bfit)


        # Kves = np.ones(8)*self.Krxn[43-1]   # should have different rate constants for each vibrational level
        

        # correct for different vibrational populations
        Kves = Kves*n2v_n2
        QVE = np.sum(Kves*dU_J)*ne*nN2

        # find tau_vt : # p[atm]*tau[s] = exp(A(T**(-1/3) - 0.015 mu**(1/4) - 18.42)) [atm*s]
        # simple just N2-N2 relaxation only for VT
        n2tau_vt_simple = self.ptauVT(Tg,220.0,14.0)/p_atm

        # more complete VT relaxation:
        # find effective tau by averaging
        sum_x_taus = p_atm*( XN2s / ptauVT_Park(Tg,self.VT_N2_ABs['N2'][0],self.VT_N2_ABs['N2'][1])
                            + XO2s / ptauVT_Park(Tg,self.VT_N2_ABs['O2'][0],self.VT_N2_ABs['O2'][1])
                            + XNOs / ptauVT_Park(Tg,self.VT_N2_ABs['NO'][0],self.VT_N2_ABs['NO'][1])
                            + XOs / ptauVT_Park(Tg,self.VT_N2_ABs['O'][0],self.VT_N2_ABs['O'][1])
                            + XNs / ptauVT_Park(Tg,self.VT_N2_ABs['N'][0],self.VT_N2_ABs['N'][1]) )
        n2tau_vt_park = XSums/sum_x_taus

        # # # compare park tauVT and simple tauVT from Millikan and white
        # print("tauVT Park : ", n2tau_vt_park, "tauVT Millikan : ", n2tau_vt_simple)
        # # # show the mole fractions used as well
        # print("XN2s : ", XN2s, "XO2s : ", XO2s, "XNOs : ", XNOs, "XOs : ", XOs, "XNs : ", XNs,'XSums : ', XSums)



        ## Now VT for O2, just populate the self.tauVT_O2 --------------------------------
        # Define the lambda function for O2-O
        # Formula: x = T/1000; (a*x**4 + b*x**3 + c*x**2 + d*x + e) * 1e-8
        # Parameters: a=-0.005798969725915833, b=0.11728071482076785, c=-0.8698505755385431, d=3.057577147662221, e=-0.043170994873928975
        ptau_VT_o2_o = lambda T: (-0.005798969725915833 * (T/1000)**4 + 
                                0.11728071482076785 * (T/1000)**3 + 
                                -0.8698505755385431 * (T/1000)**2 + 
                                3.057577147662221 * (T/1000) + 
                                -0.043170994873928975) * 1e-8

        sum_x_taus_o2 = p_atm*( XN2s / ptauVT_Park(Tg,self.VT_O2_ABs['N2'][0],self.VT_O2_ABs['N2'][1])
                            + XO2s / ptauVT_Park(Tg,self.VT_O2_ABs['O2'][0],self.VT_O2_ABs['O2'][1])
                            + XNOs / ptauVT_Park(Tg,self.VT_O2_ABs['NO'][0],self.VT_O2_ABs['NO'][1])
                            + XOs / ptau_VT_o2_o(Tg) 
                            + XNs / ptauVT_Park(Tg,self.VT_O2_ABs['N'][0],self.VT_O2_ABs['N'][1]) )
        o2tau_vt_park = XSums/sum_x_taus_o2
        self.tauVT_O2 = o2tau_vt_park

        #### comment this if O2 vibrational relaxation is not needed ----------------


        n2tau_vt = n2tau_vt_park # or use n2tau_vt_simple
        Ev_Tv = self.vibEnergy(Tv,self.n2Tvib_c) # per particle
        Ev_T = self.vibEnergy(Tg,self.n2Tvib_c)
        QVT = (Ev_Tv - Ev_T)*nN2/n2tau_vt # J/m3-s


        # Qmodes = np.array([QET, QEV, QVE, QVT])
        Qmodes = np.array([QET, QEV, QVE, QVT])
        Qmode_names = ['ET','EV','VE','VT']

        self.Qmode_names = Qmode_names
        # dictionary for Qmodes with keys and Qmode_names
        Qmodes_dict = dict(zip(Qmode_names,Qmodes))
        return Qmodes_dict



# Now the derived class for the plasma kinetics
class N2Plasma_54Rxn(System):
    '''
    This is the plasma kinetics for N2.
    '''
    nrxn_ = 54
    mech_ = 'plasmaN2.yaml'

    # Extra things needed
    #SubGroup enthalpy: N2A
    # Nasa7 polynomials use with NASA7Enthalpy() -> J/mol
    N2Avg1NASA7 = np.array([[3.07390924e+00,1.99120676e-03,-2.08768566e-06,2.62439456e-09,-1.34639011e-12,7.05970674e+04,6.42271362e+00],
                    [3.50673966e+00,1.30229712e-03,-6.69945124e-07,1.24320721e-10,-7.97758691e-15,7.03934476e+04,3.91162118e+00]])
    N2Avg2NASA7 = np.array([[2.73767346e+00,4.27761848e-03,-6.82914221e-06,6.93754842e-09,-2.81752532e-12,8.05427151e+04,7.90284209e+00],
                    [3.77161049e+00,1.00425875e-03,-5.74156450e-07,1.11907972e-10,-7.44792998e-15,8.02048573e+04,2.47921910e+00]])
    N2Avg3NASA7 = np.array([[2.64595791e+00,4.87994632e-03,-7.18000185e-06,6.35412823e-09,-2.33438726e-12,8.94758420e+04,8.34579376e+00],
                    [4.10955639e+00,2.57529645e-04,8.95120840e-09,-1.09217976e-11,5.27900491e-16,8.90514137e+04,8.01508324e-01]])

    # Vibrational energy of N2 in eV for different vibrational level v= 0-8 - values in eV
    N2v_UeV = np.array([0.14576868,0.43464128,0.71995946,1.00172153,1.27992582,1.55457065,1.82565433,2.09317519,2.35713153])

    n2Tvib_c = 3393.456 # K is the characteristic temperature for vibrational excitation of N2 - data from matlab code and calculated with : omega_e*100*h*c/kB
    
    def __init__(self, Ysp, T,laserObj=None, verbose = False):
        self.laser = LaserModel(switch=0)
        System.__init__(self, Ysp, T, self.mech_, self.nrxn_, verbose)

        if laserObj is not None:
            self.laser = laserObj        

        

    # # Implement the abstract methods
    # def Qrxn_(self):
    #     '''
    #     Calculate the heating rate for each reaction in J/m3-s.
    #     dhrxn is in J/kmol and Ysp is in number density.
    #     Wrxn is in numDensity/s and Temp is in K.
    #     -- can be put in the base class --
    #     '''
    #     gas = self.gas

    #     # Qrxn = -(self.Wrxn/Na)*self.dhrxn*1.0e-3 # J/m3-s

    #     QrxnTg = -self.Wrxn/Na*self.dhrxn[0,:]*1.0e-3 # J/m3-s
    #     QrxnTv = -self.Wrxn/Na*self.dhrxn[1,:]*1.0e-3 # J/m3-s
    #     QrxnTe = -self.Wrxn/Na*self.dhrxn[2,:]*1.0e-3 # J/m3-s

    #     Qrxn = np.array([QrxnTg, QrxnTv, QrxnTe])


    #     ## Qrxn changes here so should be abstract method
    #     # sum all the reactions, so sum over the rows
    #     # J/m3-s
    #     # Qdot = np.sum(Qrxn, axis = 1)

    #     Qdotg = np.sum(QrxnTg)
    #     Qdotv = np.sum(QrxnTv)
    #     Qdote = np.sum(QrxnTe)

    #     Qdot = np.array([Qdotg, Qdotv, Qdote])
        
    #     # validate shape of Qdot should have 3 elements
    #     if Qdot.shape[0] != 3:
    #         raise ValueError('Qdot should have 3 elements')

    #     # if self.verbose:
    #     #     print('Qdot = ', Qdot)
    #     #     print('Shape of Qdot = ', Qdot.shape)
    #     #     print ('Qrxn = ', Qrxn)
    #     #     print('Shape of Qrxn = ', Qrxn.shape)
    #     #     print('dhrxn = ', self.dhrxn)
    #     #     print('Wrxn = ', self.Wrxn)


    #     return Qrxn, Qdot

    # New method for energy exchange without species production/loss
    # Eg, RXN = 41,42,43,54 these are the processes related to : E->T, E-V, V->E, V->T
    # overwrite from the base class
    # @overrides(System)
    def Qmodes_(self):
        '''
        Calculate the heating rate for specific reaction which involve only energy exchange and no kinetics in J/m3-s.
        These are categorized as, ET, EV, VE, VT. Here reactions correspond to : 41, 42, 43, 54
        Positive means heating and negative means cooling. And the naming shows the direction of energy flow. e.g. ET means energy from electron to thermal.
        Note: the corresponding HRxn should be made zero. --> Check dhrxn_()
        '''
        gas = self.gas
        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]
        p_atm = self.p*(1.0/101325.0) # atm

        meanEe = 1.5*Te/11600.0 # mean energy of electron in eV   

        eID = gas.species_index('ele')
        n2ID = gas.species_index('N2')

        m_n2 = gas.molecular_weights[n2ID]/(Na*1000.0) # kg per particle

        ne = self.Ysp[eID]
        nN2 = self.Ysp[n2ID]

        

        # energy for vibrational excitation from ground state of N2
        dU_eV = self.N2v_UeV[1:] - self.N2v_UeV[0]

        # change to J per molecule
        dU_J = dU_eV*1.60217662e-19

        # population ratio of vibrational levels : look Boyd - nonequiiibrium gas dynamics - page 119
        vs = np.arange(1,9)
        n2v_n2 = np.exp(-vs*self.n2Tvib_c/Tv)*(1.0 - np.exp(-self.n2Tvib_c/Tv))

        # For VE excitation
        # group v = 0 and 1 but v=0 cannot be exchanged
        # n2v_n2[0] = n2v_n2[0] + 1.0*(1.0 - np.exp(-self.n2Tvib_c/Tv))

        # ET : rxn 41 :
        # Rxn 41 is elastic collsion : Here it is rxn 40 :1.0e22 per m3 BolsigFit ( results in SI ): -29.87      0.2346     -0.6179      0.9290E-01 -0.4900E-02
        r40Bolsig = np.array([-29.87, 0.2346, -0.6179, 0.9290E-01, -0.4900E-02]) # m3/s

        # Qet with maxwellean distribution 0 - 5ev: -29.97      0.3164     -0.4094      0.4047E-01 -0.1363E-02
        r40Bolsig_m = np.array([-29.97, 0.3164, -0.4094, 0.4047E-01, -0.1363E-02]) # m3/s
        Kr40 = self.bolsigFit(meanEe, r40Bolsig)
        # Kr40 = self.bolsigFit(meanEe, r40Bolsig_m)
        neu_el = Kr40*nN2 # only elastic collisions
        del_n2 = 2.0*m_e/m_n2

        # add coulomb collisions
        neu_el = neu_el + 2.91e-12*ne*(Te/11600.0)**(-1.5)*10.0

        QET = 1.5*kB*ne*neu_el*(Te-Tg)*del_n2

        # EV : Has 8 reaction rate constants : Rxn : 42
        # 8 bolsig fits
        # v1 has two : -29.19      -1.851      -6.737       1.286     -0.7418E-01
        # v1 :  -36.24      0.5479     -0.6106     -0.5111E-01  0.6544E-02
        # [higher value] v1 0.29 eV , which is resonant ?: -26.82      -2.782      -9.153      0.2181      0.8506E-01
        # v2, 0.59 eV : -25.40      -3.553      -12.10     -0.7369E-01  0.1797   
        # v3,0.88 eV  :-32.80      -1.082       4.705      -13.03       2.778
        # v4, 1.17 eV : -28.85      -2.729      -3.144      -12.11       2.901   
        # v5, 1.47 eV :  -46.02       3.238       38.00      -45.25       11.16  
        # v6, 1.76 eV :  -69.11       11.00       99.19      -104.3       28.15 
        # v7, 2.06 eV :  -80.44       14.60       128.8      -135.5       38.17   
        # v8, 2.35 eV :  -22.70      -4.335      -49.60       87.94      -65.61 
        ev1bfit = np.array([-29.19, -1.851, -6.737, 1.286, -0.7418E-01])
        ev1_2bfit = np.array([-36.24, 0.5479, -0.6106, -0.5111E-01, 0.6544E-02])
        ev2bfit = np.array([-25.40, -3.553, -12.10, -0.7369E-01, 0.1797])
        ev3bfit = np.array([-32.80, -1.082, 4.705, -13.03, 2.778])
        ev4bfit = np.array([-28.85, -2.729, -3.144, -12.11, 2.901])
        ev5bfit = np.array([-46.02, 3.238, 38.00, -45.25, 11.16])
        ev6bfit = np.array([-69.11, 11.00, 99.19, -104.3, 28.15])
        ev7bfit = np.array([-80.44, 14.60, 128.8, -135.5, 38.17])
        ev8bfit = np.array([-22.70, -4.335, -49.60, 87.94, -65.61])

        # for low n_e, Bolsig:
        # v1_1 :  -35.46      0.1119      -1.535      0.1146     -0.3288E-02
        # v1_2 : -22.11      -5.144      -15.38       1.584     -0.6550E-01
        # v2 :  -20.56      -6.158      -17.49      0.1047      0.2667

        ev1bfit = np.array([-35.46, 0.1119, -1.535, 0.1146, -0.3288E-02])
        ev1_2bfit = np.array([-22.11, -5.144, -15.38, 1.584, -0.6550E-01])
        ev2bfit = np.array([-20.56, -6.158, -17.49, 0.1047, 0.2667])


        # maxwellian distribution more stable at low temperatures
        # v1 -1 :  -36.39      0.5623     -0.7357      0.2483E-01 -0.5307E-03
        #v1 - 2 :   -30.11      -1.434      -3.586      0.1274     -0.6987E-02
        # if meanEe < 0.5:
        # ev1bfit = np.array([-36.39, 0.5623, -0.7357, 0.2483E-01, -0.5307E-03]) # problematic one - this one is better
        # ev1_2bfit = np.array([-30.11, -1.434, -3.586, 0.1274, -0.6987E-02])

        # The fits for eV exchange are notorious at low temperatures or low average energy of electron
        # So include one more parameter below wich the minimum energy of electron is limited to the ones
        # for which the fits are valid
        # mean energy limits
        eVlimts = np.array([0.1,0.2,0.2,0.3,0.3,0.4,0.5,0.6,0.8])

        # if minimum than the limits
        notMin = (meanEe > eVlimts )*1.0

        # make notMin an array
        notMin = np.array(notMin)

        Kevs = np.zeros(8)
        Kevs[0] = self.bolsigFit(meanEe, ev1bfit)+self.bolsigFit(meanEe, ev1_2bfit)
        Kevs[1] = self.bolsigFit(meanEe, ev2bfit)*notMin[2]
        Kevs[2] = self.bolsigFit(meanEe, ev3bfit)*notMin[3]
        Kevs[3] = self.bolsigFit(meanEe, ev4bfit)*notMin[4]
        Kevs[4] = self.bolsigFit(meanEe, ev5bfit)*notMin[5]
        Kevs[5] = self.bolsigFit(meanEe, ev6bfit)*notMin[6]
        Kevs[6] = self.bolsigFit(meanEe, ev7bfit)*notMin[7]
        Kevs[7] = self.bolsigFit(meanEe, ev8bfit)*notMin[8]

        # Kevs = np.ones(8)*self.Krxn[42-1]       # should have different rate constants for each vibrational level
  
        
        QEV = np.sum(Kevs*dU_J)*ne*nN2

        # VE : rxn 43 , Also has 8 reaction rate constants
        # 8 bolsig fits for inverse of rate constants
        # v1 has two : -37.08      0.8902      0.9820     -0.3074      0.2145E-01
        # v1, 0.29 eV : -27.09      -2.702      -7.682      0.9072     -0.4804E-01
        # v2, 0.59 eV : -28.30      -2.443      -5.490      0.5449     -0.2777E-01
        # v3, 0.88 eV :  -29.24      -2.231      -3.871      0.3577     -0.1886E-01
        # v4, 1.17 eV :  -29.83      -2.139      -3.121      0.2757     -0.1466E-01
        # v5, 1.47 eV :  -30.25      -2.029      -2.586      0.2638     -0.1429E-01
        # v6, 1.76 eV :  -30.55      -1.963      -2.219      0.2365     -0.1313E-01
        # v7, 2.06 eV :  -31.24      -1.914      -2.049      0.2576     -0.1440E-01
        # v8, 2.35 eV :  -32.05      -1.868      -1.885      0.2699     -0.1544E-01
        evi1bfit = np.array([-37.08, 0.8902, 0.9820, -0.3074, 0.2145E-01])
        evi1_2bfit = np.array([-27.09, -2.702, -7.682, 0.9072, -0.4804E-01])

        evi2bfit = np.array([-28.30, -2.443, -5.490, 0.5449, -0.2777E-01])
        evi3bfit = np.array([-29.24, -2.231, -3.871, 0.3577, -0.1886E-01])
        evi4bfit = np.array([-29.83, -2.139, -3.121, 0.2757, -0.1466E-01])
        evi5bfit = np.array([-30.25, -2.029, -2.586, 0.2638, -0.1429E-01])
        evi6bfit = np.array([-30.55, -1.963, -2.219, 0.2365, -0.1313E-01])
        evi7bfit = np.array([-31.24, -1.914, -2.049, 0.2576, -0.1440E-01])
        evi8bfit = np.array([-32.05, -1.868, -1.885, 0.2699, -0.1544E-01])


        # at low ne:
        # v1 1 :  -35.79      0.2432     -0.6925      0.6016E-01 -0.1544E-02
        # v1 2 :  -25.34      -3.678      -9.555       1.016     -0.5366E-01
        # v2 :   -27.49      -2.916      -6.225      0.4342     -0.1674E-01
        evi1bfit = np.array([-35.79, 0.2432, -0.6925, 0.6016E-01, -0.1544E-02])
        evi1_2bfit = np.array([-25.34, -3.678, -9.555, 1.016, -0.5366E-01])
        evi2bfit = np.array([-27.49, -2.916, -6.225, 0.4342, -0.1674E-01])


        Kves = np.zeros(8)
        Kves[0] = self.bolsigFit(meanEe, evi1bfit) + self.bolsigFit(meanEe, evi1_2bfit)
        Kves[1] = self.bolsigFit(meanEe, evi2bfit)
        Kves[2] = self.bolsigFit(meanEe, evi3bfit)
        Kves[3] = self.bolsigFit(meanEe, evi4bfit)
        Kves[4] = self.bolsigFit(meanEe, evi5bfit)
        Kves[5] = self.bolsigFit(meanEe, evi6bfit)
        Kves[6] = self.bolsigFit(meanEe, evi7bfit)
        Kves[7] = self.bolsigFit(meanEe, evi8bfit)






        # Kves = np.ones(8)*self.Krxn[43-1]   # should have different rate constants for each vibrational level
        

        # correct for different vibrational populations
        Kves = Kves*n2v_n2
        QVE = np.sum(Kves*dU_J)*ne*nN2

        # VT : rxn 54
        # find tau_vt : # p[atm]*tau[s] = exp(A(T**(-1/3) - 0.015 mu**(1/4) - 18.42)) [atm*s]
        # Systematics of Vibrational Relaxation  # Cite as: J. Chem. Phys. 39, 3209 (1963); https://doi.org/10.1063/1.1734182 --Roger C. Millikan and Donald R. White
        # for N2: A = 220.0 mu = 14.0
        # n2tau_vt = np.exp(220.0*(Tv**(-1.0/3.0) - 0.015*14.0**(1.0/4.0) - 18.42))/p_atm
        n2tau_vt = self.ptauVT(Tv,220.0,14.0)/p_atm
        Ev_Tv = self.vibEnergy(Tv,self.n2Tvib_c) # per particle
        Ev_T = self.vibEnergy(Tg,self.n2Tvib_c)
        QVT = (Ev_Tv - Ev_T)*nN2/n2tau_vt # J/m3-s

        # Qmodes = np.array([QET, QEV, QVE, QVT])
        Qmodes = np.array([QET, QEV, QVE, QVT])
        Qmode_names = ['ET','EV','VE','VT']

        self.Qmode_names = Qmode_names
        # dictionary for Qmodes with keys and Qmode_names
        Qmodes_dict = dict(zip(Qmode_names,Qmodes))
        return Qmodes_dict



    def dhrxn_(self):
        '''
        Calculate the enthalpy of reaction in J/kmol for each reaction.
        For reactions included in Qmodes_() the enthalpy of reaction should be zero.
        '''
        nrxn = self.nrxn
        hsp = self.hsp


        # get from the makePlasmaKinetics_II.py library
        # Requires definition of enthalpy for new subgroup , e.g. N2(A,v>9)
        HRxn = np.zeros((3, nrxn))

        # SubGroup enthalpy: N2A
        # [hN2A_vG1, hN2A_vG2, hN2A_vG3] = N2A(v=0-4) , N2A(v=5-9) , N2A(v>9)
        # Nasa7 polynomials use with NASA7Enthalpy(x,T,Tswitch=None): - > J/mol but needed in J/kmol, T needs to be array
        # NASA7Enthalpy outputs an array
        hN2A_vG1 = NASA7Enthalpy(self.N2Avg1NASA7, [self.Temp[0]], 1000.0)[0]*1.0e3
        hN2A_vG2 = NASA7Enthalpy(self.N2Avg2NASA7, [self.Temp[0]], 1000.0)[0]*1.0e3
        hN2A_vG3 = NASA7Enthalpy(self.N2Avg3NASA7, [self.Temp[0]], 1000.0)[0]*1.0e3



        # The ones with zero are not changed
        # Tg ----
        HRxn[0][0] = -2.0*hsp[0] + hsp[3]
        HRxn[0][1] = 0
        HRxn[0][2] = 0
        HRxn[0][3] = 0
        HRxn[0][4] = 0.7*hsp[1] - 1.4*hsp[2] + 0.7*hsp[3]
        HRxn[0][5] = 0
        HRxn[0][6] = (hsp[1] + hsp[2] - hsp[8] - hsp[9])*1.0
        HRxn[0][7] = (hsp[1] + hsp[3] - hsp[8] - hsp[9])*1.0
        HRxn[0][8] = (hsp[1] + hsp[4] - hsp[8] - hsp[9])*1.0
        HRxn[0][9] = 0
        HRxn[0][10] = hsp[0] + hsp[1] - hsp[7] - hsp[9]
        HRxn[0][11] = 2.0*hsp[0] - hsp[6] - hsp[9]
        HRxn[0][12] = 0
        HRxn[0][13] = hsp[1] - hsp[6] - hsp[9]
        HRxn[0][14] = 0
        HRxn[0][15] = 0
        HRxn[0][16] = 0
        HRxn[0][17] = hsp[0] - hsp[5] - hsp[9]
        HRxn[0][18] = 0
        HRxn[0][19] = 2.0*hsp[1] - hsp[2] + hsp[5] - hsp[7]
        HRxn[0][20] = hsp[0] - hsp[2] - hsp[6] + hsp[7]
        HRxn[0][21] = hsp[0] + hsp[1] - hsp[2] + hsp[5] - hsp[6]
        HRxn[0][22] = hsp[1] - hsp[2]
        HRxn[0][23] = hsp[1] - hsp[2]
        HRxn[0][24] = 0
        HRxn[0][25] = 0
        HRxn[0][26] = hsp[1] + hsp[6] - hsp[8] ## ??
        HRxn[0][27] = -hsp[0] + 2.0*hsp[1] + hsp[5] - hsp[8]
        HRxn[0][28] = -hsp[0] + hsp[1] + hsp[7] - hsp[8]
        HRxn[0][29] = -hsp[0] + hsp[1] + hsp[6] - hsp[7]
        HRxn[0][30] = hsp[1] + hsp[5] - hsp[7]
        HRxn[0][31] = hsp[0] + hsp[5] - hsp[6]
        HRxn[0][32] = hsp[0] - hsp[1] - hsp[6] + hsp[7]
        HRxn[0][33] = -hsp[0] + hsp[1] + hsp[5] - hsp[6]
        HRxn[0][34] = -hsp[1] - hsp[6] + hsp[8]       # Quenching paper shows opposite energy change - says released 1.06eV
        # HRxn[0][34] = - (1.06*1.6e-19*6.022e23*1.0e3) # Quenching paper shows opposite energy change - says released 1.06eV
        HRxn[0][35] = -hsp[0] - hsp[6] + hsp[7]
        HRxn[0][36] = hsp[0] - hsp[1] - hsp[5] + hsp[6]
        HRxn[0][37] = -hsp[1] - hsp[5] + hsp[7]
        HRxn[0][38] = -hsp[0] - hsp[5] + hsp[6]
        HRxn[0][39] = -96472440.0

        # # # reaction 6 - 18 except 11 make zero
        # zeroIDs = [6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18,23,28,29,35,37,38]
        # for i in zeroIDs:
        #     HRxn[0][i] = 0.0
        # zeroIDs = np.array([1,24,38]) -1
        # for i in zeroIDs:
        #     HRxn[0][i] = 0.0

        # nonZerosRxn = np.array([5,12,40]) - 1
        # # except nonZerosRxn, make zero
        # for i in range(nrxn):
        #     if i not in (nonZerosRxn):
        #         HRxn[0][i] = 0.0

        # Tv ----
        HRxn[1][0] = 0

        HRxn[1][4] = 0.3*hsp[1] - 0.6*hsp[2] + 0.3*hsp[3]
        HRxn[1][5] = hsp[1] - 2.0*hsp[2] + hsp[4]

        HRxn[1][24] = hsp[2] - hsp[3]
        HRxn[1][25] = hsp[3] - hsp[4]

        # # super recombination
        # # energy might go to vibration, photon radiation and bulk gas, do not know
        # HRxn[1][6] = (hsp[1] + hsp[2] - hsp[8] - hsp[9])*0.2
        # HRxn[1][7] = (hsp[1] + hsp[3] - hsp[8] - hsp[9])*0.2
        # HRxn[1][8] = (hsp[1] + hsp[4] - hsp[8] - hsp[9])*0.2

        # HRxn[1][41] = -27525341.0 # included in Qmodes
        # HRxn[1][42] = 27525341.0 # included in Qmodes

        # Te ----
        HRxn[2][9] = 2.0*hsp[1] - hsp[8] - hsp[9] # recombination
        HRxn[2][10] = 0
        HRxn[2][11] = 0
        HRxn[2][12] = 0
        HRxn[2][13] = 0
        HRxn[2][14] = hsp[1] - hsp[6] - hsp[9] # recombination
        HRxn[2][15] = -hsp[1] + hsp[6] + hsp[9]
        HRxn[2][16] = 0
        HRxn[2][17] = 0
        HRxn[2][18] = hsp[0] - hsp[5] - hsp[9] # recombination
        HRxn[2][39] = 2.0*hsp[0] - hsp[1] + 96472440.0
        HRxn[2][40] = 0
        HRxn[2][41] = 0
        HRxn[2][42] = 0
        HRxn[2][43] = hN2A_vG1 - hsp[1]
        HRxn[2][44] = -hN2A_vG1 + hsp[1]
        HRxn[2][45] = hN2A_vG2 - hsp[1]
        HRxn[2][46] = -hN2A_vG2 + hsp[1]
        HRxn[2][47] = -hsp[1] + hsp[3]
        HRxn[2][48] = hsp[1] - hsp[3]
        HRxn[2][49] = hN2A_vG3 - hsp[1]
        HRxn[2][50] = -hN2A_vG3 + hsp[1]
        HRxn[2][51] = -hsp[1] + hsp[4]
        HRxn[2][52] = hsp[1] - hsp[4]
        HRxn[2][53] = 0


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

        K_rxn = np.zeros(nrxn)


        # Bolsig fit for required reactions : Rxn ID is 1 more than the index as the first index is zero in python
        K44_Bparm = np.array([-24.62, -3.201, -54.41, 84.72, -55.07])
        K45_Bparm = np.array([-36.53, 0.4430, -0.9333, 0.1975, -0.1179E-01])
        K46_Bparm = np.array([-22.83, -3.286, -56.28, 87.69, -56.95])
        K47_Bparm = np.array([-34.60, 0.2789, -1.035, 0.2008, -0.1158E-01])
        K48_Bparm = np.array([-20.25, -3.846, -58.44, 89.11, -57.33])
        K49_Bparm = np.array([-32.76, -0.6230E-01, -0.9100, 0.1488, -0.8068E-02])
        K50_Bparm = np.array([-22.16, -3.462, -59.72, 92.99, -60.27])
        K51_Bparm = np.array([-34.23, 0.1644, -1.304, 0.2360, -0.1320E-01])
        K52_Bparm = np.array([-14.24, -5.166, -83.42, 127.1, -80.94])
        K53_Bparm = np.array([-30.64, -0.3308, -2.036, 0.3760, -0.2144E-01])



        # Rename with the reaction number = RxnID - 1
        K43 = bolsigFit(EeV, K44_Bparm)
        K44 = bolsigFit(EeV, K45_Bparm)
        K45 = bolsigFit(EeV, K46_Bparm)
        K46 = bolsigFit(EeV, K47_Bparm)
        K47 = bolsigFit(EeV, K48_Bparm)#*0.0
        K48 = bolsigFit(EeV, K49_Bparm)
        K49 = bolsigFit(EeV, K50_Bparm)#*0.0
        K50 = bolsigFit(EeV, K51_Bparm)
        K51 = bolsigFit(EeV, K52_Bparm)#*0.0
        K52 = bolsigFit(EeV, K53_Bparm)

        # # Use the output from the makePlasmaKinetics_II.py library
        # # If you need to change any reaction rate constants, e.g. using bolsigFit do it here
        K_rxn[0] = 8.27e-46*np.exp(500/Tg)
        K_rxn[1] = 0.5
        K_rxn[2] = 152000.0 ## peters
        # K_rxn[2] = 2.0e5 ## quenching paper
        K_rxn[3] = 26900000.0 ## peters
        # K_rxn[3] = 2.45e7 ## quenching paper
        K_rxn[4] = 2.9e-15*np.sqrt(Tg/300) # tuned values
        # K_rxn[4] = 1.1e-17*np.sqrt(Tg/300)
        K_rxn[5] = 2.6e-16*np.sqrt(Tg/300) # tuned values
        # K_rxn[5] = 2.1e-17*np.sqrt(Tg/300)
        K_rxn[6] = 6.0e-14*np.sqrt(300/Te) # tuned values - as peters --------
        # K_rxn[6] = 0.5*0.02*1.4e-12*(300.0/Te)**0.41 # tuned on alexs-----------------
        # K_rxn[6] = 0.02*1.4e-12*(300.0/Te)**0.41 # exact on alexs-----------------
        K_rxn[7] = 2.088e-12*np.sqrt(300/Te) # tuned values - as peters ----
        # K_rxn[7] = 0.5*0.87*1.4e-12*(300.0/Te)**0.41 # tuned on aleks-----------------
        # K_rxn[7] = 0.87*1.4e-12*(300.0/Te)**0.41 # exact on aleks-----------------
        K_rxn[8] = 2.64e-13*np.sqrt(300/Te) # tuned values - as peters ----
        # K_rxn[8] = 0.11*1.4e-12*(300.0/Te)**0.41 # exact on aleks-----------------
        # K_rxn[8] = 0.5*0.11*1.4e-12*(300.0/Te)**0.41 # tuned on aleks-----------------
        # K_rxn[8] = 4.6e-12*np.sqrt(300/Te)
        K_rxn[9] = 7.0e-32*(300/Te)**4.5 ## heating effect in Te ----
        K_rxn[10] = 2.0e-13*np.sqrt(300/Te)
        # K_rxn[11] = 2.8e-13*np.sqrt(300/Te)
        K_rxn[11] = 1.8e-13*(300.0/Te)**0.39 # aleks-----------------
        K_rxn[12] = 4.0e-18*(300/Te)**0.7
        K_rxn[13] = 6.0e-39*(300/Te)**1.5
        # K_rxn[14] = 1.0e-31*(300/Te)**4.5
        K_rxn[14] = 2.0e-31*(300/Te)**4.5 # aleks-----------------
        K_rxn[15] = (5.05e-17*(np.sqrt(Te) + 1.1e-5*Te**1.5))/np.exp(182000.0/Te)
        K_rxn[16] = 3.5e-18*(300/Te)**0.7
        K_rxn[17] = 6.0e-39*(300/Te)**1.5
        # K_rxn[18] = 1.0e-31*(300/Te)**4.5
        K_rxn[18] = 2.0e-31*(300/Te)**4.5 # aleks-----------------
        K_rxn[19] = 6.0e-16
        K_rxn[20] = 3.0e-16
        K_rxn[21] = 4.0e-16
        K_rxn[22] = 2.0e-23 # tuned values
        # K_rxn[22] = 3.0e-25
        K_rxn[23] = 6.2e-17*(300/Tg)**0.666666666666667 # tuned values
        # K_rxn[23] = 5.0e-18*(300/Tg)**0.666666666666667
        K_rxn[24] = 1.2e-17 # tuned values
        # K_rxn[24] = 1.6e-18
        K_rxn[25] = 1.2e-17*(300/Tg)**0.33 # tuned values
        # K_rxn[25] = 1.2e-17*(300/Tg)**0.33
        K_rxn[26] = 2.1e-22*np.exp(Tg/121)
        K_rxn[27] = 1.0e-17
        K_rxn[28] = 1.0e-15
        K_rxn[29] = 6.6e-17
        K_rxn[30] = 6.0e-16
        K_rxn[31] = 1.2e-17
        K_rxn[32] = 5.5e-18
        K_rxn[33] = 7.2e-19*np.exp(300/Tg) # different values - peters
        # K_rxn[33] = 2.4e-21*Tg # different values - quenching paper
        K_rxn[34] = 5e-41*(300/Tg)**1.64 # aleks-----------------
        K_rxn[35] = 9.0e-42*np.exp(400/Tg)
        K_rxn[36] = 1.0e-19
        K_rxn[37] = 2.0e-41*(300/Tg)**2.0
        K_rxn[38] = 1.0e-41*(300/Tg)
        K_rxn[39] = (1.2e+19/((6.022e+23*Te**1.6)))*np.exp(-113200/Te)
        K_rxn[40] = 8.25e-14
        K_rxn[41] = 8.45e-15
        K_rxn[42] = 2.69e-17
        K_rxn[43] = 1.59e-18
        K_rxn[44] = 2.67e-17
        K_rxn[45] = 5.63e-18
        K_rxn[46] = 1.8e-16
        K_rxn[47] = 2.35e-17
        K_rxn[48] = 9.85e-16
        K_rxn[49] = 4.06e-18
        K_rxn[50] = 2.15e-16
        K_rxn[51] = 6.18e-18
        K_rxn[52] = 4.06e-15
        K_rxn[53] = 3.05e-27

        # # update with bolsig rates
        K_rxn[43] = K43
        K_rxn[44] = K44
        K_rxn[45] = K45
        K_rxn[46] = K46
        K_rxn[47] = K47
        K_rxn[48] = K48
        K_rxn[49] = K49
        K_rxn[50] = K50
        K_rxn[51] = K51
        K_rxn[52] = K52


        return K_rxn

    def Wrxn_(self):
        '''
        Calculate the rate of progress in concentration units / s.
        '''
        nrxn = self.nrxn
        K_rxn = self.Krxn
        Ysp = self.Ysp

        W_rxn = np.zeros(nrxn)
        # Use the output from the makePlasmaKinetics_II.py library
        # If you need to change any reaction rate constants, e.g. using bolsigFit do it here
        W_rxn[0] = K_rxn[0]*Ysp[0]**2*Ysp[1]
        W_rxn[1] = K_rxn[1]*Ysp[2]
        W_rxn[2] = K_rxn[2]*Ysp[3]
        W_rxn[3] = K_rxn[3]*Ysp[4]
        W_rxn[4] = K_rxn[4]*Ysp[2]**2
        W_rxn[5] = K_rxn[5]*Ysp[2]**2
        W_rxn[6] = K_rxn[6]*Ysp[8]*Ysp[9]
        W_rxn[7] = K_rxn[7]*Ysp[8]*Ysp[9]
        W_rxn[8] = K_rxn[8]*Ysp[8]*Ysp[9]
        W_rxn[9] = K_rxn[9]*Ysp[8]*Ysp[9]**2
        W_rxn[10] = K_rxn[10]*Ysp[7]*Ysp[9]
        W_rxn[11] = K_rxn[11]*Ysp[6]*Ysp[9]
        W_rxn[12] = K_rxn[12]*Ysp[6]*Ysp[9]
        W_rxn[13] = K_rxn[13]*Ysp[1]*Ysp[6]*Ysp[9]
        W_rxn[14] = K_rxn[14]*Ysp[6]*Ysp[9]**2
        W_rxn[15] = K_rxn[15]*Ysp[1]*Ysp[9]
        W_rxn[16] = K_rxn[16]*Ysp[5]*Ysp[9]
        W_rxn[17] = K_rxn[17]*Ysp[1]*Ysp[5]*Ysp[9]
        W_rxn[18] = K_rxn[18]*Ysp[5]*Ysp[9]**2
        W_rxn[19] = K_rxn[19]*Ysp[2]*Ysp[7]
        W_rxn[20] = K_rxn[20]*Ysp[2]*Ysp[6]
        W_rxn[21] = K_rxn[21]*Ysp[2]*Ysp[6]
        W_rxn[22] = K_rxn[22]*Ysp[1]*Ysp[2]
        W_rxn[23] = K_rxn[23]*Ysp[0]*Ysp[2]
        W_rxn[24] = K_rxn[24]*Ysp[1]*Ysp[3]
        W_rxn[25] = K_rxn[25]*Ysp[1]*Ysp[4]
        W_rxn[26] = K_rxn[26]*Ysp[1]*Ysp[8]
        W_rxn[27] = K_rxn[27]*Ysp[0]*Ysp[8]
        W_rxn[28] = K_rxn[28]*Ysp[0]*Ysp[8]
        W_rxn[29] = K_rxn[29]*Ysp[0]*Ysp[7]
        W_rxn[30] = K_rxn[30]*Ysp[1]*Ysp[7]
        W_rxn[31] = K_rxn[31]*Ysp[1]*Ysp[6]
        W_rxn[32] = K_rxn[32]*Ysp[1]*Ysp[6]
        W_rxn[33] = K_rxn[33]*Ysp[0]*Ysp[6]
        W_rxn[34] = K_rxn[34]*Ysp[1]**2*Ysp[6]
        W_rxn[35] = K_rxn[35]*Ysp[0]*Ysp[1]*Ysp[6]
        W_rxn[36] = K_rxn[36]*Ysp[1]*Ysp[5]
        W_rxn[37] = K_rxn[37]*Ysp[1]**2*Ysp[5]
        W_rxn[38] = K_rxn[38]*Ysp[0]*Ysp[1]*Ysp[5]
        W_rxn[39] = K_rxn[39]*Ysp[1]*Ysp[9]
        W_rxn[40] = K_rxn[40]*Ysp[1]*Ysp[9]
        W_rxn[41] = K_rxn[41]*Ysp[1]*Ysp[9]
        W_rxn[42] = K_rxn[42]*Ysp[1]*Ysp[9]
        W_rxn[43] = K_rxn[43]*Ysp[1]*Ysp[9]
        W_rxn[44] = K_rxn[44]*Ysp[2]*Ysp[9]
        W_rxn[45] = K_rxn[45]*Ysp[1]*Ysp[9]
        W_rxn[46] = K_rxn[46]*Ysp[2]*Ysp[9]
        W_rxn[47] = K_rxn[47]*Ysp[1]*Ysp[9]
        W_rxn[48] = K_rxn[48]*Ysp[3]*Ysp[9]
        W_rxn[49] = K_rxn[49]*Ysp[1]*Ysp[9]
        W_rxn[50] = K_rxn[50]*Ysp[2]*Ysp[9]
        W_rxn[51] = K_rxn[51]*Ysp[1]*Ysp[9]
        W_rxn[52] = K_rxn[52]*Ysp[4]*Ysp[9]
        W_rxn[53] = K_rxn[53]*Ysp[1]**2


        


        return W_rxn

    def dYdt_(self):
        '''
        Calculate the change in species concentration in concentration units / s -- mostly numdensity/s.
        '''
        nrxn = self.nrxn
        nsp = self.nsp
        W_rxn = self.Wrxn

        dYdt = np.zeros(nsp)
        ## From the output of the make plasma kinetics library
        
        dYdt[0] = -2*W_rxn[0] + W_rxn[10] + 2*W_rxn[11] + W_rxn[16] + W_rxn[17] + W_rxn[18] + W_rxn[20] + W_rxn[21] - W_rxn[27] - W_rxn[28] - W_rxn[29] + W_rxn[31] + W_rxn[32] - W_rxn[33] - W_rxn[35] + W_rxn[36] - W_rxn[38] + 2*W_rxn[39]
        dYdt[1] = W_rxn[10] + W_rxn[12] + W_rxn[13] + W_rxn[14] - W_rxn[15] + 2*W_rxn[19] + W_rxn[1] + W_rxn[21] + W_rxn[22] + W_rxn[23] + W_rxn[26] + 2*W_rxn[27] + W_rxn[28] + W_rxn[29] + W_rxn[30] - W_rxn[32] + W_rxn[33] - W_rxn[34] - W_rxn[36] - W_rxn[37] - W_rxn[39] - W_rxn[43] + W_rxn[44] - W_rxn[45] + W_rxn[46] - W_rxn[47] + W_rxn[48] - W_rxn[49] + W_rxn[4] + W_rxn[50] - W_rxn[51] + W_rxn[52] + W_rxn[5] + W_rxn[6] + W_rxn[7] + W_rxn[8] + 2*W_rxn[9]
        dYdt[2] = -W_rxn[19] - W_rxn[1] - W_rxn[20] - W_rxn[21] - W_rxn[22] - W_rxn[23] + W_rxn[24] + W_rxn[2] + W_rxn[43] - W_rxn[44] + W_rxn[45] - W_rxn[46] + W_rxn[49] - 2*W_rxn[4] - W_rxn[50] - 2*W_rxn[5] + W_rxn[6]
        dYdt[3] = W_rxn[0] - W_rxn[24] + W_rxn[25] - W_rxn[2] + W_rxn[3] + W_rxn[47] - W_rxn[48] + W_rxn[4] + W_rxn[7]
        dYdt[4] = -W_rxn[25] - W_rxn[3] + W_rxn[51] - W_rxn[52] + W_rxn[5] + W_rxn[8]
        dYdt[5] = -W_rxn[16] - W_rxn[17] - W_rxn[18] + W_rxn[19] + W_rxn[21] + W_rxn[27] + W_rxn[30] + W_rxn[31] + W_rxn[33] - W_rxn[36] - W_rxn[37] - W_rxn[38]
        dYdt[6] = -W_rxn[11] - W_rxn[12] - W_rxn[13] - W_rxn[14] + W_rxn[15] - W_rxn[20] - W_rxn[21] + W_rxn[26] + W_rxn[29] - W_rxn[31] - W_rxn[32] - W_rxn[33] - W_rxn[34] - W_rxn[35] + W_rxn[36] + W_rxn[38]
        dYdt[7] = -W_rxn[10] - W_rxn[19] + W_rxn[20] + W_rxn[28] - W_rxn[29] - W_rxn[30] + W_rxn[32] + W_rxn[35] + W_rxn[37]
        dYdt[8] = -W_rxn[26] - W_rxn[27] - W_rxn[28] + W_rxn[34] - W_rxn[6] - W_rxn[7] - W_rxn[8] - W_rxn[9]
        dYdt[9] = -W_rxn[10] - W_rxn[11] - W_rxn[12] - W_rxn[13] - W_rxn[14] + W_rxn[15] - W_rxn[16] - W_rxn[17] - W_rxn[18] - W_rxn[6] - W_rxn[7] - W_rxn[8] - W_rxn[9]

        return dYdt
    

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
        p = self.p

        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]

        nN2 = self.Ysp[self.gas.species_index('N2')]
        ne = self.Ysp[self.gas.species_index('ele')]

        eID = self.gas.species_index('ele')
        # Tv = self.Temp[1]
        dEv_dTv = self.dEv_dTv_(Tv,self.n2Tvib_c) # vibrational energy per particle derivative with respect to temperature

        QVT = self.Qmodes['VT']
        QET = self.Qmodes['ET']
        QEV = self.Qmodes['EV']
        QVE = self.Qmodes['VE']

        if self.verbose:
            # show these values
            print('Qdotg = ', Qdotg, 'Qdotv = ', Qdotv, 'Qdote = ', Qdote)
            print('QVT = ', QVT, 'QET = ', QET, 'QEV = ', QEV, 'QVE = ', QVE)

        # d()dt:
        Ce = 1.5*kB*ne
        dydt = np.zeros(np.shape(self.Temp))

        dydt[0] = (Qdotg + QET + QVT)/(cp*rho)
        dydt[1] = ((Qdotv + QEV - QVE - QVT)/nN2)/dEv_dTv
        dydt[2] = (Qdote + QVE - QET - QEV)/Ce # - self.dYdt[eID]*Te/ne
        # dydt[2] = (Qdote + QVE - QET - 0.0*QEV)/Ce - self.dYdt[eID]*Te/ne

        # The dne/dt term is required if solving both Ysp and Temp simultaneously.
        # When the electron density is already modified then it is not required ?

        

        # If laser heating is included
        coLn = 10.0
        # calculate the nueff
        #  neuc[cellI] = 2.91e-12*n_e[cellI]*coLn.value()*(Foam::pow(Te[cellI]/11600,-3.0/2.0));
        neuc = 2.91e-12*ne*coLn*(Te/11600.0)**(-1.5)
        # neum = 9.8e-14*p[cellI]/(kB.value()*T[cellI])*(Foam::pow(Te[cellI]/11600.0,1.0/2.0)); 
        neum = 9.8e-14*p/(kB*Tg)*(Te/11600.0)**(0.5)

        nueff = neuc + neum
        Q = 0.0

        if self.laser.switch == 1:
            # calculate the laser absorption
            Q = self.laser.laserAbsorption(ne,nueff)

            dydt[2] = dydt[2] + Q/Ce


        return dydt



        

    def JacY_(self):
        '''
        Calculate the Jacobian for the species concentration.
        '''
        pass

    def JacT_(self):
        '''
        Calculate the Jacobian for the temperature.
        '''
        pass


####----------------------------------------------------------------------------------------------------------------------------



# Now the derived class for the plasma kinetics
class AirPlasmaRxn(System):
    '''
    This is the plasma kinetics for N2-O2.
    '''
    nrxn_ = 160
    mech_ = "airPlasma/combinedAirMechEne.yaml"

    # Vibrational energy of N2 in eV for different vibrational level v= 0-8 - values in eV
    N2v_UeV = np.array([0.14576868,0.43464128,0.71995946,1.00172153,1.27992582,1.55457065,1.82565433,2.09317519,2.35713153])

    n2Tvib_c = 3393.456 # K is the characteristic temperature for vibrational excitation of N2 - data from matlab code and calculated with : omega_e*100*h*c/kB
    
    def __init__(self, Ysp, T,laserObj=None, verbose = False):
        self.laser = LaserModel(switch=0)
        System.__init__(self, Ysp, T, self.mech_, self.nrxn_, verbose)

        if laserObj is not None:
            self.laser = laserObj        

        

    # New method for energy exchange without species production/loss
    # Eg, RXN = 41,42,43,54 these are the processes related to : E->T, E-V, V->E, V->T
    # overwrite from the base class
    # @overrides(System)
    def Qmodes_(self):
        '''
        Calculate the heating rate for specific reaction which involve only energy exchange and no kinetics in J/m3-s.
        These are categorized as, ET, EV, VE, VT. Here reactions correspond to : 41, 42, 43, 54
        Positive means heating and negative means cooling. And the naming shows the direction of energy flow. e.g. ET means energy from electron to thermal.
        Note: the corresponding HRxn should be made zero. --> Check dhrxn_()
        '''
        gas = self.gas
        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]
        p_atm = self.p*(1.0/101325.0) # atm

        meanEe = 1.5*Te/11600.0 # mean energy of electron in eV   

        eID = gas.species_index('ele')
        n2ID = gas.species_index('N2')

        m_n2 = gas.molecular_weights[n2ID]/(Na*1000.0) # kg per particle

        ne = self.Ysp[eID]
        nN2 = self.Ysp[n2ID]

        

        # energy for vibrational excitation from ground state of N2
        dU_eV = self.N2v_UeV[1:] - self.N2v_UeV[0]

        # change to J per molecule
        dU_J = dU_eV*1.60217662e-19

        # population ratio of vibrational levels : look Boyd - nonequiiibrium gas dynamics - page 119
        vs = np.arange(1,9)
        n2v_n2 = np.exp(-vs*self.n2Tvib_c/Tv)*(1.0 - np.exp(-self.n2Tvib_c/Tv))

        # For VE excitation
        # group v = 0 and 1 but v=0 cannot be exchanged
        # n2v_n2[0] = n2v_n2[0] + 1.0*(1.0 - np.exp(-self.n2Tvib_c/Tv))

        # ET : rxn 41 :
        # Rxn 41 is elastic collsion : Here it is rxn 40 :1.0e22 per m3 BolsigFit ( results in SI ): -29.87      0.2346     -0.6179      0.9290E-01 -0.4900E-02
        r40Bolsig = np.array([-29.87, 0.2346, -0.6179, 0.9290E-01, -0.4900E-02]) # m3/s

        # Qet with maxwellean distribution 0 - 5ev: -29.97      0.3164     -0.4094      0.4047E-01 -0.1363E-02
        r40Bolsig_m = np.array([-29.97, 0.3164, -0.4094, 0.4047E-01, -0.1363E-02]) # m3/s
        Kr40 = self.bolsigFit(meanEe, r40Bolsig)
        # Kr40 = self.bolsigFit(meanEe, r40Bolsig_m)
        neu_el = Kr40*nN2 # only elastic collisions
        del_n2 = 2.0*m_e/m_n2
        QET = 1.5*kB*ne*neu_el*(Te-Tg)*del_n2

        # EV : Has 8 reaction rate constants : Rxn : 42
        # non maxwellian : unstable at low temperatures
        ev1bfit = np.array([-29.19, -1.851, -6.737, 1.286, -0.7418E-01])
        ev1_2bfit = np.array([-36.24, 0.5479, -0.6106, -0.5111E-01, 0.6544E-02])
        ev2bfit = np.array([-25.40, -3.553, -12.10, -0.7369E-01, 0.1797])
        ev3bfit = np.array([-32.80, -1.082, 4.705, -13.03, 2.778])
        ev4bfit = np.array([-28.85, -2.729, -3.144, -12.11, 2.901])
        ev5bfit = np.array([-46.02, 3.238, 38.00, -45.25, 11.16])
        ev6bfit = np.array([-69.11, 11.00, 99.19, -104.3, 28.15])
        ev7bfit = np.array([-80.44, 14.60, 128.8, -135.5, 38.17])
        ev8bfit = np.array([-22.70, -4.335, -49.60, 87.94, -65.61])
        
        # for low n_e, Bolsig:
        # v1_1 :  -35.46      0.1119      -1.535      0.1146     -0.3288E-02
        # v1_2 : -22.11      -5.144      -15.38       1.584     -0.6550E-01
        # v2 :  -20.56      -6.158      -17.49      0.1047      0.2667

        ev1bfit = np.array([-35.46, 0.1119, -1.535, 0.1146, -0.3288E-02])
        ev1_2bfit = np.array([-22.11, -5.144, -15.38, 1.584, -0.6550E-01])
        ev2bfit = np.array([-20.56, -6.158, -17.49, 0.1047, 0.2667])


        # using maxwellian
        ev1bfit =  np.array([-36.57, 0.7034, -0.5930, 0.1285E-01, -0.1910E-03])
        ev1_2bfit= np.array([-30.05, -1.462, -3.650, 0.1433, -0.8351E-02])
        ev2bfit = np.array([-30.52, -1.481, -3.680, 0.1211, -0.7475E-02])
        ev3bfit = np.array([-30.92, -1.487, -3.625, 0.8931E-01, -0.5492E-02])
        ev4bfit = np.array([-31.26, -1.493, -3.706, 0.6115E-01, -0.3563E-02])
        ev5bfit = np.array([-31.42, -1.492, -3.813, 0.7003E-01, -0.4394E-02])
        ev6bfit = np.array([-31.51, -1.515, -4.017, 0.8081E-01, -0.8393E-02])
        ev7bfit = np.array([-31.96, -1.562, -4.479, 0.2102, -0.2704E-01])
        ev8bfit = np.array([-32.83, -1.499, -4.361, 0.3886E-01, -0.2111E-02])





        # maxwellian distribution more stable at low temperatures
        # v1 -1 :  -36.39      0.5623     -0.7357      0.2483E-01 -0.5307E-03
        #v1 - 2 :   -30.11      -1.434      -3.586      0.1274     -0.6987E-02
        # if meanEe < 0.5:
        # ev1bfit = np.array([-36.39, 0.5623, -0.7357, 0.2483E-01, -0.5307E-03]) # problematic one - this one is better
        # ev1_2bfit = np.array([-30.11, -1.434, -3.586, 0.1274, -0.6987E-02])

        # The fits for eV exchange are notorious at low temperatures or low average energy of electron
        # So include one more parameter below wich the minimum energy of electron is limited to the ones
        # for which the fits are valid
        # mean energy limits
        eVlimts = np.array([0.1,0.2,0.2,0.3,0.3,0.4,0.5,0.6,0.8])

        # if minimum than the limits
        # notMin = (meanEe > eVlimts )*1.0
        notMin = (meanEe > eVlimts*0.01 )*1.0

        # make notMin an array
        notMin = np.array(notMin)

        Kevs = np.zeros(8)
        Kevs[0] = self.bolsigFit(meanEe, ev1bfit)+self.bolsigFit(meanEe, ev1_2bfit)
        Kevs[1] = self.bolsigFit(meanEe, ev2bfit)*notMin[2]
        Kevs[2] = self.bolsigFit(meanEe, ev3bfit)*notMin[3]
        Kevs[3] = self.bolsigFit(meanEe, ev4bfit)*notMin[4]
        Kevs[4] = self.bolsigFit(meanEe, ev5bfit)*notMin[5]
        Kevs[5] = self.bolsigFit(meanEe, ev6bfit)*notMin[6]
        Kevs[6] = self.bolsigFit(meanEe, ev7bfit)*notMin[7]
        Kevs[7] = self.bolsigFit(meanEe, ev8bfit)*notMin[8]

        # Kevs = np.ones(8)*self.Krxn[42-1]       # should have different rate constants for each vibrational level
  
        
        QEV = np.sum(Kevs*dU_J)*ne*nN2

        # VE : rxn 43 , Also has 8 reaction rate constants
        # 8 bolsig fits for inverse of rate constants
        # v1 has two : -37.08      0.8902      0.9820     -0.3074      0.2145E-01
        # v1, 0.29 eV : -27.09      -2.702      -7.682      0.9072     -0.4804E-01
        # v2, 0.59 eV : -28.30      -2.443      -5.490      0.5449     -0.2777E-01
        # v3, 0.88 eV :  -29.24      -2.231      -3.871      0.3577     -0.1886E-01
        # v4, 1.17 eV :  -29.83      -2.139      -3.121      0.2757     -0.1466E-01
        # v5, 1.47 eV :  -30.25      -2.029      -2.586      0.2638     -0.1429E-01
        # v6, 1.76 eV :  -30.55      -1.963      -2.219      0.2365     -0.1313E-01
        # v7, 2.06 eV :  -31.24      -1.914      -2.049      0.2576     -0.1440E-01
        # v8, 2.35 eV :  -32.05      -1.868      -1.885      0.2699     -0.1544E-01
        evi1bfit = np.array([-37.08, 0.8902, 0.9820, -0.3074, 0.2145E-01])
        evi1_2bfit = np.array([-27.09, -2.702, -7.682, 0.9072, -0.4804E-01])
        evi2bfit = np.array([-28.30, -2.443, -5.490, 0.5449, -0.2777E-01])
        evi3bfit = np.array([-29.24, -2.231, -3.871, 0.3577, -0.1886E-01])
        evi4bfit = np.array([-29.83, -2.139, -3.121, 0.2757, -0.1466E-01])
        evi5bfit = np.array([-30.25, -2.029, -2.586, 0.2638, -0.1429E-01])
        evi6bfit = np.array([-30.55, -1.963, -2.219, 0.2365, -0.1313E-01])
        evi7bfit = np.array([-31.24, -1.914, -2.049, 0.2576, -0.1440E-01])
        evi8bfit = np.array([-32.05, -1.868, -1.885, 0.2699, -0.1544E-01])


        # at low ne:
        # v1 1 :  -35.79      0.2432     -0.6925      0.6016E-01 -0.1544E-02
        # v1 2 :  -25.34      -3.678      -9.555       1.016     -0.5366E-01
        # v2 :   -27.49      -2.916      -6.225      0.4342     -0.1674E-01
        evi1bfit = np.array([-35.79, 0.2432, -0.6925, 0.6016E-01, -0.1544E-02])
        evi1_2bfit = np.array([-25.34, -3.678, -9.555, 1.016, -0.5366E-01])
        evi2bfit = np.array([-27.49, -2.916, -6.225, 0.4342, -0.1674E-01])


        # using maxwellian
        evi1bfit =  np.array([-36.57, 0.7034, -0.1578, 0.1280E-01, -0.1881E-03])
        evi1_2bfit=np.array([-30.11, -1.442, -3.125, 0.1103, -0.4922E-02])
        evi2bfit = np.array([-30.58, -1.460, -2.699, 0.8537E-01, -0.3753E-02])
        evi3bfit = np.array([-30.96, -1.469, -2.225, 0.5941E-01, -0.2368E-02])
        evi4bfit = np.array([-31.29, -1.482, -1.901, 0.4216E-01, -0.1532E-02])
        evi5bfit = np.array([-31.46, -1.479, -1.547, 0.4624E-01, -0.1714E-02])
        evi6bfit = np.array([-31.57, -1.493, -1.267, 0.2814E-01, -0.9272E-03])
        evi7bfit = np.array([-32.18, -1.476, -0.9732, 0.1897E-01, -0.2112E-03])
        evi8bfit = np.array([-32.88, -1.479, -0.7599, 0.1671E-01, -0.1877E-03])


        Kves = np.zeros(8)
        Kves[0] = self.bolsigFit(meanEe, evi1bfit) + self.bolsigFit(meanEe, evi1_2bfit)
        Kves[1] = self.bolsigFit(meanEe, evi2bfit)
        Kves[2] = self.bolsigFit(meanEe, evi3bfit)
        Kves[3] = self.bolsigFit(meanEe, evi4bfit)
        Kves[4] = self.bolsigFit(meanEe, evi5bfit)
        Kves[5] = self.bolsigFit(meanEe, evi6bfit)
        Kves[6] = self.bolsigFit(meanEe, evi7bfit)
        Kves[7] = self.bolsigFit(meanEe, evi8bfit)






        # Kves = np.ones(8)*self.Krxn[43-1]   # should have different rate constants for each vibrational level
        

        # correct for different vibrational populations
        Kves = Kves*n2v_n2
        QVE = np.sum(Kves*dU_J)*ne*nN2

        # VT : rxn 54
        # find tau_vt : # p[atm]*tau[s] = exp(A(T**(-1/3) - 0.015 mu**(1/4) - 18.42)) [atm*s]
        # Systematics of Vibrational Relaxation  # Cite as: J. Chem. Phys. 39, 3209 (1963); https://doi.org/10.1063/1.1734182 --Roger C. Millikan and Donald R. White
        # for N2: A = 220.0 mu = 14.0
        # n2tau_vt = np.exp(220.0*(Tv**(-1.0/3.0) - 0.015*14.0**(1.0/4.0) - 18.42))/p_atm
        n2tau_vt = self.ptauVT(Tv,220.0,14.0)/p_atm
        Ev_Tv = self.vibEnergy(Tv,self.n2Tvib_c) # per particle
        Ev_T = self.vibEnergy(Tg,self.n2Tvib_c)
        QVT = (Ev_Tv - Ev_T)*nN2/n2tau_vt # J/m3-s

        # Qmodes = np.array([QET, QEV, QVE, QVT])
        Qmodes = np.array([QET, QEV, QVE, QVT])
        Qmode_names = ['ET','EV','VE','VT']

        self.Qmode_names = Qmode_names
        # dictionary for Qmodes with keys and Qmode_names
        Qmodes_dict = dict(zip(Qmode_names,Qmodes))
        return Qmodes_dict


    def dhrxn_(self):
        '''
        Calculate the enthalpy of reaction in J/kmol for each reaction.
        For reactions included in Qmodes_() the enthalpy of reaction should be zero.
        '''
        nrxn = self.nrxn
        hsp = self.hsp


        # get from the makePlasmaKinetics_II.py library
        # Requires definition of enthalpy for new subgroup , e.g. N2(A,v>9)
        HRxn = np.zeros((3, nrxn))

        # SubGroup enthalpy: N2A
        # [hN2A_vG1, hN2A_vG2, hN2A_vG3] = N2A(v=0-4) , N2A(v=5-9) , N2A(v>9)
        # Nasa7 polynomials use with NASA7Enthalpy(x,T,Tswitch=None): - > J/mol but needed in J/kmol, T needs to be array
        # NASA7Enthalpy outputs an array

        # hN2A_vG1 = NASA7Enthalpy(self.N2Avg1NASA7, [self.Temp[0]], 1000.0)[0]*1.0e3
        # hN2A_vG2 = NASA7Enthalpy(self.N2Avg2NASA7, [self.Temp[0]], 1000.0)[0]*1.0e3
        # hN2A_vG3 = NASA7Enthalpy(self.N2Avg3NASA7, [self.Temp[0]], 1000.0)[0]*1.0e3


        # Heat of reaction Tg:
        HRxn[0][0] = -2.0*hsp[0] + hsp[3]
        HRxn[0][1] = 0
        HRxn[0][2] = 0
        HRxn[0][3] = 0
        HRxn[0][4] = 0.7*hsp[1] - 1.4*hsp[2] + 0.7*hsp[3]
        HRxn[0][5] = 0
        HRxn[0][6] = -hsp[14] + hsp[1] - hsp[21] + hsp[2]
        HRxn[0][7] = -hsp[14] + hsp[1] - hsp[21] + hsp[3]
        HRxn[0][8] = -hsp[14] + hsp[1] - hsp[21] + hsp[4]
        HRxn[0][9] = 0
        HRxn[0][10] = hsp[0] - hsp[13] + hsp[1] - hsp[21]
        HRxn[0][11] = 2.0*hsp[0] - hsp[12] - hsp[21]
        HRxn[0][12] = 0
        HRxn[0][13] = -hsp[12] + hsp[1] - hsp[21]
        HRxn[0][14] = 0
        HRxn[0][15] = 0
        HRxn[0][16] = 0
        HRxn[0][17] = hsp[0] - hsp[11] - hsp[21]
        HRxn[0][18] = 0
        HRxn[0][19] = hsp[11] - hsp[13] + 2.0*hsp[1] - hsp[2]
        HRxn[0][20] = hsp[0] - hsp[12] + hsp[13] - hsp[2]
        HRxn[0][21] = hsp[0] + hsp[11] - hsp[12] + hsp[1] - hsp[2]
        HRxn[0][22] = hsp[1] - hsp[2]
        HRxn[0][23] = hsp[1] - hsp[2]
        HRxn[0][24] = 0
        HRxn[0][25] = 0
        HRxn[0][26] = hsp[12] - hsp[14] + hsp[1]
        HRxn[0][27] = -hsp[0] + hsp[11] - hsp[14] + 2.0*hsp[1]
        HRxn[0][28] = -hsp[0] + hsp[13] - hsp[14] + hsp[1]
        HRxn[0][29] = -hsp[0] + hsp[12] - hsp[13] + hsp[1]
        HRxn[0][30] = hsp[11] - hsp[13] + hsp[1]
        HRxn[0][31] = hsp[0] + hsp[11] - hsp[12]
        HRxn[0][32] = hsp[0] - hsp[12] + hsp[13] - hsp[1]
        HRxn[0][33] = -hsp[0] + hsp[11] - hsp[12] + hsp[1]
        HRxn[0][34] = -hsp[12] + hsp[14] - hsp[1]
        HRxn[0][35] = -hsp[0] - hsp[12] + hsp[13]
        HRxn[0][36] = hsp[0] - hsp[11] + hsp[12] - hsp[1]
        HRxn[0][37] = -hsp[11] + hsp[13] - hsp[1]
        HRxn[0][38] = -hsp[0] - hsp[11] + hsp[12]
        HRxn[0][39] = -96472440.0
        HRxn[0][40] = 0
        HRxn[0][41] = 0
        HRxn[0][42] = 0
        HRxn[0][43] = 0
        HRxn[0][44] = 0
        HRxn[0][45] = 0
        HRxn[0][46] = 0
        HRxn[0][47] = 0
        HRxn[0][48] = 0
        HRxn[0][49] = 0
        HRxn[0][50] = hsp[3] - hsp[5]
        HRxn[0][51] = hsp[1] - hsp[5] + hsp[6] - hsp[7] + hsp[8]
        HRxn[0][52] = 0
        HRxn[0][53] = 0
        HRxn[0][54] = hsp[14] + hsp[21] - hsp[2] - hsp[5]
        HRxn[0][55] = hsp[14] + hsp[21] - 2.0*hsp[5]
        HRxn[0][56] = -hsp[15] - hsp[21] + 2.0*hsp[6]
        HRxn[0][57] = -443773224.0
        HRxn[0][58] = hsp[0] - hsp[18] - hsp[21] + hsp[6]
        HRxn[0][59] = 0
        HRxn[0][60] = 0
        HRxn[0][61] = -hsp[15] - hsp[21] + hsp[7]
        HRxn[0][62] = -hsp[15] - hsp[21] + hsp[7]
        HRxn[0][63] = -hsp[12] + hsp[1] - hsp[21]
        HRxn[0][64] = -hsp[16] - hsp[21] + hsp[6]
        HRxn[0][65] = -hsp[16] - hsp[21] + hsp[6]
        HRxn[0][66] = hsp[20] - hsp[21] - hsp[7]
        HRxn[0][67] = hsp[20] - hsp[21] - hsp[7]
        HRxn[0][68] = hsp[20] - hsp[21] - hsp[7]
        HRxn[0][69] = hsp[19] - hsp[21] - hsp[6]
        HRxn[0][70] = 0
        HRxn[0][71] = hsp[19] - hsp[21] + hsp[7] - hsp[9]
        HRxn[0][72] = -hsp[20] + hsp[21] + hsp[7]
        HRxn[0][73] = -hsp[20] + hsp[21] + hsp[7]
        HRxn[0][74] = hsp[1] - hsp[20] + hsp[21] - hsp[2] + hsp[7]
        HRxn[0][75] = hsp[1] - hsp[20] + hsp[21] - hsp[3] + hsp[7]
        HRxn[0][76] = -hsp[19] + hsp[1] + hsp[21] - hsp[2] + hsp[6]
        HRxn[0][77] = -hsp[19] + hsp[1] + hsp[21] - hsp[3] + hsp[6]
        HRxn[0][78] = -hsp[20] + hsp[21] - hsp[6] + hsp[9]
        HRxn[0][79] = -hsp[0] + hsp[10] - hsp[19] + hsp[21]
        HRxn[0][80] = -hsp[19] + hsp[21] - hsp[7] + hsp[9]
        HRxn[0][81] = hsp[1] - hsp[2] + 2.0*hsp[6] - hsp[7]
        HRxn[0][82] = hsp[1] - hsp[2]
        HRxn[0][83] = hsp[1] - hsp[2] - hsp[6] + hsp[8]
        HRxn[0][84] = hsp[1] - hsp[2]
        HRxn[0][85] = hsp[0] + hsp[10] - hsp[2] - hsp[6]
        HRxn[0][86] = hsp[1] - hsp[3] + 2.0*hsp[6] - hsp[7]
        HRxn[0][87] = hsp[2] - hsp[3]
        HRxn[0][88] = hsp[1] - hsp[4] + hsp[6] - hsp[7] + hsp[8]
        HRxn[0][89] = hsp[6] - hsp[8]
        HRxn[0][90] = hsp[6] - hsp[8]
        HRxn[0][91] = hsp[19] - hsp[20] - hsp[6] + hsp[7]
        HRxn[0][92] = -hsp[19] + hsp[20] + hsp[6] - hsp[7]
        HRxn[0][93] = -hsp[14] + hsp[15] + 2.0*hsp[1] - hsp[7]
        HRxn[0][94] = -hsp[12] + hsp[15] + hsp[1] - hsp[7]
        HRxn[0][95] = -hsp[15] + hsp[17] - hsp[7]
        HRxn[0][96] = -hsp[15] + hsp[17] - hsp[7]
        HRxn[0][97] = -hsp[12] + hsp[14] - hsp[1]
        HRxn[0][98] = hsp[15] - hsp[17] + hsp[7]
        HRxn[0][99] = hsp[15] - hsp[17] + hsp[7]
        HRxn[0][100] = -hsp[0] - hsp[16] + hsp[18]
        HRxn[0][101] = -hsp[0] - hsp[16] + hsp[18]
        HRxn[0][102] = hsp[0] - hsp[11] + hsp[15] - hsp[7]
        HRxn[0][103] = -hsp[11] + hsp[18] + hsp[6] - hsp[7]
        HRxn[0][104] = hsp[0] - hsp[11] + hsp[16] - hsp[6]
        HRxn[0][105] = -hsp[11] + hsp[18] + hsp[7] - hsp[9]
        HRxn[0][106] = hsp[0] - hsp[10] - hsp[11] + hsp[18]
        HRxn[0][107] = -hsp[10] - hsp[11] + hsp[12] + hsp[6]
        HRxn[0][108] = -hsp[10] - hsp[11] + hsp[16] + hsp[1]
        HRxn[0][109] = hsp[0] - hsp[16] + hsp[18] - hsp[1]
        HRxn[0][110] = hsp[0] - hsp[12] + hsp[18] - hsp[6]
        HRxn[0][111] = -hsp[12] + hsp[16] + hsp[1] - hsp[6]
        HRxn[0][112] = -hsp[12] + hsp[15] + hsp[1] + hsp[6] - hsp[9]
        HRxn[0][113] = -hsp[10] - hsp[12] + hsp[18] + hsp[1]
        HRxn[0][114] = hsp[10] - hsp[15] + hsp[18] - hsp[1]
        HRxn[0][115] = -hsp[0] - hsp[15] + hsp[18] + hsp[6]
        HRxn[0][116] = -hsp[10] - hsp[15] + hsp[18] + hsp[7]
        HRxn[0][117] = -hsp[14] + hsp[16] + 2.0*hsp[1] - hsp[6]
        HRxn[0][118] = -hsp[10] - hsp[14] + hsp[18] + 2.0*hsp[1]
        HRxn[0][119] = hsp[15] - hsp[17] - hsp[6] + hsp[9]
        HRxn[0][120] = -hsp[10] - hsp[17] + hsp[18] + 2.0*hsp[7]
        HRxn[0][121] = -hsp[15] - hsp[20] + 2.0*hsp[6] + hsp[7]
        HRxn[0][122] = -hsp[15] - hsp[20] + 2.0*hsp[6] + hsp[7]
        HRxn[0][123] = -hsp[17] - hsp[20] + 2.0*hsp[6] + 2.0*hsp[7]
        HRxn[0][124] = -hsp[12] + hsp[1] - hsp[20] + hsp[7]
        HRxn[0][125] = -hsp[12] + hsp[1] - hsp[20] + hsp[7]
        HRxn[0][126] = -hsp[15] - hsp[20] + 2.0*hsp[7]
        HRxn[0][127] = -hsp[15] - hsp[20] + 2.0*hsp[7]
        HRxn[0][128] = hsp[0] - hsp[11] - hsp[20] + hsp[7]
        HRxn[0][129] = hsp[0] - hsp[11] - hsp[20] + hsp[7]
        HRxn[0][130] = -hsp[16] - hsp[20] + hsp[6] + hsp[7]
        HRxn[0][131] = -hsp[16] - hsp[20] + hsp[6] + hsp[7]
        HRxn[0][132] = hsp[10] - hsp[18] - hsp[20] + hsp[7]
        HRxn[0][133] = hsp[10] - hsp[18] - hsp[20] + hsp[7]
        HRxn[0][134] = -hsp[12] - hsp[19] + hsp[1] + hsp[6]
        HRxn[0][135] = -hsp[12] - hsp[19] + hsp[1] + hsp[6]
        HRxn[0][136] = -hsp[15] - hsp[19] + hsp[6] + hsp[7]
        HRxn[0][137] = -hsp[15] - hsp[19] + hsp[6] + hsp[7]
        HRxn[0][138] = hsp[0] - hsp[11] - hsp[19] + hsp[6]
        HRxn[0][139] = hsp[0] - hsp[11] - hsp[19] + hsp[6]
        HRxn[0][140] = -hsp[16] - hsp[19] + 2.0*hsp[6]
        HRxn[0][141] = -hsp[16] - hsp[19] + 2.0*hsp[6]
        HRxn[0][142] = hsp[10] - hsp[18] - hsp[19] + hsp[6]
        HRxn[0][143] = hsp[10] - hsp[18] - hsp[19] + hsp[6]
        HRxn[0][144] = hsp[10] - hsp[11] - hsp[19]
        HRxn[0][145] = hsp[10] - hsp[11] - hsp[19]
        HRxn[0][146] = -hsp[16] - hsp[19] + hsp[7]
        HRxn[0][147] = -hsp[16] - hsp[19] + hsp[7]
        HRxn[0][148] = -hsp[0] + hsp[10] + hsp[6] - hsp[7]
        HRxn[0][149] = -hsp[0] + hsp[10] + hsp[7] - hsp[9]
        HRxn[0][150] = -hsp[0] - hsp[10] + hsp[1] + hsp[6]
        HRxn[0][151] = -hsp[6] + 2.0*hsp[7] - hsp[9]
        HRxn[0][152] = -2.0*hsp[0] + hsp[2]
        HRxn[0][153] = -2.0*hsp[0] + hsp[1]
        HRxn[0][154] = -2.0*hsp[6] + hsp[7]
        HRxn[0][155] = -2.0*hsp[6] + hsp[7]
        HRxn[0][156] = -hsp[6] - hsp[7] + hsp[9]
        HRxn[0][157] = -hsp[6] - hsp[7] + hsp[9]
        HRxn[0][158] = -75248503.2
        HRxn[0][159] = -121555274.4
        # -----------------------------------------------------
        # Heat of reaction Tv:
        HRxn[1][0] = 0
        HRxn[1][1] = 0
        HRxn[1][2] = 0
        HRxn[1][3] = 0
        HRxn[1][4] = 0.3*hsp[1] - 0.6*hsp[2] + 0.3*hsp[3]
        HRxn[1][5] = hsp[1] - 2.0*hsp[2] + hsp[4]
        HRxn[1][6] = 0
        HRxn[1][7] = 0
        HRxn[1][8] = 0
        HRxn[1][9] = 0
        HRxn[1][10] = 0
        HRxn[1][11] = 0
        HRxn[1][12] = 0
        HRxn[1][13] = 0
        HRxn[1][14] = 0
        HRxn[1][15] = 0
        HRxn[1][16] = 0
        HRxn[1][17] = 0
        HRxn[1][18] = 0
        HRxn[1][19] = 0
        HRxn[1][20] = 0
        HRxn[1][21] = 0
        HRxn[1][22] = 0
        HRxn[1][23] = 0
        HRxn[1][24] = hsp[2] - hsp[3]
        HRxn[1][25] = hsp[3] - hsp[4]
        HRxn[1][26] = 0
        HRxn[1][27] = 0
        HRxn[1][28] = 0
        HRxn[1][29] = 0
        HRxn[1][30] = 0
        HRxn[1][31] = 0
        HRxn[1][32] = 0
        HRxn[1][33] = 0
        HRxn[1][34] = 0
        HRxn[1][35] = 0
        HRxn[1][36] = 0
        HRxn[1][37] = 0
        HRxn[1][38] = 0
        HRxn[1][39] = 0
        HRxn[1][40] = 0
        HRxn[1][41] = 0
        HRxn[1][42] = 0
        HRxn[1][43] = 0
        HRxn[1][44] = 0
        HRxn[1][45] = 0
        HRxn[1][46] = 0
        HRxn[1][47] = 0
        HRxn[1][48] = 0
        HRxn[1][49] = 0
        HRxn[1][50] = 0
        HRxn[1][51] = 0
        HRxn[1][52] = 0
        HRxn[1][53] = 0
        HRxn[1][54] = 0
        HRxn[1][55] = 0
        HRxn[1][56] = 0
        HRxn[1][57] = 0
        HRxn[1][58] = 0
        HRxn[1][59] = 0
        HRxn[1][60] = 0
        HRxn[1][61] = 0
        HRxn[1][62] = 0
        HRxn[1][63] = 0
        HRxn[1][64] = 0
        HRxn[1][65] = 0
        HRxn[1][66] = 0
        HRxn[1][67] = 0
        HRxn[1][68] = 0
        HRxn[1][69] = 0
        HRxn[1][70] = 0
        HRxn[1][71] = 0
        HRxn[1][72] = 0
        HRxn[1][73] = 0
        HRxn[1][74] = 0
        HRxn[1][75] = 0
        HRxn[1][76] = 0
        HRxn[1][77] = 0
        HRxn[1][78] = 0
        HRxn[1][79] = 0
        HRxn[1][80] = 0
        HRxn[1][81] = 0
        HRxn[1][82] = 0
        HRxn[1][83] = 0
        HRxn[1][84] = 0
        HRxn[1][85] = 0
        HRxn[1][86] = 0
        HRxn[1][87] = 0
        HRxn[1][88] = 0
        HRxn[1][89] = 0
        HRxn[1][90] = 0
        HRxn[1][91] = 0
        HRxn[1][92] = 0
        HRxn[1][93] = 0
        HRxn[1][94] = 0
        HRxn[1][95] = 0
        HRxn[1][96] = 0
        HRxn[1][97] = 0
        HRxn[1][98] = 0
        HRxn[1][99] = 0
        HRxn[1][100] = 0
        HRxn[1][101] = 0
        HRxn[1][102] = 0
        HRxn[1][103] = 0
        HRxn[1][104] = 0
        HRxn[1][105] = 0
        HRxn[1][106] = 0
        HRxn[1][107] = 0
        HRxn[1][108] = 0
        HRxn[1][109] = 0
        HRxn[1][110] = 0
        HRxn[1][111] = 0
        HRxn[1][112] = 0
        HRxn[1][113] = 0
        HRxn[1][114] = 0
        HRxn[1][115] = 0
        HRxn[1][116] = 0
        HRxn[1][117] = 0
        HRxn[1][118] = 0
        HRxn[1][119] = 0
        HRxn[1][120] = 0
        HRxn[1][121] = 0
        HRxn[1][122] = 0
        HRxn[1][123] = 0
        HRxn[1][124] = 0
        HRxn[1][125] = 0
        HRxn[1][126] = 0
        HRxn[1][127] = 0
        HRxn[1][128] = 0
        HRxn[1][129] = 0
        HRxn[1][130] = 0
        HRxn[1][131] = 0
        HRxn[1][132] = 0
        HRxn[1][133] = 0
        HRxn[1][134] = 0
        HRxn[1][135] = 0
        HRxn[1][136] = 0
        HRxn[1][137] = 0
        HRxn[1][138] = 0
        HRxn[1][139] = 0
        HRxn[1][140] = 0
        HRxn[1][141] = 0
        HRxn[1][142] = 0
        HRxn[1][143] = 0
        HRxn[1][144] = 0
        HRxn[1][145] = 0
        HRxn[1][146] = 0
        HRxn[1][147] = 0
        HRxn[1][148] = 0
        HRxn[1][149] = 0
        HRxn[1][150] = 0
        HRxn[1][151] = 0
        HRxn[1][152] = 0
        HRxn[1][153] = 0
        HRxn[1][154] = 0
        HRxn[1][155] = 0
        HRxn[1][156] = 0
        HRxn[1][157] = 0
        HRxn[1][158] = 0
        HRxn[1][159] = 0
        # -----------------------------------------------------
        # Heat of reaction Te:
        HRxn[2][0] = 0
        HRxn[2][1] = 0
        HRxn[2][2] = 0
        HRxn[2][3] = 0
        HRxn[2][4] = 0
        HRxn[2][5] = 0
        HRxn[2][6] = 0
        HRxn[2][7] = 0
        HRxn[2][8] = 0
        HRxn[2][9] = -hsp[14] + 2.0*hsp[1] - hsp[21]
        HRxn[2][10] = 0
        HRxn[2][11] = 0
        HRxn[2][12] = 0
        HRxn[2][13] = 0
        HRxn[2][14] = -hsp[12] + hsp[1] - hsp[21]
        HRxn[2][15] = hsp[12] - hsp[1] + hsp[21]
        HRxn[2][16] = 0
        HRxn[2][17] = 0
        HRxn[2][18] = hsp[0] - hsp[11] - hsp[21]
        HRxn[2][19] = 0
        HRxn[2][20] = 0
        HRxn[2][21] = 0
        HRxn[2][22] = 0
        HRxn[2][23] = 0
        HRxn[2][24] = 0
        HRxn[2][25] = 0
        HRxn[2][26] = 0
        HRxn[2][27] = 0
        HRxn[2][28] = 0
        HRxn[2][29] = 0
        HRxn[2][30] = 0
        HRxn[2][31] = 0
        HRxn[2][32] = 0
        HRxn[2][33] = 0
        HRxn[2][34] = 0
        HRxn[2][35] = 0
        HRxn[2][36] = 0
        HRxn[2][37] = 0
        HRxn[2][38] = 0
        HRxn[2][39] = 2.0*hsp[0] - hsp[1] + 96472440.0
        HRxn[2][40] = 595234954.8
        HRxn[2][41] = -595234954.8
        HRxn[2][42] = 675307080.0
        HRxn[2][43] = -675307080.0
        HRxn[2][44] = -hsp[1] + hsp[3]
        HRxn[2][45] = hsp[1] - hsp[3]
        HRxn[2][46] = 752485032.0
        HRxn[2][47] = -752485032.0
        HRxn[2][48] = -hsp[1] + hsp[4]
        HRxn[2][49] = hsp[1] - hsp[4]
        HRxn[2][50] = 0
        HRxn[2][51] = 0
        HRxn[2][52] = -hsp[1] + hsp[5]
        HRxn[2][53] = hsp[15] + hsp[21] - hsp[7]
        HRxn[2][54] = 0
        HRxn[2][55] = 0
        HRxn[2][56] = 0
        HRxn[2][57] = 0
        HRxn[2][58] = 0
        HRxn[2][59] = -hsp[15] - hsp[21] + 2.0*hsp[6]
        HRxn[2][60] = -hsp[16] - hsp[21] + hsp[6]
        HRxn[2][61] = 0
        HRxn[2][62] = 0
        HRxn[2][63] = 0
        HRxn[2][64] = 0
        HRxn[2][65] = 0
        HRxn[2][66] = 0
        HRxn[2][67] = 0
        HRxn[2][68] = 0
        HRxn[2][69] = 0
        HRxn[2][70] = hsp[20] - hsp[21] + hsp[6] - hsp[9]
        HRxn[2][71] = 0
        HRxn[2][72] = 0
        HRxn[2][73] = 0
        HRxn[2][74] = 0
        HRxn[2][75] = 0
        HRxn[2][76] = 0
        HRxn[2][77] = 0
        HRxn[2][78] = 0
        HRxn[2][79] = 0
        HRxn[2][80] = 0
        HRxn[2][81] = 0
        HRxn[2][82] = 0
        HRxn[2][83] = 0
        HRxn[2][84] = 0
        HRxn[2][85] = 0
        HRxn[2][86] = 0
        HRxn[2][87] = 0
        HRxn[2][88] = 0
        HRxn[2][89] = 0
        HRxn[2][90] = 0
        HRxn[2][91] = 0
        HRxn[2][92] = 0
        HRxn[2][93] = 0
        HRxn[2][94] = 0
        HRxn[2][95] = 0
        HRxn[2][96] = 0
        HRxn[2][97] = 0
        HRxn[2][98] = 0
        HRxn[2][99] = 0
        HRxn[2][100] = 0
        HRxn[2][101] = 0
        HRxn[2][102] = 0
        HRxn[2][103] = 0
        HRxn[2][104] = 0
        HRxn[2][105] = 0
        HRxn[2][106] = 0
        HRxn[2][107] = 0
        HRxn[2][108] = 0
        HRxn[2][109] = 0
        HRxn[2][110] = 0
        HRxn[2][111] = 0
        HRxn[2][112] = 0
        HRxn[2][113] = 0
        HRxn[2][114] = 0
        HRxn[2][115] = 0
        HRxn[2][116] = 0
        HRxn[2][117] = 0
        HRxn[2][118] = 0
        HRxn[2][119] = 0
        HRxn[2][120] = 0
        HRxn[2][121] = 0
        HRxn[2][122] = 0
        HRxn[2][123] = 0
        HRxn[2][124] = 0
        HRxn[2][125] = 0
        HRxn[2][126] = 0
        HRxn[2][127] = 0
        HRxn[2][128] = 0
        HRxn[2][129] = 0
        HRxn[2][130] = 0
        HRxn[2][131] = 0
        HRxn[2][132] = 0
        HRxn[2][133] = 0
        HRxn[2][134] = 0
        HRxn[2][135] = 0
        HRxn[2][136] = 0
        HRxn[2][137] = 0
        HRxn[2][138] = 0
        HRxn[2][139] = 0
        HRxn[2][140] = 0
        HRxn[2][141] = 0
        HRxn[2][142] = 0
        HRxn[2][143] = 0
        HRxn[2][144] = 0
        HRxn[2][145] = 0
        HRxn[2][146] = 0
        HRxn[2][147] = 0
        HRxn[2][148] = 0
        HRxn[2][149] = 0
        HRxn[2][150] = 0
        HRxn[2][151] = 0
        HRxn[2][152] = 0
        HRxn[2][153] = 0
        HRxn[2][154] = 0
        HRxn[2][155] = 0
        HRxn[2][156] = 0
        HRxn[2][157] = 0
        HRxn[2][158] = 2.0*hsp[6] - hsp[7] + 75248503.2
        HRxn[2][159] = hsp[6] - hsp[7] + hsp[8] + 121555274.4



        return HRxn

    def Krxn_(self):
        '''
        Calculate the reaction rate constants in SI units.
        '''
        import math
        nrxn = self.nrxn
        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]

        EeV = 1.5*Te/11600.0 # mean electron energy in eV

        K_rxn = np.zeros(nrxn)

        K_rxn[0] = 8.27e-46*math.exp(500/Tg)
        K_rxn[1] = 0.5
        K_rxn[2] = 152000.0
        K_rxn[3] = 26900000.0
        K_rxn[4] = 9.66666666666667e-17*math.sqrt(3)*math.sqrt(Tg)
        K_rxn[5] = 8.66666666666667e-18*math.sqrt(3)*math.sqrt(Tg)
        K_rxn[6] = 2.90253687419286e-13*(1/Te)**0.41
        K_rxn[7] = 1.26260354027389e-11*(1/Te)**0.41
        K_rxn[8] = 1.59639528080607e-12*(1/Te)**0.41
        K_rxn[9] = 9.82072807891554e-21*(1/Te)**4.5
        K_rxn[10] = 2.0e-12*math.sqrt(3)*math.sqrt(1/Te)
        K_rxn[11] = 1.6647529549595e-12*(1/Te)**0.39
        K_rxn[12] = 2.16792807522129e-16*(1/Te)**0.7
        K_rxn[13] = 3.11769145362398e-35*(1/Te)**1.5
        K_rxn[14] = 2.80592230826158e-20*(1/Te)**4.5
        K_rxn[15] = (5.05e-17*math.sqrt(Te) + 5.555e-22*Te**1.5)*math.exp(-182000.0/Te)
        K_rxn[16] = 1.89693706581863e-16*(1/Te)**0.7
        K_rxn[17] = 3.11769145362398e-35*(1/Te)**1.5
        K_rxn[18] = 2.80592230826158e-20*(1/Te)**4.5
        K_rxn[19] = 6.0e-16
        K_rxn[20] = 3.0e-16
        K_rxn[21] = 4.0e-16
        K_rxn[22] = 2.0e-23
        K_rxn[23] = 2.77847094286545e-15*(1/Tg)**0.666666666666667
        K_rxn[24] = 1.2e-17
        K_rxn[25] = 7.88190616265049e-17*(1/Tg)**0.33
        K_rxn[26] = 2.1e-22*math.exp((1/121)*Tg)
        K_rxn[27] = 1.0e-17
        K_rxn[28] = 1.0e-15
        K_rxn[29] = 6.6e-17
        K_rxn[30] = 6.0e-16
        K_rxn[31] = 1.2e-17
        K_rxn[32] = 5.5e-18
        K_rxn[33] = 7.2e-19*math.exp(300/Tg)
        K_rxn[34] = 5.77362884565987e-37*(1/Tg)**1.64
        K_rxn[35] = 9.0e-42*math.exp(400/Tg)
        K_rxn[36] = 1.0e-19
        K_rxn[37] = 1.8e-36*(1/Tg)**2.0
        K_rxn[38] = 3.0e-39/Tg
        K_rxn[39] = 1.99269345732315e-5*Te**(-1.6)*math.exp(-113200/Te)
        # K_rxn[40] = Bolsig
        # K_rxn[41] = Bolsig
        # K_rxn[42] = Bolsig
        # K_rxn[43] = Bolsig
        # K_rxn[44] = Bolsig
        # K_rxn[45] = Bolsig
        # K_rxn[46] = Bolsig
        # K_rxn[47] = Bolsig
        # K_rxn[48] = Bolsig
        # K_rxn[49] = Bolsig
        K_rxn[50] = 2.8000000000000007e-19
        K_rxn[51] = 2.8000000000000006e-17
        # K_rxn[52] = Bolsig
        # K_rxn[53] = Bolsig

        K_rxn[40] = bolsigFit(1.5*Te/11600.0, [-24.62, -3.201, -54.41, 84.72, -55.07])
        K_rxn[41] = bolsigFit(1.5*Te/11600.0, [-36.53, 0.443, -0.9333, 0.1975, -0.01179])
        K_rxn[42] = bolsigFit(1.5*Te/11600.0, [-22.83, -3.286, -56.28, 87.69, -56.95])
        K_rxn[43] = bolsigFit(1.5*Te/11600.0, [-34.6, 0.2789, -1.035, 0.2008, -0.01158])
        K_rxn[44] = bolsigFit(1.5*Te/11600.0, [-20.25, -3.846, -58.44, 89.11, -57.33])
        K_rxn[45] = bolsigFit(1.5*Te/11600.0, [-32.76, -0.0623, -0.91, 0.1488, -0.008068])
        K_rxn[46] = bolsigFit(1.5*Te/11600.0, [-22.16, -3.462, -59.72, 92.99, -60.27])
        K_rxn[47] = bolsigFit(1.5*Te/11600.0, [-34.23, 0.1644, -1.304, 0.236, -0.0132])
        K_rxn[48] = bolsigFit(1.5*Te/11600.0, [-14.24, -5.166, -83.42, 127.1, -80.94])
        K_rxn[49] = bolsigFit(1.5*Te/11600.0, [-30.64, -0.3308, -2.036, 0.376, -0.02144])
        K_rxn[52] = bolsigFit(1.5*Te/11600.0, [-30.92, -0.9741, -26.67, 32.07, -21.76])
        K_rxn[53] = bolsigFit(1.5*Te/11600.0, [-27.14, -0.6358, -59.16, 94.0, -61.56])


        K_rxn[54] = 5.0000000000000012e-17
        K_rxn[55] = 2.0000000000000005e-16
        K_rxn[56] = 1.05686493667038e-11*(1/Te)**0.7
        K_rxn[57] = 7.27461339178928e-11*math.sqrt(1/Te)
        K_rxn[58] = 2.07846096908265e-9*(1/Te)**1.5
        K_rxn[59] = 2.8059223082615806e-20*Te**(-4.5)
        K_rxn[60] = 1.4000000000000002e-20*Te**(-4.5)
        K_rxn[61] = 3.1000000000000005e-35*Te**(-1.5)
        K_rxn[62] = 3.1000000000000005e-35*Te**(-1.5)
        K_rxn[63] = 3.1000000000000005e-35*Te**(-1.5)
        K_rxn[64] = 3.1000000000000005e-35*Te**(-1.5)
        K_rxn[65] = 3.1000000000000005e-35*Te**(-1.5)
        K_rxn[66] = 4.2e-39*math.exp(-600/Tg)*math.exp((700*Te - 700*Tg)/(Te*Tg))/Te
        K_rxn[67] = 3.21e-41*math.exp(-70/Tg)*math.exp((1500*Te - 1500*Tg)/(Te*Tg))/Te
        K_rxn[68] = 1.0e-43
        K_rxn[69] = 1.0e-43
        K_rxn[70] = 2.3987678116899939e-9*Te**(-1.5)*math.exp(-15080.0/Te)
        K_rxn[71] = 7.3337328409480539e-9*Te**(-1.5)*math.exp(-18444.0/Te)
        K_rxn[72] = 1.09696551146029e-19*math.sqrt(Tg)*math.exp(-4990.0/Tg)
        K_rxn[73] = 1.55884572681199e-17*math.sqrt(Tg)*math.exp(-5590.0/Tg)
        K_rxn[74] = 2.1e-15
        K_rxn[75] = 2.5e-15
        K_rxn[76] = 2.2e-15
        K_rxn[77] = 1.9e-15
        K_rxn[78] = 1.5e-16
        K_rxn[79] = 2.6e-16
        K_rxn[80] = 5.0e-21
        K_rxn[81] = 1.7000000000000004e-18
        K_rxn[82] = 7.5000000000000019e-19
        K_rxn[83] = 3.0000000000000006e-17
        K_rxn[84] = 2.1000000000000003e-17
        K_rxn[85] = 7.0000000000000015e-18
        K_rxn[86] = 3.0000000000000006e-16
        K_rxn[87] = 2.4000000000000005e-16
        K_rxn[88] = 2.5000000000000007e-16
        K_rxn[89] = 4.0000000000000006e-17
        K_rxn[90] = 2.6000000000000006e-17
        K_rxn[91] = 3.3e-16
        K_rxn[92] = 1.0e-16
        K_rxn[93] = 2.5e-16
        K_rxn[94] = 6.0e-17
        K_rxn[95] = 2.0276792496566723e-34*(1/Tg)**3.2
        K_rxn[96] = 3.3000000000000007e-47*Tg**(-3.2)
        K_rxn[97] = 5.0000000000000009e-41
        K_rxn[98] = 0.02673*(1/Tg)**4.0*math.exp(-5030.0/Tg)
        K_rxn[99] = 0.02673*(1/Tg)**4.0*math.exp(-5030.0/Tg)
        K_rxn[100] = 1.0e-41
        K_rxn[101] = 1.0e-41
        K_rxn[102] = 2.8e-16
        K_rxn[103] = 2.5e-16
        K_rxn[104] = 1.0e-18
        K_rxn[105] = 5.0e-16
        K_rxn[106] = 8.0e-16
        K_rxn[107] = 3.0e-18
        K_rxn[108] = 1.0e-18
        K_rxn[109] = 3.0e-18*math.exp(-0.00311*Tg)
        K_rxn[110] = 2.25166604983954e-9*math.sqrt(1/Tg)
        K_rxn[111] = 3.1291346445319e-17*(1/Tg)**0.2
        K_rxn[112] = 1.0e-16
        K_rxn[113] = 3.3e-16
        K_rxn[114] = 1.0e-23
        K_rxn[115] = 1.2e-16
        K_rxn[116] = 4.4e-16
        K_rxn[117] = 2.5e-16
        K_rxn[118] = 4.0e-16
        K_rxn[119] = 3.0e-16
        K_rxn[120] = 1.0e-16
        K_rxn[121] = 3.11769145362398e-31*(1/Tg)**2.5
        K_rxn[122] = 3.11769145362398e-31*(1/Tg)**2.5
        K_rxn[123] = 3.11769145362398e-31*(1/Tg)**2.5
        K_rxn[124] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[125] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[126] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[127] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[128] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[129] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[130] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[131] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[132] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[133] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[134] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[135] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[136] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[137] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[138] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[139] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[140] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[141] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[142] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[143] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[144] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[145] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[146] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[147] = 3.1176914536239803e-31*(1/Tg)**2.5
        K_rxn[148] = 1.1e-20*Tg*math.exp(-3150.0/Tg)
        K_rxn[149] = 2.0e-22
        K_rxn[150] = 1.05e-18*math.sqrt(Tg)
        K_rxn[151] = 2.0e-17*math.exp(-2300.0/Tg)
        K_rxn[152] = 4.38e-46*math.exp(500/Tg)
        K_rxn[153] = 4.38e-46*math.exp(500/Tg)
        K_rxn[154] = 2.76e-46*math.exp(720.0/Tg)
        K_rxn[155] = 2.45e-43*Tg**(-0.63)
        K_rxn[156] = 5.6000000000000016e-41*Tg**(-2.0)
        K_rxn[157] = 8.6000000000000017e-43*Tg**(-1.25)
        K_rxn[158] = 1.0000000000000002e-6*10.0**(-8.31 - 8.4/(2.13859690844233e-9*Te**2 + 6.50097596833211e-5*Te**1.178222))
        K_rxn[159] = 1.0000000000000002e-6*10.0**(-7.86 - 17.2/(2.13859690844233e-9*Te**2 + 6.50097596833211e-5*Te**1.178222))



        return K_rxn

    def Wrxn_(self):
        '''
        Calculate the rate of progress in concentration units / s.
        '''
        nrxn = self.nrxn
        K_rxn = self.Krxn
        Ysp = self.Ysp

        W_rxn = np.zeros(nrxn)
        # Use the output from the makePlasmaKinetics_II.py library
        # If you need to change any reaction rate constants, e.g. using bolsigFit do it here
        W_rxn[0] = K_rxn[0]*Ysp[0]**2*Ysp[1]
        W_rxn[1] = K_rxn[1]*Ysp[2]
        W_rxn[2] = K_rxn[2]*Ysp[3]
        W_rxn[3] = K_rxn[3]*Ysp[4]
        W_rxn[4] = K_rxn[4]*Ysp[2]**2
        W_rxn[5] = K_rxn[5]*Ysp[2]**2
        W_rxn[6] = K_rxn[6]*Ysp[14]*Ysp[21]
        W_rxn[7] = K_rxn[7]*Ysp[14]*Ysp[21]
        W_rxn[8] = K_rxn[8]*Ysp[14]*Ysp[21]
        W_rxn[9] = K_rxn[9]*Ysp[14]*Ysp[21]**2
        W_rxn[10] = K_rxn[10]*Ysp[13]*Ysp[21]
        W_rxn[11] = K_rxn[11]*Ysp[12]*Ysp[21]
        W_rxn[12] = K_rxn[12]*Ysp[12]*Ysp[21]
        W_rxn[13] = K_rxn[13]*Ysp[12]*Ysp[1]*Ysp[21]
        W_rxn[14] = K_rxn[14]*Ysp[12]*Ysp[21]**2
        W_rxn[15] = K_rxn[15]*Ysp[1]*Ysp[21]
        W_rxn[16] = K_rxn[16]*Ysp[11]*Ysp[21]
        W_rxn[17] = K_rxn[17]*Ysp[11]*Ysp[1]*Ysp[21]
        W_rxn[18] = K_rxn[18]*Ysp[11]*Ysp[21]**2
        W_rxn[19] = K_rxn[19]*Ysp[13]*Ysp[2]
        W_rxn[20] = K_rxn[20]*Ysp[12]*Ysp[2]
        W_rxn[21] = K_rxn[21]*Ysp[12]*Ysp[2]
        W_rxn[22] = K_rxn[22]*Ysp[1]*Ysp[2]
        W_rxn[23] = K_rxn[23]*Ysp[0]*Ysp[2]
        W_rxn[24] = K_rxn[24]*Ysp[1]*Ysp[3]
        W_rxn[25] = K_rxn[25]*Ysp[1]*Ysp[4]
        W_rxn[26] = K_rxn[26]*Ysp[14]*Ysp[1]
        W_rxn[27] = K_rxn[27]*Ysp[0]*Ysp[14]
        W_rxn[28] = K_rxn[28]*Ysp[0]*Ysp[14]
        W_rxn[29] = K_rxn[29]*Ysp[0]*Ysp[13]
        W_rxn[30] = K_rxn[30]*Ysp[13]*Ysp[1]
        W_rxn[31] = K_rxn[31]*Ysp[12]*Ysp[1]
        W_rxn[32] = K_rxn[32]*Ysp[12]*Ysp[1]
        W_rxn[33] = K_rxn[33]*Ysp[0]*Ysp[12]
        W_rxn[34] = K_rxn[34]*Ysp[12]*Ysp[1]**2
        W_rxn[35] = K_rxn[35]*Ysp[0]*Ysp[12]*Ysp[1]
        W_rxn[36] = K_rxn[36]*Ysp[11]*Ysp[1]
        W_rxn[37] = K_rxn[37]*Ysp[11]*Ysp[1]**2
        W_rxn[38] = K_rxn[38]*Ysp[0]*Ysp[11]*Ysp[1]
        W_rxn[39] = K_rxn[39]*Ysp[1]*Ysp[21]
        W_rxn[40] = K_rxn[40]*Ysp[1]*Ysp[21]
        W_rxn[41] = K_rxn[41]*Ysp[21]*Ysp[2]
        W_rxn[42] = K_rxn[42]*Ysp[1]*Ysp[21]
        W_rxn[43] = K_rxn[43]*Ysp[21]*Ysp[2]
        W_rxn[44] = K_rxn[44]*Ysp[1]*Ysp[21]
        W_rxn[45] = K_rxn[45]*Ysp[21]*Ysp[3]
        W_rxn[46] = K_rxn[46]*Ysp[1]*Ysp[21]
        W_rxn[47] = K_rxn[47]*Ysp[21]*Ysp[2]
        W_rxn[48] = K_rxn[48]*Ysp[1]*Ysp[21]
        W_rxn[49] = K_rxn[49]*Ysp[21]*Ysp[4]
        W_rxn[50] = K_rxn[50]*Ysp[1]*Ysp[5]
        W_rxn[51] = K_rxn[51]*Ysp[5]*Ysp[7]
        W_rxn[52] = K_rxn[52]*Ysp[1]*Ysp[21]
        W_rxn[53] = K_rxn[53]*Ysp[21]*Ysp[7]
        W_rxn[54] = K_rxn[54]*Ysp[2]*Ysp[5]
        W_rxn[55] = K_rxn[55]*Ysp[5]**2
        W_rxn[56] = K_rxn[56]*Ysp[15]*Ysp[21]
        W_rxn[57] = K_rxn[57]*Ysp[17]*Ysp[21]
        W_rxn[58] = K_rxn[58]*Ysp[18]*Ysp[21]
        W_rxn[59] = K_rxn[59]*Ysp[15]*Ysp[21]**2
        W_rxn[60] = K_rxn[60]*Ysp[16]*Ysp[21]**2
        W_rxn[61] = K_rxn[61]*Ysp[15]*Ysp[21]*Ysp[7]
        W_rxn[62] = K_rxn[62]*Ysp[15]*Ysp[1]*Ysp[21]
        W_rxn[63] = K_rxn[63]*Ysp[12]*Ysp[21]*Ysp[7]
        W_rxn[64] = K_rxn[64]*Ysp[16]*Ysp[1]*Ysp[21]
        W_rxn[65] = K_rxn[65]*Ysp[16]*Ysp[21]*Ysp[7]
        W_rxn[66] = K_rxn[66]*Ysp[21]*Ysp[7]**2
        W_rxn[67] = K_rxn[67]*Ysp[1]*Ysp[21]*Ysp[7]
        W_rxn[68] = K_rxn[68]*Ysp[21]*Ysp[6]*Ysp[7]
        W_rxn[69] = K_rxn[69]*Ysp[21]*Ysp[6]*Ysp[7]
        W_rxn[70] = K_rxn[70]*Ysp[21]*Ysp[9]
        W_rxn[71] = K_rxn[71]*Ysp[21]*Ysp[9]
        W_rxn[72] = K_rxn[72]*Ysp[1]*Ysp[20]
        W_rxn[73] = K_rxn[73]*Ysp[20]*Ysp[7]
        W_rxn[74] = K_rxn[74]*Ysp[20]*Ysp[2]
        W_rxn[75] = K_rxn[75]*Ysp[20]*Ysp[3]
        W_rxn[76] = K_rxn[76]*Ysp[19]*Ysp[2]
        W_rxn[77] = K_rxn[77]*Ysp[19]*Ysp[3]
        W_rxn[78] = K_rxn[78]*Ysp[20]*Ysp[6]
        W_rxn[79] = K_rxn[79]*Ysp[0]*Ysp[19]
        W_rxn[80] = K_rxn[80]*Ysp[19]*Ysp[7]
        W_rxn[81] = K_rxn[81]*Ysp[2]*Ysp[7]
        W_rxn[82] = K_rxn[82]*Ysp[2]*Ysp[7]
        W_rxn[83] = K_rxn[83]*Ysp[2]*Ysp[6]
        W_rxn[84] = K_rxn[84]*Ysp[10]*Ysp[2]
        W_rxn[85] = K_rxn[85]*Ysp[2]*Ysp[6]
        W_rxn[86] = K_rxn[86]*Ysp[3]*Ysp[7]
        W_rxn[87] = K_rxn[87]*Ysp[10]*Ysp[3]
        W_rxn[88] = K_rxn[88]*Ysp[4]*Ysp[7]
        W_rxn[89] = K_rxn[89]*Ysp[7]*Ysp[8]
        W_rxn[90] = K_rxn[90]*Ysp[1]*Ysp[8]
        W_rxn[91] = K_rxn[91]*Ysp[20]*Ysp[6]
        W_rxn[92] = K_rxn[92]*Ysp[19]*Ysp[7]
        W_rxn[93] = K_rxn[93]*Ysp[14]*Ysp[7]
        W_rxn[94] = K_rxn[94]*Ysp[12]*Ysp[7]
        W_rxn[95] = K_rxn[95]*Ysp[15]*Ysp[7]**2
        W_rxn[96] = K_rxn[96]*Ysp[15]*Ysp[1]*Ysp[7]
        W_rxn[97] = K_rxn[97]*Ysp[12]*Ysp[1]*Ysp[7]
        W_rxn[98] = K_rxn[98]*Ysp[17]*Ysp[1]
        W_rxn[99] = K_rxn[99]*Ysp[17]*Ysp[7]
        W_rxn[100] = K_rxn[100]*Ysp[0]*Ysp[16]*Ysp[7]
        W_rxn[101] = K_rxn[101]*Ysp[0]*Ysp[16]*Ysp[1]
        W_rxn[102] = K_rxn[102]*Ysp[11]*Ysp[7]
        W_rxn[103] = K_rxn[103]*Ysp[11]*Ysp[7]
        W_rxn[104] = K_rxn[104]*Ysp[11]*Ysp[6]
        W_rxn[105] = K_rxn[105]*Ysp[11]*Ysp[9]
        W_rxn[106] = K_rxn[106]*Ysp[10]*Ysp[11]
        W_rxn[107] = K_rxn[107]*Ysp[10]*Ysp[11]
        W_rxn[108] = K_rxn[108]*Ysp[10]*Ysp[11]
        W_rxn[109] = K_rxn[109]*Ysp[16]*Ysp[1]
        W_rxn[110] = K_rxn[110]*Ysp[12]*Ysp[6]
        W_rxn[111] = K_rxn[111]*Ysp[12]*Ysp[6]
        W_rxn[112] = K_rxn[112]*Ysp[12]*Ysp[9]
        W_rxn[113] = K_rxn[113]*Ysp[10]*Ysp[12]
        W_rxn[114] = K_rxn[114]*Ysp[15]*Ysp[1]
        W_rxn[115] = K_rxn[115]*Ysp[0]*Ysp[15]
        W_rxn[116] = K_rxn[116]*Ysp[10]*Ysp[15]
        W_rxn[117] = K_rxn[117]*Ysp[14]*Ysp[6]
        W_rxn[118] = K_rxn[118]*Ysp[10]*Ysp[14]
        W_rxn[119] = K_rxn[119]*Ysp[17]*Ysp[6]
        W_rxn[120] = K_rxn[120]*Ysp[10]*Ysp[17]
        W_rxn[121] = K_rxn[121]*Ysp[15]*Ysp[20]*Ysp[7]
        W_rxn[122] = K_rxn[122]*Ysp[15]*Ysp[1]*Ysp[20]
        W_rxn[123] = K_rxn[123]*Ysp[17]*Ysp[20]*Ysp[7]
        W_rxn[124] = K_rxn[124]*Ysp[12]*Ysp[1]*Ysp[20]
        W_rxn[125] = K_rxn[125]*Ysp[12]*Ysp[20]*Ysp[7]
        W_rxn[126] = K_rxn[126]*Ysp[15]*Ysp[1]*Ysp[20]
        W_rxn[127] = K_rxn[127]*Ysp[15]*Ysp[20]*Ysp[7]
        W_rxn[128] = K_rxn[128]*Ysp[11]*Ysp[1]*Ysp[20]
        W_rxn[129] = K_rxn[129]*Ysp[11]*Ysp[20]*Ysp[7]
        W_rxn[130] = K_rxn[130]*Ysp[16]*Ysp[1]*Ysp[20]
        W_rxn[131] = K_rxn[131]*Ysp[16]*Ysp[20]*Ysp[7]
        W_rxn[132] = K_rxn[132]*Ysp[18]*Ysp[1]*Ysp[20]
        W_rxn[133] = K_rxn[133]*Ysp[18]*Ysp[20]*Ysp[7]
        W_rxn[134] = K_rxn[134]*Ysp[12]*Ysp[19]*Ysp[1]
        W_rxn[135] = K_rxn[135]*Ysp[12]*Ysp[19]*Ysp[7]
        W_rxn[136] = K_rxn[136]*Ysp[15]*Ysp[19]*Ysp[1]
        W_rxn[137] = K_rxn[137]*Ysp[15]*Ysp[19]*Ysp[7]
        W_rxn[138] = K_rxn[138]*Ysp[11]*Ysp[19]*Ysp[1]
        W_rxn[139] = K_rxn[139]*Ysp[11]*Ysp[19]*Ysp[7]
        W_rxn[140] = K_rxn[140]*Ysp[16]*Ysp[19]*Ysp[1]
        W_rxn[141] = K_rxn[141]*Ysp[16]*Ysp[19]*Ysp[7]
        W_rxn[142] = K_rxn[142]*Ysp[18]*Ysp[19]*Ysp[1]
        W_rxn[143] = K_rxn[143]*Ysp[18]*Ysp[19]*Ysp[7]
        W_rxn[144] = K_rxn[144]*Ysp[11]*Ysp[19]*Ysp[1]
        W_rxn[145] = K_rxn[145]*Ysp[11]*Ysp[19]*Ysp[7]
        W_rxn[146] = K_rxn[146]*Ysp[16]*Ysp[19]*Ysp[1]
        W_rxn[147] = K_rxn[147]*Ysp[16]*Ysp[19]*Ysp[7]
        W_rxn[148] = K_rxn[148]*Ysp[0]*Ysp[7]
        W_rxn[149] = K_rxn[149]*Ysp[0]*Ysp[9]
        W_rxn[150] = K_rxn[150]*Ysp[0]*Ysp[10]
        W_rxn[151] = K_rxn[151]*Ysp[6]*Ysp[9]
        W_rxn[152] = K_rxn[152]*Ysp[0]**2*Ysp[7]
        W_rxn[153] = K_rxn[153]*Ysp[0]**2*Ysp[7]
        W_rxn[154] = K_rxn[154]*Ysp[1]*Ysp[6]**2
        W_rxn[155] = K_rxn[155]*Ysp[6]**2*Ysp[7]
        W_rxn[156] = K_rxn[156]*Ysp[1]*Ysp[6]*Ysp[7]
        W_rxn[157] = K_rxn[157]*Ysp[6]*Ysp[7]**2
        W_rxn[158] = K_rxn[158]*Ysp[21]*Ysp[7]
        W_rxn[159] = K_rxn[159]*Ysp[21]*Ysp[7]


        


        return W_rxn

    def dYdt_(self):
        '''
        Calculate the change in species concentration in concentration units / s -- mostly numdensity/s.
        '''
        nrxn = self.nrxn
        nsp = self.nsp
        W_rxn = self.Wrxn

        dYdt = np.zeros(nsp)
        ## From the output of the make plasma kinetics library

        dYdt[0] = -2*W_rxn[0] - W_rxn[100] - W_rxn[101] + W_rxn[102] + W_rxn[104] + W_rxn[106] + W_rxn[109] + W_rxn[10] + W_rxn[110] - W_rxn[115] + 2*W_rxn[11] + W_rxn[128] + W_rxn[129] + W_rxn[138] + W_rxn[139] - W_rxn[148] - W_rxn[149] - W_rxn[150] - 2*W_rxn[152] - 2*W_rxn[153] + W_rxn[16] + W_rxn[17] + W_rxn[18] + W_rxn[20] + W_rxn[21] - W_rxn[27] - W_rxn[28] - W_rxn[29] + W_rxn[31] + W_rxn[32] - W_rxn[33] - W_rxn[35] + W_rxn[36] - W_rxn[38] + 2*W_rxn[39] + W_rxn[58] - W_rxn[79] + W_rxn[85]
        dYdt[1] = W_rxn[108] - W_rxn[109] + W_rxn[10] + W_rxn[111] + W_rxn[112] + W_rxn[113] - W_rxn[114] + 2*W_rxn[117] + 2*W_rxn[118] + W_rxn[124] + W_rxn[125] + W_rxn[12] + W_rxn[134] + W_rxn[135] + W_rxn[13] + W_rxn[14] + W_rxn[150] + W_rxn[153] - W_rxn[15] + 2*W_rxn[19] + W_rxn[1] + W_rxn[21] + W_rxn[22] + W_rxn[23] + W_rxn[26] + 2*W_rxn[27] + W_rxn[28] + W_rxn[29] + W_rxn[30] - W_rxn[32] + W_rxn[33] - W_rxn[34] - W_rxn[36] - W_rxn[37] - W_rxn[39] - W_rxn[40] + W_rxn[41] - W_rxn[42] + W_rxn[43] - W_rxn[44] + W_rxn[45] - W_rxn[46] + W_rxn[47] - W_rxn[48] + W_rxn[49] + W_rxn[4] + W_rxn[51] - W_rxn[52] + W_rxn[5] + W_rxn[63] + W_rxn[6] + W_rxn[74] + W_rxn[75] + W_rxn[76] + W_rxn[77] + W_rxn[7] + W_rxn[81] + W_rxn[82] + W_rxn[83] + W_rxn[84] + W_rxn[86] + W_rxn[88] + W_rxn[8] + 2*W_rxn[93] + W_rxn[94] - W_rxn[97] + 2*W_rxn[9]
        dYdt[2] = W_rxn[152] - W_rxn[19] - W_rxn[1] - W_rxn[20] - W_rxn[21] - W_rxn[22] - W_rxn[23] + W_rxn[24] + W_rxn[2] + W_rxn[40] - W_rxn[41] + W_rxn[42] - W_rxn[43] + W_rxn[46] - W_rxn[47] - 2*W_rxn[4] - W_rxn[54] - 2*W_rxn[5] + W_rxn[6] - W_rxn[74] - W_rxn[76] - W_rxn[81] - W_rxn[82] - W_rxn[83] - W_rxn[84] - W_rxn[85] + W_rxn[87]
        dYdt[3] = W_rxn[0] - W_rxn[24] + W_rxn[25] - W_rxn[2] + W_rxn[3] + W_rxn[44] - W_rxn[45] + W_rxn[4] + W_rxn[50] - W_rxn[75] - W_rxn[77] + W_rxn[7] - W_rxn[86] - W_rxn[87]
        dYdt[4] = -W_rxn[25] - W_rxn[3] + W_rxn[48] - W_rxn[49] + W_rxn[5] - W_rxn[88] + W_rxn[8]
        dYdt[5] = -W_rxn[50] - W_rxn[51] + W_rxn[52] - W_rxn[54] - 2*W_rxn[55]
        dYdt[6] = W_rxn[103] - W_rxn[104] + W_rxn[107] - W_rxn[110] - W_rxn[111] + W_rxn[112] + W_rxn[115] - W_rxn[117] - W_rxn[119] + 2*W_rxn[121] + 2*W_rxn[122] + 2*W_rxn[123] + W_rxn[130] + W_rxn[131] + W_rxn[134] + W_rxn[135] + W_rxn[136] + W_rxn[137] + W_rxn[138] + W_rxn[139] + 2*W_rxn[140] + 2*W_rxn[141] + W_rxn[142] + W_rxn[143] + W_rxn[148] + W_rxn[150] - W_rxn[151] - 2*W_rxn[154] - 2*W_rxn[155] - W_rxn[156] - W_rxn[157] + 2*W_rxn[158] + W_rxn[159] + W_rxn[51] + 2*W_rxn[56] + W_rxn[58] + 2*W_rxn[59] + W_rxn[60] + W_rxn[64] + W_rxn[65] - W_rxn[69] + W_rxn[70] + W_rxn[76] + W_rxn[77] - W_rxn[78] + 2*W_rxn[81] - W_rxn[83] - W_rxn[85] + 2*W_rxn[86] + W_rxn[88] + W_rxn[89] + W_rxn[90] - W_rxn[91] + W_rxn[92]
        dYdt[7] = -W_rxn[102] - W_rxn[103] + W_rxn[105] + W_rxn[116] + 2*W_rxn[120] + W_rxn[121] + W_rxn[122] + 2*W_rxn[123] + W_rxn[124] + W_rxn[125] + 2*W_rxn[126] + 2*W_rxn[127] + W_rxn[128] + W_rxn[129] + W_rxn[130] + W_rxn[131] + W_rxn[132] + W_rxn[133] + W_rxn[136] + W_rxn[137] + W_rxn[146] + W_rxn[147] - W_rxn[148] + W_rxn[149] + 2*W_rxn[151] + W_rxn[154] + W_rxn[155] - W_rxn[156] - W_rxn[157] - W_rxn[158] - W_rxn[159] - W_rxn[51] - W_rxn[53] + 2*W_rxn[57] + W_rxn[61] + W_rxn[62] - W_rxn[66] - W_rxn[67] - W_rxn[68] + W_rxn[71] + W_rxn[72] + W_rxn[73] + W_rxn[74] + W_rxn[75] - W_rxn[80] - W_rxn[81] - W_rxn[86] - W_rxn[88] + W_rxn[91] - W_rxn[92] - W_rxn[93] - W_rxn[94] - W_rxn[95] - W_rxn[96] + W_rxn[98] + W_rxn[99]
        dYdt[8] = W_rxn[159] + W_rxn[51] + W_rxn[83] + W_rxn[88] - W_rxn[89] - W_rxn[90]
        dYdt[9] = -W_rxn[105] - W_rxn[112] + W_rxn[119] - W_rxn[149] - W_rxn[151] + W_rxn[156] + W_rxn[157] - W_rxn[70] - W_rxn[71] + W_rxn[78] + W_rxn[80]
        dYdt[10] = -W_rxn[106] - W_rxn[107] - W_rxn[108] - W_rxn[113] + W_rxn[114] - W_rxn[116] - W_rxn[118] - W_rxn[120] + W_rxn[132] + W_rxn[133] + W_rxn[142] + W_rxn[143] + W_rxn[144] + W_rxn[145] + W_rxn[148] + W_rxn[149] - W_rxn[150] + W_rxn[79] + W_rxn[85]
        dYdt[11] = -W_rxn[102] - W_rxn[103] - W_rxn[104] - W_rxn[105] - W_rxn[106] - W_rxn[107] - W_rxn[108] - W_rxn[128] - W_rxn[129] - W_rxn[138] - W_rxn[139] - W_rxn[144] - W_rxn[145] - W_rxn[16] - W_rxn[17] - W_rxn[18] + W_rxn[19] + W_rxn[21] + W_rxn[27] + W_rxn[30] + W_rxn[31] + W_rxn[33] - W_rxn[36] - W_rxn[37] - W_rxn[38]
        dYdt[12] = W_rxn[107] - W_rxn[110] - W_rxn[111] - W_rxn[112] - W_rxn[113] - W_rxn[11] - W_rxn[124] - W_rxn[125] - W_rxn[12] - W_rxn[134] - W_rxn[135] - W_rxn[13] - W_rxn[14] + W_rxn[15] - W_rxn[20] - W_rxn[21] + W_rxn[26] + W_rxn[29] - W_rxn[31] - W_rxn[32] - W_rxn[33] - W_rxn[34] - W_rxn[35] + W_rxn[36] + W_rxn[38] - W_rxn[63] - W_rxn[94] - W_rxn[97]
        dYdt[13] = -W_rxn[10] - W_rxn[19] + W_rxn[20] + W_rxn[28] - W_rxn[29] - W_rxn[30] + W_rxn[32] + W_rxn[35] + W_rxn[37]
        dYdt[14] = -W_rxn[117] - W_rxn[118] - W_rxn[26] - W_rxn[27] - W_rxn[28] + W_rxn[34] + W_rxn[54] + W_rxn[55] - W_rxn[6] - W_rxn[7] - W_rxn[8] - W_rxn[93] + W_rxn[97] - W_rxn[9]
        dYdt[15] = W_rxn[102] + W_rxn[112] - W_rxn[114] - W_rxn[115] - W_rxn[116] + W_rxn[119] - W_rxn[121] - W_rxn[122] - W_rxn[126] - W_rxn[127] - W_rxn[136] - W_rxn[137] + W_rxn[53] - W_rxn[56] - W_rxn[59] - W_rxn[61] - W_rxn[62] + W_rxn[93] + W_rxn[94] - W_rxn[95] - W_rxn[96] + W_rxn[98] + W_rxn[99]
        dYdt[16] = -W_rxn[100] - W_rxn[101] + W_rxn[104] + W_rxn[108] - W_rxn[109] + W_rxn[111] + W_rxn[117] - W_rxn[130] - W_rxn[131] - W_rxn[140] - W_rxn[141] - W_rxn[146] - W_rxn[147] - W_rxn[60] - W_rxn[64] - W_rxn[65]
        dYdt[17] = -W_rxn[119] - W_rxn[120] - W_rxn[123] - W_rxn[57] + W_rxn[95] + W_rxn[96] - W_rxn[98] - W_rxn[99]
        dYdt[18] = W_rxn[100] + W_rxn[101] + W_rxn[103] + W_rxn[105] + W_rxn[106] + W_rxn[109] + W_rxn[110] + W_rxn[113] + W_rxn[114] + W_rxn[115] + W_rxn[116] + W_rxn[118] + W_rxn[120] - W_rxn[132] - W_rxn[133] - W_rxn[142] - W_rxn[143] - W_rxn[58]
        dYdt[19] = -W_rxn[134] - W_rxn[135] - W_rxn[136] - W_rxn[137] - W_rxn[138] - W_rxn[139] - W_rxn[140] - W_rxn[141] - W_rxn[142] - W_rxn[143] - W_rxn[144] - W_rxn[145] - W_rxn[146] - W_rxn[147] + W_rxn[69] + W_rxn[71] - W_rxn[76] - W_rxn[77] - W_rxn[79] - W_rxn[80] + W_rxn[91] - W_rxn[92]
        dYdt[20] = -W_rxn[121] - W_rxn[122] - W_rxn[123] - W_rxn[124] - W_rxn[125] - W_rxn[126] - W_rxn[127] - W_rxn[128] - W_rxn[129] - W_rxn[130] - W_rxn[131] - W_rxn[132] - W_rxn[133] + W_rxn[66] + W_rxn[67] + W_rxn[68] + W_rxn[70] - W_rxn[72] - W_rxn[73] - W_rxn[74] - W_rxn[75] - W_rxn[78] - W_rxn[91] + W_rxn[92]
        dYdt[21] = -W_rxn[10] - W_rxn[11] - W_rxn[12] - W_rxn[13] - W_rxn[14] + W_rxn[15] - W_rxn[16] - W_rxn[17] - W_rxn[18] + W_rxn[53] + W_rxn[54] + W_rxn[55] - W_rxn[56] - W_rxn[57] - W_rxn[58] - W_rxn[59] - W_rxn[60] - W_rxn[61] - W_rxn[62] - W_rxn[63] - W_rxn[64] - W_rxn[65] - W_rxn[66] - W_rxn[67] - W_rxn[68] - W_rxn[69] - W_rxn[6] - W_rxn[70] - W_rxn[71] + W_rxn[72] + W_rxn[73] + W_rxn[74] + W_rxn[75] + W_rxn[76] + W_rxn[77] + W_rxn[78] + W_rxn[79] - W_rxn[7] + W_rxn[80] - W_rxn[8] - W_rxn[9]

        return dYdt
    

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
        p = self.p

        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]

        nN2 = self.Ysp[self.gas.species_index('N2')]
        ne = self.Ysp[self.gas.species_index('ele')]

        eID = self.gas.species_index('ele')
        # Tv = self.Temp[1]
        dEv_dTv = self.dEv_dTv_(Tv,self.n2Tvib_c) # vibrational energy per particle derivative with respect to temperature

        QVT = self.Qmodes['VT']
        QET = self.Qmodes['ET']
        QEV = self.Qmodes['EV']
        QVE = self.Qmodes['VE']

        if self.verbose:
            # show these values
            print('Qdotg = ', Qdotg, 'Qdotv = ', Qdotv, 'Qdote = ', Qdote)
            print('QVT = ', QVT, 'QET = ', QET, 'QEV = ', QEV, 'QVE = ', QVE)

        # d()dt:
        Ce = 1.5*kB*ne
        dydt = np.zeros(np.shape(self.Temp))

        dydt[0] = (Qdotg + QET + QVT)/(cp*rho)
        dydt[1] = ((Qdotv + QEV - QVE - QVT)/nN2)/dEv_dTv
        dydt[2] = (Qdote + QVE - QET - QEV)/Ce # - self.dYdt[eID]*Te/ne
        # dydt[2] = (Qdote + QVE - QET - 0.0*QEV)/Ce - self.dYdt[eID]*Te/ne

        # The dne/dt term is required if solving both Ysp and Temp simultaneously.
        # When the electron density is already modified then it is not required ?

        

        # If laser heating is included
        coLn = 10.0
        # calculate the nueff
        #  neuc[cellI] = 2.91e-12*n_e[cellI]*coLn.value()*(Foam::pow(Te[cellI]/11600,-3.0/2.0));
        neuc = 2.91e-12*ne*coLn*(Te/11600.0)**(-1.5)
        # neum = 9.8e-14*p[cellI]/(kB.value()*T[cellI])*(Foam::pow(Te[cellI]/11600.0,1.0/2.0)); 
        neum = 9.8e-14*p/(kB*Tg)*(Te/11600.0)**(0.5)

        nueff = neuc + neum
        Q = 0.0

        if self.laser.switch == 1:
            # calculate the laser absorption
            Q = self.laser.laserAbsorption(ne,nueff)

            dydt[2] = dydt[2] + Q/Ce


        return dydt



        

    def JacY_(self):
        '''
        Calculate the Jacobian for the species concentration.
        '''
        pass

    def JacT_(self):
        '''
        Calculate the Jacobian for the temperature.
        '''
        pass


####------------------------------------------------------------------------------------------------------------------####

## Create a new plasmaSystem. 
# Derived from the N2Plasma_54Rxn class.
class N2Plasma_54Rxn_2(N2Plasma_54Rxn):
    '''
    The heat of reaction from the recombination heating goes to bulk gas here while in the main class, i.e 
    N2Plasma_54Rxn, it goes to the electron temperature.

    Method Changes:
    ---------------
    1. The heat of reaction calculation only changes and other methods remain the same.

    '''

    def __init__(self, Ysp, T, verbose = False):
        N2Plasma_54Rxn.__init__(self, Ysp, T, verbose)


    # changed method is dhrxn_
    def dhrxn_(self):
        '''
        Calculate the enthalpy of reaction in J/kmol for each reaction.
        For reactions included in Qmodes_() the enthalpy of reaction should be zero.
        '''
        nrxn = self.nrxn
        hsp = self.hsp


        # get from the makePlasmaKinetics_II.py library
        # Requires definition of enthalpy for new subgroup , e.g. N2(A,v>9)
        HRxn = np.zeros((3, nrxn))

        # SubGroup enthalpy: N2A
        # [hN2A_vG1, hN2A_vG2, hN2A_vG3] = N2A(v=0-4) , N2A(v=5-9) , N2A(v>9)
        # Nasa7 polynomials use with NASA7Enthalpy(x,T,Tswitch=None): - > J/mol but needed in J/kmol, T needs to be array
        # NASA7Enthalpy outputs an array
        hN2A_vG1 = NASA7Enthalpy(self.N2Avg1NASA7, [self.Temp[0]], 1000.0)[0]*1.0e3
        hN2A_vG2 = NASA7Enthalpy(self.N2Avg2NASA7, [self.Temp[0]], 1000.0)[0]*1.0e3
        hN2A_vG3 = NASA7Enthalpy(self.N2Avg3NASA7, [self.Temp[0]], 1000.0)[0]*1.0e3

        fRecTg = 0.4  # fraction of recombination energy going to bulk gas

        # The ones with zero are not changed
        # Tg ----
        HRxn[0][0] = -2.0*hsp[0] + hsp[3]
        HRxn[0][1] = 0
        HRxn[0][2] = 0
        HRxn[0][3] = 0
        HRxn[0][4] = 0.7*hsp[1] - 1.4*hsp[2] + 0.7*hsp[3]
        HRxn[0][5] = 0
        HRxn[0][6] = hsp[1] + hsp[2] - hsp[8] - hsp[9]
        HRxn[0][7] = hsp[1] + hsp[3] - hsp[8] - hsp[9]
        HRxn[0][8] = hsp[1] + hsp[4] - hsp[8] - hsp[9]
        HRxn[0][9] = fRecTg*(2.0*hsp[1] - hsp[8] - hsp[9])
        HRxn[0][10] = hsp[0] + hsp[1] - hsp[7] - hsp[9]
        HRxn[0][11] = 2.0*hsp[0] - hsp[6] - hsp[9]
        HRxn[0][12] = 0
        HRxn[0][13] = hsp[1] - hsp[6] - hsp[9]
        HRxn[0][14] = hsp[1] - hsp[6] - hsp[9]
        HRxn[0][15] = 0
        HRxn[0][16] = 0
        HRxn[0][17] = hsp[0] - hsp[5] - hsp[9]
        HRxn[0][18] = hsp[0] - hsp[5] - hsp[9]
        HRxn[0][19] = 2.0*hsp[1] - hsp[2] + hsp[5] - hsp[7]
        HRxn[0][20] = hsp[0] - hsp[2] - hsp[6] + hsp[7]
        HRxn[0][21] = hsp[0] + hsp[1] - hsp[2] + hsp[5] - hsp[6]
        HRxn[0][22] = hsp[1] - hsp[2]
        HRxn[0][23] = hsp[1] - hsp[2]
        HRxn[0][24] = 0
        HRxn[0][25] = 0
        HRxn[0][26] = hsp[1] + hsp[6] - hsp[8]  ## this heats
        HRxn[0][27] = -hsp[0] + 2.0*hsp[1] + hsp[5] - hsp[8]
        HRxn[0][28] = -hsp[0] + hsp[1] + hsp[7] - hsp[8]
        HRxn[0][29] = -hsp[0] + hsp[1] + hsp[6] - hsp[7]
        HRxn[0][30] = hsp[1] + hsp[5] - hsp[7]
        HRxn[0][31] = hsp[0] + hsp[5] - hsp[6]
        HRxn[0][32] = hsp[0] - hsp[1] - hsp[6] + hsp[7]
        HRxn[0][33] = -hsp[0] + hsp[1] + hsp[5] - hsp[6]
        HRxn[0][34] = -hsp[1] - hsp[6] + hsp[8]
        HRxn[0][35] = -hsp[0] - hsp[6] + hsp[7]
        HRxn[0][36] = hsp[0] - hsp[1] - hsp[5] + hsp[6]
        HRxn[0][37] = -hsp[1] - hsp[5] + hsp[7]
        HRxn[0][38] = -hsp[0] - hsp[5] + hsp[6]
        HRxn[0][39] = -96472440.0

        # # # reaction 6 - 18 except 11 make zero
        # zeroIDs = [6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18,23,28,29,35,37,38]
        # for i in zeroIDs:
        #     HRxn[0][i] = 0.0
        # zeroIDs = np.array([1,24,38]) -1
        # for i in zeroIDs:
        #     HRxn[0][i] = 0.0

        # nonZerosRxn = np.array([5,12,40]) - 1
        # # except nonZerosRxn, make zero
        # for i in range(nrxn):
        #     if i not in (nonZerosRxn):
        #         HRxn[0][i] = 0.0

        # Tv ----
        HRxn[1][0] = 0

        HRxn[1][4] = 0.3*hsp[1] - 0.6*hsp[2] + 0.3*hsp[3]
        HRxn[1][5] = hsp[1] - 2.0*hsp[2] + hsp[4]

        HRxn[1][24] = hsp[2] - hsp[3]
        HRxn[1][25] = hsp[3] - hsp[4]

        # HRxn[1][41] = -27525341.0 # included in Qmodes
        # HRxn[1][42] = 27525341.0 # included in Qmodes

        # Te ----
        HRxn[2][15] = -hsp[1] + hsp[6] + hsp[9]
        HRxn[2][39] = 2.0*hsp[0] - hsp[1] + 96472440.0
        HRxn[2][43] = hN2A_vG1 - hsp[1]
        HRxn[2][44] = -hN2A_vG1 + hsp[1]
        HRxn[2][45] = hN2A_vG2 - hsp[1]
        HRxn[2][46] = -hN2A_vG2 + hsp[1]
        HRxn[2][47] = -hsp[1] + hsp[3]
        HRxn[2][48] = hsp[1] - hsp[3]
        HRxn[2][49] = hN2A_vG3 - hsp[1]
        HRxn[2][50] = -hN2A_vG3 + hsp[1]

        HRxn[2][51] = -hsp[1] + hsp[4]
        HRxn[2][52] = hsp[1] - hsp[4]

        HRxn[2][9] = (1.0 - fRecTg)*(2.0*hsp[1] - hsp[8] - hsp[9]) # from gas temperature


        return HRxn


#####_-------------------------------------







###-----------------------------------------
# Now the derived class for the plasma kinetics
class N2PlasmaAleks(System):
    '''
    This is the plasma kinetics for N2.
    '''
    nrxn_ = 67
    mech_ = 'N2PlasmaMechAleks_build.yaml'

    # Extra things needed
    #SubGroup enthalpy: N2A
    # Nasa7 polynomials use with NASA7Enthalpy() -> J/mol
    N2Avg1NASA7 = np.array([[3.07390924e+00,1.99120676e-03,-2.08768566e-06,2.62439456e-09,-1.34639011e-12,7.05970674e+04,6.42271362e+00],
                    [3.50673966e+00,1.30229712e-03,-6.69945124e-07,1.24320721e-10,-7.97758691e-15,7.03934476e+04,3.91162118e+00]])
    N2Avg2NASA7 = np.array([[2.73767346e+00,4.27761848e-03,-6.82914221e-06,6.93754842e-09,-2.81752532e-12,8.05427151e+04,7.90284209e+00],
                    [3.77161049e+00,1.00425875e-03,-5.74156450e-07,1.11907972e-10,-7.44792998e-15,8.02048573e+04,2.47921910e+00]])
    N2Avg3NASA7 = np.array([[2.64595791e+00,4.87994632e-03,-7.18000185e-06,6.35412823e-09,-2.33438726e-12,8.94758420e+04,8.34579376e+00],
                    [4.10955639e+00,2.57529645e-04,8.95120840e-09,-1.09217976e-11,5.27900491e-16,8.90514137e+04,8.01508324e-01]])

    # Vibrational energy of N2 in eV for different vibrational level v= 0-8 - values in eV
    N2v_UeV = np.array([0.14576868,0.43464128,0.71995946,1.00172153,1.27992582,1.55457065,1.82565433,2.09317519,2.35713153])

    n2Tvib_c = 3393.456 # K is the characteristic temperature for vibrational excitation of N2 - data from matlab code and calculated with : omega_e*100*h*c/kB
    
    def __init__(self, Ysp, T,laserObj=None, verbose = False):
        self.laser = LaserModel(switch=0)
        System.__init__(self, Ysp, T, self.mech_, self.nrxn_, verbose)

        if laserObj is not None:
            self.laser = laserObj        

        

    # # Implement the abstract methods
    # def Qrxn_(self):
    #     '''
    #     Calculate the heating rate for each reaction in J/m3-s.
    #     dhrxn is in J/kmol and Ysp is in number density.
    #     Wrxn is in numDensity/s and Temp is in K.
    #     -- can be put in the base class --
    #     '''
    #     gas = self.gas

    #     # Qrxn = -(self.Wrxn/Na)*self.dhrxn*1.0e-3 # J/m3-s

    #     QrxnTg = -self.Wrxn/Na*self.dhrxn[0,:]*1.0e-3 # J/m3-s
    #     QrxnTv = -self.Wrxn/Na*self.dhrxn[1,:]*1.0e-3 # J/m3-s
    #     QrxnTe = -self.Wrxn/Na*self.dhrxn[2,:]*1.0e-3 # J/m3-s

    #     Qrxn = np.array([QrxnTg, QrxnTv, QrxnTe])


    #     ## Qrxn changes here so should be abstract method
    #     # sum all the reactions, so sum over the rows
    #     # J/m3-s
    #     # Qdot = np.sum(Qrxn, axis = 1)

    #     Qdotg = np.sum(QrxnTg)
    #     Qdotv = np.sum(QrxnTv)
    #     Qdote = np.sum(QrxnTe)

    #     Qdot = np.array([Qdotg, Qdotv, Qdote])
        
    #     # validate shape of Qdot should have 3 elements
    #     if Qdot.shape[0] != 3:
    #         raise ValueError('Qdot should have 3 elements')

    #     # if self.verbose:
    #     #     print('Qdot = ', Qdot)
    #     #     print('Shape of Qdot = ', Qdot.shape)
    #     #     print ('Qrxn = ', Qrxn)
    #     #     print('Shape of Qrxn = ', Qrxn.shape)
    #     #     print('dhrxn = ', self.dhrxn)
    #     #     print('Wrxn = ', self.Wrxn)


    #     return Qrxn, Qdot

    # New method for energy exchange without species production/loss
    # Eg, RXN = 41,42,43,54 these are the processes related to : E->T, E-V, V->E, V->T
    # overwrite from the base class
    # @overrides(System)
    def Qmodes_(self):
        '''
        Calculate the heating rate for specific reaction which involve only energy exchange and no kinetics in J/m3-s.
        These are categorized as, ET, EV, VE, VT. Here reactions correspond to : 41, 42, 43, 54
        Positive means heating and negative means cooling. And the naming shows the direction of energy flow. e.g. ET means energy from electron to thermal.
        Note: the corresponding HRxn should be made zero. --> Check dhrxn_()
        '''
        gas = self.gas
        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]
        p_atm = self.p*(1.0/101325.0) # atm

        meanEe = 1.5*Te/11600.0 # mean energy of electron in eV   

        eID = gas.species_index('ele')
        n2ID = gas.species_index('N2')

        m_n2 = gas.molecular_weights[n2ID]/(Na*1000.0) # kg per particle

        ne = self.Ysp[eID]
        nN2 = self.Ysp[n2ID]

        

        # energy for vibrational excitation from ground state of N2
        dU_eV = self.N2v_UeV[1:] - self.N2v_UeV[0]

        # change to J per molecule
        dU_J = dU_eV*1.60217662e-19

        # population ratio of vibrational levels : look Boyd - nonequiiibrium gas dynamics - page 119
        vs = np.arange(1,9)
        n2v_n2 = np.exp(-vs*self.n2Tvib_c/Tv)*(1.0 - np.exp(-self.n2Tvib_c/Tv))

        # For VE excitation
        # group v = 0 and 1 but v=0 cannot be exchanged
        # n2v_n2[0] = n2v_n2[0] + 1.0*(1.0 - np.exp(-self.n2Tvib_c/Tv))

        # ET : rxn 41 :
        # Rxn 41 is elastic collsion : Here it is rxn 40 :1.0e22 per m3 BolsigFit ( results in SI ): -29.87      0.2346     -0.6179      0.9290E-01 -0.4900E-02
        r40Bolsig = np.array([-29.87, 0.2346, -0.6179, 0.9290E-01, -0.4900E-02]) # m3/s

        # Qet with maxwellean distribution 0 - 5ev: -29.97      0.3164     -0.4094      0.4047E-01 -0.1363E-02
        r40Bolsig_m = np.array([-29.97, 0.3164, -0.4094, 0.4047E-01, -0.1363E-02]) # m3/s
        Kr40 = self.bolsigFit(meanEe, r40Bolsig)
        # Kr40 = self.bolsigFit(meanEe, r40Bolsig_m)
        neu_el = Kr40*nN2 # only elastic collisions
        del_n2 = 2.0*m_e/m_n2
        QET = 1.5*kB*ne*neu_el*(Te-Tg)*del_n2

        # EV : Has 8 reaction rate constants : Rxn : 42
        # 8 bolsig fits
        # v1 has two : -29.19      -1.851      -6.737       1.286     -0.7418E-01
        # v1 :  -36.24      0.5479     -0.6106     -0.5111E-01  0.6544E-02
        # [higher value] v1 0.29 eV , which is resonant ?: -26.82      -2.782      -9.153      0.2181      0.8506E-01
        # v2, 0.59 eV : -25.40      -3.553      -12.10     -0.7369E-01  0.1797   
        # v3,0.88 eV  :-32.80      -1.082       4.705      -13.03       2.778
        # v4, 1.17 eV : -28.85      -2.729      -3.144      -12.11       2.901   
        # v5, 1.47 eV :  -46.02       3.238       38.00      -45.25       11.16  
        # v6, 1.76 eV :  -69.11       11.00       99.19      -104.3       28.15 
        # v7, 2.06 eV :  -80.44       14.60       128.8      -135.5       38.17   
        # v8, 2.35 eV :  -22.70      -4.335      -49.60       87.94      -65.61 
        ev1bfit = np.array([-29.19, -1.851, -6.737, 1.286, -0.7418E-01])
        ev1_2bfit = np.array([-36.24, 0.5479, -0.6106, -0.5111E-01, 0.6544E-02])
        ev2bfit = np.array([-25.40, -3.553, -12.10, -0.7369E-01, 0.1797])
        ev3bfit = np.array([-32.80, -1.082, 4.705, -13.03, 2.778])
        ev4bfit = np.array([-28.85, -2.729, -3.144, -12.11, 2.901])
        ev5bfit = np.array([-46.02, 3.238, 38.00, -45.25, 11.16])
        ev6bfit = np.array([-69.11, 11.00, 99.19, -104.3, 28.15])
        ev7bfit = np.array([-80.44, 14.60, 128.8, -135.5, 38.17])
        ev8bfit = np.array([-22.70, -4.335, -49.60, 87.94, -65.61])

        # for low n_e, Bolsig:
        # v1_1 :  -35.46      0.1119      -1.535      0.1146     -0.3288E-02
        # v1_2 : -22.11      -5.144      -15.38       1.584     -0.6550E-01
        # v2 :  -20.56      -6.158      -17.49      0.1047      0.2667

        ev1bfit = np.array([-35.46, 0.1119, -1.535, 0.1146, -0.3288E-02])
        ev1_2bfit = np.array([-22.11, -5.144, -15.38, 1.584, -0.6550E-01])
        ev2bfit = np.array([-20.56, -6.158, -17.49, 0.1047, 0.2667])


        # maxwellian distribution more stable at low temperatures
        # v1 -1 :  -36.39      0.5623     -0.7357      0.2483E-01 -0.5307E-03
        #v1 - 2 :   -30.11      -1.434      -3.586      0.1274     -0.6987E-02
        # if meanEe < 0.5:
        # ev1bfit = np.array([-36.39, 0.5623, -0.7357, 0.2483E-01, -0.5307E-03]) # problematic one - this one is better
        # ev1_2bfit = np.array([-30.11, -1.434, -3.586, 0.1274, -0.6987E-02])

        # The fits for eV exchange are notorious at low temperatures or low average energy of electron
        # So include one more parameter below wich the minimum energy of electron is limited to the ones
        # for which the fits are valid
        # mean energy limits
        eVlimts = np.array([0.1,0.2,0.2,0.3,0.3,0.4,0.5,0.6,0.8])

        # if minimum than the limits
        notMin = (meanEe > eVlimts )*1.0

        # make notMin an array
        notMin = np.array(notMin)

        Kevs = np.zeros(8)
        Kevs[0] = self.bolsigFit(meanEe, ev1bfit)+self.bolsigFit(meanEe, ev1_2bfit)
        Kevs[1] = self.bolsigFit(meanEe, ev2bfit)*notMin[2]
        Kevs[2] = self.bolsigFit(meanEe, ev3bfit)*notMin[3]
        Kevs[3] = self.bolsigFit(meanEe, ev4bfit)*notMin[4]
        Kevs[4] = self.bolsigFit(meanEe, ev5bfit)*notMin[5]
        Kevs[5] = self.bolsigFit(meanEe, ev6bfit)*notMin[6]
        Kevs[6] = self.bolsigFit(meanEe, ev7bfit)*notMin[7]
        Kevs[7] = self.bolsigFit(meanEe, ev8bfit)*notMin[8]

        # Kevs = np.ones(8)*self.Krxn[42-1]       # should have different rate constants for each vibrational level
  
        
        QEV = np.sum(Kevs*dU_J)*ne*nN2

        # VE : rxn 43 , Also has 8 reaction rate constants
        # 8 bolsig fits for inverse of rate constants
        # v1 has two : -37.08      0.8902      0.9820     -0.3074      0.2145E-01
        # v1, 0.29 eV : -27.09      -2.702      -7.682      0.9072     -0.4804E-01
        # v2, 0.59 eV : -28.30      -2.443      -5.490      0.5449     -0.2777E-01
        # v3, 0.88 eV :  -29.24      -2.231      -3.871      0.3577     -0.1886E-01
        # v4, 1.17 eV :  -29.83      -2.139      -3.121      0.2757     -0.1466E-01
        # v5, 1.47 eV :  -30.25      -2.029      -2.586      0.2638     -0.1429E-01
        # v6, 1.76 eV :  -30.55      -1.963      -2.219      0.2365     -0.1313E-01
        # v7, 2.06 eV :  -31.24      -1.914      -2.049      0.2576     -0.1440E-01
        # v8, 2.35 eV :  -32.05      -1.868      -1.885      0.2699     -0.1544E-01
        evi1bfit = np.array([-37.08, 0.8902, 0.9820, -0.3074, 0.2145E-01])
        evi1_2bfit = np.array([-27.09, -2.702, -7.682, 0.9072, -0.4804E-01])

        evi2bfit = np.array([-28.30, -2.443, -5.490, 0.5449, -0.2777E-01])
        evi3bfit = np.array([-29.24, -2.231, -3.871, 0.3577, -0.1886E-01])
        evi4bfit = np.array([-29.83, -2.139, -3.121, 0.2757, -0.1466E-01])
        evi5bfit = np.array([-30.25, -2.029, -2.586, 0.2638, -0.1429E-01])
        evi6bfit = np.array([-30.55, -1.963, -2.219, 0.2365, -0.1313E-01])
        evi7bfit = np.array([-31.24, -1.914, -2.049, 0.2576, -0.1440E-01])
        evi8bfit = np.array([-32.05, -1.868, -1.885, 0.2699, -0.1544E-01])


        # at low ne:
        # v1 1 :  -35.79      0.2432     -0.6925      0.6016E-01 -0.1544E-02
        # v1 2 :  -25.34      -3.678      -9.555       1.016     -0.5366E-01
        # v2 :   -27.49      -2.916      -6.225      0.4342     -0.1674E-01
        evi1bfit = np.array([-35.79, 0.2432, -0.6925, 0.6016E-01, -0.1544E-02])
        evi1_2bfit = np.array([-25.34, -3.678, -9.555, 1.016, -0.5366E-01])
        evi2bfit = np.array([-27.49, -2.916, -6.225, 0.4342, -0.1674E-01])


        Kves = np.zeros(8)
        Kves[0] = self.bolsigFit(meanEe, evi1bfit) + self.bolsigFit(meanEe, evi1_2bfit)
        Kves[1] = self.bolsigFit(meanEe, evi2bfit)
        Kves[2] = self.bolsigFit(meanEe, evi3bfit)
        Kves[3] = self.bolsigFit(meanEe, evi4bfit)
        Kves[4] = self.bolsigFit(meanEe, evi5bfit)
        Kves[5] = self.bolsigFit(meanEe, evi6bfit)
        Kves[6] = self.bolsigFit(meanEe, evi7bfit)
        Kves[7] = self.bolsigFit(meanEe, evi8bfit)






        # Kves = np.ones(8)*self.Krxn[43-1]   # should have different rate constants for each vibrational level
        

        # correct for different vibrational populations
        Kves = Kves*n2v_n2
        QVE = np.sum(Kves*dU_J)*ne*nN2

        # VT : rxn 54
        # find tau_vt : # p[atm]*tau[s] = exp(A(T**(-1/3) - 0.015 mu**(1/4) - 18.42)) [atm*s]
        # Systematics of Vibrational Relaxation  # Cite as: J. Chem. Phys. 39, 3209 (1963); https://doi.org/10.1063/1.1734182 --Roger C. Millikan and Donald R. White
        # for N2: A = 220.0 mu = 14.0
        # n2tau_vt = np.exp(220.0*(Tv**(-1.0/3.0) - 0.015*14.0**(1.0/4.0) - 18.42))/p_atm
        n2tau_vt = self.ptauVT(Tv,220.0,14.0)/p_atm
        Ev_Tv = self.vibEnergy(Tv,self.n2Tvib_c) # per particle
        Ev_T = self.vibEnergy(Tg,self.n2Tvib_c)
        QVT = (Ev_Tv - Ev_T)*nN2/n2tau_vt # J/m3-s

        # Qmodes = np.array([QET, QEV, QVE, QVT])
        Qmodes = np.array([QET, QEV, QVE, QVT])
        Qmode_names = ['ET','EV','VE','VT']

        self.Qmode_names = Qmode_names
        # dictionary for Qmodes with keys and Qmode_names
        Qmodes_dict = dict(zip(Qmode_names,Qmodes))
        return Qmodes_dict



    def dhrxn_(self):
        '''
        Calculate the enthalpy of reaction in J/kmol for each reaction.
        For reactions included in Qmodes_() the enthalpy of reaction should be zero.
        '''
        nrxn = self.nrxn
        hsp = self.hsp


        # get from the makePlasmaKinetics_II.py library
        # Requires definition of enthalpy for new subgroup , e.g. N2(A,v>9)
        HRxn = np.zeros((3, nrxn))

        # SubGroup enthalpy: N2A
        # [hN2A_vG1, hN2A_vG2, hN2A_vG3] = N2A(v=0-4) , N2A(v=5-9) , N2A(v>9)
        # Nasa7 polynomials use with NASA7Enthalpy(x,T,Tswitch=None): - > J/mol but needed in J/kmol, T needs to be array
        # NASA7Enthalpy outputs an array
        hN2A_vG1 = NASA7Enthalpy(self.N2Avg1NASA7, [self.Temp[0]], 1000.0)[0]*1.0e3
        hN2A_vG2 = NASA7Enthalpy(self.N2Avg2NASA7, [self.Temp[0]], 1000.0)[0]*1.0e3
        hN2A_vG3 = NASA7Enthalpy(self.N2Avg3NASA7, [self.Temp[0]], 1000.0)[0]*1.0e3



        # The ones with zero are not changed
        # Tg ----
        HRxn[0][0] = -2.0*hsp[0] + hsp[3]
        HRxn[0][1] = 0
        HRxn[0][2] = 0
        HRxn[0][3] = 0
        HRxn[0][4] = 0.7*hsp[1] - 1.4*hsp[2] + 0.7*hsp[3]
        HRxn[0][5] = 0
        HRxn[0][6] = -hsp[10] - hsp[14] + hsp[1] + hsp[2]
        HRxn[0][7] = -hsp[10] - hsp[14] + hsp[1] + hsp[3]
        HRxn[0][8] = -hsp[10] - hsp[14] + hsp[1] + hsp[4]
        HRxn[0][9] = 0
        HRxn[0][10] = hsp[0] - hsp[14] + hsp[1] - hsp[9]
        HRxn[0][11] = 2.0*hsp[0] - hsp[14] - hsp[8]
        HRxn[0][12] = 0
        HRxn[0][13] = -hsp[14] + hsp[1] - hsp[8]
        HRxn[0][14] = 0
        HRxn[0][15] = 0
        HRxn[0][16] = 0
        HRxn[0][17] = hsp[0] - hsp[14] - hsp[7]
        HRxn[0][18] = 0
        HRxn[0][19] = 2.0*hsp[1] - hsp[2] + hsp[7] - hsp[9]
        HRxn[0][20] = hsp[0] - hsp[2] - hsp[8] + hsp[9]
        HRxn[0][21] = hsp[0] + hsp[1] - hsp[2] + hsp[7] - hsp[8]
        HRxn[0][22] = hsp[1] - hsp[2]
        HRxn[0][23] = hsp[1] - hsp[2]
        HRxn[0][24] = 0
        HRxn[0][25] = 0
        HRxn[0][26] = -hsp[10] + hsp[1] + hsp[8]
        HRxn[0][27] = -hsp[0] - hsp[10] + 2.0*hsp[1] + hsp[7]
        HRxn[0][28] = -hsp[0] - hsp[10] + hsp[1] + hsp[9]
        HRxn[0][29] = -hsp[0] + hsp[1] + hsp[8] - hsp[9]
        HRxn[0][30] = hsp[1] + hsp[7] - hsp[9]
        HRxn[0][31] = hsp[0] + hsp[7] - hsp[8]
        HRxn[0][32] = hsp[0] - hsp[1] - hsp[8] + hsp[9]
        HRxn[0][33] = -hsp[0] + hsp[1] + hsp[7] - hsp[8]
        HRxn[0][34] = hsp[10] - hsp[1] - hsp[8]
        HRxn[0][35] = -hsp[0] - hsp[8] + hsp[9]
        HRxn[0][36] = hsp[0] - hsp[1] - hsp[7] + hsp[8]
        HRxn[0][37] = -hsp[1] - hsp[7] + hsp[9]
        HRxn[0][38] = -hsp[0] - hsp[7] + hsp[8]
        HRxn[0][39] = -96472440.0
        HRxn[0][40] = 0
        HRxn[0][41] = 0
        HRxn[0][42] = 0
        HRxn[0][43] = hN2A_vG1 - hsp[1]
        HRxn[0][44] = 0
        HRxn[0][45] = hN2A_vG2 - hsp[1]
        HRxn[0][46] = 0
        HRxn[0][47] = -hsp[1] + hsp[3]
        HRxn[0][48] = 0
        HRxn[0][49] = hN2A_vG3 - hsp[1]
        HRxn[0][50] = 0
        HRxn[0][51] = -hsp[1] + hsp[4]
        HRxn[0][52] = 0
        HRxn[0][53] = 0
        HRxn[0][54] = -hsp[11] - hsp[14] + 2.0*hsp[5]
        HRxn[0][55] = -hsp[12] - hsp[14] + 2.0*hsp[6]
        HRxn[0][56] = -hsp[11] - hsp[14] + hsp[6]
        HRxn[0][57] = -hsp[11] - hsp[14] + hsp[6]
        HRxn[0][58] = -hsp[14] + hsp[1] - hsp[8]
        HRxn[0][59] = hsp[11] - hsp[12] + hsp[6]
        HRxn[0][60] = -hsp[13] - hsp[14] + hsp[1] + hsp[6]
        HRxn[0][61] = hsp[11] - hsp[13] + hsp[1]
        HRxn[0][62] = hsp[12] - hsp[13] + hsp[1] - hsp[6]
        HRxn[0][63] = -hsp[11] + hsp[12] - hsp[6]
        HRxn[0][64] = -hsp[11] + hsp[13] - hsp[1]
        HRxn[0][65] = -hsp[11] - hsp[14] + hsp[6]
        HRxn[0][66] = -196803777.6

        # # # reaction 6 - 18 except 11 make zero
        # zeroIDs = [6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18,23,28,29,35,37,38]
        # for i in zeroIDs:
        #     HRxn[0][i] = 0.0
        # zeroIDs = np.array([1,24,38]) -1
        # for i in zeroIDs:
        #     HRxn[0][i] = 0.0

        # nonZerosRxn = np.array([5,12,40]) - 1
        # # except nonZerosRxn, make zero
        # for i in range(nrxn):
        #     if i not in (nonZerosRxn):
        #         HRxn[0][i] = 0.0

        # Tv ----
        HRxn[1][4] = 0.3*hsp[1] - 0.6*hsp[2] + 0.3*hsp[3]
        HRxn[1][5] = hsp[1] - 2.0*hsp[2] + hsp[4]
        HRxn[1][24] = hsp[2] - hsp[3]
        HRxn[1][25] = hsp[3] - hsp[4]

        # # super recombination
        # # energy might go to vibration, photon radiation and bulk gas, do not know
        # HRxn[1][6] = (hsp[1] + hsp[2] - hsp[8] - hsp[9])*0.2
        # HRxn[1][7] = (hsp[1] + hsp[3] - hsp[8] - hsp[9])*0.2
        # HRxn[1][8] = (hsp[1] + hsp[4] - hsp[8] - hsp[9])*0.2

        # HRxn[1][41] = -27525341.0 # included in Qmodes
        # HRxn[1][42] = 27525341.0 # included in Qmodes

        # Te ----

        HRxn[2][9] = -hsp[10] - hsp[14] + 2.0*hsp[1]
        HRxn[2][10] = 0
        HRxn[2][11] = 0
        HRxn[2][12] = 0
        HRxn[2][13] = 0
        HRxn[2][14] = -hsp[14] + hsp[1] - hsp[8]
        HRxn[2][15] = hsp[14] - hsp[1] + hsp[8]
        HRxn[2][16] = 0
        HRxn[2][17] = 0
        HRxn[2][18] = hsp[0] - hsp[14] - hsp[7]

        HRxn[2][39] = 2.0*hsp[0] - hsp[1] + 96472440.0
        HRxn[2][40] = 0
        HRxn[2][41] = 0
        HRxn[2][42] = 0
        HRxn[2][43] = 0
        HRxn[2][44] = -hN2A_vG1 + hsp[1]
        HRxn[2][45] = 0
        HRxn[2][46] = -hN2A_vG2 + hsp[1]
        HRxn[2][47] = 0
        HRxn[2][48] = hsp[1] - hsp[3]
        HRxn[2][49] = 0
        HRxn[2][50] = -hN2A_vG3 + hsp[1]
        HRxn[2][51] = 0
        HRxn[2][52] = hsp[1] - hsp[4]

        HRxn[2][66] = 2.0*hsp[5] - hsp[6] + 196803777.6


        return HRxn

    def Krxn_(self):
        '''
        Calculate the reaction rate constants in SI units.
        '''
        import math
        nrxn = self.nrxn
        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]

        EeV = 1.5*Te/11600.0 # mean electron energy in eV

        K_rxn = np.zeros(nrxn)


        # Bolsig fit for required reactions : Rxn ID is 1 more than the index as the first index is zero in python
        K44_Bparm = np.array([-24.62, -3.201, -54.41, 84.72, -55.07])
        K45_Bparm = np.array([-36.53, 0.4430, -0.9333, 0.1975, -0.1179E-01])
        K46_Bparm = np.array([-22.83, -3.286, -56.28, 87.69, -56.95])
        K47_Bparm = np.array([-34.60, 0.2789, -1.035, 0.2008, -0.1158E-01])
        K48_Bparm = np.array([-20.25, -3.846, -58.44, 89.11, -57.33])
        K49_Bparm = np.array([-32.76, -0.6230E-01, -0.9100, 0.1488, -0.8068E-02])
        K50_Bparm = np.array([-22.16, -3.462, -59.72, 92.99, -60.27])
        K51_Bparm = np.array([-34.23, 0.1644, -1.304, 0.2360, -0.1320E-01])
        K52_Bparm = np.array([-14.24, -5.166, -83.42, 127.1, -80.94])
        K53_Bparm = np.array([-30.64, -0.3308, -2.036, 0.3760, -0.2144E-01])



        # Rename with the reaction number = RxnID - 1
        K43 = bolsigFit(EeV, K44_Bparm)
        K44 = bolsigFit(EeV, K45_Bparm)
        K45 = bolsigFit(EeV, K46_Bparm)
        K46 = bolsigFit(EeV, K47_Bparm)
        K47 = bolsigFit(EeV, K48_Bparm)#*0.0
        K48 = bolsigFit(EeV, K49_Bparm)
        K49 = bolsigFit(EeV, K50_Bparm)#*0.0
        K50 = bolsigFit(EeV, K51_Bparm)
        K51 = bolsigFit(EeV, K52_Bparm)#*0.0
        K52 = bolsigFit(EeV, K53_Bparm)

        # # Use the output from the makePlasmaKinetics_II.py library
        # # If you need to change any reaction rate constants, e.g. using bolsigFit do it here
        K_rxn[0] = 8.27e-46*np.exp(500/Tg)
        K_rxn[1] = 0.5
        K_rxn[2] = 152000.0
        K_rxn[3] = 26900000.0
        K_rxn[4] = 2.9e-15*np.sqrt(Tg/300) # tuned values
        # K_rxn[4] = 1.1e-17*np.sqrt(Tg/300)
        K_rxn[5] = 2.6e-16*np.sqrt(Tg/300) # tuned values
        # K_rxn[5] = 2.1e-17*np.sqrt(Tg/300)
        # K_rxn[6] = 6.0e-14*np.sqrt(300/Te) # tuned values - as peters --------
        # K_rxn[6] = 0.5*0.02*1.4e-12*(300.0/Te)**0.41 # tuned on alexs-----------------
        K_rxn[6] = 0.025*1.4e-12*(300.0/Te)**0.41 # exact on alexs-----------------
        # K_rxn[6] = 0.025*1.4e-12*(300.0/Te)**0.41 # o2 add on alexs-----------------
        # K_rxn[7] = 2.088e-12*np.sqrt(300/Te) # tuned values - as peters ----
        # K_rxn[7] = 0.5*0.87*1.4e-12*(300.0/Te)**0.41 # tuned on aleks-----------------
        K_rxn[7] = 0.87*1.4e-12*(300.0/Te)**0.41 # exact on aleks-----------------
        # K_rxn[7] = 0.87*1.4e-12*(300.0/Te)**0.41 # o2 on aleks-----------------
        # K_rxn[8] = 2.64e-13*np.sqrt(300/Te) # tuned values - as peters ----
        # K_rxn[8] = 0.5*0.11*1.4e-12*(300.0/Te)**0.41 # tuned on aleks-----------------
        K_rxn[8] = 0.11*1.4e-12*(300.0/Te)**0.41 # exact on aleks-----------------
        # K_rxn[8] = 0.105*1.4e-12*(300.0/Te)**0.41 # o2 on aleks-----------------
        # K_rxn[8] = 4.6e-12*np.sqrt(300/Te)
        K_rxn[9] = 7.0e-32*(300/Te)**4.5 ## heating effect in Te ----
        K_rxn[10] = 2.0e-13*np.sqrt(300/Te)
        # K_rxn[11] = 2.8e-13*np.sqrt(300/Te)
        K_rxn[11] = 1.8e-13*(300.0/Te)**0.39 # aleks-----------------
        K_rxn[12] = 4.0e-18*(300/Te)**0.7
        K_rxn[13] = 6.0e-39*(300/Te)**1.5
        # K_rxn[14] = 1.0e-31*(300/Te)**4.5
        K_rxn[14] = 2.0e-31*(300/Te)**4.5 # aleks-----------------
        K_rxn[15] = (5.05e-17*(np.sqrt(Te) + 1.1e-5*Te**1.5))/np.exp(182000.0/Te)
        K_rxn[16] = 3.5e-18*(300/Te)**0.7
        K_rxn[17] = 6.0e-39*(300/Te)**1.5
        # K_rxn[18] = 1.0e-31*(300/Te)**4.5
        K_rxn[18] = 2.0e-31*(300/Te)**4.5 # aleks-----------------
        K_rxn[19] = 6.0e-16
        K_rxn[20] = 3.0e-16
        K_rxn[21] = 4.0e-16
        K_rxn[22] = 2.0e-23 # tuned values
        # K_rxn[22] = 3.0e-25
        # K_rxn[23] = 6.2e-17*(300/Tg)**0.666666666666667 # tuned values
        K_rxn[23] = 5.0e-18*(300/Tg)**0.666666666666667
        # K_rxn[24] = 1.2e-17 # tuned values
        K_rxn[24] = 1.6e-18
        K_rxn[25] = 1.2e-17*(300/Tg)**0.33 # tuned values
        # K_rxn[25] = 1.2e-17*(300/Tg)**0.33
        K_rxn[26] = 2.1e-22*np.exp(Tg/121)
        K_rxn[27] = 1.0e-17
        K_rxn[28] = 1.0e-15
        K_rxn[29] = 6.6e-17
        K_rxn[30] = 6.0e-16
        K_rxn[31] = 1.2e-17
        K_rxn[32] = 5.5e-18
        K_rxn[33] = 7.2e-19*np.exp(300/Tg)
        K_rxn[34] = 6.8e-41*(300/Tg)**1.64
        K_rxn[35] = 9.0e-42*np.exp(400/Tg)
        K_rxn[36] = 1.0e-19
        K_rxn[37] = 2.0e-41*(300/Tg)**2.0
        K_rxn[38] = 1.0e-41*(300/Tg)
        K_rxn[39] = (1.2e+19/((6.022e+23*Te**1.6)))*np.exp(-113200/Te)
        K_rxn[40] = 8.25e-14
        K_rxn[41] = 8.45e-15
        K_rxn[42] = 2.69e-17
        K_rxn[43] = 1.59e-18
        K_rxn[44] = 2.67e-17
        K_rxn[45] = 5.63e-18
        K_rxn[46] = 1.8e-16
        K_rxn[47] = 2.35e-17
        K_rxn[48] = 9.85e-16
        K_rxn[49] = 4.06e-18
        K_rxn[50] = 2.15e-16
        K_rxn[51] = 6.18e-18
        K_rxn[52] = 4.06e-15
        K_rxn[53] = 3.05e-27
        K_rxn[54] = 1.95e-13*(300.0/Te)**0.7
        K_rxn[55] = 4.2e-12*np.sqrt(300.0/Te)
        K_rxn[56] = 1.0e-12*(3.1e-23*Te**(-1.5))
        K_rxn[57] = 1.0e-12*(3.1e-23*Te**(-1.5))
        K_rxn[58] = 1.0e-12*(3.1e-23*Te**(-1.5))
        K_rxn[59] = 1.0e-6*((3.3e-6*(300.0/Tg)**4.0)*np.exp(-5030.0/Tg))
        K_rxn[60] = 1.0e-6*(1.3e-6*np.sqrt(300.0/Te))
        K_rxn[61] = 1.0e-6*((1.1e-6*(300.0/Tg)**5.3)*np.exp(-2357.0/Tg))
        K_rxn[62] = 1.0e-15
        K_rxn[63] = 1.0e-12*(2.4e-30*(300.0/Tg)**3.2)
        K_rxn[64] = 1.0e-12*(9.0e-31*(300.0/Tg)**2.0)
        # K_rxn[65] = 1.0e-12*(2.0e-19*(Te/300.0)**(-4.5))
        K_rxn[65] = 1.0e-12*(1.0e-19*(Te/300.0)**(-4.5))
        K_rxn[66] = 1.0e-6*np.exp(-3.05447914292227e+16)

        # # update with bolsig rates
        K_rxn[43] = K43
        K_rxn[44] = K44
        K_rxn[45] = K45
        K_rxn[46] = K46
        K_rxn[47] = K47
        K_rxn[48] = K48
        K_rxn[49] = K49
        K_rxn[50] = K50
        K_rxn[51] = K51
        K_rxn[52] = K52


        return K_rxn

    def Wrxn_(self):
        '''
        Calculate the rate of progress in concentration units / s.
        '''
        nrxn = self.nrxn
        K_rxn = self.Krxn
        Ysp = self.Ysp

        W_rxn = np.zeros(nrxn)
        # Use the output from the makePlasmaKinetics_II.py library
        # If you need to change any reaction rate constants, e.g. using bolsigFit do it here
        W_rxn[0] = K_rxn[0]*Ysp[0]**2*Ysp[1]
        W_rxn[1] = K_rxn[1]*Ysp[2]
        W_rxn[2] = K_rxn[2]*Ysp[3]
        W_rxn[3] = K_rxn[3]*Ysp[4]
        W_rxn[4] = K_rxn[4]*Ysp[2]**2
        W_rxn[5] = K_rxn[5]*Ysp[2]**2
        W_rxn[6] = K_rxn[6]*Ysp[10]*Ysp[14]
        W_rxn[7] = K_rxn[7]*Ysp[10]*Ysp[14]
        W_rxn[8] = K_rxn[8]*Ysp[10]*Ysp[14]
        W_rxn[9] = K_rxn[9]*Ysp[10]*Ysp[14]**2
        W_rxn[10] = K_rxn[10]*Ysp[14]*Ysp[9]
        W_rxn[11] = K_rxn[11]*Ysp[14]*Ysp[8]
        W_rxn[12] = K_rxn[12]*Ysp[14]*Ysp[8]
        W_rxn[13] = K_rxn[13]*Ysp[14]*Ysp[1]*Ysp[8]
        W_rxn[14] = K_rxn[14]*Ysp[14]**2*Ysp[8]
        W_rxn[15] = K_rxn[15]*Ysp[14]*Ysp[1]
        W_rxn[16] = K_rxn[16]*Ysp[14]*Ysp[7]
        W_rxn[17] = K_rxn[17]*Ysp[14]*Ysp[1]*Ysp[7]
        W_rxn[18] = K_rxn[18]*Ysp[14]**2*Ysp[7]
        W_rxn[19] = K_rxn[19]*Ysp[2]*Ysp[9]
        W_rxn[20] = K_rxn[20]*Ysp[2]*Ysp[8]
        W_rxn[21] = K_rxn[21]*Ysp[2]*Ysp[8]
        W_rxn[22] = K_rxn[22]*Ysp[1]*Ysp[2]
        W_rxn[23] = K_rxn[23]*Ysp[0]*Ysp[2]
        W_rxn[24] = K_rxn[24]*Ysp[1]*Ysp[3]
        W_rxn[25] = K_rxn[25]*Ysp[1]*Ysp[4]
        W_rxn[26] = K_rxn[26]*Ysp[10]*Ysp[1]
        W_rxn[27] = K_rxn[27]*Ysp[0]*Ysp[10]
        W_rxn[28] = K_rxn[28]*Ysp[0]*Ysp[10]
        W_rxn[29] = K_rxn[29]*Ysp[0]*Ysp[9]
        W_rxn[30] = K_rxn[30]*Ysp[1]*Ysp[9]
        W_rxn[31] = K_rxn[31]*Ysp[1]*Ysp[8]
        W_rxn[32] = K_rxn[32]*Ysp[1]*Ysp[8]
        W_rxn[33] = K_rxn[33]*Ysp[0]*Ysp[8]
        W_rxn[34] = K_rxn[34]*Ysp[1]**2*Ysp[8]
        W_rxn[35] = K_rxn[35]*Ysp[0]*Ysp[1]*Ysp[8]
        W_rxn[36] = K_rxn[36]*Ysp[1]*Ysp[7]
        W_rxn[37] = K_rxn[37]*Ysp[1]**2*Ysp[7]
        W_rxn[38] = K_rxn[38]*Ysp[0]*Ysp[1]*Ysp[7]
        W_rxn[39] = K_rxn[39]*Ysp[14]*Ysp[1]
        W_rxn[40] = K_rxn[40]*Ysp[14]*Ysp[1]
        W_rxn[41] = K_rxn[41]*Ysp[14]*Ysp[1]
        W_rxn[42] = K_rxn[42]*Ysp[14]*Ysp[1]
        W_rxn[43] = K_rxn[43]*Ysp[14]*Ysp[1]
        W_rxn[44] = K_rxn[44]*Ysp[14]*Ysp[2]
        W_rxn[45] = K_rxn[45]*Ysp[14]*Ysp[1]
        W_rxn[46] = K_rxn[46]*Ysp[14]*Ysp[2]
        W_rxn[47] = K_rxn[47]*Ysp[14]*Ysp[1]
        W_rxn[48] = K_rxn[48]*Ysp[14]*Ysp[3]
        W_rxn[49] = K_rxn[49]*Ysp[14]*Ysp[1]
        W_rxn[50] = K_rxn[50]*Ysp[14]*Ysp[2]
        W_rxn[51] = K_rxn[51]*Ysp[14]*Ysp[1]
        W_rxn[52] = K_rxn[52]*Ysp[14]*Ysp[4]
        W_rxn[53] = K_rxn[53]*Ysp[1]**2
        W_rxn[54] = K_rxn[54]*Ysp[11]*Ysp[14]
        W_rxn[55] = K_rxn[55]*Ysp[12]*Ysp[14]
        W_rxn[56] = K_rxn[56]*Ysp[11]*Ysp[14]*Ysp[6]
        W_rxn[57] = K_rxn[57]*Ysp[11]*Ysp[14]*Ysp[1]
        W_rxn[58] = K_rxn[58]*Ysp[14]*Ysp[6]*Ysp[8]
        W_rxn[59] = K_rxn[59]*Ysp[12]*Ysp[6]
        W_rxn[60] = K_rxn[60]*Ysp[13]*Ysp[14]
        W_rxn[61] = K_rxn[61]*Ysp[13]*Ysp[1]
        W_rxn[62] = K_rxn[62]*Ysp[13]*Ysp[6]
        W_rxn[63] = K_rxn[63]*Ysp[11]*Ysp[6]**2
        W_rxn[64] = K_rxn[64]*Ysp[11]*Ysp[1]**2
        W_rxn[65] = K_rxn[65]*Ysp[11]*Ysp[14]**2
        W_rxn[66] = K_rxn[66]*Ysp[14]*Ysp[6]



        


        return W_rxn

    def dYdt_(self):
        '''
        Calculate the change in species concentration in concentration units / s -- mostly numdensity/s.
        '''
        nrxn = self.nrxn
        nsp = self.nsp
        W_rxn = self.Wrxn

        dYdt = np.zeros(nsp)
        ## From the output of the make plasma kinetics library
        dYdt[0] = -2*W_rxn[0] + W_rxn[10] + 2*W_rxn[11] + W_rxn[16] + W_rxn[17] + W_rxn[18] + W_rxn[20] + W_rxn[21] - W_rxn[27] - W_rxn[28] - W_rxn[29] + W_rxn[31] + W_rxn[32] - W_rxn[33] - W_rxn[35] + W_rxn[36] - W_rxn[38] + 2*W_rxn[39]
        dYdt[1] = W_rxn[10] + W_rxn[12] + W_rxn[13] + W_rxn[14] - W_rxn[15] + 2*W_rxn[19] + W_rxn[1] + W_rxn[21] + W_rxn[22] + W_rxn[23] + W_rxn[26] + 2*W_rxn[27] + W_rxn[28] + W_rxn[29] + W_rxn[30] - W_rxn[32] + W_rxn[33] - W_rxn[34] - W_rxn[36] - W_rxn[37] - W_rxn[39] - W_rxn[43] + W_rxn[44] - W_rxn[45] + W_rxn[46] - W_rxn[47] + W_rxn[48] - W_rxn[49] + W_rxn[4] + W_rxn[50] - W_rxn[51] + W_rxn[52] + W_rxn[58] + W_rxn[5] + W_rxn[60] + W_rxn[61] + W_rxn[62] - W_rxn[64] + W_rxn[6] + W_rxn[7] + W_rxn[8] + 2*W_rxn[9]
        dYdt[2] = -W_rxn[19] - W_rxn[1] - W_rxn[20] - W_rxn[21] - W_rxn[22] - W_rxn[23] + W_rxn[24] + W_rxn[2] + W_rxn[43] - W_rxn[44] + W_rxn[45] - W_rxn[46] + W_rxn[49] - 2*W_rxn[4] - W_rxn[50] - 2*W_rxn[5] + W_rxn[6]
        dYdt[3] = W_rxn[0] - W_rxn[24] + W_rxn[25] - W_rxn[2] + W_rxn[3] + W_rxn[47] - W_rxn[48] + W_rxn[4] + W_rxn[7]
        dYdt[4] = -W_rxn[25] - W_rxn[3] + W_rxn[51] - W_rxn[52] + W_rxn[5] + W_rxn[8]
        dYdt[5] = 2*W_rxn[54] + 2*W_rxn[66]
        dYdt[6] = 2*W_rxn[55] + W_rxn[56] + W_rxn[57] + W_rxn[59] + W_rxn[60] - W_rxn[62] - W_rxn[63] + W_rxn[65] - W_rxn[66]
        dYdt[7] = -W_rxn[16] - W_rxn[17] - W_rxn[18] + W_rxn[19] + W_rxn[21] + W_rxn[27] + W_rxn[30] + W_rxn[31] + W_rxn[33] - W_rxn[36] - W_rxn[37] - W_rxn[38]
        dYdt[8] = -W_rxn[11] - W_rxn[12] - W_rxn[13] - W_rxn[14] + W_rxn[15] - W_rxn[20] - W_rxn[21] + W_rxn[26] + W_rxn[29] - W_rxn[31] - W_rxn[32] - W_rxn[33] - W_rxn[34] - W_rxn[35] + W_rxn[36] + W_rxn[38] - W_rxn[58]
        dYdt[9] = -W_rxn[10] - W_rxn[19] + W_rxn[20] + W_rxn[28] - W_rxn[29] - W_rxn[30] + W_rxn[32] + W_rxn[35] + W_rxn[37]
        dYdt[10] = -W_rxn[26] - W_rxn[27] - W_rxn[28] + W_rxn[34] - W_rxn[6] - W_rxn[7] - W_rxn[8] - W_rxn[9]
        dYdt[11] = -W_rxn[54] - W_rxn[56] - W_rxn[57] + W_rxn[59] + W_rxn[61] - W_rxn[63] - W_rxn[64] - W_rxn[65]
        dYdt[12] = -W_rxn[55] - W_rxn[59] + W_rxn[62] + W_rxn[63]
        dYdt[13] = -W_rxn[60] - W_rxn[61] - W_rxn[62] + W_rxn[64]
        dYdt[14] = -W_rxn[10] - W_rxn[11] - W_rxn[12] - W_rxn[13] - W_rxn[14] + W_rxn[15] - W_rxn[16] - W_rxn[17] - W_rxn[18] - W_rxn[54] - W_rxn[55] - W_rxn[56] - W_rxn[57] - W_rxn[58] - W_rxn[60] - W_rxn[65] - W_rxn[6] - W_rxn[7] - W_rxn[8] - W_rxn[9]
            
        return dYdt
    

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
        p = self.p

        Tg = self.Temp[0]
        Tv = self.Temp[1]
        Te = self.Temp[2]

        nN2 = self.Ysp[self.gas.species_index('N2')]
        ne = self.Ysp[self.gas.species_index('ele')]

        eID = self.gas.species_index('ele')
        # Tv = self.Temp[1]
        dEv_dTv = self.dEv_dTv_(Tv,self.n2Tvib_c) # vibrational energy per particle derivative with respect to temperature

        QVT = self.Qmodes['VT']
        QET = self.Qmodes['ET']
        QEV = self.Qmodes['EV']
        QVE = self.Qmodes['VE']

        if self.verbose:
            # show these values
            print('Qdotg = ', Qdotg, 'Qdotv = ', Qdotv, 'Qdote = ', Qdote)
            print('QVT = ', QVT, 'QET = ', QET, 'QEV = ', QEV, 'QVE = ', QVE)

        # d()dt:
        Ce = 1.5*kB*ne
        dydt = np.zeros(np.shape(self.Temp))

        dydt[0] = (Qdotg + QET + QVT)/(cp*rho)
        dydt[1] = ((Qdotv + QEV - QVE - QVT)/nN2)/dEv_dTv
        dydt[2] = (Qdote + QVE - QET - QEV)/Ce # - self.dYdt[eID]*Te/ne ## no need to upt the dnedt part as electron energy also goes down with change in ne
        # dydt[2] = (Qdote + QVE - QET - 0.0*QEV)/Ce - self.dYdt[eID]*Te/ne

        # The dne/dt term is required if solving both Ysp and Temp simultaneously.
        # When the electron density is already modified then it is not required ?

        

        # If laser heating is included
        coLn = 10.0
        # calculate the nueff
        #  neuc[cellI] = 2.91e-12*n_e[cellI]*coLn.value()*(Foam::pow(Te[cellI]/11600,-3.0/2.0));
        neuc = 2.91e-12*ne*coLn*(Te/11600.0)**(-1.5)
        # neum = 9.8e-14*p[cellI]/(kB.value()*T[cellI])*(Foam::pow(Te[cellI]/11600.0,1.0/2.0)); 
        neum = 9.8e-14*p/(kB*Tg)*(Te/11600.0)**(0.5)

        nueff = neuc + neum
        Q = 0.0

        if self.laser.switch == 1:
            # calculate the laser absorption
            Q = self.laser.laserAbsorption(ne,nueff)

            dydt[2] = dydt[2] + Q/Ce


        return dydt



        

    def JacY_(self):
        '''
        Calculate the Jacobian for the species concentration.
        '''
        pass

    def JacT_(self):
        '''
        Calculate the Jacobian for the temperature.
        '''
        pass


####----------------------------------------------------------------------------------------------------------------------------