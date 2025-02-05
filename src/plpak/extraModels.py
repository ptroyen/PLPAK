import numpy as np
from abc import ABC, abstractmethod

# add current directory to the path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# This has all the constants
# Import the plasma kinetics library
from makePlasmaKinetics import *    ## Has constants and other stuff

'''
This file contains the extra models that do not fall under the plasma kinetics. For example, in most of the laser induced plamsas
the parameters of the lasers are important. And the absorption of the laser might have different implementations. Thus it is necessary to create such models.
Another model might be the radiation model. All of these models can be added on top of the plasmaSystems. In this sense
the plasmaSystems act as the barebones of the plasma which accounts for all the plasma evolution and decay, and any 
additional terms in the equations can be added on top of it.

'''

# Laser model
class LaserModel:
    '''
    Parameters of the laser are included here.
    Choice for what type of absorption is used is also included here.
    Properties like temporal width, waist size, wavelength, delay, etc. are included here.
    '''

    #  use dictionary to unpack the parameters
    def __init__(self, **kwargs):
        #update the parameters : all in SI units
        self.lam = kwargs.get('lam', 1064.0e-9) # wavelength of the laser
        self.w0 = kwargs.get('w0', 100.0e-6) # waist size of the laser
        self.fwhm = kwargs.get('fwhm', 10.0e-9) # temporal width of the laser: FWHM
        # self.wt = kwargs.get('tau', 8.0e-9) # 
        self.tau_t = self.fwhm/2.355 # tau for pure gaussian: f(t) = exp(-t^2/(2*tau^2))
        self.delay = kwargs.get('delay', 0.0) # delay of the laser from t=0
        self.In0 = kwargs.get('In0', 1.0e14) # intensity of the laser
        self.switch = kwargs.get('switch', 0) # switch for the laser 0/1 : 0 is off, 1 is on, by default it is off

        self.In = self.In0 # intensity of the laser - current value and changes with time

        # there could be many shapes, so use appropriate shape in time
        self.shapeGauss = 'gauss'
        self.shapeShaped = 'shaped' # if shaped first_slow is used
        self.shapeConstant = 'constant'
        # based on the shape, use the appropriate function
        self.useShape = kwargs.get('shape', self.shapeGauss) # by default use gauss
        self.updateIn = self.updateInConstant
        if self.useShape == self.shapeShaped:
            self.updateIn = self.updateInShaped
        if self.useShape == self.shapeConstant:
            self.updateIn = self.updateInConstant

        self.first_slow = kwargs.get('first_slow', 1) # 1 or -1 to determine which side is slow, 1 means first slow , -1 means first fast , only for shaped

    # print the parameters
    def parms(self):
        print('Laser parameters:')
        print('Wavelength: ', self.lam, 'm')
        print('Waist size: ', self.w0, 'm')
        print('Temporal width: ', '{:.2e}'.format(self.fwhm), 's')
        print('Delay: ', self.delay, 's')
        print('Intensity: ', '{:.2e}'.format(self.In), 'W/m2')
        print('Switch: ', self.switch)


    # Laser absorption model
    # This needs to couple with the plasma model and need the following
    # n_e, nu_eff
    def laserAbsorption(self, ne, nueff):
        '''
        Classical approach to laser absorption.
        Q = [J/m3-s]
        '''
        # nueff = plasmaSystem.nueff
        # ne = plasmaSystem.ne


        # # show all the properties to check
        # print('Laser properties:')
        # print('Intensity: ', '{:.2e}'.format(self.In), 'W/m2')
        # print('ne: ', '{:.2e}'.format(ne), 'm-3')
        # print('nueff: ', '{:.2e}'.format(nueff), 's-1')

        wl = 2.0*CO_pi*CO_c/self.lam # angular frequency of the laser

        # Calculate absorption
        Q = 0.0
        if self.switch == 1:
            Q = CO_eC**2*self.In*nueff*ne/(m_e*c*eps0*wl**2)

        # print("self.switch: ", self.switch)
        # print('Laser absorption: ', '{:.2e}'.format(Q), 'J/m3-s')

        return Q
    
    # # get new intensity based on time
    # def updateIn(self, t):
    #     '''
    #     Get the intensity of the laser based on the time
    #     '''

    #     ## use the function defined in self.shape to update with appropriate shape of the intensity
    #     updateFunc = self.shape


    #     # get the intensity
    #     In0 = self.In0

    #     # get the delay
    #     delay = self.delay
    #     # calculate tau_t from fwhm
    #     self.tau_t = self.fwhm/2.355
    #     # sigma = self.tau_t

    #     In = 0.0

    #     if self.switch == 1:
    #         In = In0*np.exp(-(t-delay)**2/(2.0*self.tau_t**2))

    #     self.In = In

    #     return In

        # get new intensity based on time
    def updateInGauss(self, t):
        '''
        Get the intensity of the laser based on the time
        '''
        # get the intensity
        In0 = self.In0

        # get the delay
        delay = self.delay
        # calculate tau_t from fwhm
        self.tau_t = self.fwhm/2.355
        # sigma = self.tau_t

        In = 0.0

        if self.switch == 1:
            In = In0*np.exp(-(t-delay)**2/(2.0*self.tau_t**2))

        self.In = In

        return In

    def updateInConstant(self, t):
        '''
        Get the intensity of the laser based on the time, square pulse,
        delay here is the peak of the pulse as this is a square pulse
        '''
        # get the intensity
        In0 = self.In0

        # get the delay
        delay = self.delay
        # calculate tau_t from fwhm
        self.tau_t = self.fwhm/2.355
        # sigma = self.tau_t

        # ## for delay at the middle of the pulse
        # start_t = delay - self.fwhm*0.5
        # end_t = delay + self.fwhm*0.5

        ## for delay at the start of the pulse
        start_t = delay
        end_t = delay + self.fwhm

        In = 0.0

        if self.switch == 1:
            if t >= start_t and t <= end_t:
                In = In0

        self.In = In

        return In

    # get new intensity based on time
    def updateInShaped(self, t):
        '''
        Get the intensity of the laser based on the time
            """
            Analytical function with slow rise and sharp decay using hyperbolic tangent.

            Parameters:
            - x: Input variable (nanoseconds)
            - c: Center of the transition : this is delay
            - t1: Controls the rate of rise : use 1.0e-9
            - t2: Controls the width (sharpness) of the Gaussian = tau_t
            - p: Either 1 or -1 to determine which side falls sharply
            -first_slow: 1 or -1 to determine which side is slow, 1 means first slow , -1 means first fast

            Returns:
            - y: Output of the function
            """
            func =  (1/0.9779)*0.5 * (1 - p * np.tanh((x - c) / t1)) * np.exp(-0.5*(x - c) ** 2 / (t2) ** 2)
        '''
        # get the intensity
        In0 = self.In0

        # get the delay
        delay = self.delay
        # calculate tau_t from fwhm
        self.tau_t = self.fwhm/2.355
        sharp_t = 1.0e-9
        # sigma = self.tau_t

        # based on first_slow, if 1 get 1 else -1
        p = self.first_slow


        In = 0.0

        # scalePeak = 1.05327
        scalePeak = 1.04950 # This produces same In_peak as gaussian where 10ns (fwhm) of gaussian pw is equal to 19.057 ns of the shaped pulse
        sdel = 2.0e-9*p

        if self.switch == 1:
            # In = In0*np.exp(-(t-delay)**2/(2.0*self.tau_t**2))
            In = In0*scalePeak*0.5*(1.0 - p*np.tanh((t-delay-sdel)/sharp_t))*np.exp(-0.5*(t-delay-sdel)**2/(self.tau_t)**2)

        self.In = In

        return In

    # get new intensity based on time
    def updateInShape1(self, t):
        '''
        Get the intensity of the laser based on the time
            """
            Analytical function with slow rise and sharp decay using hyperbolic tangent.

            Parameters:
            - x: Input variable (nanoseconds)
            - c: Center of the transition : this is delay
            - t1: Controls the rate of rise : use 1.0e-9
            - t2: Controls the width (sharpness) of the Gaussian = tau_t
            - p: Either 1 or -1 to determine which side falls sharply
            -first_slow: 1 or -1 to determine which side is slow, 1 means first slow , -1 means first fast

            Returns:
            - y: Output of the function
            """
            func =  (1/0.9779)*0.5 * (1 - p * np.tanh((x - c) / t1)) * np.exp(-0.5*(x - c) ** 2 / (t2) ** 2)
        '''
        # get the intensity
        In0 = self.In0

        # get the delay
        delay = self.delay
        # calculate tau_t from fwhm
        self.tau_t = self.fwhm/2.355
        sharp_t = 1.0e-9
        # sigma = self.tau_t

        # based on first_slow, if 1 get 1 else -1
        p = self.first_slow


        In = 0.0

        scalePeak = 1.05327
        sdel = 2.0e-9*p

        if self.switch == 1:
            # In = In0*np.exp(-(t-delay)**2/(2.0*self.tau_t**2))
            In = In0*scalePeak*0.5*(1.0 - p*np.tanh((t-delay-sdel)/sharp_t))*np.exp(-0.5*(t-delay-sdel)**2/(self.tau_t)**2)

        self.In = In

        return In


# Test
def testLaser():
    # create a laser
    laser = LaserModel(In0=1.0e14,delay=10.0e-9,fwhm=10.0e-9,switch=1)
    laser.parms()

    ne0 = 1.0e23
    nueff0 = 1.0e12
    Ins = np.linspace(1.0e13,1.0e18,100)
    Qs = np.zeros(100)
    for i in range(100):
        laser.In = Ins[i]
        Qs[i] = laser.laserAbsorption(ne0,nueff0)

    import matplotlib.pyplot as plt
    plt.plot(Ins,Qs)
    # log scale
    plt.yscale('log')
    plt.xscale('log')
    # labels
    plt.xlabel('Intensity [W/m2]')
    plt.ylabel('Absorption - Q [J/m3-s]')

    # in a text box show the laser parameters
    # laser parameters are printed from laser.parms()
    textstr = '\n'.join((
        r'$\lambda=%.2e\ m$' % (laser.lam, ),
        r'$w_0=%.2e\ m$' % (laser.w0, ),
        r'$\tau=%.2e\ s$' % (laser.tau_t, ),
        r'$\Delta t=%.2e\ s$' % (laser.delay, ),
        r'$f(t)= exp(-t^2/(2*tau^2))$'
        ))
    props = dict(boxstyle='round', alpha=0.25, facecolor=None)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    plt.show()


    # show the laser in time
    nts=400
    ts = np.linspace(0.0,20.0e-9,nts)
    Ins = np.zeros(nts)
    for i in range(nts):
        Ins[i] = laser.updateIn(ts[i])


    # make a new plot
    fig_t, ax_t = plt.subplots()
    ax_t.plot(ts,Ins)
    # labels
    ax_t.set_xlabel('Time [s]')
    ax_t.set_ylabel('Intensity [W/m2]')
    # show the plot
    plt.show()






# Now make a new class which interfaces the barebones plasmamodel ( unknown right now) with the laser model and other models like such
class CombinedModels:
    '''
    Here you select the 
    1. Plasma Kinetics Model for plasma Decay
    2. Supply the laser models
    3 ... etc
    
    '''

    # initialize
    def __init__(self,plasma, laser):
        # plasma kinetics model
        self.plasma = plasma
        self.laser = laser
        self.gas = plasma.gas
        



    
    # Extension of the methods of the plasmaModel -- havent checked
    # Need to override dTdt_ only such that all method of the base is completely run and we perform some additional operations
    # Other than that the methods acceeed by users directly need to be wrapped around the plasmaModel methods

    # override the dTdt_ method
    def dTdt_(self):
        # call the base class method
        dydts_base = self.plasma.dTdt_()

        # add the laser absorption
        eID = self.gas.species_index('ele')
        ne = self.plasma.Ysp[eID]
        Tg = self.plasma.Temp[0]
        Tv = self.plasma.Temp[1]
        Te = self.plasma.Temp[2]
        p = self.plasma.p


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
        

        # add the laser absorption to the dTdt[2]
        dydts_base[2] += Q

        return dydts_base
    
    # interface other methods of the plasmaModel except dTdt_
    # This is done by wrapping the plasmaModel methods
    def __getattr__(self, attr):
        all_attrs = getattr(self.plasma, attr)

        # modified here
        modd = ['dTdt_']
        if callable(all_attrs) and attr not in modd:
            def wrapper(*args, **kwargs):
                return all_attrs(*args, **kwargs)
            return wrapper
        else:
            return all_attrs
        
    # interface the plasmaModel methods which are not callable
    def __getitem__(self, key):
        return self.plasma[key]
    



### A model for photo-ionization from the straight line fit of the PPT model:
def photoIonizePPT(IUSE,Ns,**kwargs):
    '''
    This is a model for photo-ionization from the straight line fit of the PPT model.
    The model has the refractive index for 800 nm pulse for now.

    Parameters:
        - IUSE: Intensity of the laser
        - Ns: A list with number density of N2 and O2, note the order
        - kwargs: Additional parameters for the model
            - PW: Pulse width of the laser = 100 fs

    Returns:
        [N2+, O2+] : number density of N2+, and O2+ at the end of the pulse, sum to get electrons

    --- some known results ---
    IUSE = 8.2e17 # W/m2 :e =  4.090796778008972e+23, N2+ =  8.112343830191564e+22, O2+ =  3.2795623949898156e+23 
    IUSE = 5.395e17 # W/m2 : e =  1.0050203977821561e+23, N2+ =  1.4756006490408548e+22, O2+ =  8.574603328780707e+22

    --- UPDATES ---
    * For tight focusing cases, a decoupling threshold limits the electron density, so the time to clamp is calculated based on this threshold.

    '''

    import scipy.integrate as integrate
    import matplotlib.pyplot as plt

    # constants
    eps0 = 8.854187817e-12 # F/m
    c = 299792458.0 # m/s
    h = 6.62607015e-34 # J s
    e = 1.602176634e-19 # C
    me = 9.1093837015e-31 # kg

    # read PW or PW ( length of the pulse)
    PW = kwargs.get('PW', 100.0e-15) # s
    clampTime = kwargs.get('clampTime', 20.0e-15) # s
    NTOT = Ns[0] + Ns[1] # total number density of N2 and O2
    N20 = Ns[0]
    O20 = Ns[1]


    # Rates for N2 and O2
    RTN2 = 2.5e4 # per s
    RTO2 = 2.8e6 # per s
    alpN2 = 7.5
    alpO2 = 6.5
    # normalizing intensity
    IT = 1.0e17 # W/m2
    # Calculate MPI rates for N2 and O2
    RN2 = RTN2*(IUSE/IT)**alpN2
    RO2 = RTO2*(IUSE/IT)**alpO2

    # details of the laser
    lam = 800.0e-9 # m
    nu = c/lam # Hz
    omega = 2.0*np.pi*nu # rad/s
    NeCr = eps0*me/(e**2)*omega**2 # critical electron density

    # from N2
    eta2_n2 = 3.19e-23 # m2/W for N2 at 800 nm and atmospheric pressure
    # eta2 = 7.4e-24 # m2/W for N2 at 800 nm and atmospheric pressure another reference

    # from O2
    eta2_o2 = 6.7857e-23 # m2/W for O2 at 800 nm and atmospheric pressure


    # define a function to find In critical, iterative
    def findInCrit(gIn):
        wn2 = (RTN2*(gIn/IT)**alpN2)
        wo2 = (RTO2*(gIn/IT)**alpO2)
        w = wn2 + wo2

        # In = NTOT*w*PW/(2.0*NeCr)/eta2

        In = N20*wn2*PW/(2.0*NeCr)/eta2_n2 + O20*wo2*PW/(2.0*NeCr)/eta2_o2

        print('w = ',w,)
        print('In = ',In,' W/m2',' gIn = ',gIn)
        return In

    # Find critical/clamping intensity
    print('Finding critical intensity...')
    # initial guess:
    In_guess0 = 1.0e17 # W/m2
    Inold = In_guess0
    while True:
        In = findInCrit(Inold)
        if np.abs(In-Inold)/In < 1.0e-5:
            break
        # update with relaxation factor
        Inold = Inold - 1.0e-1*(In-Inold)
        print('Inold = ',Inold,' Diff = ',Inold - In)

    print('Critical intensity = ',In,' W/m2')
    ICLAMP = Inold

    # calculate the electron density for Intesnity = IUSE
    def ode_dydt(y,t):
        # first N2 and second O2
        # Calculate MPI rates for N2 and O2
        RN2 = RTN2*(IUSE/IT)**alpN2
        RO2 = RTO2*(IUSE/IT)**alpO2
        N2 = y[0]
        O2 = y[1]

        dydt = [-RN2*N2, -RO2*O2]

        return dydt

    # initial conditions
    y0 = [N20, O20]


    DT = 100.0e-15 # s

    N20im = N20
    O20im = O20
    t = [0.0]
    y1 = [y0]
    # clampTime = 20.0e-15 # s from kwargs, default is 20.0e-15
    # pulse width = 100.0e-15 # s from kwargs, default is 100.0e-15

    # # clampTime = 10.0e-15 # s---------------
    ## This is the approach for tight focusing case #############################
    ## find the decoupling threshold critical electron dnesity first : 5.0e23 m-3
    decThresNe = 9.0e23
    # to determine the clamp time first solve the ODE with given IUSE
    tempt = np.linspace(0, DT/100, 500)
    yTemp = integrate.odeint(ode_dydt, y0, tempt)
    # density which defines when
    clamp_crit_ne = 0.277*decThresNe
    # n2p = N20 - y[:, 0]
    # o2p = O20 - y[:, 1]
    # tot = n2p + o2p
    allNes = N20 - yTemp[:, 0] + O20 - yTemp[:, 1]
    
    # print('allNes = ',allNes)
    # find t closest to clamp_crit_ne
    idx = np.abs(allNes - clamp_crit_ne).argmin()
    # print('idx = ',idx)
    clampTime = tempt[idx] + 1.0e-40
        
    # print('clampTime = ',clampTime)
    #############################    


    if clampTime > 1.0e-40:
        # # clampTime = 10.0e-15 # s---------------
        # solve till clampTime
        ns = int(1000*(DT/clampTime)) + 10
        t = np.linspace(0, clampTime, ns)
        print("INUSE = ",IUSE," W/m2")
        y1 = integrate.odeint(ode_dydt, y0, t)
        # get the final values
        N20im = y1[-1,0]
        O20im = y1[-1,1]
        ###--------------------------------------
    else:
        t = [0.0]
        y1 = [y0]
        N20im = N20
        O20im = O20


    # ### Removed ---------------- // update tight focused
    # # solve till clampTime
    # ns = int(1000*(DT/clampTime)) + 10
    # t = np.linspace(0, clampTime, ns)
    # print("INUSE0 = ",IUSE," W/m2")
    # y1 = integrate.odeint(ode_dydt, y0, t)

    # # get the final values
    # N20im = y1[-1,0]
    # O20im = y1[-1,1]
    # ###--------------------------------------

    # Now solve the rest part after clampTime
    if IUSE > ICLAMP:
        IUSE = ICLAMP
    else:
        IUSE = IUSE
    # initial conditions
    y0 = [N20im, O20im]
    t2 = np.linspace(clampTime, DT, 1000)
    # show INUSE
    print('In Clamp = ',ICLAMP,' W/m2')
    print('INUSE after Clamp = ',IUSE,' W/cm2')
    y2 = integrate.odeint(ode_dydt, y0, t2)

    # merge the two
    y = np.concatenate((y1,y2),axis=0)
    t = np.concatenate((t,t2),axis=0)

    n2p = N20 - y[:, 0]
    o2p = O20 - y[:, 1]
    tot = n2p + o2p
    # print the final values
    print('N2+ = ', n2p[-1])
    print('O2+ = ', o2p[-1])
    print('e = ', tot[-1])

    # # plot results
    # plt.plot(t, n2p, 'r-', linewidth=2, label='N2+')
    # plt.plot(t, o2p, 'b--', linewidth=2, label='O2+')
    # plt.plot(t, tot, 'k:', linewidth=2, label='e')

    # # grid
    # plt.grid()

    # # plot a vertical line at 1.0e-13 s
    # plt.axvline(x=1.0e-13, color='k', linestyle='--')

    # plt.xlabel('time')
    # plt.ylabel('y(t)')

    # # log
    # plt.yscale('log')
    # plt.xscale('log')

    # plt.legend()
    # plt.show()    

    return [n2p[-1],o2p[-1]]


# test the photo-ionization model for a range of percentages of O2, starting from 0.0 to 1.0
def testPhotoIonizePPT():
    IUSE = 5.395e17 # W/m2 : this gives 1.0e23 1/m3 for electrons in air , 22% O2 and 78% N2

    IUSE = 5.0*IUSE
    nPoints = 300
    ntot=2.45e25
    perO2_ = np.linspace(0.0,0.3,nPoints)
    ions = np.zeros((nPoints,2))


    for ii, perO2 in enumerate(perO2_):
        N2 = ntot*(1.0 - perO2)
        O2 = ntot*perO2
        Ns = [N2,O2]
        ions_ = photoIonizePPT(IUSE,Ns)
        ions[ii,:] = ions_
            

    plotPerO2 = 100.0*perO2_
    # plot the results
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # import scienceplots
    # plt.style.use('science')
    # # plt.style.use(['science','ieee'])
    # plt.rcParams.update({'figure.dpi': '100'})

    fmts = ['png','pdf']



    ww = 4
    hh = 4
    fig, ax = plt.subplots(figsize=(ww,hh))
    # plot results
    ax.plot(plotPerO2,ions[:,0]+ions[:,1],'k-',linewidth=2,label='$e^-$')
    ax.plot(plotPerO2,ions[:,0],'b:',linewidth=2,label='$N_2^+$')
    ax.plot(plotPerO2,ions[:,1],'r--',linewidth=2,label='$O_2^+$')

    # legend
    ax.legend()

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$O_2$ percentage')
    ax.set_ylabel(r'Number density [$m^{-3}$]')

    fig.tight_layout()
    plt.show()

    # save the plots
    for fmt in fmts:
        fname = 'fsPPT' + str(int(IUSE/1.0e17)) + 'e17' + '.' + fmt
        fig.savefig(fname, format=fmt, dpi=300)




if __name__ == '__main__':
    # testLaser()

    # # check photo-ionization model
    # IUSE = 5.395e17 # W/m2 : this gives 1.0e23 for electrons in air , 22% O2 and 78% N2
    # ntot=2.45e25
    # perO2 = 0.0
    # # perO2 = 0.20982
    # N2 = ntot*(1.0 - perO2)
    # O2 = ntot*perO2
    # Ns = [N2,O2]
    # ions = photoIonizePPT(IUSE,Ns)

    testPhotoIonizePPT()