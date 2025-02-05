import numpy as np
import sys
import os


dir = os.path.dirname(".\plpak")
sys.path.append(dir)
import plpak as pl


# Make a class to hold the solution object
class Solution:
    def __init__(self, t, n2Plasma,laser=None,laserDetachmentFunc=None,laserIonizationFunc=None):

        self.t = []
        self.Ysp = []
        self.Temp = []
        self.Wrxn = []
        self.Qrxn = []
        self.dhrxn = []
        self.Qmodes = []
        self.Qdot = []
        self.cp_mix = []
        self.dNdtRxn = []   # dNet/dt for each reaction

        # additional variables which might be present if there is a laser model
        self.laser_In = []  # intensity
        self.laser_ionization = []  # ionization rate : [N2, O2 , NO] : sum to get the total ionization rate
        self.laser_detach = []  # detachment rate : total  : detachment rate
        self.laser_energyExchange = []  # energy exchange rate : [total absorbed, loss on ionization]

        self.laser = laser
        self.laser_funcDetach = laserDetachmentFunc
        self.laser_funcIon = laserIonizationFunc

        # get gas object from n2Plasma
        self.gas = n2Plasma.gas

        # find reactant stoichiometric coefficients
        self.effStoic = n2Plasma.effStoic

        # assume last species is electron
        self.spe_idx = self.gas.n_species - 1

        # find ID of electrons
        if 'E' in self.gas.species_names:
            spe_idx = self.gas.species_index('E')
        if 'e' in self.gas.species_names:
            spe_idx = self.gas.species_index('e')
        if 'ele' in self.gas.species_names:
            spe_idx = self.gas.species_index('ele')
        # for electrons only find the dNdtRxn = Wrxn*effStoic[spe_idx]
        dNdt = n2Plasma.Wrxn*self.effStoic[:,self.spe_idx]

        self.t.append(t)
        # self.Ysp = n2Plasma.Ysp
        # self.Temp = n2Plasma.Temp
        # self.Wrxn = n2Plasma.Wrxn
        self.Ysp.append(n2Plasma.Ysp.copy())
        self.Temp.append(n2Plasma.Temp.copy())
        self.Wrxn.append(n2Plasma.Wrxn.copy())
        # Add other variables here if needed
        self.Qrxn.append(n2Plasma.Qrxn)
        self.dhrxn.append(n2Plasma.dhrxn)
        self.Qmodes.append(n2Plasma.Qmodes)
        self.Qdot.append(n2Plasma.Qdot)
        self.cp_mix.append(n2Plasma.cp_mix)

        # zeros dydt
        dydt = np.zeros(len(n2Plasma.Ysp)+3)

        # initialize others to zero : laser related
        if self.laser is not None and self.laser_funcDetach is not None and self.laser_funcIon is not None:
            self.laser_In.append(self.laser.updateIn(t))
            ion_rate, energy_rate = self.laser_funcIon(dydt,n2Plasma,self.laser)
            self.laser_ionization.append(ion_rate)
            self.laser_energyExchange.append(energy_rate)
            detach_rate = self.laser_funcDetach(dydt,n2Plasma,self.laser)
            self.laser_detach.append(detach_rate)

            # # update dNdt, dNdt is for all reactions
            # dNdt = dNdt + np.sum(ion_rate,axis=0) + detach_rate

        self.dNdtRxn.append(dNdt)


    # appending to the solution
    def solnPush(self, t, n2Plasma):
        self.t.append(t)
        self.Ysp.append(n2Plasma.Ysp.copy())
        self.Temp.append(n2Plasma.Temp.copy())
        self.Wrxn.append(n2Plasma.Wrxn[:])
        # Add other variables here if needed
        self.Qrxn.append(n2Plasma.Qrxn)
        self.dhrxn.append(n2Plasma.dhrxn)
        self.Qmodes.append(n2Plasma.Qmodes)
        self.Qdot.append(n2Plasma.Qdot)
        self.cp_mix.append(n2Plasma.cp_mix)

        # other after calculation
        dNdt = n2Plasma.Wrxn*self.effStoic[:,self.spe_idx]

        # zeros dydt
        dydt = np.zeros(len(n2Plasma.Ysp)+3)
        # laser related
        if self.laser is not None and self.laser_funcDetach is not None and self.laser_funcIon is not None:
            self.laser_In.append(self.laser.updateIn(t))
            ion_rate, energy_rate = self.laser_funcIon(dydt,n2Plasma,self.laser)
            self.laser_ionization.append(ion_rate)
            self.laser_energyExchange.append(energy_rate)
            detach_rate = self.laser_funcDetach(dydt,n2Plasma,self.laser)
            self.laser_detach.append(detach_rate)

            # # update dNdt , 
            # dNdt = dNdt + np.sum(ion_rate,axis=0) + detach_rate


        self.dNdtRxn.append(dNdt)

    # # a method that updates other properties calculated from n2Plasma
    # def update(self, n2Plasma):
    #     # update dNdtRxn


    # converto to numpy array
    def soln2np(self):
        self.t = np.array(self.t)
        self.Ysp = np.array(self.Ysp)
        self.Temp = np.array(self.Temp)
        self.Wrxn = np.array(self.Wrxn)
        self.Qrxn = np.array(self.Qrxn)
        self.dhrxn = np.array(self.dhrxn)
        self.Qmodes = np.array(self.Qmodes)
        self.Qdot = np.array(self.Qdot)
        self.cp_mix = np.array(self.cp_mix)
        self.dNdtRxn = np.array(self.dNdtRxn)

        # laser related
        if self.laser is not None and self.laser_funcDetach is not None and self.laser_funcIon is not None:
            self.laser_In = np.array(self.laser_In)
            self.laser_ionization = np.array(self.laser_ionization)
            self.laser_energyExchange = np.array(self.laser_energyExchange)
            self.laser_detach = np.array(self.laser_detach)

    # save the solution to numpy file
    def solnSave(self, file_name):
        ''' save to npz file 
        Header: t, Ysp, Temp, Wrxn, Qrxn, dhrxn, Qdot, Qmodes
        Note that some of the components are vectors ( arrays) like temperature, species, etc.
        '''
        # convert to array first
        self.soln2np()
        if self.laser is not None and self.laser_funcDetach is not None and self.laser_funcIon is not None:
            np.savez(file_name, t=self.t, Ysp=self.Ysp, Temp=self.Temp, Wrxn=self.Wrxn,
                 Qrxn=self.Qrxn, dhrxn=self.dhrxn,Qdot=self.Qdot,Qmodes=self.Qmodes, cp_mix=self.cp_mix, dNdtRxn=self.dNdtRxn,
                 laser_In=self.laser_In,laser_ionization=self.laser_ionization,laser_detach=self.laser_detach,laser_energyExchange=self.laser_energyExchange)
        else:
            np.savez(file_name, t=self.t, Ysp=self.Ysp, Temp=self.Temp, Wrxn=self.Wrxn,
                 Qrxn=self.Qrxn, dhrxn=self.dhrxn,Qdot=self.Qdot,Qmodes=self.Qmodes, cp_mix=self.cp_mix, dNdtRxn=self.dNdtRxn)
        

    # static method to plot the solution object saved in time and npz file
    # function to plot the solution object saved in time
    @staticmethod
    def plotSoln(fname='solnPlasma',**kwargs):
        import numpy as np
        import matplotlib.pyplot as plt

        # import os
        # import sys
        # # add this path "E:\POKHAREL_SAGAR\gits\pyblish\plots" to the system path explicitly
        # pathName = "E:\POKHAREL_SAGAR\gits\pyblish\plots"
        # sys.path.append(pathName)
        # import publish    


        # how many reactions to plot
        prx = kwargs.get('prx',10)
        fmax = kwargs.get('fmax',True)

        grp = kwargs.get('grp',0)    # when fmax is true, if grp = None gives max, grp = 1 gives second set of max and so on
        

        


        # show the reaction numbers + 1
        print('Reactions to plot:',np.array(prx)+1)

        # vars to plot
        pvars = kwargs.get('pvars',['t','Ysp','Temp','Wrxn','Qrxn','dhrxn'])

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

        # make absolute values and eV for dhrxn
        solndhrxn = (solndhrxn/1.602e-19)/6.023e26 # convert to eV
        # solnQrxn = np.abs(solnQrxn) # make absolute values
        # solnQdot = np.abs(solnQdot) # make absolute values
        # solnQmodes = np.abs(solnQmodes) # make absolute values

        # What type is Qmodes?, size of Qmodes
        print('Qmodes type = ',type(solnQmodes))
        print('Qmodes shape = ',solnQmodes.shape)

        Qmodes_array = np.zeros((len(solnQmodes),4))
        keys = solnQmodes[0].keys()

        label_keys = []
        # Extract different modes in arrays
        for i,key in enumerate(keys):
            Qmodes_array[:,i] = np.array([solnQmodes[j][key] for j in range(len(solnQmodes))])
            label_keys.append(key)
        
        print(Qmodes_array)

        if fmax:
            # find maxWrxn in time and make an array
            maxWrxn = np.max(solnWrxn,axis=0)

            # print ('maxWrxn = ',maxWrxn,'shape = ',maxWrxn.shape)
            # take top prx reactions , based on specified group, group 0 is the max, gropu 1 is the second set of max and so on
            if grp == 0:
                idPrx = np.argsort(maxWrxn)[-prx:]
            else:
                # use grp value
                idPrx = np.argsort(maxWrxn)[-prx*(grp+1):-prx*grp]


            # idPrx = np.argsort(maxWrxn)[-prx:]
            # use only the first prx reactions to plot
            prx = idPrx
        else:
            prx = np.array(prx)-1 # subtract 1 to get the index in python
        




        # # show Qdot and Qmodes
        # print(solnQdot)
        # print(solnQmodes)

        # plot the data
        # 6 plots in a single figure: three rows and two columns
        fig, axs = plt.subplots(3,2,figsize=(18,26))

        # define a function to plot given the variable and axes
        def plotVar(var,ax):
            if var == 't':
                ax.plot(solt,solt)
                ax.set_xlabel('t')
                ax.set_ylabel('t')
                # log scale
                ax.set_yscale('log')
                ax.set_xscale('log')

            elif var == 'Ysp':
                ax.plot(solt,solnYsp)
                ax.set_xlabel('t')
                ax.set_ylabel('Ysp')
                ax.set_yscale('log')
                ax.set_xscale('log')
            elif var == 'Temp':
                ax.plot(solt,solnTemp[:,0],'-')
                ax.plot(solt,solnTemp[:,1],'--')
                ax.plot(solt,solnTemp[:,2],':')
                ax.set_xlabel('t')
                ax.set_ylabel('Temp')
                ax.set_yscale('log')
                ax.set_xscale('log')
            elif var == 'Wrxn':
                for i in prx:
                    ax.plot(solt,solnWrxn[:,i],label='Wrxn'+str(i+1))
                # ax.plot(solt,solnWrxn[:,prx])
                ax.set_xlabel('t')
                ax.set_ylabel('Wrxn')
                ax.set_yscale('log')
                ax.set_xscale('log')
                # grid on
                ax.grid(True)

            elif var == 'Qrxn':
                for i in prx:
                    ax.plot(solt,solnQrxn[:,0,i],'-',label='Qrxn'+str(i+1))
                    ax.plot(solt,solnQrxn[:,1,i],'--')#,label='Qrxn'+str(i))
                    ax.plot(solt,solnQrxn[:,2,i],':')#,label='Qrxn'+str(i))
                # ax.plot(solt,solnQrxn[:,0,prx],'-')
                # ax.plot(solt,solnQrxn[:,1,prx],'--')
                # ax.plot(solt,solnQrxn[:,2,prx],':')
                ax.set_xlabel('t')
                ax.set_ylabel('Qrxn')

                ax.set_yscale('symlog')
                ax.set_xscale('log')
            elif var == 'dhrxn':
                for i in prx:
                    ax.plot(solt,solndhrxn[:,0,i],'-',label='dhrxn'+str(i+1))
                    ax.plot(solt,solndhrxn[:,1,i],'--')#,label='dhrxn'+str(i))
                    ax.plot(solt,solndhrxn[:,2,i],':')#,label='dhrxn'+str(i))
                # ax.plot(solt,solndhrxn[:,0,prx],'-')
                # ax.plot(solt,solndhrxn[:,1,prx],'--')
                # ax.plot(solt,solndhrxn[:,2,prx],':')
                ax.set_xlabel('t')
                ax.set_ylabel('dhrxn')

                # ax.set_yscale('log')
                # ax.set_xscale('log')
            elif var == 'Qdot':
                ax.plot(solt,solnQdot[:,0],'-')
                ax.plot(solt,solnQdot[:,1],'--')
                ax.plot(solt,solnQdot[:,2],':')
                ax.set_xlabel('t')
                ax.set_ylabel('Qdot')

                ax.set_yscale('symlog')
                ax.set_xscale('log')
            elif var == 'Qmodes':
                # plot Qmodes_array
                ax.plot(solt,Qmodes_array[:,0],'-',label=label_keys[0])
                ax.plot(solt,Qmodes_array[:,1],'--',label=label_keys[1])
                ax.plot(solt,Qmodes_array[:,2],':',label=label_keys[2])
                ax.plot(solt,Qmodes_array[:,3],'-.',label=label_keys[3])

                # log scale
                ax.set_yscale('log')
                ax.set_xscale('log')
                    

                ax.set_xlabel('t')
                ax.set_ylabel('Qmodes')
            elif var == 'dhrxn_Te':
                for i in prx:
                    # ax.plot(solt,solndhrxn[:,0,i],'-',label='dhrxn'+str(i+1))
                    # ax.plot(solt,solndhrxn[:,1,i],'--')#,label='dhrxn'+str(i+1))
                    ax.plot(solt,solndhrxn[:,2,i],'-',label='dhrxn'+str(i+1))
                # ax.plot(solt,solndhrxn[:,0,prx],'-')
                # ax.plot(solt,solndhrxn[:,1,prx],'--')
                # ax.plot(solt,solndhrxn[:,2,prx],':')
                ax.set_xlabel('t')
                ax.set_ylabel('dhrxn')

                # ax.set_yscale('log')
                # ax.set_xscale('log')
            else:
                print('Variable not found')
        




        # plot the list in vars
        for i in range(len(pvars)):
            plotVar(pvars[i],axs[i//2,i%2])
            axs[i//2,i%2].legend()

        # save the figure
        fig.savefig(fname+'.png',dpi=300)

        # show the figure
        plt.show()


        # plot new figure just cp_mix
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.plot(solt,solncp_mix,'k-',label='cp_mix')
        ax.set_xlabel('t')
        ax.set_ylabel('cp_mix')

        # log in y
        # ax.set_yscale('log')
        ax.set_xscale('log')
        plt.show()


        


# plot the variable chosen, just a single one
    @staticmethod
    def plotProduction(fname='solnPlasma',**kwargs):
        import numpy as np
        import matplotlib.pyplot as plt
        
        # select cylcer from 30 colors in cmap
        import matplotlib as mpl
        # 10 different colors
        
        # update the cycler
        # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

        # import scienceplots
        # plt.style.use('science')
        # # plt.style.use(['science','ieee'])
        # plt.rcParams.update({'figure.dpi': '100'})


        # colors = plt.cm.tab10(np.linspace(0,1,10))
        colors = plt.cm.tab20(np.linspace(0,1,15))
        custom_colors = [
        "#b9eb0d",  # Lime Green
        "#5ad7bb",  # Aqua
        "#f9ee37",  # Yellow
        "#c1c1c3",  # Cool Gray
        "#f03daf",  # Hot Pink
        "#ff5733",  # Orange
        "#4a90e2",  # Blue
        "#a85fd1",  # Purple
        "#ff8c00",  # Dark Orange
        "#00bfff",  # Deep Sky Blue
        ]
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

        # increase default line width to 1.5
        plt.rcParams['lines.linewidth'] = 1.5

        # make different line styles with symbols for each line
        styles = ['--', '-.', ':']
        markers = ['o','s','^','v','<','D','d','P','X']
        lineStyles_combo = []
        for i in range(len(styles)):
            for j in range(len(markers)):
                lineStyles_combo.append(styles[i]+markers[j])



        ww = 5
        hh = 5

        fmts = ['png','pdf']

    



        prx = kwargs.get('prx',10)
        mechFile = kwargs.get('mechFile',None)

        plotBoth = kwargs.get('plotBoth',True)

        reactions = []
        plasmaSystem = pl.PlasmaSolver
        plasmaSys = plasmaSystem(mechFile,verbose=False)
        # get all the reactions from the mechanism file
        if mechFile is not None:
                # Direct Solve with give mechanism file
                plasmaSys = plasmaSystem(mechFile,verbose=False)
                nRxn = len(plasmaSys.gas.reactions())




        # read the data
        data = np.load(fname+'.npz',allow_pickle=True)
        plot_varName = kwargs.get('plotVarName','dNdtRxn')

        # load time and dNdtRxn
        solt = data['t']
        # solndNdtRxn = data['dNdtRxn']

        ## if plot variable name starts with Y_X then X is the name of the species and we need the production rate of the spcies to be plotted
        if plot_varName.startswith('Y_'):
            # get the species name
            speName = plot_varName.split('_')[1]
            spe_idx = plasmaSys.gas.species_index(speName)

            ## now need to calculate the production rate of the species as all of this is not saved 
            dNdt = data['Wrxn']*plasmaSys.effStoic[:,spe_idx]
            solndNdtRxn = dNdt
        else:
            solndNdtRxn = data[plot_varName]

        # show size of solndNdtRxn and solt
        print('solndNdtRxn shape = ',solndNdtRxn.shape)
        print('solt shape = ',solt.shape)

        # make a figure
        fig, ax = plt.subplots(1,1,figsize=(ww,hh))



        ## to find the max and min, use the data at the half simulation time or at time = atTime
        atTime = kwargs.get('atTime',None)
        if atTime is not None:
            ## do sth
            idTime = np.argmin(np.abs(solt-atTime))
            #max
            maxdNdtRxn = (solndNdtRxn[idTime,:])
            # min
            mindNdtRxn = (-solndNdtRxn[idTime,:])

            # maxdNdtRxn_global = np.max(solndNdtRxn,axis=0)
            # # for minimum find max after making positive
            # mindNdtRxn_global = np.max(-solndNdtRxn,axis=0)

            # # average them out
            # maxdNdtRxn = (0.001*maxdNdtRxn + 0.999*maxdNdtRxn_global)
            # mindNdtRxn = (0.001*mindNdtRxn + 0.999*mindNdtRxn_global)
        else:
            # dNRxn has data for all reactions, 
            # find the max for positive and min for negative
            maxdNdtRxn = np.max(solndNdtRxn,axis=0)

            # for minimum find max after making positive
            mindNdtRxn = np.max(-solndNdtRxn,axis=0)


        

        # find the ids of the maximum prx reactions
        # argsort returns the indices of the sorted array in ascending order so the max is at the end
        idPrxPos = np.argsort(maxdNdtRxn)[-prx:]
        idPrxNeg = np.argsort(mindNdtRxn)[-prx:]


        RxnPOS = []
        RxnNeg = []
        # populate the reactions process
        for i in range(prx):
            RxnPOS.append(plasmaSys.gas.reaction(idPrxPos[i]))
            RxnNeg.append(plasmaSys.gas.reaction(idPrxNeg[i]))

        # show the positive and negative reactions with their reaction id
        print('Positive Reactions:')
        for i in range(prx):
            print('R'+str(idPrxPos[i]+1),RxnPOS[i])
        print('Negative Reactions:')
        for i in range(prx):
            print('R'+str(idPrxNeg[i]+1),RxnNeg[i])

        # Now plot
        if plotBoth:
            for i in np.arange(prx):
                ax.plot(solt,np.abs(solndNdtRxn[:,idPrxPos[i]]),'-',label='+R'+str(idPrxPos[i]+1))

        for i in np.arange(prx):
            # ax.plot(solt,np.abs(solndNdtRxn[:,idPrxNeg[i]]),'--',label='-R'+str(idPrxNeg[i]+1))
            # change linestyles every other line
            ax.plot(solt,np.abs(solndNdtRxn[:,idPrxNeg[i]]),lineStyles_combo[i],label='-R'+str(idPrxNeg[i]+1),color=colors[i],markevery=0.1,alpha=1.0)


        # set limit 
        ax.set_xlim([1e-13,solt[-1]])
        ax.set_ylim([1e22,1e33])


        # set labels
        ax.set_xlabel('$Time~[s]$')
        ax.set_ylabel('$dN_e/dt~[m^-3/s]$')

        # log scale
        ax.set_yscale('log')
        ax.set_xscale('log')

        # legend
        ax.legend(ncol=2)

        # tight layout
        fig.tight_layout()

        # show the figure
        # fig.savefig(fname+'_dNdtRxn.png',dpi=300)

        # save
        for fmt in fmts:
            fig.savefig(fname+'_dNdtRxn.'+fmt,dpi=300)

        plt.show()

    ## Another method to plot the species number density of selected species
    @staticmethod
    def plotNumberDensities(fname='solnPlasma',**kwargs):
        import numpy as np
        import matplotlib.pyplot as plt

        # select cylcer from 30 colors in cmap
        import matplotlib as mpl

        # # from tab20 import tab20_data
        # colors = plt.cm.tab20(np.linspace(0,1,20))
        # # update the cycler
        # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

        # import scienceplots
        # plt.style.use('science')
        # # plt.style.use(['science','ieee'])
        # plt.rcParams.update({'figure.dpi': '100'})



        prx = kwargs.get('prx',10)
        mechFile = kwargs.get('mechFile',None)
        species = kwargs.get('species',None)

        plasmaSystem = pl.PlasmaSolver3T
        plasmaSys = plasmaSystem(mechFile,verbose=False)
        # get all the reactions from the mechanism file
        if mechFile is not None:
                # Direct Solve with give mechanism file
                plasmaSys = plasmaSystem(mechFile,verbose=False)
                nRxn = len(plasmaSys.gas.reactions())

        # read the data
        data = np.load(fname+'.npz',allow_pickle=True)

        # load time and dNdtRxn
        solt = data['t']
        solnYsp = data['Ysp']

        # make a figure
        fig, ax = plt.subplots(1,1,figsize=(6,6))

        plotVars = kwargs.get('plotVars',['ele'])
        plotLabels = kwargs.get('plotLabels',['$e^-$'])
        # plot the species
        for i in range(len(plotVars)):
            spe_idx = plasmaSys.gas.species_index(plotVars[i])
            # if species is electron put a dark black solid line, if ends in small "p and m" put a dashed line
            if plotVars[i] == 'ele':
                ax.plot(solt,solnYsp[:,spe_idx],'-',label=plotLabels[i],marker='o',markevery=0.1,markersize=2,markerfacecolor='k',markeredgecolor='k')
            elif plotVars[i][-1] == 'p':
                ax.plot(solt,solnYsp[:,spe_idx],'--',label=plotLabels[i])
            elif plotVars[i][-1] == 'm':
                ax.plot(solt,solnYsp[:,spe_idx],'-.',label=plotLabels[i])
            else:
                ax.plot(solt,solnYsp[:,spe_idx],'-',label=plotLabels[i])

        # set labels
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Number Density [m$^{-3}$]')
        # log scale
        ax.set_yscale('log')    
        ax.set_xscale('log')

        # limit x
        ax.set_xlim([1e-13,solt[-1]])
        # ylim
        ax.set_ylim([1e16,1.1*solnYsp.max()])

        # make grids both majot and minor
        ax.grid(True, which='both', ls='--', alpha=0.5)


        # legend, columns should be num_vars//4
        ax.legend(ncol=len(plotVars)//4)

        # show the figure
        for fmt in ['png','pdf']:
            fig.savefig(fname+'_Ysp.'+fmt,dpi=300)
        plt.show()

    ## show reactions given fname and ID of the reaction
    @staticmethod
    def printReactions(**kwargs):
        # plasmaSys.gas.reaction
        mechFile = kwargs.get('mechFile',None)
        rxnIds = kwargs.get('pids',None)

        plasmaSystem = pl.PlasmaSolver3T
        plasmaSys = plasmaSystem(mechFile,verbose=False)

        # if pids = 'All', print all reactions
        
        if rxnIds == 'All':
            nRxn = len(plasmaSys.gas.reactions())
            for i in range(nRxn):
                print('R'+str(i+1),plasmaSys.gas.reaction(i))
        else:

            # print the reactions in rxnIds
            for i in rxnIds:
                print('R'+str(i+1),plasmaSys.gas.reaction(i))


 

    @staticmethod
    def plotProduction_electrons(fname='solnPlasma',**kwargs):
        
        prx = kwargs.get('prx',10)
        mechFile = kwargs.get('mechFile',None)

        # call plotProduction_generic
        Solution.plotProduction_generic(fname=fname,prx=prx,mechFile=mechFile,plotVarName='dNdtRxn')

    @staticmethod
    def plotProduction_generic(fname='solnPlasma',**kwargs):
        import numpy as np
        import matplotlib.pyplot as plt

        # select cylcer from 30 colors in cmap
        import matplotlib as mpl
        # from tab20 import tab20_data
        colors = plt.cm.tab20(np.linspace(0,1,20))
        # update the cycler
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

        prx = kwargs.get('prx',10)
        mechFile = kwargs.get('mechFile',None)
        pos = kwargs.get('pos',1) # 1 is positive, if -1 then negative for max

        reactions = []
        plasmaSystem = pl.PlasmaSolver
        plasmaSys = plasmaSystem(mechFile,verbose=False)
        # get all the reactions from the mechanism file
        if mechFile is not None:
                # Direct Solve with give mechanism file
                plasmaSys = plasmaSystem(mechFile,verbose=False)
                nRxn = len(plasmaSys.gas.reactions())


        Na = 6.022e23

        # read the data
        data = np.load(fname+'.npz',allow_pickle=True)

        # load time and dNdtRxn
        solt = data['t']
        # solndNdtRxn_def = data['dNdtRxn']
        plot_varName = kwargs.get('plotVarName','dNdtRxn')

        ## if plot variable name starts with Y_X then X is the name of the species and we need the production rate of the spcies to be plotted
        if plot_varName.startswith('Y_'):
            # get the species name
            speName = plot_varName.split('_')[1]
            spe_idx = plasmaSys.gas.species_index(speName)

            ## now need to calculate the production rate of the species as all of this is not saved 
            dNdt = data['Wrxn']*plasmaSys.effStoic[:,spe_idx]
            solndNdtRxn = dNdt*pos



        id_col = kwargs.get('id_col',0) # default is 0, if there are multiple columns

        if plot_varName == 'Qrxn_I':
            wrxn_data = data['Wrxn']
            print('wrxn_data shape = ',wrxn_data.shape, 'dhrxn shape = ',data['dhrxn'].shape)

            solndNdtRxn = -1.0e-3*(wrxn_data/Na)*data['dhrxn'][:,id_col,:]*pos # J/m3-s
        else:
            solndNdtRxn = data[plot_varName]*pos


        # # if there are multiple columns, select the id_col
        # if len(solndNdtRxn.shape) > 2:
        #     solndNdtRxn = solndNdtRxn[:,id_col]


        # make a figure
        fig, ax = plt.subplots(1,1,figsize=(6,6))

        # dNRxn has data for all reactions, find the max for positive and min for negative
        maxdNdtRxn = np.max(solndNdtRxn,axis=0)
        
        # for minimum find max after making positive
        mindNdtRxn = np.max(-solndNdtRxn,axis=0)

        # find the ids of the maximum prx reactions
        # argsort returns the indices of the sorted array in ascending order so the max is at the end
        idPrxPos = np.argsort(maxdNdtRxn)[-prx:]
        idPrxNeg = np.argsort(mindNdtRxn)[-prx:]


        RxnPOS = []
        RxnNeg = []
        # populate the reactions process
        for i in range(prx):
            RxnPOS.append(plasmaSys.gas.reaction(idPrxPos[i]))
            RxnNeg.append(plasmaSys.gas.reaction(idPrxNeg[i]))

        # show the positive and negative reactions with their reaction id
        print('Positive Reactions:')
        for i in range(prx):
            print('R'+str(idPrxPos[i]+1),RxnPOS[i])
        print('Negative Reactions:')
        for i in range(prx):
            print('R'+str(idPrxNeg[i]+1),RxnNeg[i])

        # Now plot
        for i in np.arange(prx):
            ax.plot(solt,np.abs(solndNdtRxn[:,idPrxPos[i]]),'-',label='+R'+str(idPrxPos[i]+1))

        for i in np.arange(prx):
            ax.plot(solt,np.abs(solndNdtRxn[:,idPrxNeg[i]]),'--',label='-R'+str(idPrxNeg[i]+1))

        # set labels
        ax.set_xlabel('t')
        ax.set_ylabel(plot_varName+'_'+str(id_col))

        # log scale
        ax.set_yscale('log')
        ax.set_xscale('log')

        # legend
        ax.legend()

        fig.tight_layout()

        # show the figure
        fig.savefig(fname+'_'+plot_varName+str(id_col)+'.png',dpi=300)
        plt.show()

    # static method for plot_production for Qrxn_I
    @staticmethod
    def plotProduction_Qrxn_I(fname='solnPlasma',**kwargs):
        # call plotProduction_generic
        # needs id_col: 0,1,2 for Tg, Tv and Te
        Solution.plotProduction_generic(fname=fname,plotVarName='Qrxn_I',**kwargs)


# Define other methods needed here
# this is better to include with integrator rather than here
def updateFromYsp(self):
    '''
    Update the system object to new state based on Ysp change only.
    This should be used to update the sate while integrating the species equation as temeprature is not changing.
    '''
    # update from base class
    self.p = self.pressure()        # pressure in Pa
    # self.rho = self.density()       # density in kg/m3
    self.X = self.numbDensity2X(self.Ysp) # mole fraction
    # set UV
    # self.gas.UV = self.Ug , 1.0/self.rho
    # # put electrons zero
    # self.X[-1] = 0.0
    self.gas.TPX = self.Temp[0], self.p, self.X
    
    # self.gas.TDX = self.Temp[0], self.rho, self.X
    self.Krxn = self.Krxn_()
    self.Wrxn = self.Wrxn_()
    self.dYdt = self.dYdt_()
    

