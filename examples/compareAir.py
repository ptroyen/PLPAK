import sys
import os

dir_plpak = os.path.dirname("E:\POKHAREL_SAGAR\gits\plasmaKinetics\PLPAK")
sys.path.append(dir_plpak)
import PLPAK as pl
# add path to LateX : C:\Users\pokharel_sagar\AppData\Local\Programs\MiKTeX\miktex\bin\x64
# sys.path.append(r"C:\Users\pokharel_sagar\AppData\Local\Programs\MiKTeX\miktex\bin\x64")
# C:\Program Files\MiKTeX\miktex\bin\x64
sys.path.append(r"C:\Program Files\MiKTeX\miktex\bin\x64")

import cantera as ct
import numpy as np
import scipy as sc

# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

# from solveWithMech import dYdt_all, directSolve

# list of plasmasolution files to compare
# list of the labels to put
# list of refFiles to compare with
# list of refLabels to put




# define a function to plot from saved data
def plotCompare(dfList, labels, refFiles, refLabels, refTimeScale, refValueScale, plotTemp=False,**kwargs):

    import os
    import sys

    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')
    # plt.style.use(['science','ieee'])
    plt.rcParams.update({'figure.dpi': '100'})

    fmtsSave = ['png','pdf']

    # check the number of files in the list
    nplots = len(dfList)

    # check number of reference files
    nrefs = len(refFiles)

    # color for each plot
    colors = ['black','tab:red','blue','tab:cyan','olive','tab:gray','tab:olive','tab:cyan']    
    colors_ref = ['blue','tab:orange','tab:red','m','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    romans = ['I','II','III','IV','V','VI','VII','VIII']

    # choose most distinct colors, from set1
    colors = plt.get_cmap("tab10").colors
    # HEX FOR BLACK : #000000
    colors = [
            "#00FFFF",   # Cyan
            "#FF00FF",  # Magenta
            "#FF0000",  # Red
            "#0000FF",  # Orange 0000FF
            "#0000FF",  # Blue
            "#000000",  # Black
            "#00FF00"  # Green
    ]

    # # make gradual colors from cmaps like : viridis, inferno, plasma, magma
    # # single hue progression cmap like : 'cividis', 'cool', 'spring', 'summer', 'autumn', 'winter'
    # colors = plt.cm.Set1(np.linspace(0,1,nplots))

    # from kwargs
    expAlpha = kwargs.get('expAlpha',0.6)

    saveName = kwargs.get('saveName',dfList[0][:-5])
    yScale = kwargs.get('yScale','linear')
    xScale = kwargs.get('xScale','linear')

    normalize = kwargs.get('normalize',False)
    ylabel = kwargs.get('ylabel',None)


    ww = 4
    hh = 4


    

    # # import the reference data from Aleksandrov_1atm_N2femtosecondDecay.dat
    # refdata = np.loadtxt("Aleksandrov_1atm_N2femtosecondDecay.dat", skiprows=1)
    # reft = refdata[:,0]*1e-9
    # refne = refdata[:,1]

    # refData2 = np.loadtxt("N2Data.csv")
    # refne2 = refData2[:,1]
    # reft2 = refData2[:,0]*1e-9

    tf = 1e6 # time factor to multiply the time in us
    # plot the data
    fig, ax = plt.subplots()
    fig.set_size_inches(ww,hh)

    # list of color you wanna use


    # plot the reference data
    for i in range(nrefs):

        # read the data
        fname = refFiles[i]
        expData = np.loadtxt(fname)
        ext = expData[:,0]*refTimeScale[i]
        exne = expData[:,1]*refValueScale[i]

        # if fname has shneider then without marker
        if 'shneider' in fname:
            ax.plot(tf*ext, exne, '-', label=refLabels[i], lw=1.5, color= colors_ref[i])
        else:
            # plot the experimental data
            ax.plot(tf*ext, exne, 'o', label=refLabels[i], markersize=6, color= colors_ref[i],alpha=expAlpha,markerfacecolor="None")
    


    for i in range(nplots):
        
        # read the data
        # load the data
        fname = dfList[i]
        data = np.load(fname+'.npz',allow_pickle=True)
        solntimes = data['t']
        solnYsp = data['Ysp']
        solnne = solnYsp[:,-1]



        # show ne
        # print('ne = ',solnne)
        labelI = labels[i]
        if normalize:
            solnne = solnne/np.max(solnne)
        # ax.plot(tf*solntimes, solnne,'-',lw=1.5, label=str(romans[i]),color=colors[i])
        ax.plot(tf*solntimes, solnne,'-',lw=1.5, label=labelI,color=colors[i])


        # # sum N , N_D and N_P if they are present
        # if solnYsp.shape[1] > 14:
        #     solnNall = solnYsp[:,0]+solnYsp[:,-2]+solnYsp[:,-3]
        # else:
        #     solnNall = solnYsp[:,0]
        # # plot atomic nitrogen
        # ax.plot(tf*solntimes, solnNall,'--',lw=1.5,color=colors[i])

    


    # # For coloring reference use the same color used for similar initiali conditions
    # ax.plot(tf*reft, refne, 'o', label='Aleksandrov', markersize=6, color= colors[0],alpha=0.7,markerfacecolor="None")
    # # ax.plot(tf*reft, refne, 'o', label='Aleksandrov', markersize=6, color= colors[0],alpha=0.7)

    # # plot another reference data
    # # ax.plot(tf*reft2, refne2, 's', label='Chizhov', markersize=6, color=colors[1],alpha=0.7,markerfacecolor="None")
    # ax.plot(tf*reft2, refne2, 's', label='Chizhov', markersize=6, color=colors[1],alpha=0.8)


    # # # plot the experimental data
    # # ax.errorbar(tf*ext, exnen, yerr=exneErr, xerr=None, fmt='o', label='Exp-normalized', markersize=8, color='blue',alpha=0.7)

    # # # not normalized ne
    # # ax.errorbar(tf*ext, exne, yerr=exneErr, xerr=None, fmt='o', label='Exp', markersize=8, color='red',alpha=0.4)


    ax.set_xlabel('Time [$\mu$s]')
    ax.set_ylabel('Electron Density [$m^{-3}$]')
    ax.legend()

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # grids
    # ax.grid(True, which='both', ls='--', alpha=0.5)

    # # put the number labels on both sides for y, so right and left
    # ax.yaxis.set_ticks_position('both')
    # # tick labels on both sides
    # # ax.tick_params(which='both', direction='in', top=True, right=True)
    # ax.tick_params(right=True, top=True, labelright=True, labeltop=True,rotation=0)


    # log x and y axis
    ax.set_yscale(yScale)
    ax.set_xscale(xScale)

    # ax.set_xlim(1.0e-12*tf, tf*solntimes[-1][-1])
    # ax.set_xlim(1.0e-12*tf, tf*80.0e-9)
    ax.set_xlim(1.0e-10*tf, tf*80.0e-9)
    # # set y lim to min and max of the data
    # ax.set_ylim(1.0e19, 1.1*np.max(solnne))


    plt.tight_layout()

    # save plot with the same name as the data
    for fmt in fmtsSave:
        saveTo = saveName+'.'+fmt
        fig.savefig(saveTo,dpi=300)
        # fig.savefig('test.'+fmt,dpi=300)
        # print the name of the file
        # print(dflist[0][:-5]+'.'+fmt)

    # fig.savefig(fname+'.png',dpi=300)
    plt.show()

    # make a new plot for temperature for dfList[0]

    # figure
    fig, ax = plt.subplots()
    fig.set_size_inches(ww,hh)



    if plotTemp:
        # read
        fname = dfList[0]
        data = np.load(fname+'.npz',allow_pickle=True)
        solntimes = data['t']
        solnT = data['Temp']
        Tg = solnT[:,0]
        Tv = solnT[:,1]
        Te = solnT[:,2]

        # plot
        ax.plot(tf*solntimes, Tg, 'k-', label='Tg')
        ax.plot(tf*solntimes, Tv, 'b--', label='Tv')
        ax.plot(tf*solntimes, Te, 'r:', label='Te')

        ax.set_xlabel('Time [$\mu$s]')
        ax.set_ylabel('Temperature [K]')
        ax.legend()

        # log scale y
        ax.set_yscale('log')
        ax.set_xscale('log')

        # limit x
        ax.set_xlim(1.0e-12*tf, tf*np.max(solntimes))

        # save
        for fmt in fmtsSave:
            fig.savefig(fname+'Temp.'+fmt,dpi=300)

        plt.show()





def compareExperiments():

    dflist = ["airPlasmaSolnPapeer", "airPlasmaSolnChizhov", "airPlasmaSolnAleks"]
    refFiles = ["dataExp/papeerAir14e21normalized.csv", "dataExp/airData.csv","dataExp/Aleksandrov_1atm_N2femtosecondDecay.dat"]
    refLabels = ["Papeer", "Chizhov", "Aleks"]
    refTimeScale = [1.0, 1.0e-9, 1.0e-9]
    refValueScale = [1.4e22, 1.0, 1.0]
    labels = ["I", "II", "III"]

    # # # skip the aleksandrov data
    dflist = dflist[:-1]
    labels = labels[:-1]
    refFiles = refFiles[:-1]
    refLabels = refLabels[:-1]
    refTimeScale = refTimeScale[:-1]
    refValueScale = refValueScale[:-1]


    # call 
    plotCompare(dflist, labels, refFiles, refLabels, refTimeScale, refValueScale, plotTemp=False,expAlpha=0.8)


# compare old and new N2 models
def compareN2Models():
    dflist = ["N2SolnsOld", "N2SolnsNew"]
    refFiles = ["dataExp/papeerN21e21normalized.csv"]

    # # include Shneider data
    # dflist = ["N2SolnsOld", "N2SolnsNew", "fracO2Change22p0O2_1e23"]
    # refFiles = ["dataExp/papeerN21e21normalized.csv", "dataExp/shneiderAirDecay.csv"]

    # # read dataExp/shneiderAirDecay.csv and change from comma delimited to space/tab delimited
    # import pandas as pd
    # csvDF = pd.read_csv("dataExp/shneiderAirDecay.csv",delimiter=',')
    # csvDF.to_csv("dataExp/shneiderAirDecay.csv",sep='\t',index=False)

    refLabels = ["Papeer", "Shneider"]
    refTimeScale = [1.0, 1.0]
    refValueScale = [1.0e21, 1.0]
    labels = ["Peters", "This work: $N_2$", "This work: Air"]
    xScale = 'log'
    yScale = 'log'

    # # # # skip the aleksandrov data
    # dflist = dflist[:-1]
    # labels = labels[:-1]
    # refFiles = refFiles[:-1]
    # refLabels = refLabels[:-1]
    # refTimeScale = refTimeScale[:-1]
    # refValueScale = refValueScale[:-1]

    #call 
    plotCompare(dflist, labels, refFiles, refLabels, refTimeScale, refValueScale, xScale=xScale,yScale=yScale, saveName='compModelsN2',
                 plotTemp=False,expAlpha=0.8)



# compare same intensity different percent of o2
def compareO2Percent():
    # files : fracO2Change22p0O2
    # dflist = ["fracO2Change0p0O2", "fracO2Change0p1O2", "fracO2Change2p0O2", "fracO2Change22p0O2"]
    dflist = ["fracO2Change0p0O2", "fracO2Change0p1O2", "fracO2Change2p0O2", "fracO2Change22p0O2",'fracO2Change22p0O2_1e23']
    # dflist = ["fracO2Change0p0O2","fracO2Change22p0O2_1e23"]
    refFiles = []
    refLabels = []
    refTimeScale = []
    refValueScale = []
    normalize = False
    ylabel = 'Electron Density [$m^{-3}$]'
    labels = ["$N_2$","$N_2+0.1\%O_2$","$N_2+2\%O_2$","Air","Air, $N_e = 10^{23} [m^{-3}]$"]
    # labels = ["1","2","3","4"]
    # labels = ["$N_2$", "$Air$"]
    saveName='NeCompN2AirAll_add_lin'
    yScale = 'log'
    xScale = 'linear'
    # normalize = True
    # ylabel = 'Normalized Electron Density'

    # call 
    plotCompare(dflist, labels, refFiles, refLabels, refTimeScale, refValueScale, plotTemp=False,normalize=normalize,ylabel=ylabel,
                saveName=saveName,yScale=yScale,xScale=xScale,expAlpha=0.8)

if __name__ == "__main__":

    # # list of plasmasolution files to compare
    # # list of the labels to put
    # # list of refFiles to compare with
    # # list of refLabels to put

    # compareExperiments()
    compareN2Models()
    # compareO2Percent()



