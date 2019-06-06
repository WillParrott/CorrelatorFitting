
import collections
import sys
import h5py
import gvar as gv
import numpy as np
import corrfitter as cf
import corrbayes
import copy
import matplotlib.pyplot as plt
import os.path
import pickle

################ F PARAMETERS  #############################
F = collections.OrderedDict()
F['conf'] = 'F'
F['filename'] = 'HstoEtasfine5Neg.gpl'
F['masses'] = ['0.449','0.566','0.683','0.8']
F['twists'] = ['0','0.4281','1.282','2.141','2.570','2.993']
F['m_s'] = '0.0376'
F['Ts'] = [14,17,20]
F['tp'] = 96
F['L'] = 32
F['tminG'] = 2
F['tminNG'] = 2
F['tminD'] = 3
F['Stmin'] = 2
F['Vtmin'] = 1
F['an'] = '0.2(2)'
F['Vn'] = '0.01(15)'
F['V0'] = '0.1(5)' 
F['loosener'] = 0.5                         #Loosener on V_nn[0][0]
F['Mloosener'] = 0.1                        #Loosener on ground state 
F['oMloosener'] = 0.2                       #Loosener on oscillating ground state
F['a'] = 0.1715/(1.9006*0.1973)
F['goldTag'] = 'meson.m{0}_m{1}'
F['nonGoldTag'] = 'meson-G5T.m{0}_m{1}'
F['daugterTag'] = ['etas','etas_p0.0728','etas_p0.218','etas_p0.364','etas_p0.437','etas_p0.509'] 
F['threePtTag'] = ['{0}.T{1}_m{2}_m{3}_m{2}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}']

                
################ SF PARAMETERS #############################
SF = collections.OrderedDict()
SF['conf'] = 'SF'
SF['filename'] = 'SFCopy.gpl'
SF['masses'] = ['0.274','0.450','0.6','0.8']
SF['twists'] = ['0','1.261','2.108','2.946','3.624']
SF['m_s'] = '0.0234'
SF['Ts'] = [20,25,30]
SF['tp'] = 144
SF['L'] = 48
SF['tminG'] = 2
SF['tminNG'] = 3
SF['tminD'] = 2
SF['Stmin'] = 2
SF['Vtmin'] = 2
SF['an'] = '0.1(3)'
SF['Vn'] = '0.01(15)'
SF['V0'] = '0.1(4)'
SF['loosener'] = 0.7                         #Loosener on V_nn[0][0]
SF['Mloosener'] = 0.1                        #Loosener on ground state 
SF['oMloosener'] = 0.2                       #Loosener on oscillating ground state
SF['a'] = 0.1715/(2.8960*0.1973)
SF['goldTag'] = 'meson.m{0}_m{1}'
SF['nonGoldTag'] = 'meson2G5T.m{0}_m{1}'
SF['daugterTag'] = ['etas_p0','etas_p0.143','eta_s_tw2.108_m0.0234','etas_p0.334','eta_s_tw3.624_m0.0234']
SF['threePtTag'] = ['{0}.T{1}_m{2}_m{3}_m{2}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}']

################ USER INPUTS ################################
#############################################################
DoFit = True
FitAll = False
TestData = False
Fit = F                                               # Choose to fit F, SF or UF
FitMasses = [0,1,2,3]                                 # Choose which masses to fit
FitTwists = [0,1,2,3,4]                               # Choose which twists to fit
FitTs = [0,1,2]
FitCorrs = ['G','NG','D','S','V']  # Choose which corrs to fit ['G','NG','D','S','V']
Chained = False
CorrBayes = False
SaveFit = True
svdnoise = True
priornoise = True
ResultPlots = 'N'         # Tell what to plot against, "Q", "N","Log(GBF)", False
AutoSvd = True
SvdFactor = 1.0           # Multiplies saved SVD 
Nmax = 7                               # Number of exp to fit nterm dictates which will not be marginalised 
                      
##############################################################
##############################################################

def make_params(FitMasses,FitTwists,FitTs):
    TwoPts = collections.OrderedDict()
    ThreePts = collections.OrderedDict()
    masses = []
    twists = []
    Ts = []    
    m_s = Fit['m_s']
    filename = Fit['filename']          
    for i in FitMasses:
        masses.append(Fit['masses'][i])
    for j in FitTwists:
        twists.append(Fit['twists'][j])
    for k in FitTs:
        Ts.append(Fit['Ts'][k])
    for twist in Fit['twists']:
        TwoPts['Dtw{0}'.format(twist)] = Fit['daugterTag'][Fit['twists'].index(twist)]
    for mass in Fit['masses']:
        TwoPts['Gm{0}'.format(mass)] = Fit['goldTag'].format(m_s,mass)
        TwoPts['NGm{0}'.format(mass)] = Fit['nonGoldTag'].format(m_s,mass)
        for twist in Fit['twists']:
            for T in Ts:
                ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)] = Fit['threePtTag'][Fit['twists'].index(twist)].format('current-scalar',T,m_s,mass,twist)
                ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)] = Fit['threePtTag'][Fit['twists'].index(twist)].format('current-vector',T,m_s,mass,twist)
                
    return(TwoPts,ThreePts,masses,twists,Ts)



def make_data(filename,N):
    if CorrBayes == True:
        dset = cf.read_dataset(filename)
        Autoprior,new_dset = corrbayes.get_prior(dset,15,N,loosener = 1.0)        
        return(Autoprior,gv.dataset.avg_data(new_dset))
    else:
        Autoprior = 0
        return(Autoprior,gv.dataset.avg_data(cf.read_dataset(filename)))



def eff_calc():
    T = Ts[0]
    tp = Fit['tp']
    #Make this do plots
    M_effs = collections.OrderedDict()
    A_effs = collections.OrderedDict()
    V_effs = collections.OrderedDict()       
    M_eff = collections.OrderedDict()
    A_eff = collections.OrderedDict()
    V_eff = collections.OrderedDict()    
    #plt.figure(1)
    for mass in Fit['masses']:
        M_effs['Gm{0}'.format(mass)] = []
        M_effs['NGm{0}'.format(mass)] = []
        M_eff['Gm{0}'.format(mass)] = 0
        M_eff['NGm{0}'.format(mass)] = 0
        A_effs['Gm{0}'.format(mass)] = []
        A_effs['NGm{0}'.format(mass)] = []
        A_eff['Gm{0}'.format(mass)] = 0
        A_eff['NGm{0}'.format(mass)] = 0
        #plt.figure(mass)
        for t in range(2,tp-2):
            G = (data[TwoPts['Gm{0}'.format(mass)]][t-2] + data[TwoPts['Gm{0}'.format(mass)]][t+2])/(2*data[TwoPts['Gm{0}'.format(mass)]][t])
            if G >= 1:
                M_effs['Gm{0}'.format(mass)].append(gv.arccosh(G)/2)
            else:
                M_effs['Gm{0}'.format(mass)].append(0)
                
            NG = (data[TwoPts['NGm{0}'.format(mass)]][t-2] + data[TwoPts['NGm{0}'.format(mass)]][t+2])/(2*data[TwoPts['NGm{0}'.format(mass)]][t])
            if NG >= 1:
                M_effs['NGm{0}'.format(mass)].append(gv.arccosh(NG)/2)
            else:
                M_effs['NGm{0}'.format(mass)].append(0)            
           
        #plt.errorbar(M_effs['Gm{0}'.format(mass)][:].mean, yerr=M_effV[:].sdev, fmt='ko')
        #plt.errorbar(M_effs['Gm{0}'.format(mass)][:].mean, yerr=M_effS[:].sdev, fmt='ro')
    for twist in Fit['twists']:
        M_effs['Dtw{0}'.format(twist)] = []
        M_eff['Dtw{0}'.format(twist)] = 0
        A_effs['Dtw{0}'.format(twist)] = []
        A_eff['Dtw{0}'.format(twist)] = 0
        for t in range(2,tp-2):
            D = (data[TwoPts['Dtw{0}'.format(twist)]][t-2] + data[TwoPts['Dtw{0}'.format(twist)]][t+2])/(2*data[TwoPts['Dtw{0}'.format(twist)]][t])
            if D >= 1:
                M_effs['Dtw{0}'.format(twist)].append(gv.arccosh(D)/2)
            else:
                M_effs['Dtw{0}'.format(twist)].append(0)
    #print(M_effs)
        #plt.errorbar(M_effs['Dtw{0}'.format(twist)][:].mean, yerr=M_effS[:].sdev, fmt='ro')
    #plt.title('M_eff')
    #print('M',M_effs)
    for mass in Fit['masses']:
        denomG = 0
        denomNG = 0
        for i in range(int(3*tp/8)-4,int(3*tp/8)+4):
            M_eff['Gm{0}'.format(mass)] += M_effs['Gm{0}'.format(mass)][i]
            if M_effs['Gm{0}'.format(mass)][i] != 0:
                denomG += 1
            M_eff['NGm{0}'.format(mass)] += M_effs['NGm{0}'.format(mass)][i]
            if M_effs['NGm{0}'.format(mass)][i] != 0:
                denomNG += 1
        M_eff['Gm{0}'.format(mass)] = M_eff['Gm{0}'.format(mass)]/denomG
        M_eff['NGm{0}'.format(mass)] = M_eff['NGm{0}'.format(mass)]/denomNG
    for twist in Fit['twists']:
        denomD = 0
        for i in range(int(3*tp/8)-4,int(3*tp/8)+4):
            M_eff['Dtw{0}'.format(twist)] += M_effs['Dtw{0}'.format(twist)][i]
            if M_effs['Dtw{0}'.format(twist)][i] != 0:
                denomD +=1
        M_eff['Dtw{0}'.format(twist)] = M_eff['Dtw{0}'.format(twist)]/denomD
        p = np.sqrt(3)*np.pi*float(twist)/Fit['L']
        M_effTheory = gv.sqrt(M_eff['Dtw0']**2 + p**2)
        if abs(((M_eff['Dtw{0}'.format(twist)]-M_effTheory)/M_effTheory).mean) > 0.1:
            print('Substituted M_effTheory for twist',twist, 'Difference:',(M_eff['Dtw{0}'.format(twist)]-M_effTheory)/M_effTheory,'old:',M_eff['Dtw{0}'.format(twist)],'New:',M_effTheory)
            M_eff['Dtw{0}'.format(twist)] = copy.deepcopy(M_effTheory)
            
    #print('M_eff',M_eff)        
    #plt.figure(2)
    for mass in Fit['masses']:
        for t in range(1,tp-2):
            numerator = data[TwoPts['Gm{0}'.format(mass)]][t]
            if numerator >= 0:
                A_effs['Gm{0}'.format(mass)].append(gv.sqrt(numerator/(gv.exp(-M_eff['Gm{0}'.format(mass)]*t)+gv.exp(-M_eff['Gm{0}'.format(mass)]*(tp-t)))))
            else:
                A_effs['Gm{0}'.format(mass)].append(0)
            numerator = data[TwoPts['NGm{0}'.format(mass)]][t]
            if numerator >= 0:
                A_effs['NGm{0}'.format(mass)].append(gv.sqrt(numerator/(gv.exp(-M_eff['NGm{0}'.format(mass)]*t)+gv.exp(-M_eff['NGm{0}'.format(mass)]*(tp-t)))))
            else:
                A_effs['NGm{0}'.format(mass)].append(0)

                
    for twist in Fit['twists']:
        for t in range(1,tp-2):
            numerator = data[TwoPts['Dtw{0}'.format(twist)]][t]
            if numerator >= 0:
                A_effs['Dtw{0}'.format(twist)].append(gv.sqrt(numerator/(np.exp(-M_eff['Dtw{0}'.format(twist)]*t)+np.exp(-M_eff['Dtw{0}'.format(twist)]*(tp-t)))))
            else:
                A_effs['Dtw{0}'.format(twist)].append(0)
               
    #print('A',A_effs)          
    for mass in Fit['masses']:
        denomG = 0
        denomNG = 0
        for i in range(int(3*tp/8)-4,int(3*tp/8)+4):
            A_eff['Gm{0}'.format(mass)] += A_effs['Gm{0}'.format(mass)][i]
            if A_effs['Gm{0}'.format(mass)][i] != 0:
                denomG += 1
            A_eff['NGm{0}'.format(mass)] += A_effs['NGm{0}'.format(mass)][i]
            if A_effs['NGm{0}'.format(mass)][i] != 0:
                denomNG += 1
        A_eff['Gm{0}'.format(mass)] = A_eff['Gm{0}'.format(mass)]/denomG
        A_eff['NGm{0}'.format(mass)] = A_eff['NGm{0}'.format(mass)]/denomNG
    for twist in Fit['twists']:
        denomD = 0
        for i in range(int(3*tp/8)-4,int(3*tp/8)+4):         
            A_eff['Dtw{0}'.format(twist)] += A_effs['Dtw{0}'.format(twist)][i]
            if A_effs['Dtw{0}'.format(twist)][i] != 0:
                denomD += 1
        A_eff['Dtw{0}'.format(twist)] = A_eff['Dtw{0}'.format(twist)]/denomD
    #print('A_eff',A_eff)
            #plt.errorbar(t, A_effV[t-1].mean, yerr=A_effV[t-1].sdev, fmt='ko')
        #plt.errorbar(t, A_effS[t-1].mean, yerr=A_effS[t-1].sdev, fmt='ro')
    #plt.title('A_eff')
    #plt.savefig('AeffSf_m{0}_tw{1}'.format(mass,twist))
    #plt.show()
    #print(M_eff)    
    #plt.show()
    
    for mass in Fit['masses']:
        for twist in Fit['twists']:        
            V_effs['Sm{0}_tw{1}'.format(mass,twist)] = []
            V_effs['Vm{0}_tw{1}'.format(mass,twist)] = []
            V_eff['Sm{0}_tw{1}'.format(mass,twist)] = 0
            V_eff['Vm{0}_tw{1}'.format(mass,twist)] = 0
            for t in range(T):
                V_effs['Sm{0}_tw{1}'.format(mass,twist)].append(A_eff['Gm{0}'.format(mass)]*A_eff['Dtw{0}'.format(twist)]*data[ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['Dtw{0}'.format(twist)]][t]*data[TwoPts['Gm{0}'.format(mass)]][T-t]))
                V_effs['Vm{0}_tw{1}'.format(mass,twist)].append(A_eff['NGm{0}'.format(mass)]*A_eff['Dtw{0}'.format(twist)]*data[ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['Dtw{0}'.format(twist)]][t]*data[TwoPts['NGm{0}'.format(mass)]][T-t]))
                

    for mass in Fit['masses']:
        for twist in Fit['twists']:           
            for t in range(int(T/2-2),int(T/2+2)):
                V_eff['Sm{0}_tw{1}'.format(mass,twist)]  += (1/4)*V_effs['Sm{0}_tw{1}'.format(mass,twist)][t]
                V_eff['Vm{0}_tw{1}'.format(mass,twist)]  += (1/4)*V_effs['Vm{0}_tw{1}'.format(mass,twist)][t]
            if V_eff['Sm{0}_tw{1}'.format(mass,twist)] < 0.1 or V_eff['Sm{0}_tw{1}'.format(mass,twist)] > 1.5:
                V_eff['Sm{0}_tw{1}'.format(mass,twist)] = gv.gvar('0.5(5)')
            if V_eff['Vm{0}_tw{1}'.format(mass,twist)] < 0.1 or V_eff['Vm{0}_tw{1}'.format(mass,twist)] > 1.5:
                V_eff['Vm{0}_tw{1}'.format(mass,twist)] = gv.gvar('0.5(5)')
    #print(V_effs)
    #print(V_eff)
    return(M_eff,A_eff,V_eff)





def make_prior(N,M_eff,A_eff,V_eff,Autoprior):    
    Lambda = 0.5    ###Set Lambda_QCD in GeV
    an = Fit['an']
    Vn = Fit['Vn']
    V0 = Fit['V0']
    a = Fit['a']
    loosener = Fit['loosener']
    Mloosener = Fit['Mloosener']                 
    oMloosener = Fit['oMloosener']
    prior = gv.BufferDict()
    if CorrBayes == True:
        TwoKeys,ThreeKeys = makeKeys()
        for key in Autoprior:
            if key in TwoKeys:
                prior[key] = Autoprior[key]
                if key in ThreeKeys:
                    prior[key] = Autoprior[key]
    else:        
        if 'D' in FitCorrs:
            for twist in twists:        
                # Daughter
                prior['log({0}:a)'.format(TwoPts['Dtw{0}'.format(twist)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:{0})'.format(TwoPts['Dtw{0}'.format(twist)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,0.5*Lambda*a)]))
                prior['log({0}:a)'.format(TwoPts['Dtw{0}'.format(twist)])][0] = gv.log(gv.gvar(A_eff['Dtw{0}'.format(twist)].mean,loosener*A_eff['Dtw{0}'.format(twist)].mean))
                prior['log(dE:{0})'.format(TwoPts['Dtw{0}'.format(twist)])][0] = gv.log(gv.gvar(M_eff['Dtw{0}'.format(twist)].mean,Mloosener*M_eff['Dtw{0}'.format(twist)].mean))
            #prior['log(etas:dE)'][1] = gv.log(gv.gvar(EtaE1[str(twist)]))
        
                # Daughter -- oscillating part
                if twist!='0':                      
                    prior['log(o{0}:a)'.format(TwoPts['Dtw{0}'.format(twist)])] = gv.log(gv.gvar(N * [an]))
                    prior['log(dE:o{0})'.format(TwoPts['Dtw{0}'.format(twist)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,0.5*Lambda*a)]))
                #prior['log(o{0}:a)'.format(TwoPts['Dtw{0}'.format(twist)])][0] = gv.log(gv.gvar(A_eff['Dtw{0}'.format(twist)].mean,loosener*A_eff['Dtw{0}'.format(twist)].mean))
                    prior['log(dE:o{0})'.format(TwoPts['Dtw{0}'.format(twist)])][0] = gv.log(gv.gvar(M_eff['Dtw{0}'.format(twist)].mean+Lambda*a,oMloosener*(M_eff['Dtw{0}'.format(twist)].mean+Lambda*a)))
           
        if 'G' in FitCorrs:
            for mass in masses:
                # Goldstone
                prior['log({0}:a)'.format(TwoPts['Gm{0}'.format(mass)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:{0})'.format(TwoPts['Gm{0}'.format(mass)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,0.5*Lambda*a)]))
                prior['log({0}:a)'.format(TwoPts['Gm{0}'.format(mass)])][0] = gv.log(gv.gvar(A_eff['Gm{0}'.format(mass)].mean,loosener*A_eff['Gm{0}'.format(mass)].mean))
                prior['log(dE:{0})'.format(TwoPts['Gm{0}'.format(mass)])][0] = gv.log(gv.gvar(M_eff['Gm{0}'.format(mass)].mean,Mloosener*M_eff['Gm{0}'.format(mass)].mean))
        

                # Goldstone -- oscillating part
                prior['log(o{0}:a)'.format(TwoPts['Gm{0}'.format(mass)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:o{0})'.format(TwoPts['Gm{0}'.format(mass)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,0.5*Lambda*a)]))
                #prior['log(o{0}:a)'.format(TwoPts['Gm{0}'.format(mass)])][0] = gv.log(gv.gvar(A_eff['Gm{0}'.format(mass)].mean,loosener*A_eff['Gm{0}'.format(mass)].mean))
                prior['log(dE:o{0})'.format(TwoPts['Gm{0}'.format(mass)])][0] = gv.log(gv.gvar(M_eff['Gm{0}'.format(mass)].mean+Lambda*a,oMloosener*(M_eff['Gm{0}'.format(mass)].mean+Lambda*a)))
        if 'NG' in FitCorrs:
            for mass in masses:
                # Non-Goldstone
                prior['log({0}:a)'.format(TwoPts['NGm{0}'.format(mass)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:{0})'.format(TwoPts['NGm{0}'.format(mass)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,0.5*Lambda*a)]))
                #prior['log({0}:a)'.format(TwoPts['NGm{0}'.format(mass)])][0] = gv.log(gv.gvar(A_eff['NGm{0}'.format(mass)].mean,loosener*A_eff['NGm{0}'.format(mass)].mean))
                prior['log(dE:{0})'.format(TwoPts['NGm{0}'.format(mass)])][0] = gv.log(gv.gvar(M_eff['NGm{0}'.format(mass)].mean,Mloosener*M_eff['NGm{0}'.format(mass)].mean))
        

                # Non-Goldstone -- oscillating part
                prior['log(o{0}:a)'.format(TwoPts['NGm{0}'.format(mass)])] = gv.log(gv.gvar(N * [an]))
                prior['log(dE:o{0})'.format(TwoPts['NGm{0}'.format(mass)])] = gv.log(gv.gvar(N * ['{0}({1})'.format(Lambda*a,0.5*Lambda*a)]))
                #prior['log(o{0}:a)'.format(TwoPts['NGm{0}'.format(mass)])][0] = gv.log(gv.gvar(A_eff['NGm{0}'.format(mass)].mean,loosener*A_eff['NGm{0}'.format(mass)].mean))
                prior['log(dE:o{0})'.format(TwoPts['NGm{0}'.format(mass)])][0] = gv.log(gv.gvar(M_eff['NGm{0}'.format(mass)].mean+Lambda*a,oMloosener*(M_eff['NGm{0}'.format(mass)].mean+Lambda*a)))
    if 'S' in FitCorrs:
        for mass in masses: 
            for twist in twists:
                prior['SVnn_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [Vn]])
                prior['SVnn_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(V_eff['Sm{0}_tw{1}'.format(mass,twist)].mean,loosener*V_eff['Sm{0}_tw{1}'.format(mass,twist)].mean)
                prior['SVno_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [Vn]])
                prior['SVno_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(V0)
                if twist != '0':                    
                    prior['SVon_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [Vn]])
                    prior['SVon_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(V0)
                    prior['SVoo_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [Vn]])
                    prior['SVoo_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(V0)
    if 'V' in FitCorrs:
        for mass in masses:
            for twist in twists:
                prior['VVnn_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [Vn]])
                prior['VVnn_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(V_eff['Vm{0}_tw{1}'.format(mass,twist)].mean,loosener*V_eff['Sm{0}_tw{1}'.format(mass,twist)].mean)
                prior['VVno_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [Vn]])
                prior['VVno_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(V0)
                if twist !=0:                    
                    prior['VVon_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [Vn]])
                    prior['VVon_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(V0)
                    prior['VVoo_m{0}_tw{1}'.format(mass,twist)] = gv.gvar(N * [N * [Vn]])
                    prior['VVoo_m{0}_tw{1}'.format(mass,twist)][0][0] = gv.gvar(V0)
    return(prior)



def make_models():
    print('Masses:',masses,'Twists:',twists, 'Ts:',Ts,'Corrs:',FitCorrs)
    """ Create models to fit data. """
    tminG = Fit['tminG']
    tminNG = Fit['tminNG']
    tminD = Fit['tminD']
    Stmin = Fit['Stmin']
    Vtmin = Fit['Vtmin']
    tp = Fit['tp']
    twopts  = []
    Sthreepts = []
    Vthreepts = []
    if 'G' in FitCorrs:
        for mass in masses:        
            GCorrelator = copy.deepcopy(TwoPts['Gm{0}'.format(mass)])
            twopts.append(cf.Corr2(datatag=GCorrelator, tp=tp, tmin=tminG, a=('{0}:a'.format(GCorrelator), 'o{0}:a'.format(GCorrelator)), b=('{0}:a'.format(GCorrelator), 'o{0}:a'.format(GCorrelator)), dE=('dE:{0}'.format(GCorrelator), 'dE:o{0}'.format(GCorrelator)),s=(1.,-1.)))
    if 'NG' in FitCorrs:
        for mass in masses:
            NGCorrelator = copy.deepcopy(TwoPts['NGm{0}'.format(mass)])
            twopts.append(cf.Corr2(datatag=NGCorrelator, tp=tp, tmin=tminNG, a=('{0}:a'.format(NGCorrelator), 'o{0}:a'.format(NGCorrelator)), b=('{0}:a'.format(NGCorrelator), 'o{0}:a'.format(NGCorrelator)), dE=('dE:{0}'.format(NGCorrelator), 'dE:o{0}'.format(NGCorrelator)),s=(1.,-1.)))
    if 'D' in FitCorrs:
        for twist in twists:
            DCorrelator = copy.deepcopy(TwoPts['Dtw{0}'.format(twist)])
            if twist != '0':                
                twopts.append(cf.Corr2(datatag=DCorrelator, tp=tp, tmin=tminD, a=('{0}:a'.format(DCorrelator), 'o{0}:a'.format(DCorrelator)), b=('{0}:a'.format(DCorrelator), 'o{0}:a'.format(DCorrelator)), dE=('dE:{0}'.format(DCorrelator), 'dE:o{0}'.format(DCorrelator)),s=(1.,-1.)))
            else:
                twopts.append(cf.Corr2(datatag=DCorrelator, tp=tp, tmin=tminD, a=('{0}:a'.format(DCorrelator)), b=('{0}:a'.format(DCorrelator)), dE=('dE:{0}'.format(DCorrelator))))
                
    if 'S' in FitCorrs:
        for mass in masses:
            GCorrelator = copy.deepcopy(TwoPts['Gm{0}'.format(mass)])
            NGCorrelator = copy.deepcopy(TwoPts['NGm{0}'.format(mass)])
            for twist in twists:
                DCorrelator = copy.deepcopy(TwoPts['Dtw{0}'.format(twist)])
                for T in Ts:
                    if twist != '0':
                        Sthreepts.append(cf.Corr3(datatag=ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)], T=T, tmin=Stmin,  a=('{0}:a'.format(DCorrelator), 'o{0}:a'.format(DCorrelator)), dEa=('dE:{0}'.format(DCorrelator), 'dE:o{0}'.format(DCorrelator)), sa=(1,-1), b=('{0}:a'.format(GCorrelator), 'o{0}:a'.format(GCorrelator)), dEb=('dE:{0}'.format(GCorrelator), 'dE:o{0}'.format(GCorrelator)), sb=(1,-1), Vnn='SVnn_m'+str(mass)+'_tw'+str(twist), Vno='SVno_m'+str(mass)+'_tw'+str(twist), Von='SVon_m'+str(mass)+'_tw'+str(twist), Voo='SVoo_m'+str(mass)+'_tw'+str(twist)))
                    else:
                        Sthreepts.append(cf.Corr3(datatag=ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)], T=T, tmin=Stmin,  a=('{0}:a'.format(DCorrelator)), dEa=('dE:{0}'.format(DCorrelator)), b=('{0}:a'.format(GCorrelator), 'o{0}:a'.format(GCorrelator)), dEb=('dE:{0}'.format(GCorrelator), 'dE:o{0}'.format(GCorrelator)), sb=(1,-1), Vnn='SVnn_m'+str(mass)+'_tw'+str(twist), Vno='SVno_m'+str(mass)+'_tw'+str(twist)))
                        
    if 'V' in FitCorrs:
        for mass in masses:
            GCorrelator = copy.deepcopy(TwoPts['Gm{0}'.format(mass)])
            NGCorrelator = copy.deepcopy(TwoPts['NGm{0}'.format(mass)])
            for twist in twists:
                DCorrelator = copy.deepcopy(TwoPts['Dtw{0}'.format(twist)])
                for T in Ts:
                    if twist != '0':
                        Vthreepts.append(cf.Corr3(datatag=ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)], T=T, tmin=Vtmin, a=('{0}:a'.format(DCorrelator), 'o{0}:a'.format(DCorrelator)), dEa=('dE:{0}'.format(DCorrelator), 'dE:o{0}'.format(DCorrelator)), sa=(1,-1), b=('{0}:a'.format(NGCorrelator), 'o{0}:a'.format(NGCorrelator)), dEb=('dE:{0}'.format(NGCorrelator), 'dE:o{0}'.format(NGCorrelator)), sb=(1,-1), Vnn='VVnn_m'+str(mass)+'_tw'+str(twist), Vno='VVno_m'+str(mass)+'_tw'+str(twist), Von='VVon_m'+str(mass)+'_tw'+str(twist), Voo='VVoo_m'+str(mass)+'_tw'+str(twist)))
                    else:
                        Vthreepts.append(cf.Corr3(datatag=ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)], T=T, tmin=Vtmin, a=('{0}:a'.format(DCorrelator)), dEa=('dE:{0}'.format(DCorrelator)), b=('{0}:a'.format(NGCorrelator), 'o{0}:a'.format(NGCorrelator)), dEb=('dE:{0}'.format(NGCorrelator), 'dE:o{0}'.format(NGCorrelator)), sb=(1,-1), Vnn='VVnn_m'+str(mass)+'_tw'+str(twist), Vno='VVno_m'+str(mass)+'_tw'+str(twist)))
                        
    if Chained == True:            
        twopts = tuple(twopts)
        Sthreepts =tuple(Sthreepts)
        Vthreepts =tuple(Vthreepts)
        return(twopts,Sthreepts,Vthreepts)
    else:
        twopts.extend(Sthreepts)
        twopts.extend(Vthreepts)
        return(twopts)

    
    
def modelsandsvd(N):
    if Chained == True:     
        twopts,Sthreepts,Vthreepts = make_models()
        #models = [twopts, dict(nterm=(N,N)), Sthreepts, Vthreepts]
        models = []
        if 'S' in FitCorrs or 'V' in FitCorrs: 
            models.append(twopts)
            models.append(dict(nterm=(N,N)))
        else:
            models.extend(twopts)
        if 'S' in FitCorrs:
            models.append(Sthreepts)
        if 'V' in FitCorrs:
            models.append(Vthreepts)
    else:
        models = make_models()   
    print('Models made: ', models)
    File = 'Ps/{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}.pickle'.format(Fit['conf'],FitMasses,FitTwists,FitTs,FitCorrs,Fit['Stmin'],Fit['Vtmin'],Fit['tminG'],Fit['tminNG'],Fit['tminD'],Chained)
    if AutoSvd == True:        
        if os.path.isfile(File) == True:
            pickle_off = open(File,"rb")
            trueSvd = pickle.load(pickle_off)
            svdcut = SvdFactor*trueSvd
            print('Used existing svdcut {0} times factor {1}:'.format(trueSvd,SvdFactor), svdcut)
        else:
            print('Calculating svd')
            s = gv.dataset.svd_diagnosis(cf.read_dataset(filename), models=models)
            s.plot_ratio(show=True)
            var = input("Hit enter to accept svd = {0}, or else type svd here:".format(s.svdcut))
            if var == '':
                trueSvd = s.svdcut
                svdcut = trueSvd*SvdFactor
                print('Used calculated svdcut {0}, times factor {1}:'.format(trueSvd,SvdFactor),svdcut)
            else:
                trueSvd = float(var)
                svdcut = SvdFactor*trueSvd
                print('Used alternative svdcut {0}, times factor {1}:'.format(float(var),SvdFactor), svdcut)                
            pickling_on = open(File, "wb")
            pickle.dump(trueSvd,pickling_on)
            pickling_on.close()
    else:
        if os.path.isfile(File) == True:
            pickle_off = open(File,"rb")
            previous = pickle.load(pickle_off)
            var = input('Hit enter to use previously chosen svd {0}, times factor {1} or type new one:'.format(previous,SvdFactor))
            if var == '':
                trueSvd =  previous
                svdcut = SvdFactor*trueSvd
            else:
                trueSvd = float(var)
                svdcut = SvdFactor*trueSvd               
        else:
            var = input('Type new svd:')
            trueSvd = float(var)
            svdcut = SvdFactor*trueSvd
        print('Using svdcut {0}, times factor {1}:'.format(trueSvd,SvdFactor),svdcut)
        pickling_on = open(File, "wb")
        pickle.dump(trueSvd,pickling_on)
        pickling_on.close()
    return(models,svdcut)

        
def main(Autoprior,data): 
    M_eff,A_eff,V_eff = eff_calc()
    TwoKeys,ThreeKeys = makeKeys()
    p0 = collections.OrderedDict()
######################### CHAINED ###################################
    if Chained == True:
        N=1
        GBF1 = -1e21
        GBF2 = -1e20
        while N <= Nmax-2:
            if GBF2-GBF1 > 0.5:
                models,svdcut = modelsandsvd(N)
                Fitter = cf.CorrFitter(models=models, svdcut=svdcut, fitter='gsl_multifit', alg='subspace2D', solver='cholesky', maxit=5000, fast=False)
                fname = 'Ps/{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}'.format(Fit['conf'],FitMasses,FitTwists,FitTs,FitCorrs,Fit['Stmin'],Fit['Vtmin'],Fit['tminG'],Fit['tminNG'],Fit['tminD'],Chained,Nmax)
                if os.path.isfile('{0}{1}.pickle'.format(fname,N)) == True:
                    pickle_off = open('{0}{1}.pickle'.format(fname,N),"rb")
                    p0 = pickle.load(pickle_off)
                    print('Used existing p0 for N')
                elif os.path.isfile('{0}{1}.pickle'.format(fname,N+1)) == True:
                    pickle_off = open('{0}{1}.pickle'.format(fname,N+1),"rb")
                    p0 = pickle.load(pickle_off)
                    print('Used existing p0 for N+1')                
                GBF1 = GBF2
                print('Making Prior')              
                prior = make_prior(Nmax,M_eff,A_eff,V_eff,Autoprior)        
                print(30 * '=', 'Chained-Marginalised', 'nterm =','({0},{0})'.format(N), 'Nmax =',Nmax)
                fit = Fitter.chained_lsqfit(data=data, prior=prior,  p0=p0, add_svdnoise=svdnoise, add_priornoise=priornoise)
                GBF2 = fit.logGBF
                if GBF2-GBF1 > 0:
                    if ResultPlots == 'Q':
                        plots(fit.Q,fit.p,N)
                    if ResultPlots == 'GBF':
                        plots(GBF2,fit.p,N)
                    if ResultPlots == 'N':
                        plots(N,fit.p,fit.Q)
                    print(fit.formatall(pstyle='v'))
                    print('N = ({0},{0})'.format(N))
                    print('Q = {0:.2f}'.format(fit.Q))
                    print('log(GBF) = {0:.1f}, up {1:.1f}'.format(GBF2,GBF2-GBF1))   
                    print('chi2/dof = {0:.2f}'.format(fit.chi2/fit.dof))
                    print('SVD noise = {0} Prior noise = {1}'.format(svdnoise,priornoise))        
                    if fit.Q >= 0.05:
                        if SaveFit == True:
                            gv.dump(fit.p,'Fits/{6}5_Q{5:.2f}_Nmax{0}_Stmin{1}_Vtmin{2}_svd{3:.5f}_nterm{4}_chi{7:.3f}'.format(Nmax,Fit['Stmin'],Fit['Vtmin'],svdcut,N,fit.Q,Fit['conf'],fit.chi2/fit.dof))
                            f = open('Fits/{6}5_Q{5:.2f}_Nmax{0}_Stmin{1}_Vtmin{2}_svd{3:.5f}_nterm{4}_chi{7:.3f}.txt'.format(Nmax,Fit['Stmin'],Fit['Vtmin'],svdcut,N,fit.Q,Fit['conf'],fit.chi2/fit.dof), 'w')
                            f.write(fit.format(pstyle='v'))
                            f.close()

                else:
                    print('log(GBF) had gone down by {2:.2f} from {0:.2f} to {1:.2f}'.format(GBF1,GBF2,GBF1-GBF2))            
                if fit.Q >= 0.01:
                    pickling_on = open('{0}{1}.pickle'.format(fname,N), "wb")
                    pickle.dump(fit.pmean,pickling_on)
                    pickling_on.close()
                    p0=fit.pmean   
                if GBF2-GBF1 < 0:
                    return()
            print(100 * '+')
            print(100 * '+')
            N += 1 
########################## Unchained ######################################                
    else:
        #print('Initial p0', p0)        
        Nexp = 3
        GBF1 = -1e21
        GBF2 = -1e20
        models,svdcut = modelsandsvd('somenumber')
        Fitter = cf.CorrFitter(models=models, svdcut=svdcut, fitter='gsl_multifit', alg='subspace2D', solver='cholesky', maxit=5000, fast=False, tol=(1e-6,0.0,0.0))
        cond = (lambda: Nexp <= 8) if FitAll else (lambda: GBF2 - GBF1 > 0.01)
        while cond():           
            fname = 'Ps/{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}'.format(Fit['conf'],FitMasses,FitTwists,FitTs,FitCorrs,Fit['Stmin'],Fit['Vtmin'],Fit['tminG'],Fit['tminNG'],Fit['tminD'],Chained)            
            p0 = load_p0(p0,Nexp,fname,TwoKeys,ThreeKeys)                    
            GBF1 = GBF2
            print('Making Prior')
            if CorrBayes == True:
                Autoprior,data = make_data(filename,Nexp)       
            prior = make_prior(Nexp,M_eff,A_eff,V_eff,Autoprior)
            print(30 * '=','Unchained-Unmarginalised','Nexp =',Nexp)            
            fit = Fitter.lsqfit(data=data, prior=prior,  p0=p0, add_svdnoise=svdnoise, add_priornoise=priornoise)            
            GBF2 = fit.logGBF
            if fit.Q>=0.05:
                pickling_on = open('{0}{1}.pickle'.format(fname,Nexp), "wb")
                pickle.dump(fit.pmean,pickling_on)
                pickling_on.close()
            if cond():
                if FitAll == False:
                    if ResultPlots == 'Q':
                        plots(fit.Q,fit.p,Nexp)
                    if ResultPlots == 'GBF':
                        plots(GBF2,fit.p,Nexp)
                    if ResultPlots == 'N':
                        plots(Nexp,fit.p,fit.Q)
                print(fit)
                #print(fit.format(pstyle=None if Nexp<3 else'v'))
                print('Nexp = ',Nexp)
                print('Q = {0:.2f}'.format(fit.Q))
                print('log(GBF) = {0:.1f}, up {1:.1f}'.format(GBF2,GBF2-GBF1))       
                print('chi2/dof = {0:.2f}'.format(fit.chi2/fit.dof))
                print('SVD noise = {0} Prior noise = {1}'.format(svdnoise,priornoise))
                if fit.Q >= 0.05:
                    p0=fit.pmean
                    save_p0(p0,Nexp,fname,TwoKeys,ThreeKeys)
                    if SaveFit == True:
                        gv.dump(fit.p,'Fits/{5}5_Q{4:.2f}_Nexp{0}_Stmin{1}_Vtmin{2}_svd{3:.5f}_chi{6:.3f}'.format(Nexp,Fit['Stmin'],Fit['Vtmin'],svdcut,fit.Q,Fit['conf'],fit.chi2/fit.dof))
                        f = open('Fits/{5}5_Q{4:.2f}_Nexp{0}_Stmin{1}_Vtmin{2}_svd{3:.5f}_chi{6:.3f}.txt'.format(Nexp,Fit['Stmin'],Fit['Vtmin'],svdcut,fit.Q,Fit['conf'],fit.chi2/fit.dof), 'w')
                        f.write(fit.format(pstyle=None if Nexp<3 else'v'))
                        f.close()
            else:
                print('log(GBF) had gone down by {2:.2f} from {0:.2f} to {1:.2f}'.format(GBF1,GBF2,GBF1-GBF2))                
            print(100 * '+')
            print(100 * '+')
            Nexp += 1           
    print_results(fit.p)
    return()






def plots(Q,p,Nexp):
    if ResultPlots == 'Q':
        xlab = 'Q'
        lab = 'N'       
    if ResultPlots == 'GBF':
        xlab = 'Log(GBF)'
        lab = 'N'
    if ResultPlots == 'N':
        xlab = 'N'
        lab = 'Q'
    for twist in twists:
        if 'D' in FitCorrs:
            result = p['dE:{0}'.format(TwoPts['Dtw{0}'.format(twist)])][0]
            y = result.mean
            err = result.sdev    
            plt.figure(twist)
            plt.errorbar(Q,y,yerr=err, capsize=2, fmt='o', mfc='none', label=('{0} = {1:.2f}'.format(lab,Nexp)))
            plt.legend()
            plt.xlabel('{0}'.format(xlab))
            plt.ylabel('dE:{0}'.format(TwoPts['Dtw{0}'.format(twist)]))
    for mass in masses:
        if 'G' in FitCorrs:
            result = p['dE:{0}'.format(TwoPts['Gm{0}'.format(mass)])][0]
            y = result.mean
            err = result.sdev    
            plt.figure(mass)
            plt.errorbar(Q,y,yerr=err, capsize=2, fmt='o', mfc='none', label=('G {0} = {1:.2f}'.format(lab,Nexp)))
            plt.legend()
            plt.xlabel('{0}'.format(xlab))
            plt.ylabel('dE:{0}'.format(mass))
        if 'NG' in FitCorrs:
            result = p['dE:{0}'.format(TwoPts['NGm{0}'.format(mass)])][0]
            y = result.mean
            err = result.sdev    
            plt.figure(mass)
            plt.errorbar(Q,y,yerr=err, capsize=2, fmt='o', mfc='none', label=('NG {0} = {1:.2f}'.format(lab,Nexp)))
            plt.legend()
            plt.xlabel('{0}'.format(xlab))
            plt.ylabel('dE:{0}'.format(mass))
            for twist in twists:
                if 'S' in FitCorrs:
                    result = p['SVnn_m{0}_tw{1}'.format(mass,twist)][0][0]
                    y = result.mean
                    err = result.sdev    
                    plt.figure('SVnn_m{0}_tw{1}'.format(mass,twist))
                    plt.errorbar(Q,y,yerr=err, capsize=2, fmt='o', mfc='none', label=('{0} = {1:.2f}'.format(lab,Nexp)))
                    plt.legend()
                    plt.xlabel('{0}'.format(xlab))
                    plt.ylabel('SVnn_m{0}_tw{1}'.format(mass,twist))
                if 'V' in FitCorrs:
                    result = p['VVnn_m{0}_tw{1}'.format(mass,twist)][0][0]
                    y = result.mean
                    err = result.sdev    
                    plt.figure('VVnn_m{0}_tw{1}'.format(mass,twist))
                    plt.errorbar(Q,y,yerr=err, capsize=2, fmt='o', mfc='none', label=('{0} = {1:.2f}'.format(lab,Nexp)))
                    plt.legend()
                    plt.xlabel('{0}'.format(xlab))
                    plt.ylabel('VVnn_m{0}_tw{1}'.format(mass,twist))
    return()


def test_data():
    if 'D' in FitCorrs:
        for twist in twists:
            plt.figure(twist)
            for t in range(Fit['tp']):                
                plt.errorbar(t, gv.log(data[TwoPts['Dtw{0}'.format(twist)]][t]).mean, yerr=gv.log(data[TwoPts['Dtw{0}'.format(twist)]][t]).sdev, fmt='ko')
            plt.title('D Twist = {0}'.format(twist))
        #plt.legend()
        
    if 'G' in FitCorrs:                
        for mass in masses:
            plt.figure(mass)
            for t in range(Fit['tp']):                
                plt.errorbar(t, gv.log(data[TwoPts['Gm{0}'.format(mass)]][t]).mean, yerr=gv.log(data[TwoPts['Gm{0}'.format(mass)]][t]).sdev, fmt='ko')
            plt.title('G NG Mass = {0}'.format(mass))
        #plt.legend()
    if 'NG' in FitCorrs:                
        for mass in masses:
            plt.figure(mass)
            for t in range(Fit['tp']):                
                plt.errorbar(t, gv.log(data[TwoPts['NGm{0}'.format(mass)]][t]).mean, yerr=gv.log(data[TwoPts['NGm{0}'.format(mass)]][t]).sdev, fmt='ro')
            plt.title('G NG Mass = {0}'.format(mass))
        #plt.legend()

    colours = ['r','k','b']
    if 'S' in FitCorrs:        
        for mass in masses:
            for twist in twists:                    
                plt.figure('S{0}{1}'.format(mass,twist))
                for i,T in enumerate(Ts):
                    for t in range(T):
                        plt.errorbar(t, gv.log(data[ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]).mean, yerr=gv.log(data[ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]).sdev,  fmt='{0}o'.format(colours[i]))
                plt.title('S Mass = {0}, Twist = {1}'.format(mass, twist))
                #plt.legend()

                plt.figure('SRat{0}{1}'.format(mass,twist))
                for i,T in enumerate(Ts):
                    for t in range(T):
                        plt.errorbar(t, (data[ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['Dtw{0}'.format(twist)]][t]*data[TwoPts['Gm{0}'.format(mass)]][T-t])).mean, yerr=(data[ThreePts['Sm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['Dtw{0}'.format(twist)]][t]*data[TwoPts['Gm{0}'.format(mass)]][T-t])).sdev, fmt='{0}o'.format(colours[i]))
                plt.title('S Ratio Mass = {0}, Twist = {1}'.format(mass, twist))
                #plt.legend()
        
    if 'V' in FitCorrs:
        for mass in masses:
            for twist in twists:                    
                plt.figure('V{0}{1}'.format(mass,twist))
                for i,T in enumerate(Ts):
                    for t in range(T):
                        plt.errorbar(t, gv.log(data[ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]).mean, yerr=gv.log(data[ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]).sdev, fmt='{0}o'.format(colours[i]))
                plt.title('V Mass = {0}, Twist = {1}'.format(mass, twist))
                #plt.legend()

                plt.figure('VRat{0}{1}'.format(mass,twist))
                for i,T in enumerate(Ts):
                    for t in range(T):
                        plt.errorbar(t, (data[ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['Dtw{0}'.format(twist)]][t]*data[TwoPts['NGm{0}'.format(mass)]][T-t])).mean, yerr=(data[ThreePts['Vm{0}_tw{1}_T{2}'.format(mass,twist,T)]][t]/(data[TwoPts['Dtw{0}'.format(twist)]][t]*data[TwoPts['NGm{0}'.format(mass)]][T-t])).sdev, fmt='{0}o'.format(colours[i]))
                plt.title('V Ratio Mass = {0}, Twist = {1}'.format(mass, twist))
                #plt.legend()
    plt.show()
    return()


def makeKeys():
    TwoKeys = []
    ThreeKeys = []
    if 'D' in FitCorrs:
        for twist in twists:
            TwoKeys.append('log({0}:a)'.format(TwoPts['Dtw{0}'.format(twist)]))
            TwoKeys.append('log(dE:{0})'.format(TwoPts['Dtw{0}'.format(twist)]))
            if twist != '0':            
                TwoKeys.append('log(dE:o{0})'.format(TwoPts['Dtw{0}'.format(twist)]))
                TwoKeys.append('log(o{0}:a)'.format(TwoPts['Dtw{0}'.format(twist)]))
    if 'G' in FitCorrs:
        for mass in masses:
            TwoKeys.append('log({0}:a)'.format(TwoPts['Gm{0}'.format(mass)]))
            TwoKeys.append('log(o{0}:a)'.format(TwoPts['Gm{0}'.format(mass)]))
            TwoKeys.append('log(dE:{0})'.format(TwoPts['Gm{0}'.format(mass)]))
            TwoKeys.append('log(dE:o{0})'.format(TwoPts['Gm{0}'.format(mass)]))
    if 'NG' in FitCorrs:
        for mass in masses:
            TwoKeys.append('log({0}:a)'.format(TwoPts['NGm{0}'.format(mass)]))
            TwoKeys.append('log(o{0}:a)'.format(TwoPts['NGm{0}'.format(mass)]))
            TwoKeys.append('log(dE:{0})'.format(TwoPts['NGm{0}'.format(mass)]))
            TwoKeys.append('log(dE:o{0})'.format(TwoPts['NGm{0}'.format(mass)]))
    if 'S' in FitCorrs:
        for mass in masses:
            for twist in twists:
                ThreeKeys.append('SVnn_m{0}_tw{1}'.format(mass,twist))
                ThreeKeys.append('SVno_m{0}_tw{1}'.format(mass,twist))
                if twist != '0':
                    ThreeKeys.append('SVon_m{0}_tw{1}'.format(mass,twist))
                    ThreeKeys.append('SVoo_m{0}_tw{1}'.format(mass,twist))
    if 'V' in FitCorrs:
        for mass in masses:
            for twist in twists:
                ThreeKeys.append('VVnn_m{0}_tw{1}'.format(mass,twist))
                ThreeKeys.append('VVon_m{0}_tw{1}'.format(mass,twist))
                if twist != '0':
                    ThreeKeys.append('VVno_m{0}_tw{1}'.format(mass,twist))
                    ThreeKeys.append('VVoo_m{0}_tw{1}'.format(mass,twist))
    #print(TwoKeys,ThreeKeys)
    return(TwoKeys,ThreeKeys)



def save_p0(p0,Nexp,fname,TwoKeys,ThreeKeys):                   
    if os.path.exists('Ps/{0}.pickle'.format(Fit['conf'])):
        rglobalpickle = open('Ps/{0}.pickle'.format(Fit['conf']), "rb")
        p1 = pickle.load(rglobalpickle)                    
        for key in TwoKeys:                        
            if key in p1.keys():
                if len(p0[key]) >= len(p1[key]):      
                    p1.pop(key,None)
                    p1[key]=p0[key]
                    print('Replaced element of global p0:', key)
            else:
                p1[key]=p0[key]
                print('Added new element to global p0:',key)
        for key in ThreeKeys:
            if key in p1.keys():
                if np.shape(p0[key])[0] >= np.shape(p1[key])[0]:      
                    p1.pop(key,None)
                    p1[key]=p0[key]
                    print('Replaced element of global p0:', key)
            else:
                p1[key]=p0[key]
                print('Added new element to global p0:',key)
        wglobalpickle = open('Ps/{0}.pickle'.format(Fit['conf']), "wb")
        pickle.dump(p1,wglobalpickle)
        wglobalpickle.close()
    else:
        p2 = collections.OrderedDict()
        for key in TwoKeys:                        
            p2[key] = copy.deepcopy(p0[key])
        for key in ThreeKeys:                        
            p2[key] = copy.deepcopy(p0[key])

        wglobalpickle = open('Ps/{0}.pickle'.format(Fit['conf']), "wb")
        pickle.dump(p2,wglobalpickle)
        wglobalpickle.close()
#################################### p0 Nexp ###########################             
    if os.path.exists('Ps/{0}{1}.pickle'.format(Fit['conf'],Nexp)):
        rglobalpickle = open('Ps/{0}{1}.pickle'.format(Fit['conf'],Nexp), "rb")
        p1 = pickle.load(rglobalpickle)                    
        for key in TwoKeys:                        
            if key in p1.keys():                      
                p1.pop(key,None)
                p1[key]=p0[key]
                #print('Replaced element of global p0 Nexp={0}:'.format(Nexp), key)
            else:
                p1[key]=p0[key]
                print('Added new element to global p0 Nexp={0}:'.format(Nexp),key)
        for key in ThreeKeys:
            if key in p1.keys():                      
                p1.pop(key,None)
                p1[key]=p0[key]
                #print('Replaced element of global p0 Nexp={0}:'.format(Nexp), key)
            else:
                p1[key]=p0[key]
                print('Added new element to global p0 Nexp={0}:'.format(Nexp),key)
        wglobalpickle = open('Ps/{0}{1}.pickle'.format(Fit['conf'],Nexp), "wb")
        pickle.dump(p1,wglobalpickle)
        wglobalpickle.close()
    else:
        p2 = collections.OrderedDict()
        for key in TwoKeys:                        
            p2[key] = copy.deepcopy(p0[key])
        for key in ThreeKeys:                        
            p2[key] = copy.deepcopy(p0[key])
        wglobalpickle = open('Ps/{0}{1}.pickle'.format(Fit['conf'],Nexp), "wb")
        pickle.dump(p2,wglobalpickle)
        wglobalpickle.close()
        
    return()




def load_p0(p0,Nexp,fname,TwoKeys,ThreeKeys):
    elements1 = []
    elements2 = []
    if os.path.isfile('{0}{1}.pickle'.format(fname,Nexp)):
        pickle_off = open('{0}{1}.pickle'.format(fname,Nexp),"rb")
        p0 = pickle.load(pickle_off)
        print('Using existing p0 for Nexp')                
    elif os.path.isfile('{0}{1}.pickle'.format(fname,Nexp+1)):
        pickle_off = open('{0}{1}.pickle'.format(fname,Nexp+1),"rb")
        p1 = pickle.load(pickle_off)
        for key in TwoKeys:
            if key in p1.keys():
                #p1[key].pop(Nexp)        
                p0[key]=p1[key]
        print('Using existing p0 for Nexp+1')
    
    elif os.path.exists('Ps/{0}.pickle'.format(Fit['conf'])):
        pickle_off = open('Ps/{0}.pickle'.format(Fit['conf']),"rb")
        p1 = pickle.load(pickle_off)
        for key in TwoKeys:
            if key in p1.keys():                    
                if len(p1[key]) >= Nexp:
                    elements1.append(key)
                    p0.pop(key,None)                            
                    p0[key] = np.zeros((Nexp))                            
                    for n in range(Nexp):
                        p0[key][n] = p1[key][n]                            
        for key in ThreeKeys:
            if key in p1.keys():                    
                if np.shape(p1[key])[0] >= Nexp:
                    p0.pop(key,None)
                    p0[key]=np.zeros((Nexp,Nexp))
                    elements1.append(key)
                    for n in range(Nexp):
                        for m in range(Nexp):
                            p0[key][n][m]=p1[key][n][m]
        if os.path.exists('Ps/{0}{1}.pickle'.format(Fit['conf'],Nexp)):
            pickle_off = open('Ps/{0}{1}.pickle'.format(Fit['conf'],Nexp),"rb")
            p2 = pickle.load(pickle_off)
            for key in TwoKeys:
                if key in p2.keys():
                    p0.pop(key,None)
                    p0[key] = p2[key]                    
                    elements2.append(key)         
            for key in ThreeKeys:
                if key in p2.keys():
                   p0.pop(key,None)
                   p0[key]=p2[key]
                   elements2.append(key)
        for element in elements1:
            if element in elements2:
                print('Using element of global p0 Nexp={0}:'.format(Nexp),element)
            else:
                print('Using element of global p0:',element)
    return(p0)




def print_results(p):
    if 'D' in FitCorrs:
        for twist in twists:
            print('D tw {0} :'.format(twist),p['dE:{0}'.format(TwoPts['Dtw{0}'.format(twist)])][0])
    if 'G' in FitCorrs:
        for mass in masses:
            print('G m {0} :'.format(mass),p['dE:{0}'.format(TwoPts['Gm{0}'.format(mass)])][0])
    if 'NG' in FitCorrs:
        for mass in masses:
            print('NG m {0} :'.format(mass),p['dE:{0}'.format(TwoPts['NGm{0}'.format(mass)])][0])
    if 'S' in FitCorrs:
        for mass in masses:
            for twist in twists:
                print('SVnn m {0} tw {1} :'. format(mass, twist), p['SVnn_m{0}_tw{1}'.format(mass,twist)][0][0])
    if 'V' in FitCorrs:
        for mass in masses:
            for twist in twists:
                print('VVnn m {0} tw {1} :'. format(mass, twist), p['VVnn_m{0}_tw{1}'.format(mass,twist)][0][0])

    return()

if TestData == True:
    test_data()
    
if DoFit == True:
    filename = Fit['filename']
    Autoprior,data = make_data(filename,Nmax)
    gotnan = any(bool(a[~np.isfinite([p.mean for p in a])].size) for a in data.values())
    print('Nan or inf in data: ',gotnan)
    if FitAll == True:
        FitTs = [0]
        FitCorrs = ['G','NG','D','S','V']
        for i in range(4):
            FitMasses = [i]
            for j in range(5):                
                FitTwists = [j]
                TwoPts,ThreePts,masses,twists,Ts = make_params(FitMasses, FitTwists,FitTs)
                main(Autoprior,data)
                            
    else:        
        TwoPts,ThreePts,masses,twists,Ts = make_params(FitMasses,FitTwists,FitTs)        
        print('Nan or inf in data: ',gotnan)
        main(Autoprior,data)
        plt.show()
