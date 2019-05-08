import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import collections

################################## F PARAMETERS ##########################
F = collections.OrderedDict()
F['conf']='F'
F['filename'] = 'Fits/F5_Q1.00_Nexp4_Stmin2_Vtmin1_svd0.00411_chi0.529'
F['Masses'] = ['0.449','0.566','0.683','0.8']
F['Twists'] = ['0','0.4281','1.282','2.141','2.570','2.993']
F['m_s'] = '0.0376'
F['Ts'] = [14,17,20]
F['tp'] = 96
F['L'] = 32
F['a'] = '0.0884(6)'
F['goldTag'] = 'meson.m{0}_m{1}'
F['nonGoldTag'] = 'meson-G5T.m{0}_m{1}'
F['daugterTag'] = ['etas','etas_p0.0728','etas_p0.218','etas_p0.364','etas_p0.437','etas_p0.509'] 
F['threePtTag'] = ['{0}.T{1}_m{2}_m{3}_m{2}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}']

######################## SF PARAMETERS ####################################
SF = collections.OrderedDict()
SF['conf']='SF'
SF['filename'] = 'Fits/SF5_Q1.00_Nexp4_Stmin2_Vtmin2_svd0.00500'
SF['Masses'] = ['0.274','0.450','0.6','0.8']
SF['Twists'] = ['0','1.261','2.108','2.946','3.624']
SF['m_s'] = '0.0234'
SF['Ts'] = [20,25,30]
SF['tp'] = 144
SF['L'] = 48
SF['a'] = '0.05922(12)'
SF['goldTag'] = 'meson.m{0}_m{1}'
SF['nonGoldTag'] = 'meson2G5T.m{0}_m{1}'
SF['daugterTag'] = ['etas_p0','etas_p0.143','eta_s_tw2.108_m0.0234','etas_p0.334','eta_s_tw3.624_m0.0234']
SF['threePtTag'] = ['{0}.T{1}_m{2}_m{3}_m{2}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}']

##################### USER INPUTS ##########################################
############################################################################
Fits = [F,SF]                                         # Choose to fit F, SF or UF
FMasses = [0,1,2,3]                                     # Choose which masses to fit
FTwists = [0,1,2,3,4]
SFMasses = [0,1,2,3]
SFTwists = [0,1,2,3,4]
GeV = False
############################################################################
############################################################################
   


def make_params():
    for Fit in [F,SF]:
        Fit['masses'] = []
        Fit['twists'] = []
        Fit['momenta'] = []
    for i in FMasses:
        F['masses'].append(F['Masses'][i])
    for j in SFMasses:
        SF['masses'].append(SF['Masses'][j])
    for k in FTwists:
        F['twists'].append(F['Twists'][k])
    for l in SFTwists:
        SF['twists'].append(SF['Twists'][l])
    for Fit in Fits:
        for twist in Fit['twists']:
            Fit['momenta'].append(np.sqrt(3)*np.pi*float(twist)/Fit['L'])
        for Twist in Fit['Twists']:
            Fit['Dtw{0}'.format(Twist)] = Fit['daugterTag'][Fit['Twists'].index(Twist)]
        for mass in Fit['Masses']:
            Fit['Gm{0}'.format(mass)] = Fit['goldTag'].format(Fit['m_s'],mass)
            Fit['NGm{0}'.format(mass)] = Fit['nonGoldTag'].format(Fit['m_s'],mass)
        
    return()


def get_results(Fit):
    Vnn = collections.OrderedDict()
    p = gv.load(Fit['filename'])    
    for i in range(len(Fit['twists'])):
        Fit['M_D_tw{0}'.format(Fit['twists'][i])] =  gv.sqrt(p['dE:{0}'.format(Fit['Dtw{0}'.format(Fit['twists'][i])])][0]**2 - Fit['momenta'][i]**2)
        Fit['E_D_tw{0}'.format(Fit['twists'][i])] = p['dE:{0}'.format(Fit['Dtw{0}'.format(Fit['twists'][i])])][0]
    
    for mass in Fit['masses']:
        Fit['M_G_m{0}'.format(mass)] = p['dE:{0}'.format(Fit['Gm{0}'.format(mass)])][0]
        Fit['M_NG_m{0}'.format(mass)] = p['dE:{0}'.format(Fit['NGm{0}'.format(mass)])][0]
            
        for twist in Fit['twists']:
            Fit['Sm{0}_tw{1}'.format(mass,twist)] = 2*2*gv.sqrt(Fit['E_D_tw{0}'.format(twist)]*Fit['M_G_m{0}'.format(mass)])*p['SVnn_m{0}_tw{1}'.format(mass,twist)][0][0]
            Fit['Vm{0}_tw{1}'.format(mass,twist)] = 2*2*gv.sqrt(Fit['E_D_tw{0}'.format(twist)]*Fit['M_G_m{0}'.format(mass)])*p['VVnn_m{0}_tw{1}'.format(mass,twist)][0][0]
    return()



def plot_f(Fit,F_0,F_plus,qSq,Z,Sca,Vec):
    
    plt.figure(1)
    for k in range(len(Fit['masses'])):
        z = []
        zerr = []
        f = []
        ferr = []
        for i in range(len(qSq[Fit['masses'][k]])):
            z.append(Z[Fit['masses'][k]][i].mean)
            f.append(F_0[Fit['masses'][k]][i].mean)
            zerr.append(Z[Fit['masses'][k]][i].sdev)
            ferr.append(F_0[Fit['masses'][k]][i].sdev)        
        plt.errorbar(z,f,xerr=[zerr,zerr],yerr=[ferr,ferr], capsize=2, fmt='o', mfc='none', label=('{1} $m_h$ = {0}'.format(Fit['masses'][k],Fit['conf'])))
    plt.legend()
    plt.xlabel('$z$')
    plt.ylabel('$f_0$')


    plt.figure(2)
    for k in range(len(Fit['masses'])):
        q = []
        qerr = []
        f = []
        ferr = []
        for i in range(len(qSq[Fit['masses'][k]])):
            q.append(qSq[Fit['masses'][k]][i].mean)
            f.append(F_0[Fit['masses'][k]][i].mean)
            qerr.append(qSq[Fit['masses'][k]][i].sdev)
            ferr.append(F_0[Fit['masses'][k]][i].sdev)        
        plt.errorbar(q,f,xerr=[qerr,qerr],yerr=[ferr,ferr], capsize=2, fmt='o', mfc='none', label=('{1} $m_h$ = {0}'.format(Fit['masses'][k],Fit['conf'])))
    plt.legend()
    if GeV == True:
        plt.xlabel('$q^2 (GeV)^2$')
    else:
        plt.xlabel('$q^2$ (lattice units)')        
    plt.ylabel('$f_0$')


    plt.figure(3)           
    for k in range(len(Fit['masses'])):
        z = []
        zerr = []
        f = []
        ferr = []
        for i in range(1,len(qSq[Fit['masses'][k]])):            
                z.append(Z[Fit['masses'][k]][i].mean)
                f.append(F_plus[Fit['masses'][k]][i-1].mean)
                zerr.append(Z[Fit['masses'][k]][i].sdev)
                ferr.append(F_plus[Fit['masses'][k]][i-1].sdev)        
        plt.errorbar(z,f,xerr=[zerr,zerr],yerr=[ferr,ferr], capsize=2, fmt='o', mfc='none', label=('{1} $m_h$ = {0}'.format(Fit['masses'][k],Fit['conf'])))
    plt.legend()    
    plt.xlabel('$z$')
    plt.ylabel('$f_+$')
    #plt.xlim(right=1.1*qSq[masses[k]][0].mean)


    plt.figure(4)           
    for k in range(len(Fit['masses'])):
        q = []
        qerr = []
        f = []
        ferr = []
        for i in range(1,len(qSq[Fit['masses'][k]])):            
                q.append(qSq[Fit['masses'][k]][i].mean)
                f.append(F_plus[Fit['masses'][k]][i-1].mean)
                qerr.append(qSq[Fit['masses'][k]][i].sdev)
                ferr.append(F_plus[Fit['masses'][k]][i-1].sdev)        
        plt.errorbar(q,f,xerr=[qerr,qerr],yerr=[ferr,ferr], capsize=2, fmt='o', mfc='none', label=('{1} $m_h$ = {0}'.format(Fit['masses'][k],Fit['conf'])))
    plt.legend()
    if GeV == True:
        plt.xlabel('$q^2 (GeV)^2$')
    else:
        plt.xlabel('$q^2$ (lattice units)')
    plt.ylabel('$f_+$')
    
    return()





def main():    
    make_params()
    for Fit in Fits:
        hbar = '6.58211928(15)'
        c = 2.99792458
        a = gv.gvar(Fit['a'])/(gv.gvar(hbar)*c*1e-2)               
        print('Plot for', Fit['filename'] )
        get_results(Fit)
        F_0 = collections.OrderedDict()
        F_plus = collections.OrderedDict()        
        qSq = collections.OrderedDict()
        Z = collections.OrderedDict()
        Sca = collections.OrderedDict()
        Vec = collections.OrderedDict()
        for mass in Fit['masses']:
            F_0[mass] =[]
            F_plus[mass] =[]            
            qSq[mass] =[]
            Z[mass] =[]        
            Sca[mass] =[]
            Vec[mass] =[]
            Z_v = (float(mass) - float(Fit['m_s']))*Fit['Sm{0}_tw0'.format(mass)]/((Fit['M_G_m{0}'.format(mass)] - Fit['M_D_tw0'])*Fit['Vm{0}_tw0'.format(mass)]) 
            for twist in Fit['twists']:
                delta = (float(mass) - float(Fit['m_s']))*(Fit['M_G_m{0}'.format(mass)]-Fit['E_D_tw{0}'.format(twist)])
                qsq = (Fit['M_G_m{0}'.format(mass)]**2 + Fit['M_D_tw{0}'.format(twist)]**2 - 2*Fit['M_G_m{0}'.format(mass)]*Fit['E_D_tw{0}'.format(twist)])
                t = (Fit['M_G_m{0}'.format(mass)] + Fit['M_D_tw{0}'.format(twist)])**2
                z = (gv.sqrt(t-qsq)-gv.sqrt(t))/(gv.sqrt(t-qsq)+gv.sqrt(t))           
                if qsq.mean >= 0:
                    F0 = (float(mass) - float(Fit['m_s']))*(1/(Fit['M_G_m{0}'.format(mass)]**2 - Fit['M_D_tw{0}'.format(twist)]**2))*Fit['Sm{0}_tw{1}'.format(mass,twist)]               
                    F_0[mass].append(F0)
                    if GeV == True:
                        qSq[mass].append(qsq/(a**2))                      
                    else:
                        qSq[mass].append(qsq)                    
                    Z[mass].append(z)                        
                    Sca[mass].append(Fit['Sm{0}_tw{1}'.format(mass,twist)])
                    Vec[mass].append(Fit['Vm{0}_tw{1}'.format(mass,twist)])           
                    A = Fit['M_G_m{0}'.format(mass)] + Fit['E_D_tw{0}'.format(twist)]
                    B = (Fit['M_G_m{0}'.format(mass)]**2 - Fit['M_D_tw{0}'.format(twist)]**2)*(Fit['M_G_m{0}'.format(mass)] - Fit['E_D_tw{0}'.format(twist)])/qsq           
                    if twist != '0':
                        F_plus[mass].append((1/(A-B))*(Z_v*Fit['Vm{0}_tw{1}'.format(mass,twist)] - B*F0))       
        plot_f(Fit,F_0,F_plus,qSq,Z,Sca,Vec)
    plt.show()
    return()


main() 

    


#plot_f()


def speedtest():
    csq = collections.OrderedDict()
    for mass in masses:
        for i in range(len(twists)):
            csq['m_{0}tw_{1}'.format(mass,twists[i])] = E_eta[twists[i]]**2/(momenta[i]**2+E_eta['0']**2)
    print(csq)
    return(csq)
#speedtest()
