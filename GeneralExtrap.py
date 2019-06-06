import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

plt.rc("font",**{"size":18})
import collections
import copy
import os.path
import pickle
from collections import defaultdict
################################## F PARAMETERS ##########################
F = collections.OrderedDict()
F['conf']='F'
F['filename'] = 'Fits/F5_Q1.00_Nexp4_Stmin2_Vtmin1_svd0.00411_chi0.529'
F['Masses'] = ['0.449','0.566','0.683','0.8']
F['Twists'] = ['0','0.4281','1.282','2.141','2.570','2.993']
F['m_s'] = '0.0376'
F['m_ssea'] = 0.037
F['m_lsea'] = 0.0074
F['Ts'] = [14,17,20]
F['tp'] = 96
F['L'] = 32
F['w0/a'] = '1.9006(20)'
F['goldTag'] = 'meson.m{0}_m{1}'
F['nonGoldTag'] = 'meson-G5T.m{0}_m{1}'
F['daugterTag'] = ['etas','etas_p0.0728','etas_p0.218','etas_p0.364','etas_p0.437','etas_p0.509'] 
F['threePtTag'] = ['{0}.T{1}_m{2}_m{3}_m{2}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}','{0}.T{1}_m{2}_m{3}_m{2}_tw{4}']

######################## SF PARAMETERS ####################################
SF = collections.OrderedDict()
SF['conf']='SF'
SF['filename'] = 'Fits/SF5_Q1.00_Nexp5_Stmin2_Vtmin2_svd0.01000_chi0.199'
SF['Masses'] = ['0.274','0.450','0.6','0.8']
SF['Twists'] = ['0','1.261','2.108','2.946','3.624']
SF['m_s'] = '0.0234'
SF['m_ssea'] = 0.024
SF['m_lsea'] = 0.0048
SF['Ts'] = [20,25,30]
SF['tp'] = 144
SF['L'] = 48
SF['w0/a'] = '2.896(6)'
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
AddRho = False
svdnoise = False
priornoise = False
Pri = '0.00(2.00)'
DoFit = True
N = 3
############################################################################
############################################################################
   
def make_params():
    w0 = gv.gvar('0.1715(9)')  #fm
    hbar = gv.gvar('6.58211928(15)')
    c = 2.99792458
    for Fit in [F,SF]:
        Fit['a'] = w0/((hbar*c*1e-2)*gv.gvar(Fit['w0/a']))
        Fit['masses'] = []
        Fit['twists'] = []
        Fit['momenta'] = []
        Fit['Delta'] = 0
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
        Fit['M_Go_m{0}'.format(mass)] = p['dE:o{0}'.format(Fit['Gm{0}'.format(mass)])][0]
        Fit['M_NG_m{0}'.format(mass)] = p['dE:{0}'.format(Fit['NGm{0}'.format(mass)])][0]
        Fit['M_NGo_m{0}'.format(mass)] = p['dE:o{0}'.format(Fit['NGm{0}'.format(mass)])][0]
            
        for twist in Fit['twists']:
            Fit['Sm{0}_tw{1}'.format(mass,twist)] = 2*2*gv.sqrt(Fit['E_D_tw{0}'.format(twist)]*Fit['M_G_m{0}'.format(mass)])*p['SVnn_m{0}_tw{1}'.format(mass,twist)][0][0]
            Fit['Vm{0}_tw{1}'.format(mass,twist)] = 2*2*gv.sqrt(Fit['E_D_tw{0}'.format(twist)]*Fit['M_G_m{0}'.format(mass)])*p['VVnn_m{0}_tw{1}'.format(mass,twist)][0][0]
    return()


def make_fs(Fit):     
    make_params()    
    print('Calc for', Fit['filename'] )
    get_results(Fit)
    F_0 = collections.OrderedDict()
    F_plus = collections.OrderedDict()        
    qSq = collections.OrderedDict()
    Z = collections.OrderedDict()
    Sca = collections.OrderedDict()
    Vec = collections.OrderedDict()
    for mass in Fit['masses']:
        F_0[mass] = collections.OrderedDict()
        F_plus[mass] = collections.OrderedDict()        
        qSq[mass] = collections.OrderedDict()
        Z[mass] = collections.OrderedDict()
        Sca[mass] = collections.OrderedDict()
        Vec[mass] = collections.OrderedDict()
        Z_v = (float(mass) - float(Fit['m_s']))*Fit['Sm{0}_tw0'.format(mass)]/((Fit['M_G_m{0}'.format(mass)] - Fit['M_D_tw0'])*Fit['Vm{0}_tw0'.format(mass)]) 
        for twist in Fit['twists']:
            delta = (float(mass) - float(Fit['m_s']))*(Fit['M_G_m{0}'.format(mass)]-Fit['E_D_tw{0}'.format(twist)])
            qsq = Fit['M_G_m{0}'.format(mass)]**2 + Fit['M_D_tw{0}'.format(twist)]**2 - 2*Fit['M_G_m{0}'.format(mass)]*Fit['E_D_tw{0}'.format(twist)]
            t = (Fit['M_G_m{0}'.format(mass)] + Fit['M_D_tw{0}'.format(twist)])**2
            z = (gv.sqrt(t-qsq)-gv.sqrt(t))/(gv.sqrt(t-qsq)+gv.sqrt(t))           
            if qsq.mean >= 0:
                F0 = (float(mass) - float(Fit['m_s']))*(1/(Fit['M_G_m{0}'.format(mass)]**2 - Fit['M_D_tw{0}'.format(twist)]**2))*Fit['Sm{0}_tw{1}'.format(mass,twist)]               
                F_0[mass][twist] = F0                    
                qSq[mass][twist] = qsq                    
                Z[mass][twist] = z
                Sca[mass][twist] = Fit['Sm{0}_tw{1}'.format(mass,twist)]
                Vec[mass][twist] = Fit['Vm{0}_tw{1}'.format(mass,twist)]           
                A = Fit['M_G_m{0}'.format(mass)] + Fit['E_D_tw{0}'.format(twist)]
                B = (Fit['M_G_m{0}'.format(mass)]**2 - Fit['M_D_tw{0}'.format(twist)]**2)*(Fit['M_G_m{0}'.format(mass)] - Fit['E_D_tw{0}'.format(twist)])/qsq           
                if twist != '0':
                    F_plus[mass][twist] = (1/(A-B))*(Z_v*Fit['Vm{0}_tw{1}'.format(mass,twist)] - B*F0)       
        
    
    return(F_0,F_plus,qSq,Z)



def main():
    Metasphys = gv.gvar('0.6885(22)')
    slratio = gv.gvar('27.18(10)')
    make_params()
    f = gv.BufferDict()
    #fplus = gv.BufferDict()
    z = gv.BufferDict()
    prior = gv.BufferDict()    
    mh0val = gv.BufferDict()   
    MetacF = gv.gvar('1.366850(90)')/F['a']
    MetacSF = gv.gvar('0.896686(23)')/SF['a']
    MetacPhys = gv.gvar('2.9863(27)')
    x = MetacPhys*gv.gvar('0.1438(4)')/2   #GeV
    datatags = []
    prior['Metacphys'] = MetacPhys
    for Fit in Fits:
        prior['LQCD_{0}'.format(Fit['conf'])] = 0.5*Fit['a']
        #print(gv.evalcorr([prior['LQCD_{0}'.format(Fit['conf'])],Fit['a']]))
        F_0,F_plus,qSq,Z = make_fs(Fit)
        if Fit == F:
            prior['Metac_{0}'.format(Fit['conf'])] = MetacF          
        elif Fit == SF:
            prior['Metac_{0}'.format(Fit['conf'])] = MetacSF 
        ms0val = float(Fit['m_s'])
        Metas = Fit['M_D_tw{0}'.format(0)]/Fit['a']
        prior['mstuned_{0}'.format(Fit['conf'])] = ms0val*(Metasphys/Metas)**2
        mltuned = prior['mstuned_{0}'.format(Fit['conf'])]/slratio 
        prior['MHc_{0}'.format(Fit['conf'])] = Fit['M_G_m{0}'.format(Fit['Masses'][0])]      
        prior['mstuned_{0}'.format(Fit['conf'])] = ms0val*(Metasphys/Metas)**2
        ms0 = Fit['m_ssea']
        ml0 = Fit['m_lsea']
        prior['deltas_{0}'.format(Fit['conf'])] = ms0-prior['mstuned_{0}'.format(Fit['conf'])]     
        prior['deltasval_{0}'.format(Fit['conf'])] = ms0val-prior['mstuned_{0}'.format(Fit['conf'])]
        prior['deltal_{0}'.format(Fit['conf'])] = ml0-mltuned
        for mass in Fit['masses']:
            prior['MHs0_{0}_m{1}'.format(Fit['conf'],mass)] = Fit['M_Go_m{0}'.format(mass)]
            prior['MHh_{0}_m{1}'.format(Fit['conf'],mass)] = Fit['M_G_m{0}'.format(mass)]
            prior['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)] = prior['MHh_{0}_m{1}'.format(Fit['conf'],mass)] + x*Fit['a']/prior['MHh_{0}_m{1}'.format(Fit['conf'],mass)]
            
            
            mh0val['{0}_m{1}'.format(Fit['conf'],mass)] = float(mass)
            for twist in Fit['twists']:
                if twist in qSq[mass]:                    
                    datatag =  '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
                    datatags.append(datatag)
                    prior['Eetas_{0}_tw{1}'.format(Fit['conf'],twist)] = Fit['E_D_tw{0}'.format(twist)]
                    prior['z_{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)] = Z[mass][twist]
                    prior['qsq_{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)] = qSq[mass][twist]                    
                    f['0{0}'.format(datatag)] = F_0[mass][twist]
                    if twist !='0':
                        f['plus{0}'.format(datatag)] = F_plus[mass][twist]                                            
    if AddRho == True:
        prior['0rho'] =gv.gvar(N*[Pri])
        prior['plusrho'] =gv.gvar(N*[Pri])
        prior['plusrho'][0] = prior['0rho'][0]       
    prior['0d'] = gv.gvar(3*[3*[3*[N*[Pri]]]])
    prior['0cl'] = gv.gvar(N*[Pri])
    prior['0cs'] = gv.gvar(N*[Pri])
    prior['0cc'] = gv.gvar(N*[Pri])
    prior['0csval'] = gv.gvar(N*[Pri])
    prior['plusd'] = gv.gvar(3*[3*[3*[N*[Pri]]]])
    for i in range(3):
        prior['plusd'][i][0][0][0] = prior['0d'][i][0][0][0]
    prior['pluscl'] = gv.gvar(N*[Pri])
    prior['pluscs'] = gv.gvar(N*[Pri])
    prior['pluscc'] = gv.gvar(N*[Pri])
    prior['pluscsval'] = gv.gvar(N*[Pri])
    
    #np.save('Extraps/Datatags', datatags)   

    #plots(prior,f,Metasphys)
    if DoFit == True:
        def fcn(p):
            models = {}
            #print(datatags)
            for datatag in datatags:
                fit = datatag.split('_')[0]
                mass = datatag.split('_')[1].strip('m')
                twist = datatag.split('_')[2].strip('tw')
                if '0{0}'.format(datatag) in f:
                    models['0{0}'.format(datatag)] =  0
                if 'plus{0}'.format(datatag) in f:
                    models['plus{0}'.format(datatag)] =  0
                for n in range(N):
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                if AddRho == True: 
                                    if '0{0}'.format(datatag) in f:
                                        models['0{0}'.format(datatag)] += (1/(1-(p['qsq_{0}_m{1}_tw{2}'.format(fit,mass,twist)]/(p['MHs0_{0}_m{1}'.format(fit,mass)])**2))) * (p['z_{0}_m{1}_tw{2}'.format(fit,mass,twist)])**n * (1 + p['0rho'][n]*gv.log(p['MHh_{0}_m{1}'.format(fit,mass)]/p['MHc_{0}'.format(fit)])) * (1 + (p['0csval'][n]*p['deltasval_{0}'.format(fit)] + p['0cs'][n]*p['deltas_{0}'.format(fit)] + 2*p['0cl'][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + p['0cc'][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['0d'][i][j][k][n] * (p['LQCD_{0}'.format(fit)]/p['MHh_{0}_m{1}'.format(fit,mass)])**int(i) * (mh0val['{0}_m{1}'.format(fit,mass)]/np.pi)**int(2*j) * (p['Eetas_{0}_tw{1}'.format(fit,twist)]/np.pi)**int(2*k)
                                    if 'plus{0}'.format(datatag) in f:                                       
                                        models['plus{0}'.format(datatag)] += (1/(1-p['qsq_{0}_m{1}_tw{2}'.format(fit,mass,twist)]/p['MHsstar_{0}_m{1}'.format(fit,mass)]**2)) * ( p['z_{0}_m{1}_tw{2}'.format(fit,mass,twist)]**n - (n/N) * (-1)**(n-N) * p['z_{0}_m{1}_tw{2}'.format(fit,mass,twist)]**N)* (1 + p['plusrho'][n]*gv.log(p['MHh_{0}_m{1}'.format(fit,mass)]/p['MHc_{0}'.format(fit)])) * (1 + (p['pluscsval'][n]*p['deltasval_{0}'.format(fit)] + p['pluscs'][n]*p['deltas_{0}'.format(fit)] + 2*p['pluscl'][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + p['pluscc'][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['plusd'][i][j][k][n] * (p['LQCD_{0}'.format(fit)]/p['MHh_{0}_m{1}'.format(fit,mass)])**int(i) * (mh0val['{0}_m{1}'.format(fit,mass)]/np.pi)**int(2*j) * (p['Eetas_{0}_tw{1}'.format(fit,twist)]/np.pi)**int(2*k)
                                else:
                                    if '0{0}'.format(datatag) in f:
                                        models['0{0}'.format(datatag)] += (1/(1-(p['qsq_{0}_m{1}_tw{2}'.format(fit,mass,twist)]/(p['MHs0_{0}_m{1}'.format(fit,mass)])**2))) * (p['z_{0}_m{1}_tw{2}'.format(fit,mass,twist)])**n * (1 + (p['0csval'][n]*p['deltasval_{0}'.format(fit)] + p['0cs'][n]*p['deltas_{0}'.format(fit)] + 2*p['0cl'][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + p['0cc'][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['0d'][i][j][k][n] * (p['LQCD_{0}'.format(fit)]/p['MHh_{0}_m{1}'.format(fit,mass)])**int(i) * (mh0val['{0}_m{1}'.format(fit,mass)]/np.pi)**int(2*j) * (p['Eetas_{0}_tw{1}'.format(fit,twist)]/np.pi)**int(2*k)
                                    if 'plus{0}'.format(datatag) in f:                                       
                                        models['plus{0}'.format(datatag)] += (1/(1-p['qsq_{0}_m{1}_tw{2}'.format(fit,mass,twist)]/p['MHsstar_{0}_m{1}'.format(fit,mass)]**2)) * ( p['z_{0}_m{1}_tw{2}'.format(fit,mass,twist)]**n - (n/N) * (-1)**(n-N) * p['z_{0}_m{1}_tw{2}'.format(fit,mass,twist)]**N) * (1 + (p['pluscsval'][n]*p['deltasval_{0}'.format(fit)] + p['pluscs'][n]*p['deltas_{0}'.format(fit)] + 2*p['pluscl'][n]*p['deltal_{0}'.format(fit)])/(10*p['mstuned_{0}'.format(fit)]) + p['pluscc'][n]*((p['Metac_{0}'.format(fit)] - p['Metacphys'])/p['Metacphys'])) * p['plusd'][i][j][k][n] * (p['LQCD_{0}'.format(fit)]/p['MHh_{0}_m{1}'.format(fit,mass)])**int(i) * (mh0val['{0}_m{1}'.format(fit,mass)]/np.pi)**int(2*j) * (p['Eetas_{0}_tw{1}'.format(fit,twist)]/np.pi)**int(2*k)
                                                                                                
                            
            return(models)
    
        if os.path.exists('Extraps/{0}{1}.pickle'.format(AddRho,N)):
            pickle_off = open('Extraps/{0}{1}.pickle'.format(AddRho,N),"rb")
            p0 = pickle.load(pickle_off)
        else:
            p0 = None
        #s = gv.dataset.svd_diagnosis(f)
        #s.plot_ratio(show=True)
        fit = lsqfit.nonlinear_fit(data=f, prior=prior, p0=p0, fcn=fcn, svdcut=1e-12,add_svdnoise=svdnoise, add_priornoise=priornoise,fitter='gsl_multifit', alg='subspace2D', solver='cholesky', maxit=5000, tol=(1e-6,0.0,0.0) )
        gv.dump(fit.p,'Extraps/{0}{1}chi{2:.3f}'.format(AddRho,N,fit.chi2/fit.dof))
        print(fit.format(maxline=True))        
        #print(fit)
        savefile = open('Extraps/{0}{1}chi{2:.3f}.txt'.format(AddRho,N,fit.chi2/fit.dof),'w')
        savefile.write(fit.format(pstyle='v'))
        savefile.close()
        pickle_on = open('Extraps/{0}{1}.pickle'.format(AddRho,N),"wb")
        pickle.dump(fit.pmean,pickle_on)
        pickle_on.close
        plots(prior,f,Metasphys,fit.p,x)
    return()
    
     
       
def plots(prior,f,Metasphys,p,x):
    Z,Qsq,MBsphys,MBsstarphys,F0mean,F0upp,F0low,Fplusmean,Fplusupp,Fpluslow = plot_results(Metasphys,p,x,prior,f)
    cols = ['b','g','r','c','m','y','k','purple']
    plt.figure(1)
    Fit = F
    for i , mass in enumerate(Fit['masses']):
        xmean = [] 
        xerr = [] 
        ymean = [] 
        yerr = []
        col = cols[i]        
        for twist in Fit['twists']:
            datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)    
            if '0{0}'.format(datatag) in f:
                xmean.append(prior['z_{0}'.format(datatag)].mean)
                ymean.append((f['0{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHs0_{0}_m{1}'.format(Fit['conf'],mass)]**2))).mean)
                xerr.append(prior['z_{0}'.format(datatag)].sdev)
                yerr.append((f['0{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHs0_{0}_m{1}'.format(Fit['conf'],mass)]**2))).sdev)
        plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none')
        plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, fmt='o', mfc='none',label=('{0}_m{1}'.format(Fit['conf'],mass)))
    Fit = SF
    for i , mass in enumerate(Fit['masses']):
        xmean = [] 
        xerr = [] 
        ymean = [] 
        yerr = []
        col = cols[i+4]        
        for twist in Fit['twists']:
            datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)    
            if '0{0}'.format(datatag) in f:
                xmean.append(prior['z_{0}'.format(datatag)].mean)
                ymean.append((f['0{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHs0_{0}_m{1}'.format(Fit['conf'],mass)]**2))).mean)
                xerr.append(prior['z_{0}'.format(datatag)].sdev)
                yerr.append((f['0{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHs0_{0}_m{1}'.format(Fit['conf'],mass)]**2))).sdev)
        plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none')
        plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none',fmt='^',label=('{0}_m{1}'.format(Fit['conf'],mass)))
    plt.plot(Z,F0mean, color='k')
    #plt.plot(Z,F0upp, color='r')
    plt.fill_between(Z,F0low,F0upp, color='r')
    plt.legend()
    plt.xlabel('z',fontsize=20)
    plt.ylabel(r'$(1-\frac{q^2}{M^2_{H_{s}^0}})f_0$',fontsize=20)
    plt.savefig('Extraps/f0pole')
    plt.figure(2)
    Fit = F
    for i , mass in enumerate(Fit['masses']):
        xmean = [] 
        xerr = [] 
        ymean = [] 
        yerr = []
        col = cols[i]
        for twist in Fit['twists']:
            datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
            if 'plus{0}'.format(datatag) in f:
                xmean.append(prior['z_{0}'.format(datatag)].mean)
                ymean.append((f['plus{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2))).mean)
                xerr.append(prior['z_{0}'.format(datatag)].sdev)
                yerr.append((f['plus{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2))).sdev)
        plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none')
        plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, fmt='o',mfc='none',label=('{0}_m{1}'.format(Fit['conf'],mass)))
    Fit = SF
    for i , mass in enumerate(Fit['masses']):
        xmean = [] 
        xerr = [] 
        ymean = [] 
        yerr = []
        col = cols[4+i]
        for twist in Fit['twists']:
            datatag = '{0}_m{1}_tw{2}'.format(Fit['conf'],mass,twist)
            if 'plus{0}'.format(datatag) in f:
                xmean.append(prior['z_{0}'.format(datatag)].mean)
                ymean.append((f['plus{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2))).mean)
                xerr.append(prior['z_{0}'.format(datatag)].sdev)
                yerr.append((f['plus{0}'.format(datatag)]*(1-prior['qsq_{0}'.format(datatag)]/(prior['MHsstar_{0}_m{1}'.format(Fit['conf'],mass)]**2))).sdev)
        plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, mfc='none')
        plt.errorbar(xmean, ymean, xerr=xerr, yerr=yerr, color=col, fmt='^',mfc='none',label=('{0}_m{1}'.format(Fit['conf'],mass)))
    plt.plot(Z,Fplusmean, color='k')
    #plt.plot(Z,Fplusupp, color='g')
    plt.fill_between(Z,Fpluslow,Fplusupp, color='r')
    plt.legend()
    plt.xlabel('z',fontsize=20)
    plt.ylabel(r'$(1-\frac{q^2}{M^2_{H_{s}^*}})f_{plus}$',fontsize=20)
    plt.savefig('Extraps/fpluspole')
    plt.show()
    return()




def plot_results(Metasphys,p,x,prior,f):
    a = collections.OrderedDict()
    a['0'] = [0]*N
    a['plus'] = [0]*N
    Del = gv.gvar('0.312(15)')
    filename = 'Extraps/Testchi0.142'
    datatag0 = 'F_m0.499_tw0'
    plusdatatag = 'F_m0.499_tw0.4281'
    Metab = gv.gvar('9.3987(20)')
    MBsphys = gv.gvar('5.36688(17)')
    MDsphys = gv.gvar('1.96834(7)')
    MBs0 = gv.gvar('5.36688(17)') + Del  # Get this properly later
    #MBs0 = gv.gvar('5.36688(17)') + gv.gvar('0.3471(73)')
    MBsstarphys = gv.gvar('5.4158(15)')
    #p = gv.load(filename)
    F0meanpole = np.zeros((100))
    Fplusmeanpole = np.zeros((100))
    F0upppole = np.zeros((100))
    F0lowpole = np.zeros((100))
    Fplusupppole = np.zeros((100))
    Fpluslowpole = np.zeros((100))
    F0mean = np.zeros((100))
    Fplusmean = np.zeros((100))
    F0upp= np.zeros((100))
    F0low= np.zeros((100))
    Fplusupp = np.zeros((100))
    Fpluslow = np.zeros((100))
    qsq = []
    qsqmean = []
    #qsqmax = 26
    qsqmax = (MBsphys-Metasphys)**2
    tplus = (MBsphys+Metasphys)**2
    zmax = ((gv.sqrt(tplus-qsqmax)-gv.sqrt(tplus))/(gv.sqrt(tplus-qsqmax)+gv.sqrt(tplus))).mean
    Z = np.linspace(zmax,0,100)
    #p['LQCD'] = 0.5
    p['LQCD'] = p['LQCD_F']/F['a']
    #print(gv.evalcorr([p['LQCD'],F['a']]))
    plt.figure(3)
    
    for j in range(len(Z)):
        f0physpole = 0
        f0phys = 0
        fplusphyspole = 0
        fplusphys = 0
        qsq.append((1-((1+Z[j])/(1-Z[j]))**2)*tplus)
        qsqmean.append(((1-((1+Z[j])/(1-Z[j]))**2)*tplus).mean)
        for n in range(N):
            a['0'][n] = 0
            a['plus'][n] = 0
            for i in range(3):
                if AddRho == True:
                    a['0'][n] += p['0d'][i][0][0][n] * (p['LQCD']/MBsphys)**i * (1 + p['0rho'][n] * gv.log(MBsphys/MDsphys) )
                    a['plus'][n] += p['plusd'][i][0][0][n] * (p['LQCD']/MBsphys)**i * (1 + p['plusrho'][n] * gv.log(MBsphys/MDsphys))
                else:
                    a['0'][n] += p['0d'][i][0][0][n] * (p['LQCD']/MBsphys)**i
                    a['plus'][n] += p['plusd'][i][0][0][n] * (p['LQCD']/MBsphys)**i 
                    
            f0physpole += (1/(1 - qsq[j]/(MBs0**2))) * Z[j]**n * a['0'][n]
            f0phys +=  Z[j]**n * a['0'][n]
            fplusphyspole += (1/(1 - qsq[j]/(MBsstarphys**2))) * (Z[j]**n - (n/N) * (-1)**(n-N) * Z[j]**N) * a['plus'][n]
            fplusphys += (Z[j]**n - (n/N) * (-1)**(n-N) * Z[j]**N) * a['plus'][n]
        if j == 99:
            print('$f_0(0)$:',f0physpole)
            if AddRho:               
                inputs = {'d0000':prior['0d'][0][0][0][0],'d1000':prior['0d'][1][0][0][0],'d1000':prior['0d'][1][0][0][0],'d2000':prior['0d'][2][0][0][0],'d0001':prior['0d'][0][0][0][1],'d1001':prior['0d'][1][0][0][1],'d1001':prior['0d'][1][0][0][1],'d2001':prior['0d'][2][0][0][1],'d0002':prior['0d'][0][0][0][2],'d1002':prior['0d'][1][0][0][2],'d1002':prior['0d'][1][0][0][2],'d2002':prior['0d'][2][0][0][2],'rho0':prior['0rho'][0],'rho1':prior['0rho'][1],'rho2':prior['0rho'][2],'data':f}
                
            else:
                inputs = {'d0000':prior['0d'][0][0][0][0],'d1000':prior['0d'][1][0][0][0],'d1000':prior['0d'][1][0][0][0],'d2000':prior['0d'][2][0][0][0],'d0001':prior['0d'][0][0][0][1],'d1001':prior['0d'][1][0][0][1],'d1001':prior['0d'][1][0][0][1],'d2001':prior['0d'][2][0][0][1],'d0002':prior['0d'][0][0][0][2],'d1002':prior['0d'][1][0][0][2],'d1002':prior['0d'][1][0][0][2],'d2002':prior['0d'][2][0][0][2],'data':f}
            outputs = {'f_0(0)':f0physpole}
            print(gv.fmt_errorbudget(outputs=outputs, inputs=inputs))
        F0mean[j] = f0phys.mean
        F0upp[j] = f0phys.mean + f0phys.sdev
        F0low[j] = f0phys.mean - f0phys.sdev
                
        F0meanpole[j] = f0physpole.mean
        F0upppole[j] = f0physpole.mean + f0physpole.sdev
        F0lowpole[j] = f0physpole.mean - f0physpole.sdev
        
        Fplusmean[j] = fplusphys.mean
        Fplusupp[j] = fplusphys.mean + fplusphys.sdev
        Fpluslow[j] = fplusphys.mean - fplusphys.sdev
                
        Fplusmeanpole[j] = fplusphyspole.mean
        Fplusupppole[j] = fplusphyspole.mean + fplusphyspole.sdev
        Fpluslowpole[j] = fplusphyspole.mean - fplusphyspole.sdev
    plt.plot(Z,F0meanpole, color='b',label='$f_0$')
    #plt.plot(Z,F0mean, color='b',linestyle='--',label='$f_0$ no pole')
    plt.fill_between(Z,F0lowpole,F0upppole, color='b',alpha=0.6)
    plt.xlabel('z',fontsize=20)
    #plt.ylabel('f',fontsize=20)
    plt.plot(Z,Fplusmeanpole, color='r',label='$f_+$')
    #plt.plot(Z,Fplusmean, color='r',linestyle='--',label='$f_+$ no pole')
    plt.fill_between(Z,Fpluslowpole,Fplusupppole, color='r', alpha=0.6)
    plt.legend()
    #plt.ylabel('f',fontsize=20)
    plt.axes().tick_params(labelright=True,which='both',width=2)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().set_xlim([zmax,0])
    plt.axes().set_ylim([0,3.3])

    plt.figure(4)
    plt.plot(qsqmean,F0meanpole, color='b',label='$f_0$')
    #plt.plot(qsqmean,F0mean, color='b',linestyle='--',label='$f_0$ no pole')
    plt.fill_between(qsqmean,F0lowpole,F0upppole, color='b',alpha=0.6)
    plt.xlabel('$q^2(GeV^2)$',fontsize=20)
    #plt.ylabel('f',fontsize=20)
    plt.plot(qsqmean,Fplusmeanpole, color='r',label='$f_+$')
    #plt.plot(qsqmean,Fplusmean, color='r',linestyle='--',label='$f_+$ no pole')
    plt.fill_between(qsqmean,Fpluslowpole,Fplusupppole, color='r',alpha=0.6)
    plt.axes().tick_params(labelright=True,which='both',width=2)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(5))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().set_xlim([0,qsqmax.mean])
    plt.axes().set_ylim([0,3.3])    
    plt.legend()

    ####################################################### M_h plots ###############
    Mh = np.linspace(MDsphys.mean,MBsphys.mean,100)
    a = collections.OrderedDict()
    a['0'] = [0]*N
    a['plus'] = [0]*N
    fplusqmaxmean = np.zeros((100))
    fplusqmaxupp = np.zeros((100))
    fplusqmaxlow = np.zeros((100))
    f0qmaxmean = np.zeros((100))
    f0qmaxupp = np.zeros((100))
    f0qmaxlow = np.zeros((100))
    f0q0mean = np.zeros((100))
    f0q0upp = np.zeros((100))
    f0q0low = np.zeros((100))
    fplusq0mean = np.zeros((100))
    fplusq0upp = np.zeros((100))
    fplusq0low = np.zeros((100))
    invbetamean = np.zeros((100))
    invbetaupp = np.zeros((100))
    invbetalow = np.zeros((100))
    deltamean = np.zeros((100))
    deltaupp =np.zeros((100))
    deltalow =np.zeros((100))
    for j in range(len(Mh)):
        tpl =(Mh[j]+Metasphys)**2
        MHs0 = Mh[j] + Del
        MHsstar = Mh[j] + x/Mh[j]
        qsqmax = (Mh[j]-Metasphys)**2
        zmax = ((gv.sqrt(tpl-qsqmax)-gv.sqrt(tpl))/(gv.sqrt(tpl-qsqmax)+gv.sqrt(tpl))).mean
        f0qmax = 0
        fplusqmax = 0
        for n in range(N):
            a['0'][n] = 0
            a['plus'][n] = 0
            for i in range(3):
                if AddRho == True:
                    a['0'][n] +=  p['0d'][i][0][0][n] * (p['LQCD']/Mh[j])**i *(1 + p['0rho'][n] * gv.log(Mh[j]/MDsphys))
                    a['plus'][n] += p['plusd'][i][0][0][n] * (p['LQCD']/Mh[j])**i * (1 + p['plusrho'][n] * gv.log(Mh[j]/MDsphys))
                else:
                    a['0'][n] +=  p['0d'][i][0][0][n] * (p['LQCD']/Mh[j])**i 
                    a['plus'][n] += p['plusd'][i][0][0][n] * (p['LQCD']/Mh[j])**i 
                    
            f0qmax += (1/(1 - qsqmax/(MHs0**2))) * zmax**n * a['0'][n]
            fplusqmax += (1/(1 - qsqmax/(MHsstar**2))) * (zmax**n - (n/N) * (-1)**(n-N) * zmax**N) * a['plus'][n]
        f0q0 =  a['0'][0]
        fplusq0 = a['plus'][0]
        #print(gv.evalcorr([Metasphys,Metasphys]))
        #print(gv.evalcorr([Metasphys**2-Mh[j]**2,tpl]))
        
        
        #invbeta = (Metasphys**2-Mh[j]**2)/(fplusq0*2*tpl) * ( a['0'][0]/(MHs0**2) + a['0'][1])
        #delta = 1-((Metasphys**2-Mh[j]**2)/fplusq0 * (1/(2*tpl)) * (a['plus'][0]/MHsstar**2 - a['0'][0]/MHs0**2 + a['plus'][1] - a['0'][1]))
        invbeta = (Mh[j]**2-Metasphys**2) * ( 1/(MHs0**2) - a['0'][1]/(2*tpl*a['0'][0]))
        delta = 1-((Mh[j]**2-Metasphys**2) * (1/MHsstar**2 - 1/MHs0**2 - a['plus'][1]/(2*tpl*a['plus'][0]) + a['0'][1]/(2*tpl*a['0'][0])))
        f0qmaxmean[j] = f0qmax.mean
        f0qmaxupp[j] = f0qmax.mean + f0qmax.sdev
        f0qmaxlow[j] = f0qmax.mean - f0qmax.sdev

        fplusqmaxmean[j] = fplusqmax.mean
        fplusqmaxupp[j] = fplusqmax.mean + fplusqmax.sdev
        fplusqmaxlow[j] = fplusqmax.mean - fplusqmax.sdev
                
        f0q0mean[j] = f0q0.mean
        f0q0upp[j] = f0q0.mean + f0q0.sdev
        f0q0low[j] = f0q0.mean - f0q0.sdev

        fplusq0mean[j] = fplusq0.mean
        fplusq0upp[j] = fplusq0.mean + fplusq0.sdev
        fplusq0low[j] = fplusq0.mean - fplusq0.sdev
        
        invbetamean[j] = invbeta.mean
        invbetaupp[j] = invbeta.mean + invbeta.sdev
        invbetalow[j] = invbeta.mean - invbeta.sdev

        deltamean[j] = delta.mean
        deltaupp[j] = delta.mean + delta.sdev
        deltalow[j] = delta.mean - delta.sdev
        
    plt.figure()
    plt.plot(Mh,f0qmaxmean, color='b',label='$f_0(q^2_{max})$')
    plt.fill_between(Mh,f0qmaxupp,f0qmaxlow, color='b',alpha=0.6)
    plt.plot(Mh,fplusqmaxmean, color='r',label='$f_+(q^2_{max})$')
    plt.fill_between(Mh,fplusqmaxupp,fplusqmaxlow, color='r',alpha=0.6)
    plt.plot(Mh,f0q0mean, color='k',label='$f_0(0)$')
    plt.fill_between(Mh,f0q0upp,f0q0low, color='k',alpha=0.6)
    #plt.plot(Mh,fplusq0mean, color='purple',label='$f_+(0)$')
    #plt.fill_between(Mh,fplusq0upp,fplusq0low, color='purple',alpha=0.6)
    plt.xlabel('$M_h[GeV]$',fontsize=20)
    #plt.ylabel('$f$',fontsize=20)
    plt.legend()
    plt.axes().tick_params(labelright=True,which='both',width=2)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().set_xlim([MDsphys.mean,MBsphys.mean])
    plt.axes().set_ylim([0.2,3.3]) 

    plt.figure()

    plt.plot(Mh,invbetamean, color='b',label=r'$\beta^{-1}$')
    plt.fill_between(Mh,invbetaupp,invbetalow, color='b',alpha=0.6)
    plt.plot(Mh,deltamean, color='r',label='$\delta$')
    plt.fill_between(Mh,deltaupp,deltalow, color='r',alpha=0.6)
    plt.xlabel('$M_h[GeV]$',fontsize=20)
    #plt.ylabel('$f$',fontsize=20)
    plt.legend()
    plt.axes().tick_params(labelright=True,which='both',width=2)
    plt.axes().tick_params(which='major',length=15)
    plt.axes().tick_params(which='minor',length=8)
    plt.axes().yaxis.set_ticks_position('both')
    plt.axes().xaxis.set_major_locator(MultipleLocator(1))
    plt.axes().xaxis.set_minor_locator(MultipleLocator(0.2))
    plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.axes().set_xlim([MDsphys.mean,MBsphys.mean])
    #plt.axes().set_ylim([-2.2,0.2]) 
    

    
    return(Z,qsq,MBsphys,MBsstarphys,F0mean,F0upp,F0low,Fplusmean,Fplusupp,Fpluslow)

#plot_results()


aDelta = collections.OrderedDict()
amb = collections.OrderedDict()

afm = ['0.1583(13)','0.1595(14)','0.1247(10)','0.1264(11)','0.0878(7)']
amb['1'] = [3.4,3.4,3.6,3.6]
amb['2'] = 3.4
amb['3'] = 2.8
amb['4'] = 2.8
amb['5'] = 1.95
aDelta['1'] = ['0.310(11)','0.317(11)','0.300(15)','0.315(11)']
aDelta['2'] = '0.299(17)'
aDelta['3'] = '0.215(17)'
aDelta['4'] = '0.253(8)'
aDelta['5'] = '0.1708(48)'


def convert_Gev(a):
    hbar = '6.58211928(15)'
    c = 2.99792458
    aGev = gv.gvar(a)/(gv.gvar(hbar)*c*1e-2)
    return(aGev)

def findDelta():
    x = []
    y = []
    mean =[]
    upper= []
    lower = []
    colours =['r','g','g','b','k','purple']
    labels = ['Very Coarse', 'Coarse','Coarse', 'Fine','Fine','Superfine']
    plt.figure()
    #Delta = collections.OrderedDict()
    for c , Fit in enumerate(Fits):        
        p = gv.load(Fit['filename'])
        for mass in Fit['masses']:
            Delta = (p['dE:o{0}'.format(Fit['Gm{0}'.format(mass)])][0]-p['dE:{0}'.format(Fit['Gm{0}'.format(mass)])][0])/Fit['a']
            x.append((float(mass)/Fit['a']))
            y.append(Delta)
            
            plt.errorbar((float(mass)/Fit['a']).mean, Delta.mean, yerr=Delta.sdev,fmt='o', mfc='none', color=colours[c+4],label=(labels[c+4]))
    plt.xlabel('Heavy Mass (GeV)')
    plt.ylabel('Delta (GeV)')

    
    for i in range(4):
        a = convert_Gev(afm[1])
        delta = gv.gvar(aDelta['1'][i])/a
        mb = amb['1'][i]/a
        plt.errorbar((mb).mean, delta.mean, xerr=(mb).sdev, yerr=delta.sdev, fmt='o', mfc='none',color='r',label=(labels[0]))
        #print('mb(Gev)', a, 'Delta (Gev)', delta)
    for i in range(4):
        
        a = convert_Gev(afm[i+1])
        delta = gv.gvar(aDelta['{0}'.format(i+2)])/a
        mb = amb['{0}'.format(i+2)]/a
        plt.errorbar((mb).mean, delta.mean, xerr=(mb).sdev, yerr=delta.sdev,color=colours[i],fmt='o', mfc='none',label=(labels[i]))        
    plt.legend()

    prior = gv.BufferDict()
    prior['x'] = x
    prior['a'] = gv.gvar('-0.01(1)')
    prior['b'] = gv.gvar('0.36(2)')
 
    def func(p):
        return(p['a']*p['x']+p['b'])
    fit = lsqfit.nonlinear_fit(prior=prior, data=y, fcn=func)
    print(fit)
    p = fit.p
    p.pop('x',None)
    p['x'] = np.linspace(0.7,5,100)
    for i in range(100):
        mean.append(func(p)[i].mean)
        upper.append(func(p)[i].mean+func(p)[i].sdev)
        lower.append(func(p)[i].mean-func(p)[i].sdev)
    plt.plot(p['x'], mean, color='k',linestyle='--')
    p.pop('x',None)
    p['x'] = 4.18
    print(func(p))
    #plt.fill_between(p['x'],lower,upper, color='k',alpha=0.4)
    #print(func(p))
    plt.show()
    return()



main()


#findDelta()

