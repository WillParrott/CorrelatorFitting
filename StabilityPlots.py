import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator


#F

NMarg = [5,5,5,5,5,5,4,4]
Nexp =  [1,2,1,2,1,2,1,2]
twoptsvd = [0.0028,0.0028,0.0057,0.0057,0.0014,0.0014,0.0028,0.0028]
twoptchi = [0.28,0.28,0.25,0.25,0.32,0.32,0.30,0.30]
twoptgbf = [17017.11,17017.11,16881.06,16881.06,17145.53,17145.53,17006.30,17006.30]
threeptsvd = [0.0021,0.0021,0.0041,0.0041,0.001,0.001,0.0021,0.0021]
threeptchi = [0.35,0.32,0.30,0.27,0.42,0.37,0.39,0.39]
threeptgbf = [35016.27,35495.11,34597.52,35083.85,35406.12,35879.67,35019.03,35506.76]
exrhochi = [0.13,0.13,0.093,0.11,0.15,0.15,0.13,0.14]
extrhogbf = [98.509,105.46,92.224,99.911,103.83,110.12,101.02,110.06]
rhof00 = ['0.33(21)','0.31(20)','0.34(22)','0.32(21)','0.32(20)','0.30(19)','0.33(21)','0.30(20)']
exchi = [0.16,0.17,0.13,0.14,0.18,0.19,0.17,0.19]
extgbf = [101.26,107.89,95.098,102.55,106.94,112.76,103.95,112.33]   #worth comparing?
f00 = ['0.495(80)','0.493(78)','0.501(84)','0.502(78)','0.485(76)','0.481(77)','0.495(78)','0.498(75)']



#SF

NMarg = [6,6,6]
Nexp =  [1,2,3]
twoptsvd = [0.0029,0.0029,0.0029]
twoptchi = [0.25,0.25,0.25]
twoptgbf = [28161.11,28161.11,28161.11]
threeptsvd = [0.0046,0.0046,0.0046]
threeptchi = [0.11,0.10,0.08]
threeptgbf = [61974.62,62750.22,63102.08]
exrhochi =[0.14,0.16,0.12]
extrhogbf = [86.025,89.236,81.604]
rhof00 = ['0.310(98)','0.278(95)','0.28(10)']
exchi = [0.24,0.27,0.22]
extgbf = [88.28,91,413,83.744]   #worth comparing?
f00 = ['0.441(54)','0.435(53)','0.436(57)']

####################################################################################################################
Fit = 'SF'
#Key = 'dE:meson.m0.0234_m0.6'       #optionally choose a key else False
#Key='VVnn_m0.450_tw1.261'
Key='VVnn_m0.8_tw2.946'
#Key=False
#N=1 End result
#N=2 Nterm +1
#N=3 Nterm -1
#N=4 2svd
#N=5 0.5 svd
#N=6 2 pl
#N=7 0.5 pl
#Stmin=Vtmin=3


SFNterm = 3
SFTmin = 2

FNterm = 2
FTmin = 2

Nterm = globals()['{0}Nterm'.format(Fit)]
tmin = globals()['{0}Tmin'.format(Fit)]

Ffilenames = ['Fits/F5_3pts_Q1.00_Nexp1_NMarg5_Stmin2_Vtmin1_svd0.00157_chi0.360']
SFfilenames =['Fits/SF5_3pts_Q1.00_Nexp1_NMarg6_Stmin2_Vtmin2_svd0.00457_chi0.106_pl1.0_svdfac1.0','Fits/SF5_3pts_Q1.00_Nexp2_NMarg6_Stmin2_Vtmin2_svd0.00457_chi0.097_pl1.0_svdfac1.0','Fits/SF5_3pts_Q1.00_Nexp3_NMarg6_Stmin2_Vtmin2_svd0.00457_chi0.079_pl1.0_svdfac1.0','Fits/SF5_3pts_Q1.00_Nexp4_NMarg6_Stmin2_Vtmin2_svd0.00457_chi0.062_pl1.0_svdfac1.0','Fits/SF5_3pts_Q1.00_Nexp1_NMarg6_Stmin2_Vtmin2_svd0.00913_chi0.091_pl1.0_svdfac2.0','Fits/SF5_3pts_Q1.00_Nexp2_NMarg6_Stmin2_Vtmin2_svd0.00913_chi0.079_pl1.0_svdfac2.0','Fits/SF5_3pts_Q1.00_Nexp3_NMarg6_Stmin2_Vtmin2_svd0.00913_chi0.043_pl1.0_svdfac2.0','Fits/SF5_3pts_Q1.00_Nexp4_NMarg6_Stmin2_Vtmin2_svd0.00913_chi0.033_pl1.0_svdfac2.0']

for filename in globals()['{0}filenames'.format(Fit)]:
    #print(filename)
    Q = float(filename.split('_')[2].strip('Q'))
    Nexp = int(filename.split('_')[3].strip('Nexp'))
    NMarg = int(filename.split('_')[4].strip('NMarg'))
    Stmin = int(filename.split('_')[5].strip('Stmin'))
    Vtmin = int(filename.split('_')[6].strip('Vtmin'))
    svd = float(filename.split('_')[7].strip('svd'))
    chi = float(filename.split('_')[8].strip('chi'))
    pl = float(filename.split('_')[9].strip('pl'))
    svdfac = float(filename.split('_')[10].strip('svdfac'))
    p = gv.load(filename)
    if Key == False:  #Make this so you don't have to choose all the keys
        for key in p:
            print(key)
            if key[0] == 'l':
                key = key.strip('log(').strip(')')
            plt.figure(key)
            plt.xlabel('Test',fontsize=30)
            plt.ylabel(key,fontsize=30)
            plt.axes().tick_params(labelright=True,which='both',width=2)
            plt.axes().tick_params(which='major',length=15)
            plt.axes().tick_params(which='minor',length=8)
            plt.axes().yaxis.set_ticks_position('both')
            plt.axes().xaxis.set_major_locator(MultipleLocator(1))
            plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
            #plt.axes().yaxis.set_major_locator(MultipleLocator(0.005))
            #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.001))
            plt.axes().set_xlim([0.5,8.5])
            if key[0]!='S' and key[0]!='V':
                if Nexp== Nterm and pl==1.0 and svdfac==1.0 and Stmin==tmin:
                    plt.errorbar(1,p[key][0].mean,yerr=p[key][0].sdev,fmt='kd',ms=12,mfc='none')
                    plt.fill_between([0,9],[p[key][0].mean-p[key][0].sdev,p[key][0].mean-p[key][0].sdev],[p[key][0].mean+p[key][0].sdev,p[key][0].mean+p[key][0].sdev], color='b',alpha=0.3)
                elif Nexp== Nterm+1 and pl==1.0 and svdfac==1.0 and Stmin==tmin:
                    plt.errorbar(2,p[key][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
                elif Nexp==Nterm-1 and pl==1.0 and svdfac==1.0 and Stmin==tmin:
                    plt.errorbar(3,p[key][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
                elif Nexp== Nterm and pl==1.0 and svdfac==2.0 and Stmin==tmin:
                    plt.errorbar(4,p[key][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
                elif Nexp==Nterm and pl==1.0 and svdfac==0.5 and Stmin==tmin:
                    plt.errorbar(5,p[key][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
                elif Nexp== Nterm and pl==2.0 and svdfac==1.0 and Stmin==tmin:
                    plt.errorbar(6,p[key][0][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
                elif Nexp==Nterm and pl==0.5 and svdfac==1.0 and Stmin==tmin:
                    plt.errorbar(7,p[key][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
                elif Nexp== Nterm and pl==1.0 and svdfac==1.0 and Stmin==tmin +1:
                    plt.errorbar(8,p[key][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
                else:
                    print('Nexp:',Nexp,'pl:',pl,'svdfac:',svdfac, 'not used')
            else:
            
                if Nexp== Nterm and pl==1.0 and svdfac==1.0 and Stmin==tmin:
                    plt.errorbar(1,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='kd',ms=12,mfc='none')
                    delta = 2*p[key][0][0].sdev
                    plt.fill_between([0,9],[p[key][0][0].mean-p[key][0][0].sdev,p[key][0][0].mean-p[key][0][0].sdev],[p[key][0][0].mean+p[key][0][0].sdev,p[key][0][0].mean+p[key][0][0].sdev], color='b',alpha=0.3)
                elif Nexp== Nterm+1 and pl==1.0 and svdfac==1.0 and Stmin==tmin:
                    plt.errorbar(2,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
                elif Nexp==Nterm-1 and pl==1.0 and svdfac==1.0 and Stmin==tmin:
                    plt.errorbar(3,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
                elif Nexp== Nterm and pl==1.0 and svdfac==2.0 and Stmin==tmin:
                    plt.errorbar(4,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
                elif Nexp==Nterm and pl==1.0 and svdfac==0.5 and Stmin==tmin:
                    plt.errorbar(5,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
                elif Nexp== Nterm and pl==2.0 and svdfac==1.0 and Stmin==tmin:
                    plt.errorbar(6,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
                elif Nexp==Nterm and pl==0.5 and svdfac==1.0 and Stmin==tmin:
                    plt.errorbar(7,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
                elif Nexp== Nterm and pl==1.0 and svdfac==1.0 and Stmin==tmin+1:
                    plt.errorbar(8,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
                else:
                    print(Nexp,pl,svdfac, 'not used')
    else:
        key = Key
        plt.figure(key)
        plt.xlabel('Test',fontsize=30)
        plt.ylabel(key,fontsize=30)
        plt.axes().tick_params(labelright=True,which='both',width=2)
        plt.axes().tick_params(which='major',length=15)
        plt.axes().tick_params(which='minor',length=8)
        plt.axes().yaxis.set_ticks_position('both')
        plt.axes().xaxis.set_major_locator(MultipleLocator(1))
        plt.axes().xaxis.set_minor_locator(MultipleLocator(1))
        #plt.axes().yaxis.set_major_locator(MultipleLocator(0.005))
        #plt.axes().yaxis.set_minor_locator(MultipleLocator(0.001))
        plt.axes().set_xlim([0.5,8.5])
        if key[0]!='S' and key[0]!='V':
            if Nexp== Nterm and pl==1.0 and svdfac==1.0 and Stmin==tmin:
                plt.errorbar(1,p[key][0].mean,yerr=p[key][0].sdev,fmt='kd',ms=12,mfc='none')
                plt.fill_between([0,9],[p[key][0].mean-p[key][0].sdev,p[key][0].mean-p[key][0].sdev],[p[key][0].mean+p[key][0].sdev,p[key][0].mean+p[key][0].sdev], color='b',alpha=0.3)
            elif Nexp== Nterm+1 and pl==1.0 and svdfac==1.0 and Stmin==tmin:
                plt.errorbar(2,p[key][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
            elif Nexp==Nterm-1 and pl==1.0 and svdfac==1.0 and Stmin==tmin:
                plt.errorbar(3,p[key][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
            elif Nexp== Nterm and pl==1.0 and svdfac==2.0 and Stmin==tmin:
                plt.errorbar(4,p[key][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
            elif Nexp==Nterm and pl==1.0 and svdfac==0.5 and Stmin==tmin:
                plt.errorbar(5,p[key][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
            elif Nexp== Nterm and pl==2.0 and svdfac==1.0 and Stmin==tmin:
                plt.errorbar(6,p[key][0][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
            elif Nexp==Nterm and pl==0.5 and svdfac==1.0 and Stmin==tmin:
                plt.errorbar(7,p[key][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
            elif Nexp== Nterm and pl==1.0 and svdfac==1.0 and Stmin==tmin +1:
                plt.errorbar(8,p[key][0].mean,yerr=p[key][0].sdev,fmt='ko',ms=12,mfc='none')
            else:
                print(Nexp,pl,svdfac, 'not used')
        else:
            #plt.title(Stability)
            #print(pl)
            if Nexp== Nterm and pl==1.0 and svdfac==1.0 and Stmin==tmin:
                plt.errorbar(1,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='kd',ms=12,mfc='none')
                delta = 2*p[key][0][0].sdev
                plt.fill_between([0,9],[p[key][0][0].mean-p[key][0][0].sdev,p[key][0][0].mean-p[key][0][0].sdev],[p[key][0][0].mean+p[key][0][0].sdev,p[key][0][0].mean+p[key][0][0].sdev], color='b',alpha=0.3)
            elif Nexp== Nterm+1 and pl==1.0 and svdfac==1.0 and Stmin==tmin:
                plt.errorbar(2,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
            elif Nexp==Nterm-1 and pl==1.0 and svdfac==1.0 and Stmin==tmin:
                plt.errorbar(3,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
            elif Nexp== Nterm and pl==1.0 and svdfac==2.0 and Stmin==tmin:
                plt.errorbar(4,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
            elif Nexp==Nterm and pl==1.0 and svdfac==0.5 and Stmin==tmin:
                plt.errorbar(5,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
            elif Nexp== Nterm and pl==2.0 and svdfac==1.0 and Stmin==tmin:
                plt.errorbar(6,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
            elif Nexp==Nterm and pl==0.5 and svdfac==1.0 and Stmin==tmin:
                plt.errorbar(7,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
            elif Nexp== Nterm and pl==1.0 and svdfac==1.0 and Stmin==tmin+1:
                plt.errorbar(8,p[key][0][0].mean,yerr=p[key][0][0].sdev,fmt='ko',ms=12,mfc='none')
            else:
                print(Nexp,pl,svdfac, 'not used')


plt.show()


