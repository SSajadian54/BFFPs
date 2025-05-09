import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.figure import Figure
from IPython.display import display, Math
import emcee
from numpy import conj
import pylab
from matplotlib import cm
from matplotlib import colors
cm=colors.ListedColormap(['purple', 'blue', 'darkgreen','yellowgreen', 'orange', 'red'])
from matplotlib import gridspec
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
cmap=plt.get_cmap('viridis')
mpl.rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams["font.size"] = 11.5
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Computer Modern Sans"]
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
mpl.rcParams['text.usetex'] = True
###############################################################################
epsi=0.0001;  
nc=53
save1=open("./Table.txt","w")
save1.close(); 

nx=int(8+2)
xar=np.zeros((nx,7)) 
xar[:,0]=np.array([0.0,   1.0,  2.0, 3.0, 4.0, 4.7, 5.3,  6.0, 7.0, 8.0])# Dl
xar[:,1]=np.array([-2.0, -1.7, -1.4, -1.1 ,-0.8, -0.5, 0.0, 1.0, 1.5, 2.5])# log(semi)
xar[:,2]=np.array([-1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.2, 0.4, 0.7, 1.0])# log(Dis(RE))
xar[:,3]=np.array([-1.7, -1.3, -0.9, -0.5, -0.1,0.1, 0.4, 0.7,1.2, 1.5])# log(rho*)
xar[:,4]=np.array([-6.0,-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.2, 0.4])#log(Pirel)
xar[:,5]=np.array([0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.88])#q
xar[:,6]=np.array([-2.0, -1.7, -1.4,-1.1, -0.8, -0.5, -0.2, 0.1, 0.35, 0.85])#log10(tE)

Effi=np.zeros((nx,2,7))

def funin(pr, arry, nn):  
    Ndl=-1
    if(pr<arry[0]): Ndl=0;
    elif(pr>arry[nn-1] or pr==arry[nn-1]): Ndl=nn-1
    else: 
        for i in range(nn-1): 
            if(float((arry[i]-pr)*(arry[i+1]-pr))<0.0 or pr==arry[i]): Ndl=i  
    if(Ndl<0 or Ndl>(nn-1)): 
        print("Big error, NDL:  ", Ndl, pr, arry)
        input("Enter a number ")        
    return(Ndl)        

###############################################################################
ML=np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -1.75, -1.25, -0.25, 0.25])##14
l=0
for im in range(14):#Mass  
    f1=open("./ParamLast{0:d}.txt".format(im),"r")
    nt= sum(1 for line in f1) 
    par =np.zeros((nt,nc+17)) 
    parm=np.zeros((nt,nc+17))
    par[:nt,:]=np.loadtxt("./ParamLast{0:d}.txt".format(im))
    
    print("\n\nlimits of pirel: ", np.log10(np.min(par[:nt,30]*par[:nt,31])),   np.log10(np.max(par[:nt,30]*par[:nt,31])) )
    
    dd=0;  Fwi=0.0; Fin=0.0;  Fcl=0.0; efi=0.0
    for i in range(nt):#MonteCarlo  
        ei,Massi,LMtot, strucl, Ml,Dl      =int(par[i,0]), int(par[i,1]), par[i,2], par[i,3], par[i,4], par[i,5]  
        vl, mul1, mul2, strucs,cl,mass, Ds =par[i,6], par[i,7], par[i,8], par[i,9], par[i,10],par[i,11],par[i,12]
        Tstar, Rstar,logl,typs,vs,mus1,mus2=par[i,13],par[i,14],par[i,15],par[i,16],par[i,17],par[i,18],par[i,19]
        q, dis, lperi, lsemi,tE,RE, mul    =par[i,20],par[i,21],par[i,22],par[i,23],par[i,24],par[i,25],par[i,26]
        Vt, u0, ksi, piE, tetE ,opd, ros   =par[i,27],par[i,28],par[i,29],par[i,30],par[i,31],par[i,32],par[i,33]
        Mab1,Mab4,Map1,Map4,magb1,magb4,bl1=par[i,34],par[i,35],par[i,36],par[i,37],par[i,38],par[i,39],par[i,40]
        fb,Nb1,Nb4,Ex1,Ex4,t0, ChiPL1      =par[i,41],par[i,42],par[i,43],par[i,44],par[i,45],par[i,46],par[i,47]
        ChiPL2, ChiR, ChiB,inc,config      =par[i,48], par[i,49], par[i,50], par[i,51], int(par[i,52])
        u0f, du01, du02, t0f, dt01, dt02   =par[i,53], par[i,54], par[i,55], par[i,56], par[i,57], par[i,58]
        tEf, dtE1, dtE2, fbf, dfb1, dfb2   =par[i,59], par[i,60], par[i,61], par[i,62], par[i,63], par[i,64]
        rosf, dros1, dros2, chif, ei2      =par[i,65], par[i,66], par[i,67], par[i,68], par[i,69]
        prl=piE*tetE
        if(abs(LMtot-ML[im])>0.01 or abs(ChiR-ChiB)<90.0 or ei!=ei2 or Massi!=im or Dl>Ds or dis<0.0 or Ml<0.0): 
            print ("Error, LMtot,  chi2:  ",  LMtot, ML[im],   ChiR, ChiB, abs(ChiR-ChiB) )
            input("Enter a number ")
            
        N0= funin(Dl,           xar[:,0],nx)
        N1= funin(lsemi,        xar[:,1],nx)
        N2= funin(np.log10(dis),xar[:,2],nx)
        N3= funin(np.log10(ros),xar[:,3],nx)
        N4= funin(np.log10(prl),xar[:,4],nx)
        N5= funin(q,            xar[:,5],nx)
        N6= funin(np.log10(tE), xar[:,6],nx)
        Effi[N0,0,0]+=1.0
        Effi[N1,0,1]+=1.0
        Effi[N2,0,2]+=1.0
        Effi[N3,0,3]+=1.0
        Effi[N4,0,4]+=1.0
        Effi[N5,0,5]+=1.0
        Effi[N6,0,6]+=1.0
        Dchi=abs(chif-ChiR)
        if(Dchi>100.0):##detectable_events  
            parm[dd,:]=par[i,:]
            dd+=1;
            if(  config==3): Fwi+=1.0;
            elif(config==2): Fin+=1.0;
            elif(config==1): Fcl+=1.0;
            Effi[N0,1,0]+=1.0
            Effi[N1,1,1]+=1.0
            Effi[N2,1,2]+=1.0
            Effi[N3,1,3]+=1.0
            Effi[N4,1,4]+=1.0
            Effi[N5,1,5]+=1.0
            Effi[N6,1,6]+=1.0
    Fwi=float(Fwi*100.0/(dd+epsi))
    Fin=float(Fin*100.0/(dd+epsi))          
    Fcl=float(Fcl*100.0/(dd+epsi))
    efi=float(dd*100.0/nt)
    test=np.array([ 
    LMtot,#log10(Mtotal) 
    np.log10(np.mean(np.power(10,parm[:dd,23]))),##log10[s] 
    np.mean(parm[:dd,12]),# Ds
    np.mean(parm[:dd, 5]), #Dl
    np.log10(np.mean(parm[:dd,30]*parm[:dd,31])),# log10[pi_rel]
    np.mean(parm[:dd,20]),#q 
    np.mean(parm[:dd,21]),#dis
    np.mean(parm[:dd,33]),#ros
    np.mean(parm[:dd,24]),#tE
    efi, Fcl, Fin, Fwi ]) ##13
    print("MASS: ", im,  LMtot)
    print("log[<s>],   <Ds>, <Dl>,  q,  tE, rhos,   efficiency,  Fclose:Finter:Fwide:", test, dd, nt, Fwi, Fin, Fcl )
    print("**********************************************************************")    
    save1=open("./Table.txt","a+")      
    np.savetxt(save1,test.reshape(-1,13),
    fmt='$%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.1f$:$%.1f$:$%.1f$')#13
    save1.close()
    ###########################################################################
    
nam0=[r"$D_{\rm{l}}(\rm{kpc})$",r"$\log_{10}[s(\rm{AU})]$",r"$\log_{10}[d]$",r"$\log_{10}[\rho_{\star}]$",r"$\log_{10}[\pi_{\rm{rel}}(\rm{mas})]$", r"$q$", r"$\log_{10}[t_{\rm{E}}(\rm{days})]$"]#7 
    
for i in range(7):
    Effi[:,1,i]=Effi[:,1,i]*100.0/(Effi[:,0,i]+epsi)
    plt.clf()
    plt.cla()
    fig=plt.figure(figsize=(8,6))
    plt.step(xar[:,i],Effi[:,1,i],where='mid',c='darkblue',ls='--',lw=2.2)
    plt.xlabel(str(nam0[i]),fontsize=19)
    plt.ylabel(r"$\rm{Detection}~\rm{Efficiency}[\%]$", fontsize=19)
    plt.xticks(fontsize=19, rotation=0)
    plt.yticks(fontsize=19, rotation=0)
    #plt.grid("True")
    #plt.grid(linestyle='dashed')    
    fig=plt.gcf()
    fig.tight_layout()
    fig.savefig("./Effi{0:d}.jpg".format(i),dpi=200)








