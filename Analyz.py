###   4 /07/1404    Friday
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
col=['r', 'g', 'k', 'b']  

Efb=np.zeros((4,4,2))

##############################################################################################################


smin=pow(10.0,-2.25)
epsi=0.0001;  
ML=np.array([-1.75, -1.5,-1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])##14
beta=np.array([0.5, 1.0, 1.5, 2.0])#power-index
cadence=np.array([15.16, 10.0])



nc=int(54)
nb=int(4)
nx=int(11)

xar=np.zeros((nx,8)) 
xar[:,0]=np.array([0.0,  0.8, 1.6, 2.4, 3.3, 4.0, 4.8, 5.6, 6.4, 7.2, 8.0])# Dl;   11
xar[:,1]=np.array([-2.15, -1.8, -1.45, -1.1 ,-0.65, -0.25, 0.2, 0.6, 0.9, 1.2, 1.4])# log(semi)
xar[:,2]=np.array([-1.25, -1.05, -0.85, -0.65, -0.45,-0.25, -0.05, 0.15, 0.35, 0.55, 0.75])# log(Dis(RE))
xar[:,3]=np.array([-1.1, -0.9, -0.7, -0.5, -0.3 ,-0.1, 0.1,  0.35 , 0.65, 1.2, 1.5])# log(rho*)  [-2.5, 1.5]
xar[:,4]=np.array([-3.5, -3.1, -2.7, -2.3, -1.9, -1.5, -1.1, -0.7, -0.3, 0.1, 0.5])#log(Pirel)
xar[:,5]=np.array([0.0, 0.07, 0.14, 0.21, 0.28, 0.35, 0.42, 0.49, 0.56, 0.63, 0.7])#q
xar[:,6]=np.array([-2.5, -2.3, -2.1, -1.9 , -1.7, -1.5, -1.3, -1.0, -0.7, -0.5, -0.15])#log10(tE)]
xar[:,7]=np.array([-1.75,-1.25, -1.0, -0.75, -0.25, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0 ])#log10(Mtot)


Effi=np.zeros((nx,2,8,nb,2))


for i in range(nb):  
    save1=open("./Tabletot{0:d}.txt".format(i),"w")              
    save1.close(); 

##########################################################################################################################

nam0=[r"$D_{\rm{l}}(\rm{kpc})$",r"$\log_{10}[s(\rm{AU})]$",r"$\log_{10}[d]$",r"$\log_{10}[\rho_{\star}]$",r"$\log_{10}[\pi_{\rm{rel}}(\rm{mas})]$", r"$q$", r"$\log_{10}[t_{\rm{E}}(\rm{days})]$", r"$\log_{10}[M_{\rm{tot}}(M_{\oplus})]$"]#*

  
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

for w in range(2): ### number of cadences 
    if(w==0):  addres="./cad15/"
    if(w==1):  addres="./cad10/"
    for j in range(4):  
        save1= open("./Tabletot{0:d}.txt".format(j),"a+")
        save1.write( str(round(cadence[w],2))+"\n")
        save1.close()
    for im in range(14):#Mass  
        f1=open(addres+"ParamLast{0:d}.txt".format(im),"r")
        nt= sum(1 for line in f1) 
        par =np.zeros((nt, nc+17)) 
        par[:nt,:]=np.loadtxt(addres+"ParamLast{0:d}.txt".format(im))
        parm=np.zeros((nt, nc+17,nb))
        num=np.zeros((nb));  
        d1 =np.zeros((nb));   
        d0 =np.zeros((nb)); 
        Fwi=np.zeros((nb));   
        Fin=np.zeros((nb));   
        Fcl=np.zeros((nb)); 
        ave=np.zeros((nc+17)) 
        efi=0.0
        ###########################################################################
        for i in range(nt): 
            ei,Massi,LMtot, strucl, Ml,Dl      =int(par[i,0]), int(par[i,1]), par[i,2], par[i,3], par[i,4], par[i,5]  
            vl, mul1, mul2, strucs,cl,mass, Ds =par[i,6], par[i,7], par[i,8], par[i,9], par[i,10],par[i,11],par[i,12]
            Tstar, Rstar,logl,typs,vs,mus1,mus2=par[i,13],par[i,14],par[i,15],par[i,16],par[i,17],par[i,18],par[i,19]
            q, dis, lperi, lsemi,tE,RE, mul    =par[i,20],par[i,21],par[i,22],par[i,23],par[i,24],par[i,25],par[i,26]
            Vt, u0, ksi, piE, tetE ,opd, ros   =par[i,27],par[i,28],par[i,29],par[i,30],par[i,31],par[i,32],par[i,33]
            Mab1,Mab4,Map1,Map4,magb1,magb4,bl1=par[i,34],par[i,35],par[i,36],par[i,37],par[i,38],par[i,39],par[i,40]
            fb,Nb1,Nb4,Ex1,Ex4,t0, ChiPL1      =par[i,41],par[i,42],par[i,43],par[i,44],par[i,45],par[i,46],par[i,47]
            ChiPL2, ChiR, ChiB,inc,config,nsem =par[i,48], par[i,49], par[i,50], par[i,51], int(par[i,52]), par[i,53]
            u0f, du01, du02, t0f, dt01, dt02   =par[i,54], par[i,55], par[i,56], par[i,57], par[i,58],par[i,59]
            tEf, dtE1, dtE2, fbf, dfb1, dfb2   =par[i,60], par[i,61], par[i,62], par[i,63], par[i,64],par[i,65]
            rosf, dros1, dros2, chif, ei2      =par[i,66], par[i,67], par[i,68], par[i,69],par[i,70]
            if(abs(LMtot-ML[im])>0.01 or abs(ChiR-ChiB)<90.0 or ei!=ei2 or Massi!=im or Dl>Ds or dis<0.0 or Ml<0.0): 
                print ("Error, LMtot,  chi2:  ",  LMtot, ML[im],   ChiR, ChiB, abs(ChiR-ChiB))
                input("Enter a number")
            
        
            lprl=np.log10(piE*tetE)
            Dchi=abs(chif-ChiR)
            semi=pow(10.0,lsemi);#[AU]    
            par[i,23]=pow(10.0 , par[i,23])## semai major axis
            par[i,30]=par[i,30]*par[i,31]##pirel
            N0= funin(Dl,           xar[:,0],nx)
            N1= funin(lsemi,        xar[:,1],nx)
            N2= funin(np.log10(dis),xar[:,2],nx)
            N3= funin(np.log10(ros),xar[:,3],nx)
            N4= funin(lprl,         xar[:,4],nx)
            N5= funin(q,            xar[:,5],nx)
            N6= funin(np.log10(tE), xar[:,6],nx) 
            N7= funin(LMtot,        xar[:,7],nx) 
            
            if(dis>100.0 and Dchi>100.0): 
                print(dis, Dchi, chif, ChiR,  ei, ei2, q, LMtot, w, semi, tE, tEf, u0, u0f, Massi)
                #input("Enter a number ")    
            
            for j in range(nb): 
                we=pow(float(semi/smin),-beta[j]+1.0)#weight function  
                d0[j]+=1.0*we
                Effi[N0,0,0,j,w]+=1.0*we
                Effi[N1,0,1,j,w]+=1.0*we
                Effi[N2,0,2,j,w]+=1.0*we
                Effi[N3,0,3,j,w]+=1.0*we
                Effi[N4,0,4,j,w]+=1.0*we
                Effi[N5,0,5,j,w]+=1.0*we
                Effi[N6,0,6,j,w]+=1.0*we
                Effi[N7,0,7,j,w]+=1.0*we
                if(Dchi>100.0 and dis<100.0):##detectable_events  
                    parm[int(num[j]),:,j]=par[i,:]*we
                    num[j]+=1
                    d1[ j]+=1.0*we;
                    if(  config==3):  Fwi[j] +=1.0*we;
                    elif(config==2):  Fin[j] +=1.0*we;
                    elif(config==1):  Fcl[j] +=1.0*we;
                    Effi[N0,1,0,j,w]+=1.0*we
                    Effi[N1,1,1,j,w]+=1.0*we
                    Effi[N2,1,2,j,w]+=1.0*we
                    Effi[N3,1,3,j,w]+=1.0*we
                    Effi[N4,1,4,j,w]+=1.0*we
                    Effi[N5,1,5,j,w]+=1.0*we
                    Effi[N6,1,6,j,w]+=1.0*we
                    Effi[N7,1,7,j,w]+=1.0*we
        #######################################################################        
        for j in range(nb):             
            save1=open("./Tabletot{0:d}.txt".format(j),"a+")              
            ss=    int(num[j])
            qq=    float(d1[j]*1.0)
            Fwi[j]=float(Fwi[j]*100.0/(qq+epsi))
            Fin[j]=float(Fin[j]*100.0/(qq+epsi))          
            Fcl[j]=float(Fcl[j]*100.0/(qq+epsi))
            efi=   float(qq*100.0/d0[j])
            for k in range(nc+17):   ave[k]=np.sum(parm[:ss,k,j])/qq
            test=np.array([ LMtot,np.log10(ave[23]),ave[12],ave[5],np.log10(ave[30]),ave[28],ave[20],ave[21],ave[33],ave[24],efi,
               Fcl[j],Fin[j],Fwi[j] ])
            np.savetxt(save1,test.reshape(-1,14),
            fmt='$%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.1f$:$%.1f$:$%.1f$')
            if(im==13):   save1.write("************************************************************\n")
            save1.close()
            
                       
            if(im==3):  Efb[0,j,w]=efi*0.01## 0.1 MEarth
            if(im==7):  Efb[1,j,w]=efi*0.01## 1.0 MEarth
            if(im==11): Efb[2,j,w]=efi*0.01## 10  MErath
            if(im==13): Efb[3,j,w]=efi*0.01## 100 MEarth
            
        print("MASS: ", im,  LMtot, nt, num[0], num[1], num[2] )
        print(d1[0], d1[1], d1[2], nt, Fwi,  Fin,  Fcl)
        print("************************************************************")    
        ###########################################################################
    
##################################################################################################################
nff=np.zeros((4,3))
Mass=np.array([0.1,  1,  10.0,   100.0])
nff[:,0]=np.array([17.9,  88.3, 349.0, 1250.0])*0.09#one per star
nff[:,1]=np.array([5.13,  25.2, 83.00,  298.0])*0.09##Log-uniform
nff[:,2]=np.array([10.3,  50.5, 103.0,   68.9])*0.09 ##Fiducial
MASSF= ['one-per-star', 'log-uniform', 'Fiducial']
cad=np.array([15.16, 10.0])

sav=open("./Tablefinal.txt","w")
for k in range(3):#Johnson mass function for FFPs
    sav.write( str(MASSF[k]) +"\n")
    for i in range(4):#mass 
        for j in range(2):## cadence 
            test=np.array([ Mass[i], cad[j],  nff[i,k]*Efb[i,0,j], nff[i,k]*Efb[i,1,j], nff[i,k]*Efb[i,2,j], nff[i,k]*Efb[i,3,j]   ])
            np.savetxt(sav,  test.reshape(-1,6) , fmt='$%.1f$ & $%.1f$ & $%.1f$  &  $%.1f$  &  $%.1f$ &  $%.1f$') 
    sav.write("*********************************************\n")             
sav.close()


##################################################################################################################
lines=['--', '-'] ## 15 and 10 minute cadences 
lwe= np.array([1.7,  1.2])
for i in range(8):
    plt.clf()
    plt.cla()
    fig=plt.figure(figsize=(8,6))
    for w in range(2): 
        for j in range(nb):
            Effi[:,1,i,j,w]=Effi[:,1,i,j,w]*100.0/(Effi[:,0,i,j,w]+epsi)
            plt.scatter(xar[:,i],Effi[:,1,i,j,w],marker= "o",facecolors=col[j],edgecolors=col[j] , s=29)
            plt.step(   xar[:,i],Effi[:,1,i,j,w],where='mid',c=col[j],ls=lines[w],lw=lwe[w])
    plt.xlabel(str(nam0[i]),fontsize=19)
    plt.ylabel(r"$\rm{Roman}~\rm{Detection}~\rm{Efficiency}[\%]$", fontsize=19)
    plt.xticks(fontsize=19, rotation=0)
    plt.yticks(fontsize=19, rotation=0)
    plt.grid("True")
    plt.grid(linestyle='dashed')  
    if(i==7):   
        plt.text(-1.7, 50.0, r"$\beta=0.5$", fontsize=18, color="r")
        plt.text(-1.7, 45.0, r"$\beta=1.0$", fontsize=18, color="g")
        plt.text(-1.7, 40.0, r"$\beta=1.5$", fontsize=18, color="k")
        plt.text(-1.7, 35.0, r"$\beta=2.0$", fontsize=18, color="b")    
    #if(i==1 ):
    #    plt.legend()
    #    plt.legend(loc='best',fancybox=True, fontsize=18, framealpha=0.9)
    fig=plt.gcf()
    fig.tight_layout()
    fig.savefig(addres+"Effi{0:d}.jpg".format(i),dpi=200)







