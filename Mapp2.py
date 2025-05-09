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
mpl.rcParams['xtick.labelsize']= 20
mpl.rcParams['ytick.labelsize']= 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams["font.size"] = 11.5
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Computer Modern Sans"]
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
mpl.rcParams['text.usetex'] = True
import VBMicrolensing
vbb = VBMicrolensing.VBMicrolensing()
vbb.RelTol = 1e-04
vbb.Tol=1e-04
vbb.LoadESPLTable("./ESPL.tbl"); 
vbb.a1=0.0


###############################################################################

Rsun=6.9634*np.power(10.0,8.0)###; ///solar radius [meter]
G=6.67384*np.power(10.,-11.0);##// in [m^3/s^2*kg].
Msun=1.98892*np.power(10.,30.0);## //in [kg].
Mearth=5.972*pow(10.0,24.0)##[kg]
logme=np.log10(Mearth); 
velocity=299792458.0;##//velosity of light  m/s
AU=1.495978707*np.power(10.0,11.0);
year=float(365.2422)
pc=float(30856775814913673.0)

###############################################################################

tt=9
nx=100 
ny=100
nz=100
dx=3.5/nx/1.0
dy=5.0/ny/1.0
sep= np.zeros((nx))
mas= np.zeros((ny))
mapp=np.zeros((nx, ny,10))
v=   np.zeros((tt))
xmin=-2.0##log10(Sep[AU])
ymin=-2.5##log10(Mt/Me)

###############################################################################

def tickfun(x, start, dd0, rho):
    return((start+x*dd0)*rho)
    
def funcq(mu,sig):
    fr=1.0
    fq=0.0
    while(fr>fq): 
        q =np.random.rand(1)
        fq=np.exp(-(q-mu)*(q-mu)/(2.0*sig*sig))
        fr=np.random.rand(1)
    return(q)    
     
def funcw(q): 
    dw  =0.0;  
    dclo=0.0
    dw  =np.power(1.0+pow(q,float(1.0/3.0)),1.5)/np.sqrt(1.0+q)
    min1=1000000000.0;  
    for w in range(500): 
        dc=float(0.6+w*(0.4/500.0));
        t3=abs(pow(dc,8.0)*27.0*q - (1.0+q)*(1.0+q)*pow(1.0 - dc*dc*dc*dc , 3.0))
        if(t3<min1):  
            min1=t3;  
            dclo=dc
    return(dw,dclo)  
    
def Acaus(dis, q, ros):  
    area=[];  
    Mag=[];
    if(dis>0.01 and dis<4.0 and  q>1.0e-7 and q<1.0 and ros<6.0): 
        caustics=vbb.Caustics(dis,q)
        for cau in caustics:
            xcau,ycau=np.array(cau[0]),np.array(cau[1])
            delx=abs(np.max(xcau)- np.min(xcau))
            dely=abs(np.max(ycau)- np.min(ycau))
            area.append(abs(delx*dely))
            Astar=abs(vbb.BinaryMag2(dis, q, np.mean(xcau), np.mean(ycau), ros))
            if(Astar>500.0 or Astar<1.0): 
                print("Error, Astar: ", Astar, dis, q, np.mean(xcau), np.mean(ycau), ros)
                #plt.plot(xcau,ycau, "ro")
                #plt.show()
                #input("Enter a number")
                #Astar=1.0+2.0/(ros*ros)    
            Mag.append(Astar)
        del(caustics);
    else: 
        area.append(0.0); 
        Astar=vbb.ESPLMag2(0.5,ros)
        Mag.append(Astar)    
        if(Astar>500.0 or Astar<1.0): 
            print("Error(2), Astar: ", Astar, dis, q ,  ros)
            #input("Enter a number")    
    return(np.mean(area), np.mean(Mag))   
         

def plotSMD(Map,ii, fone):
    plt.cla()
    plt.clf()
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111)
    plt.imshow(Map,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
    if(ii==2): 
        plt.plot(fone[:,1], fone[:,0], "ro", markersize=8)
    plt.clim()
    minn=np.min(Map)
    maxx=np.max(Map)
    step=float((maxx-minn)/(tt-1.0));
    for m in range(tt):
        v[m]=round(float(minn+m*step),1)
    cbar=plt.colorbar(orientation='vertical',shrink=0.9,pad=0.01,ticks=v)
    cbar.ax.tick_params(labelsize=19)
    plt.clim(v[0]-0.005*step,v[tt-1]+0.005*step)
    plt.xticks(fontsize=22, rotation=0)
    plt.yticks(fontsize=22, rotation=0)
    plt.title(str(tit[ii]), fontsize=22)
    plt.xlim(nx*0.0 , nx*1.0)
    plt.ylim(ny*0.0 , ny*1.0)
    ticc=np.array([nx*0.1, nx*0.3, nx*0.5, nx*0.7, nx*0.9 ])
    ax.set_xticks(ticc,labels=[round(j,1) for j in tickfun(ticc, xmin,dx, 1.0 ) ])
    ax.set_yticks(ticc,labels=[round(j,1) for j in tickfun(ticc, ymin,dy, 1.0 ) ])
    #ax.set_aspect('equal', adjustable='box')
    plt.xlabel(r"$\log_{10}[s(\rm{AU})]$",fontsize=22,labelpad=0.05)
    plt.ylabel(r"$\log_{10}[M_{\rm{tot}}(M_{\oplus})]$",fontsize=22,labelpad=0.05)
    fig=plt.gcf()
    fig.tight_layout()
    fig.savefig("Mapn{0:d}.jpg".format(ii),dpi=200)     

###############################################################################
nc=int(54+2)
fil3=open('./fraction.txt',"w"); 
fil3.close()
fil4=open('./limit.txt',"w"); 
fil4.close()
fil5=open('./limit2.txt',"w"); 
fil5.close()
f2=open("./Maps/param.dat","r")
nd= sum(1 for line in f2)
par=np.zeros((nd,nc))
par=np.loadtxt("./Maps/param.dat")
i=0
limit=np.zeros((nx,3))
tim= []

fone=np.zeros((nx,5))

for x in range(nx): ## total mass
    slim=[]; nintr=0
    ratio=np.zeros((ny))
    for j in range(ny): ## semi_major axis 
        frac=0.0;   fint=0.0;   tscal=0.0;  fwid=0.0
        pteav=0.0;  qav=0.0;    disav=0.0;  peaka=0.0
        rosav=0.0;  sizav=0.0;  num=0.0
        #######################################################################
        for k in range(100): 
            ei, LMtot,Lsem, strucl, Ml, Rl        =int(par[i,0]),par[i,1], par[i,2], par[i,3], par[i,4], par[i,5]  
            Dl, vl, mul1, mul2, strucs,cl,mass=par[i,6], par[i,7], par[i,8], par[i,9], par[i,10],par[i,11],par[i,12]
            Ds, Tstar, Rstar,logl,typs,vs,mus1=par[i,13],par[i,14],par[i,15],par[i,16],par[i,17],par[i,18],par[i,19]
            mus2, q, dis, lperi, lsemi,con, tE=par[i,20],par[i,21],par[i,22],par[i,23],par[i,24],par[i,25],par[i,26]
            RE, mul, Vt, u0, ksi, piE, tetE   =par[i,27],par[i,28],par[i,29],par[i,30],par[i,31],par[i,32],par[i,33]
            opd, ros,Mab1,Mab4,Map1,Map4,magb1=par[i,34],par[i,35],par[i,36],par[i,37],par[i,38],par[i,39],par[i,40]
            magb4,blend1,fb,Nb1,Nb4,Ex1,Ex4   =par[i,41],par[i,42],par[i,43],par[i,44],par[i,45],par[i,46],par[i,47]
            t0,ChiPL1,ChiPL2,ChiR,ChiB,inc,i1,j1=par[i,48], par[i,49], par[i,50], par[i,51], par[i,52], par[i,53], par[i,54],par[i,55]
            #print("i, i1, j1, con, ", i, i1, j1,k, con, Ml,  tE, mass,   q, dis, ros)
            if(i1!=j or j1!=x or q<0.0 or dis<0.0 or ros<0.0 or Ml<0.0 or tE<0.0 or con<1 or con>3 or mass<0.0 or 
            float(LMtot-float(-2.5+x*5.0/nx))>0.04 or float(Lsem-float(-2.0+j*3.5/ny))>0.03 ): 
                print("Error: ",i1, j1, q, dis, ros, Ml);
                print(LMtot, Lsem,  float(-2.5+x*5.0/nx), float(-2.0+j*3.5/ny)); 
                input("Enter a number ")
            i+=1
            areac,magm=Acaus(dis, q, ros)
            farea= abs(areac/(np.pi*ros*ros))
            sizav+=np.sqrt(areac)
            qav  +=q
            disav+=dis
            peaka+=magm 
            frac +=farea  
            rosav+=ros
            tscal+=tE*(ros+np.sqrt(areac))*24.0
            pteav+=pow(10.0,lperi)/(tE*(ros+np.sqrt(areac)))
            if(con==2): fint+=1.0
            if(con==3): fwid+=1.0
            num+=1.0;
        #######################################################################      
        test=np.array([j,x, float(fint*100.0/num), float(fwid*100.0/num),
        np.log10(float(frac/num)+0.00001), np.log10(float(tscal/num)), float(qav/num), np.log10(float(disav/num)),
        float(peaka/num),float(rosav/num),float(sizav/num), np.log10(float(pteav/num)) ]) 
        
        ratio[j]= abs(np.log10(float(frac/num)+1.0e-5))
        
        if(float(fint*100.0/num)>70.0):
            slim.append(Lsem)   
            nintr+=1
        #print("q dis,area, ros:  ", test, LMtot, Lsem)
        fil3=open('./fraction.txt',"a")
        np.savetxt(fil3,test.reshape(-1,12), fmt="%d  %d %.10f %.10f  %.10f  %.10f %.10f  %.10f  %.10f  %.10f  %.10f   %.10f")
        fil3.close()
    ################################################
    if( np.min(ratio)<0.1):    
        dds=int(np.argmin(ratio))    
        fone[x,0], fone[x,1], fone[x,2], fone[x,3], fone[x,4]=x, dds , LMtot , float(-2.0+dds*3.5/ny) , np.min(ratio)
        fil5=open('./limit2.txt',"a")
        np.savetxt(fil5,fone[x,:].reshape(-1,5), fmt="%d   %d   %.5f   %.5f    %.8f")
        fil5.close()
    ################################################    
    limit[x,0]=LMtot    
    if(nintr>1): limit[x,1],limit[x,2]= np.min(slim), np.max(slim)                      
    fil4=open('./limit.txt',"a")
    np.savetxt(fil4,limit[x,:].reshape(-1,3), fmt="%.4f   %.10f  %.10f")
    fil4.close()
    ################################################                       
f1=open("./fraction.txt","r")
ns= sum(1 for line in f1)   
test2=np.zeros((ns,12))
test2=np.loadtxt("./fraction.txt")              
k=0
for x in range(nx):#Mass_y
    for j in range(ny):#SemiMajor_x
        mapp[x,j,:]=test2[k,2:]
        k+=1
         
tit=[r"$\mathcal{F}_{\rm{i}}[\%]$",r"$\mathcal{F}_{\rm{w}}[\%]$",r"$\log_{10}[\overline{\Delta_{\rm{c}}\big/\pi\rho_{\star}^{2}}]$",r"$\log_{10}[\overline{\tau} (\rm{hrs})]$",r"$\overline{q}$",r"$\log_{10}[ \overline{d} ]$",r"$\overline{A_{\rm{max}}}$", r"$\overline{\rho_{\star}}$", r"$\overline{L_{\rm{c}}}$", r"$\log_{10}[\overline{P/\tau}]$"]        
                          
for i in range(10):                
    plotSMD(mapp[:,:,i],i, fone[:,:2])

          
            
            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
