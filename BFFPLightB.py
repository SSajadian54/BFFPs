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
import VBMicrolensing
vbb = VBMicrolensing.VBMicrolensing()
vbb.RelTol = 1e-04
vbb.Tol=1e-04
vbb.LoadESPLTable("./ESPL.tbl"); 
vbb.a1=0.0
labels=['u0', 't0', 'tE', 'fb', 'ros']
nam0=["count", "Massi", "Mtot(Mearth)", "strucL", "Ml(Mearth)", "Dl(Kpc)", "vl(km/s)", "mul1", "mul2", "strucS", "cl", "mass(Msun)", "Ds(kpc)", "Tstar(K)", "Rstar(Rsun)", "logl", "typeS", "vs(km/s)", "mus1", "mus2", "q", "dis", "log10(period)", "log10(Semi/AU)", "tE(days)", "RE/AU", "mul*30", "Vt(km/s)", "u0", "ksi(degree)", "piE", "tetE(mas)", "opd*1.0e6", "ros" ,"Mab1","Mab4","Map1","Map4","magb1","magb4","blend1","blend4","Nblend1","Nblend4", "Ext1","Ext4","t0", "ChiPL1","ChiPL2", "ChiReal", "ChiBaseline", "inclination", "Configuration"]


###############################################################################
sm=1.0e-50
Tobs=float(62.0)
epsi=float(0.006)
nx=3000
ny=3000
cau=np.zeros((ny, nx))
angle =np.linspace(0,2*np.pi,150 ) 
xsou = np.cos(angle) 
ysou = np.sin(angle)
ndim=5
num=5000
nf=14*num
nc=53
par=np.zeros((nf,nc))
parm=np.zeros((nc+ndim*3+2))
par=np.loadtxt("./paramf.dat")   
############################################################ 

def tickfun(x, start, dd0, rho):
    return((start+x)*dd0*rho)
    
#======================================        
    
def LensEq(xl, yl, xl1, yl1, xl2, yl2, mu1, mu2):
    x1=float(xl-xl1)
    y1=float(yl-yl1)
    x2=float(xl-xl2)
    y2=float(yl-yl2)
    d1=float(x1*x1+ y1*y1+1.0e-52)
    d2=float(x2*x2+ y2*y2+1.0e-52)
    xs=float(xl-mu1*x1/d1 - mu2*x2/d2)
    ys=float(yl-mu1*y1/d1 - mu2*y2/d2)
    z=complex(xl,yl);  z1=complex(xl1,yl1);    z2=complex(xl2,yl2)
    zb=conj(z);        z1b=conj(z1);           z2b=conj(z2);
    f= mu1/((zb-z1b)**2.0) + mu2/((zb-z2b)**2.0)
    ep= abs(np.sqrt((f.real)**2.0 + (f.imag)**2.0)-1.0)
    if(ep<epsi):  
        flag=1
    else:  
        flag=0    
    return(xs, ys, flag)
#======================================    
    
def likelihood(p, tim, Magni, erra, prt):
    lp=prior(p,prt)
    if(lp<0.0):
        return(-np.inf)
    return lnlike2(p, tim, Magni , erra)
#======================================    

def prior(p, prt): 
    u01, t01, tE1, fb1, rho1 =p[0],   p[1],   p[2],   p[3],   p[4]
    u00, t00, tE0, fb0, rho0 =prt[0], prt[1], prt[2], prt[3], prt[4]
    if(u01>sm and u01<1.0 and t01>0.0 and t01<=Tobs and tE1>0.0 and tE1<150.0 and fb1>sm and fb1<1.0 and rho1>sm and rho1<100.0 and abs(u00-u01)<0.4 and abs(t01-t00)<tE0 and abs(tE1-tE0)<tE0 and abs(fb1-fb0)<0.4 and abs(rho1-rho0)<float(5.0*rho0)): 
        return(0); 
    return(-1.0);     
#======================================    

def lnlike2(p, tim, Magni, erra):
    u01, t01, tE1, fb1, rho1=p[0], p[1], p[2], p[3],  p[4]
    vbb.a1=0.0;
    chii=0.0
    for ji in range(len(tim)):  
        u=np.sqrt((tim[ji]-t01)*(tim[ji]-t01)/tE1/tE1 + u01*u01); 
        if(u>float(35.0*rho1)):  As=(u*u+2.0)/np.sqrt(u*u*(u*u+4.0))
        else:                    As=vbb.ESPLMag2(u,rho1)
        As=fb1*As+1.0-fb1
        chii+=(As-Magni[ji])**2.0/(erra[ji]*erra[ji])
    return(-1.0*chii)
#======================================    

def mapcau(q, dis):
    xcau=[]; ycau=[]
    xl1=-dis*q/(1.0+q)
    xl2=   dis/(1.0+q)
    yl1=0.0 
    yl2=0.0
    mu1=1.0/(1.0+q)
    mu2=  q/(1.0+q)
    for i0 in range(nx):
        for j0 in range(ny): cau[j0,i0]=0.0
    for i1 in range(nx):
        for j1 in range(ny):
            xl=float(xmin+i1*dx)
            yl=float(ymin+j1*dy)
            xs,ys, flag=LensEq(xl , yl, xl1, yl1, xl2, yl2, mu1, mu2)   
            ix=int((xs-xmin)/dx)           
            iy=int((ys-ymin)/dy)
            if(ix>=0 and ix<nx and iy>0 and iy<ny): cau[iy,ix]+=1.0 
            if(flag>0): xcau.append(ix);  ycau.append(iy)    
    return(np.log10(cau+0.01), xcau, ycau)
###############################################################################
detec= np.zeros((nf,nc))
k=0
for i in range(nf):  
    if(abs(par[i,49]-par[i,50])>=100.0):
        detec[k,:]=par[i,:]
        k+=1     
###############################################################################

ini=0
i=-1+num

for im in range(1):#Mass_discrete_values
    im=1
    save2=open("./ParamLast{0:d}.txt".format(im),"a+")
    save2.close()
    for k in range(num):#MonteCarlo  
        i+=1
        ei,Massi,LMtot, strucl, Ml,Dl       =int(par[i,0]), par[i,1], par[i,2], par[i,3], par[i,4], par[i,5]  
        vl, mul1, mul2, strucs,cl,mass, Ds =par[i,6], par[i,7], par[i,8], par[i,9], par[i,10],par[i,11],par[i,12]
        Tstar, Rstar,logl,typs,vs,mus1,mus2=par[i,13],par[i,14],par[i,15],par[i,16],par[i,17],par[i,18],par[i,19]
        q, dis, lperi, lsemi,tE,RE, mul    =par[i,20],par[i,21],par[i,22],par[i,23],par[i,24],par[i,25],par[i,26]
        Vt, u0, ksi, piE, tetE ,opd, ros   =par[i,27],par[i,28],par[i,29],par[i,30],par[i,31],par[i,32],par[i,33]
        Mab1,Mab4,Map1,Map4,magb1,magb4,bl1=par[i,34],par[i,35],par[i,36],par[i,37],par[i,38],par[i,39],par[i,40]
        fb,Nb1,Nb4,Ex1,Ex4,t0, ChiPL1      =par[i,41],par[i,42],par[i,43],par[i,44],par[i,45],par[i,46],par[i,47]
        ChiPL2, ChiR, ChiB,inc,config      =par[i,48], par[i,49], par[i,50], par[i,51], par[i,52]
        parm[:nc]=par[i,:]   
        if(im!=Massi or abs(LMtot-float(-2.5+im*5.0/10))>0.01 or Ml<0.0 or Dl>Ds or q<0.0 or dis<0.0 or(ei-ini-1)!=i or fb>1.0 or t0>Tobs 
        or t0<0.0 or tE<0.0 or tE>120.0 or ChiB<0.0 or ChiR<0.0 or ChiPL1<0.0 or ChiPL2<0.0  or config<1 or config>3 or LMtot!=-2.0): 
            print("Error:  ", im, Massi, LMtot,   float(-2.5+im*5.0/10.0),   Ml, Dl, Ds, q, dis, ei, fb, t0, Tobs, tE)
            print("Chis: ",  ChiB, ChiR, ChiPL1, ChiPL2, config)
            input("Enter a number ")
        if(ros<1.0):siz=2.5
        else:       siz=2.5*ros
        xmin, xmax=-1.0*siz, 1.0*siz
        ymin, ymax=-1.0*siz, 1.0*siz
        dx=float(xmax-xmin)/nx
        dy=float(ymax-ymin)/ny
        DCHIL= abs(ChiR-ChiB)
        Mtot=pow(10.0,LMtot)##Ml*(1.0+q)
        caus=vbb.Caustics(dis,q)
        print("****************************************************")
        print("Counter, ChiR, ChiB, ChiPL1, ChiPL2:   ",  ei, ChiR, ChiB, ChiPL1, ChiPL2  )
        print("t0,  u0,  tE:       ",  t0, u0, tE)
        print("Orbit_parameters:      ", lsemi, q, dis, inc, ros)
        print("Lensing detectability:  ",   DCHIL)
        print("****************************************************")
        if(DCHIL>=100.0): flagd=1
        else:             flagd=0    
        f1=open('./fil/dat_{0:d}.dat'.format(ei))
        f2=open('./fil/mod_{0:d}.dat'.format(ei))
        nd=int(len(f1.readlines()))
        nm=int(len(f2.readlines())) 
        #######################################################################             
        if(flagd>0 and nd>2 and nm>10 and ei>6967):
            Nwalkers=int(60-im)
            nstep=   int(8000- im*200)
            p0=np.zeros((Nwalkers,ndim))
            dat=np.zeros((nd,3))
            mod=np.zeros((nm,6))
            dat=np.loadtxt('./fil/dat_{0:d}.dat'.format(ei)) 
            mod=np.loadtxt('./fil/mod_{0:d}.dat'.format(ei))      
            tim=  np.zeros((nd)); 
            Magni=np.zeros((nd));   
            erra =np.zeros((nd));
            tim=  dat[:,0];    
            Magni=dat[:,1];   
            erra= dat[:,2];         
            p0[:,0]=np.abs(np.random.normal(u0,  0.5,    Nwalkers))#u0
            p0[:,1]=np.abs(np.random.normal(t0,  0.5*tE, Nwalkers))#t0
            p0[:,2]=np.abs(np.random.normal(tE,  0.5*tE, Nwalkers))#tE
            p0[:,3]=np.abs(np.random.normal(fb,  0.5,    Nwalkers))#fb
            p0[:,4]=np.abs(np.random.normal(ros, ros,    Nwalkers))#ros
            prt=np.array([u0, t0, tE, fb, ros])
            sampler=emcee.EnsembleSampler(Nwalkers,ndim,likelihood,args=(tim,Magni,erra,prt))
            sampler.run_mcmc(p0, nstep, progress=True)
            print("**END OF MCMC *************************** ")
            Chain=sampler.get_chain(flat=True)
            print(Chain.shape)
            pb=np.zeros((ndim,3))
            for j in range(ndim):
                mcmc=np.percentile(Chain[:,j], [16, 50, 84])
                pb[j,0]= mcmc[1]
                pb[j,1:] = np.diff(mcmc)
                print(str(labels[j]), pb[j,0],  pb[j,1], pb[j,2])
            chi2s=sampler.get_log_prob(flat=True)
            tt=int(np.argmin(-1.0*chi2s))
            Chibf=abs(lnlike2(Chain[tt,:],tim, Magni, erra))
            ffd=  abs(chi2s[tt])
            print("Chi2s, argmin,  param_min, chis: ",chi2s, tt, Chain[tt,:], ffd, Chibf)
            if(abs(Chibf-ffd)>0.1):  
                print("BIG ERROR: chi2s:  ",  Chibf, ffd)
                print("Best-fitted parameters:  ",  Chain[tt, :])
                print("Initial parameters:  ", prt)
                input("Enter a number ")
            u0f, t0f, tEf, fbf, rosf=Chain[tt,0], Chain[tt,1], Chain[tt,2], Chain[tt,3], Chain[tt,4]
            du0=0.5*(abs(pb[0,1])+abs(pb[0,2]));     
            dt0=0.5*(abs(pb[1,1])+abs(pb[1,2]));
            dtE=0.5*(abs(pb[2,1])+abs(pb[2,2]));     
            dfb=0.5*(abs(pb[3,1])+abs(pb[3,2]));
            dro=0.5*(abs(pb[4,1])+abs(pb[4,2]));
            parm[nc:]=np.array([u0f,pb[0,1],pb[0,2],t0f,pb[1,1],pb[1,2],tEf,pb[2,1],pb[2,2],fbf,pb[3,1],pb[3,2],rosf,pb[4,1],pb[4,2],Chibf,ei])
            save2=open("./ParamLast{0:d}.txt".format(im),"a")
            np.savetxt(save2,parm.reshape(-1,nc+17),fmt='%d  %d   %.5f  %d  %.8f  %.7f  %.5f  %.5f  %.5f  %d  %d  %.7f   %.7f  %.7f  %.7f  %.4f  %.2f   %.4f  %.4f  %.4f  %.5f  %.8f  %.5f   %.7f %.8f  %.9f  %.7f   %.8f %.6f  %.5f  %.5f  %.9f  %.6f  %.6f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f %.5f %.2f  %.2f  %.4f  %.4f  %.6f  %.5f  %.5f  %.5f  %.5f  %.5f  %d  %.6f  %.6f  %.6f %.6f  %.6f  %.6f %.6f  %.6f %.6f %.6f  %.6f  %.6f  %.8f  %.8f  %.8f  %.5f  %d')
            save2.close()
            mfit=np.zeros((nm))
            mfda=np.zeros((nd))
            u2=np.sqrt(u0f**2.0+ (mod[:,0]-t0f)**2.0/tEf/tEf)
            u3=np.sqrt(u0f**2.0+ (dat[:,0]-t0f)**2.0/tEf/tEf)
            for j in range(nm):  
                if(u2[j]>float(35.0*rosf)): mfit[j]=float(u2[j]*u2[j]+2.0)/np.sqrt(u2[j]*u2[j]*(u2[j]*u2[j]+4.0))
                else:                       mfit[j]=vbb.ESPLMag2(u2[j],rosf)
            for j in range(nd): 
                if(u3[j]>float(35.0*rosf)): mfda[j]=float(u3[j]*u3[j]+2.0)/np.sqrt(u3[j]*u3[j]*(u3[j]*u3[j]+4.0))
                else:                       mfda[j]=vbb.ESPLMag2(u3[j],rosf)
            mfit=fbf*mfit+1.0-fbf
            mfda=fbf*mfda+1.0-fbf
            ###################################################################                 
            ymin0= np.min(np.concatenate((mod[:,3],dat[:,1],mfit),axis=0))
            ymax0= np.max(np.concatenate((mod[:,3],dat[:,1],mfit),axis=0))
            xmin0= np.min(mod[:,0])#np.concatenate((mod[:,0],dat[:,0]),axis=0))
            xmax0= np.max(mod[:,0])##np.concatenate((mod[:,0],dat[:,0]),axis=0))
            plt.clf()
            plt.cla()
            fig=plt.figure(figsize=(8,6))
            spec3=gridspec.GridSpec(nrows=3, ncols=1, figure=fig)
            ax1= fig.add_subplot(spec3[:2,0])
            ax1.plot(mod[:,0],mfit,'g-.' ,label=r"$\rm{Best}-\rm{fit}~\rm{ESPL}$",lw=1.5)
            ax1.plot(mod[:,0],mod[:,3],'k-', label=r"$\rm{Real}~\rm{ESBL}$",lw=1.5)
            ax1.errorbar(dat[:,0],dat[:,1],yerr=dat[:,2],fmt=".",markersize=8.0,color='m',ecolor='gray',elinewidth=0.5, capsize=0)
            ax1.set_ylabel(r"$\rm{Magnification}$",fontsize=18)
            #ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.xticks(fontsize=17, rotation=0)
            plt.yticks(fontsize=17, rotation=0)
            ax1.set_title(
            r"$\log_{10}[M_{\rm{tot}}(M_{\oplus})]=$"+str(round(LMtot,2))+
            r"$,~s(\rm{AU})=$"+str(round(pow(10.0,lsemi),2))+
            r"$,~\rm{q}=$"+str(round(q,2))+
            r"$,~\log_{10}[d]=$"+str(round(np.log10(dis),2))+
            r"$,~u_{0}=$"+str(round(u0,2))+
            r"$,~t_{\rm{E}}\rm{(days)}=$"+str(round(tE,2))+
            "\n"+r"$\rho_{\star}=$"+str(round(ros,2))+
            r"$,~u_{0,f}=$"+str(round(u0f,2))+r"$\pm$"+str(round(du0,1))+
            r"$,~t_{\rm{E},f}\rm{(days)}=$"+str(round(tEf,2))+r"$\pm$"+str(round(dtE,2))+
            r"$,~\rho_{\star,f}=$"+str(round(rosf,2))+r"$\pm$"+str(round(dro,1))+
            r"$,~\Delta\chi^{2}_{\rm{B}}=$"+str(int(abs(Chibf-ChiR))),fontsize=14,color='k')
            #plt.text(np.min(mod[:,0])+1.0,ymax-(ymax-ymin)/2.0,str(sta), color=col , fontsize=13)
            ax1.set_ylim([ymin0-0.02*(ymax0-ymin0),ymax0+0.02*(ymax0-ymin0)])
            ax1.set_xlim([xmin0,xmax0])
            plt.setp(ax1.get_xticklabels(), visible=False)
            #ax1.grid("True")
            #ax1.grid(linestyle='dashed')
            plt.legend()
            plt.legend(loc=1,fancybox=True, shadow=True, fontsize=12, framealpha=0.8)
            ###################################################################
            '''
            left, bottom, width, height=[0.08, 0.43, 0.35, 0.35]
            ax4 = fig.add_axes([left, bottom, width, height])
            ax4.imshow(cau,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
            ax4.scatter(xcau, ycau, s=9.0,marker=".", color="y")
            for caue in caus:
                caue[0]=[val/dx+nx/2 for val in caue[0]]
                caue[1]=[val/dy+ny/2 for val in caue[1]]
                plt.plot(caue[0], caue[1], 'y-', lw=3.0)
            ax4.plot(mod[:,4]/dx+nx/2,mod[:,5]/dx+ny/2, "b--", lw=1.0)
            ax4.scatter((xsou*ros+mod[int(nm/2),4])/dx+nx/2, (ysou*ros+mod[int(nm/2),5])/dy+ny/2, s=9.0,marker=".", color="r")
            ax4.set_xlim(0.15*nx, nx*0.85)
            ax4.set_ylim(0.15*ny, ny*0.85)
            ticc=np.array([nx*0.2, nx*0.35, nx*0.5, nx*0.65, nx*0.8 ])
            ax4.set_xticks(ticc,labels=[round(j,1) for j in tickfun(ticc, -nx/2, dx,1.0 ) ])
            ax4.set_yticks(ticc,labels=[round(j,1) for j in tickfun(ticc, -ny/2, dy,1.0 ) ])
            ax4.set_aspect('equal', adjustable='box')
            ax4.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax4.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.axis('off')
            '''
            left, bottom, width, height=[0.1, 0.5, 0.33, 0.33]
            ax4 = fig.add_axes([left, bottom, width, height])
            #ax4.imshow(cau,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
            #ax4.scatter(xcau, ycau, s=9.0,marker=".", color="y")
            for caue in caus:
                #caue[0]=[val/dx+nx/2 for val in caue[0]]
                #caue[1]=[val/dy+ny/2 for val in caue[1]]
                plt.plot(caue[0], caue[1], 'c-', lw=2.0)
            ax4.plot(mod[:,4],mod[:,5], "b--", lw=1.0)
            ax4.scatter(xsou*ros+mod[int(nm/2),4], ysou*ros+mod[int(nm/2),5], s=9.0,marker=".", color="r")
            #ax4.set_xlim(0.15*nx, nx*0.85)
            #ax4.set_ylim(0.15*ny, ny*0.85)
            #ticc=np.array([nx*0.2, nx*0.35, nx*0.5, nx*0.65, nx*0.8 ])
            #ax4.set_xticks(ticc,labels=[round(j,1) for j in tickfun(ticc, -nx/2, dx,1.0 ) ])
            #ax4.set_yticks(ticc,labels=[round(j,1) for j in tickfun(ticc, -ny/2, dy,1.0 ) ])
            ax4.set_aspect('equal', adjustable='box')
            ax4.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax4.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.axis('off')
            ###################################################################
            ymin1= np.min(np.concatenate((mod[:,3]-mfit,dat[:,1]-mfda),axis=0))
            ymax1= np.max(np.concatenate((mod[:,3]-mfit,dat[:,1]-mfda),axis=0))
            ax2= fig.add_subplot(spec3[2,0],sharex=ax1) 
            ax2.errorbar(dat[:,0],dat[:,1]-mfda,yerr=dat[:,2],fmt=".",markersize=10.8,color='m',ecolor='gray',elinewidth=0.8, capsize=0) 
            ax2.plot(mod[:,0],    mod[:,3]-mfit ,"k-", lw=1.5)
            plt.xticks(fontsize=17, rotation=0)
            plt.yticks(fontsize=17, rotation=0)
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax2.set_xlabel(r"$\rm{time}(\rm{days})$",fontsize=17, labelpad=0.1)
            ax2.set_ylabel(r"$\Delta A$",  fontsize=17,labelpad=0.1)
            ax2.set_ylim([ ymin1-0.02*(ymax1-ymin1) , ymax1+0.02*(ymax1-ymin1) ])
            ax2.set_xlim([xmin0,xmax0])
            plt.subplots_adjust()
            fig=plt.gcf()
            fig.savefig("./fig/LC{0:d}_{1:d}.jpg".format(int(im), int(k+0)), dpi=200)
            ###################################################################




    
    

