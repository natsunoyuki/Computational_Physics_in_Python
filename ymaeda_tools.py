from __future__ import division
from scipy import * 
from scipy import linalg
from scipy import signal
import matplotlib.pyplot as plt
import struct #for dealing with binary data
import time

#python version of functions in ymaeda_tools.

#test directory:
main_dir='/home/yumi/kirishima_invert/inversion_results_dump_new/inversion_result_residuals117/t60_p100/x-10900y-121100z1000/'
#snapshot_dir="/media/yumi/INVERSION/SMN_EW_SMALL/PML/snapshot"
snapshot_dir="/media/yumi/INVERSION/GFDIR1/SMN_EW_SMALL/PML/snapshot"
#main_dir='/Volumes/Verbatim/kirishima_invert_reduce/inversion_results_dump_new/inversion_result_residuals117/t60_p100/x-10900y-121100z1000/'
data_obs_dir='/home/yumi/kirishima_invert/inversion_results_dump_new/inversion_result_residuals117/data_obs/inv_data_dir_supershort_filtered/'
data_obs_spectrum_dir='/home/yumi/kirishima_invert/inversion_results_dump_new/inversion_result_residuals117/data_obs_spectrum/inv_data_dir_supershort_filtered/'
stfun_dir='/home/yumi/kirishima_invert/inversion_results_dump_new/inversion_result_residuals117/'

df=0.002441406 #frequency step size used by YMAEDA_TOOLS
f=arange(0,df*2049,df) #frequency half space
F=arange(0,df*4096,df) #frequency full space

#example function calls:
"""
mt=read_Mseq1(main_dir)
mest=winv_lstsq(main_dir,w=0)
m=exifft_timeshift(mest,1000); mr=real(m)/0.1
plt.plot(mt);plt.plot(mr);plt.show()
"""

#DATA LOADING FUNCTIONS########################################################

def read_snapshot(snapshot_dir='source.Fx.t3.0000.3db'):
    """
    reads the PML snapshots generated by the FDM code in YMAEDA_TOOLS
    file name: source.Fx.t0.0000.3db 
    First 12 bytes: Binary expressions for N[0], N[1], N[2].
    Next 24 bytes: Binary expressions for x0[0], x0[1], x0[2].
    Next 24 bytes: Binary expressions for dx[0], dx[1], dx[2].
    Following each 8 bytes: Binary expressions for member value[index[i][j][k]].
    The most out loop is that with regard to "x",
    the intermediate is "y" and the most inner loop is "z".
    Separators such as tab do not appear.
    """
    #personal note: As this function relies on loading the entire grid
    #of data, processing can be very slow.
    with open(snapshot_dir,mode='rb') as File:
        fileContent=File.read()
        File.close()
        BYTELEN=8
        N=struct.unpack('iii',fileContent[0:12])
        x0=struct.unpack('ddd',fileContent[12:36])
        dx=struct.unpack('ddd',fileContent[36:60])
        data_length=int((len(fileContent)-60)/BYTELEN)
        assert data_length==N[0]*N[1]*N[2]
        GT=zeros(data_length)
        GT3D=zeros((N[2],N[1],N[0])) #python 3D indexing goes from z:y:x
        counter=0
        INDICES=[]
        for x in range(N[0]):
            for y in range(N[1]):
                for z in range(N[2]):
                    L=60+BYTELEN*counter
                    R=60+BYTELEN*counter+BYTELEN
                    GT[counter]=struct.unpack('d',fileContent[L:R])[0]
                    GT3D[z,y,x]=GT[counter]
                    INDICES.append([z,y,x])
                    counter=counter+1
    return GT,GT3D

def read_snapshot_fast(snapshot_dir='source.Fx.t3.0000.3db'):
    """
    reads the PML snapshots generated by the FDM code in YMAEDA_TOOLS
    file name: source.Fx.t0.0000.3db 
    First 12 bytes: Binary expressions for N[0], N[1], N[2].
    Next 24 bytes: Binary expressions for x0[0], x0[1], x0[2].
    Next 24 bytes: Binary expressions for dx[0], dx[1], dx[2].
    Following each 8 bytes: Binary expressions for member value[index[i][j][k]].
    The most out loop is that with regard to "x",
    the intermediate is "y" and the most inner loop is "z".
    Separators such as tab do not appear.
    """
    #unlike the previous function this one outputs the data as a 1D array
    #to speed up load and output times
    with open(snapshot_dir,mode='rb') as File:
        fileContent=File.read()
        File.close()
        BYTELEN=8
        N=struct.unpack('iii',fileContent[0:12])
        x0=struct.unpack('ddd',fileContent[12:36])
        dx=struct.unpack('ddd',fileContent[36:60])
        data_length=int((len(fileContent)-60)/BYTELEN)
        assert data_length==N[0]*N[1]*N[2]
        GT=zeros(data_length)
        for counter in range(data_length):
            L=60+BYTELEN*counter
            R=60+BYTELEN*counter+BYTELEN
            GT[counter]=struct.unpack('d',fileContent[L:R])[0]
    return GT,N 

def index_convert13(r,Nx,Ny,Nz):
    #converts the 1D index of GT to the 3D index of GT3D
    #the outer most loop: x
    #next inner loop: y
    #inner most loop: z
    #therefore z is the most folded coordinate followed by y, x
    idz=mod(r,Nz) #most folded coordinate
    idy=mod(r//Nz,Ny) #next folded coordinate
    idx=r//Nz//Ny #least folded coordinate
    #output is in python index format of z:y:x
    return idz,idy,idx

def index_convert31(idx,idy,idz,Nx,Ny,Nz):
    #converts the 3D index of GT3D to the 1D index of GT
    r=idx*Ny*Nz+idy*Nz+idz
    return r
    
def read_snapshot_loc(snapshot_dir,direction='z',X=0,Y=0,Z=0,t0=0.0,t1=12.0,dt=0.1):
    #reads the snapshot over time for a particular location given by 
    #location indices X, Y, Z over a time range of t0:t1 with dt.
    #this code reads and loads the entire 3D matrix and is therefore slow.
    """
    t,gx=read_snapshot_loc(snapshot_dir="/media/yumi/INVERSION/SMN_EW_SMALL/PML/snapshot",direction='x',X=idx,Y=idy,Z=idz);
    t,gy=read_snapshot_loc(snapshot_dir="/media/yumi/INVERSION/SMN_EW_SMALL/PML/snapshot",direction='y',X=idx,Y=idy,Z=idz);
    t,gz=read_snapshot_loc(snapshot_dir="/media/yumi/INVERSION/SMN_EW_SMALL/PML/snapshot",direction='z',X=idx,Y=idy,Z=idz);
    """
    t=arange(t0,t1+dt,dt)
    N=int((t1-t0)/dt)+1
    gt=zeros(N)
    for i in range(N):
        #snapshot_file=snapshot_dir+"/"+"source.F"+direction+".t"+str(t0+dt*i)+"000.3db"
        TIME_ZEROS=format(t0+dt*i,'0.4f')
        #TIME_ZEROS=str(t0+dt*i)[:len(str(dt))]
        #TIME_ZEROS=TIME_ZEROS+'0'*(4-(len(TIME_ZEROS)-2))
        snapshot_file=snapshot_dir+"/source.F"+direction+".t"+TIME_ZEROS+".3db"
        GT,GT3D=read_snapshot(snapshot_file)
        gt[i]=GT3D[Z,Y,X]
    return t,gt

def read_snapshot_loc_fast(snapshot_dir,direction='z',X=0,Y=0,Z=0,t0=0.0,t1=12.0,dt=0.1):
    #reads the snapshot over time for a particular location given by 
    #location indices X, Y, Z over a time range of t0:t1 with dt.
    #Faster code that deals with only 1D array structures
    """
    t,gx=read_snapshot_loc_fast(snapshot_dir="/media/yumi/INVERSION/SMN_EW_SMALL/PML/snapshot",direction='x',X=idx,Y=idy,Z=idz);
    t,gy=read_snapshot_loc_fast(snapshot_dir="/media/yumi/INVERSION/SMN_EW_SMALL/PML/snapshot",direction='y',X=idx,Y=idy,Z=idz);
    t,gz=read_snapshot_loc_fast(snapshot_dir="/media/yumi/INVERSION/SMN_EW_SMALL/PML/snapshot",direction='z',X=idx,Y=idy,Z=idz);
    """
    t=arange(t0,t1+dt,dt)
    N=int((t1-t0)/dt)+1
    gt=zeros(N)
    for i in range(N):
        #snapshot_file=snapshot_dir+"/"+"source.F"+direction+".t"+str(t0+dt*i)+"000.3db"
        TIME_ZEROS=format(t0+dt*i,'0.4f')
        #TIME_ZEROS=str(t0+dt*i)[:len(str(dt))]
        #TIME_ZEROS=TIME_ZEROS+'0'*(4-(len(TIME_ZEROS)-2))
        snapshot_file=snapshot_dir+"/source.F"+direction+".t"+TIME_ZEROS+".3db"
        GT,N0=read_snapshot_fast(snapshot_file)
        gt[i]=GT[index_convert31(X,Y,Z,N0[0],N0[1],N0[2])]
    return t,gt

def read_snapshot_all_fast(snapshot_dir,direction='z',t0=0.0,t1=12.0,dt=0.1):
    #this function reads and loads the entire snapshot over the specified
    #time range for a particular direction. THIS WILL CONSUME PLENTY OF 
    #MEMORY and should not be used lightly on slower computers.
    t=arange(t0,t1+dt,dt)
    N=int((t1-t0)/dt)+1
    G_ALL=[]
    N_ALL=[]
    for i in range(N):
        TIME_ZEROS=format(t0+dt*i,'0.4f')
        snapshot_file=snapshot_dir+"/source.F"+direction+".t"+TIME_ZEROS+".3db"
        GT,N0=read_snapshot_fast(snapshot_file)
        G_ALL.append(GT)
        N_ALL.append(N0)
    return G_ALL,N_ALL

def extract_snapshot_loc_fast(G_ALL,X=0,Y=0,Z=0):
    #extracts a time series snapshot at a particular location from the output
    #of read_snapshot_all_fast()
    return 0

def read_snapshot_loc3D(snapshot_dir,X=0,Y=0,Z=0,t0=0.0,t1=12.0,dt=0.1):     
    N=int((t1-t0)/dt)+1
    g=zeros((N,3))
    i=0
    j=['x','y','z']
    for k in j:
        t,g[:,i]=read_snapshot_loc(snapshot_dir,k,X,Y,Z,t0,t1,dt)
        i=i+1
    return t,g

def read_snapshot_loc3D_fast(snapshot_dir,X=0,Y=0,Z=0,t0=0.0,t1=12.0,dt=0.1):     
    N=int((t1-t0)/dt)+1
    g=zeros((N,3))
    i=0
    j=['x','y','z']
    for k in j:
        t,g[:,i]=read_snapshot_loc_fast(snapshot_dir,k,X,Y,Z,t0,t1,dt)
        i=i+1
    return t,g

def read_snapshot_params(snapshot_file='source.Fx.t3.0000.3db'):
    #returns only the various parameters of a snapshot.3db file
    #without outputting any data
    #N,x0,dx=read_snapshot_params(snapshot_dir='/media/yumi/INVERSION/SMN_EW_SMALL/PML/snapshot/source.Fx.t3.0000.3db')
    with open(snapshot_file,mode='rb') as File:
        fileContent=File.read()
        File.close()
        BYTELEN=8
        N=struct.unpack('iii',fileContent[0:12])
        x0=struct.unpack('ddd',fileContent[12:36])
        dx=struct.unpack('ddd',fileContent[36:60])
        data_length=int((len(fileContent)-60)/BYTELEN)
        assert data_length==N[0]*N[1]*N[2]
    return N,x0,dx

def snapshot_XYZ(N,x0,dx):
    #creates the X, Y and Z axis arrays from the output parameters from
    #read_snapshot_params()
    #X=arange(x0[0],x0[0]+dx[0]*N[0],dx[0])
    #Y=arange(x0[1],x0[1]+dx[1]*N[1],dx[1])
    #Z=arange(x0[2],x0[2]+dx[2]*N[2],dx[2])
    X=array([x0[0]+dx[0]*i for i in range(N[0])])
    Y=array([x0[1]+dx[1]*i for i in range(N[1])])
    Z=array([x0[2]+dx[2]*i for i in range(N[2])])
    return X,Y,Z   

def snapshot_stnloc(N,x0,dx,X_STN,Y_STN,Z_STN):
    """
    from the parameters of the snapshot.3db file calculate the nearest
    grid location for a specified station.
    If the specified location is outside the grid, the grid point nearest 
    to the specified location is returned and a warning is given.
    SMN: -11175, -119878, 1317
    SMW: -12295, -120893, 1110
    LP:  -10900, -121100, 1000
    """
    X,Y,Z=snapshot_XYZ(N,x0,dx)
    idx=(abs(X-X_STN)).argmin()
    idy=(abs(Y-Y_STN)).argmin()
    idz=(abs(Z-Z_STN)).argmin()
    if not(X.min()<=X_STN<=X.max()):
        print('Warning! X out of range! Returning nearest grid value...')
    if not(Y.min()<=Y_STN<=Y.max()):
        print('Warning! Y out of range! Returning nearest grid value...')
    if not(Z.min()<=Z_STN<=Z.max()):
        print('Warning! Z out of range! Returning nearest grid value...')    
    return idx,idy,idz

def read_Mseq1(main_dir):
    #reads the output M.seq1 results from YMAEDA_TOOLS in time domain
    main_dir=main_dir+'model/M.seq1'
    synm=array([])

    f=open(main_dir,'r')
    B=f.read().splitlines() #remove newline characters
    f.close()
    Size=len(B)-4
    t0=float(B[1][3:]) #initial time
    dt=float(B[2][3:]) #time step
    
    synwave_buffer=array([])
    for i in range(Size):
        synwave_buffer=hstack([synwave_buffer,float(B[i+4])])
    synm=synwave_buffer[0:Size] #bug makes synwave_buffer Size+1, remove extra line
    return synm

def read_stfunseq2(stfun_dir):
    #reads the zero padded source time function output by YMAEDA_TOOLS. 
    #some how, making fourier transform of this time series does not result
    #in the fourier transform obtained through read_stfunimseq2(stfun_dir)...
    #a,b=read_stfunseq2(stfun_dir)
    filename=stfun_dir+"stfun.seq2"
    f=open(filename,'r')
    B=f.read().splitlines()
    f.close()    
    Size=len(B)-4
    t0=float(B[1][3:]) #initial time
    dt=float(B[2][3:]) #time step
    B=B[4:]
    t=zeros(Size)
    x=zeros(Size)
    for i in range(Size):
        B[i]=B[i].split('\t')
        t[i]=float(B[i][0])
        x[i]=float(B[i][1])
    return t,x

def read_stfunimseq2(stfun_dir):
    #reads the fourier spectrum of the source time function output by
    #YMAEDA_TOOLS.
    #a,b=read_stfunimseq2(stfun_dir)
    filename=stfun_dir+"stfun_spectrum.imseq2"
    f=open(filename,'r')
    B=f.read().splitlines()
    f.close()    
    Size=len(B)-4
    t0=float(B[1][3:]) #initial time
    dt=float(B[2][3:]) #time step
    B=B[4:]
    t=zeros(Size)
    x=zeros(Size)
    y=zeros(Size)
    for i in range(Size):
        B[i]=B[i].split('\t')
        t[i]=float(B[i][0])
        x[i]=float(B[i][1])
        y[i]=float(B[i][2])
    return t,x+y*1j

def read_dseq1(data_obs_dir,file_name):
    #reads the output d.seq1 observed seismograms in time domain
    #d=read_dseq1(data_obs_dir,'EV.SMN.E.sac.seq1')
    f=open(data_obs_dir+file_name,'r')
    B=f.read().splitlines()
    f.close()
    Size=len(B)-4
    t0=float(B[1][3:]) #initial time
    dt=float(B[2][3:]) #time step
    synwave_buffer=array([])
    for i in range(Size):
        synwave_buffer=hstack([synwave_buffer,float(B[i+4])])
    synm=synwave_buffer[0:Size] #bug makes synwave_buffer Size+1, remove extra line
    return synm    

def read_dimseq2(tf_dir,file_name):
    #reads the output d.imseq2 observed seismograms in frequency domain
    #a,b=read_dimseq2(data_obs_spectrum_dir,'EV.SMN.E.sac.imseq2')
    f=open(tf_dir+file_name,'r')
    B=f.read().splitlines()
    f.close()   
    Size=len(B)-4
    f0=float(B[1][3:]) #initial frequency
    df=float(B[2][3:]) #frequency step
    B=B[4:]
    f=zeros(Size)
    C=zeros(Size,complex)
    for i in range(Size):
        B[i]=B[i].split('\t')
        f[i]=float(B[i][0])
        C[i]=float(B[i][1])+1j*float(B[i][2])
    return f,C

def read_dobs(main_dir,i):
    #reads the dX.cv data files in frequency domain, output by YMAEDA_TOOLS
    d_dir=main_dir+'d_obs/d'
    d_file=d_dir+str(i)+'.cv'
    ddata=loadtxt(d_file)
    d=ddata[1:]
    return d

def readall_dobs(main_dir,DATLEN=2049,ROW=0):
    #reads the dX.cv data files in frequency domain, output by YMAEDA_TOOLS
    #for one particular station
    #D=readall_dobs(main_dir,ROW=0)
    d_dir=main_dir+'d_obs/d'
    DOBS=zeros(DATLEN,complex)
    for i in range(DATLEN):
        d_file=d_dir+str(i)+'.cv'
        ddata=loadtxt(d_file)
        d=ddata[1:]
        d=d[0:int(len(d)/2)]+d[int(len(d)/2):]*1j
        DOBS[i]=d[ROW]
    return DOBS

def read_mest(main_dir,i):
    #reads the mX.cv data files in frequency domain output by YMAEDA_TOOLS
    m_dir=main_dir+'m_est/m'
    m_file=m_dir+str(i)+'.cv'
    mdata=loadtxt(m_file)
    m=mdata[1:]
    return m

def readall_mest(main_dir,DATLEN=2049,NM=2):
    #reads ALL the mX.cv data files in frequency domain output by YMAEDA_TOOLS
    M=zeros([DATLEN,NM],complex)
    for i in range(DATLEN):
        m=read_mest(main_dir,i)
        M[i,:]=m
    return M

def read_G(main_dir,i):
    #reads the GX.bdm binary data files output by YMAEDA_TOOLS
    G_dir=main_dir+'G/G'
    G_file=G_dir+str(i)+'.bdm'
    with open(G_file,mode='rb') as file:
        fileContent=file.read()
        file.close()
        nrows=struct.unpack("i", fileContent[:4]) #no of rows
        nrows=nrows[0]
        ncols=struct.unpack("i", fileContent[4:8]) #no of cols
        ncols=ncols[0]
        dataSize=len(fileContent)-8 #first 8 bytes are the nrows and ncols
        data=zeros(nrows*ncols) #array to hold the data
        count=0
        for ii in range(0,dataSize,8):
            B=fileContent[ii+8:ii+8*2]
            b=struct.unpack("d",B)
            b=b[0]
            #print b
            data[count]=b
            count=count+1
        G=zeros([nrows,ncols])
        count=0
        for iii in range(nrows):
            for jjj in range(ncols):
                G[iii,jjj]=data[count]
                count=count+1
    return G

def readall_G(main_dir,DATLEN=2049):
    #returns a list of arrays which needs to be unfolded
    Gstack=[]
    for i in range(DATLEN):
        G=read_G(main_dir,i)
        Gstack.append(G)
    return Gstack

def unfold_G(Gstack,INDEX=0,DATLEN=2049):
    #to be used with readall_G to unfold the individual green functions
    g=zeros(DATLEN,complex)
    for i in range(DATLEN):
        g[i]=Gstack[i][INDEX][0]-Gstack[i][INDEX][1]*1j
    return g

#FOURIER TRANSFORM#############################################################

def idft(X):
    #this DFT was constructed to match that used in YMAEDA_TOOLS and is
    #different from the version used by MATLAB, SCIPY etc.
    #This uses the traditional DFT algorithm which is very slow due to having
    #2 for loops, and can definitely be made faster by using symmetry...
    N=len(X)
    x=zeros(N,'complex')
    for n in range(0,N,1):
        for k in range(0,N,1):
            x[n]=x[n]+X[k]*exp(-1j*2*pi*k*n/N)
    return x/N

def dft(X):
    #this DFT was constructed to match that used in YMAEDA_TOOLS and is
    #different from the version used by MATLAB, SCIPY etc.
    #This uses the traditional DFT algorithm which is very slow due to having
    #2 for loops, and can definitely be made faster by using symmetry...
    N=len(X)
    x=zeros(N,'complex')
    for n in range(0,N,1):
        for k in range(0,N,1):
            x[n]=x[n]+X[k]*exp(1j*2*pi*k*n/N)
    return x

def timeshift(mest,D):
    #time shift a Fourier transform by some integer value D
    #for use on the non-complex arrays output by YMAEDA_TOOLS
    #warning! Use this only on full range Fourier transforms...
    N=len(mest)
    for k in range(N):
        W=-2*pi*k*(-D)/N
        a=mest[k,0]
        b=mest[k,1]
        mest[k,0]=a*cos(W)-b*sin(W)
        mest[k,1]=a*sin(W)+b*cos(W)
    return mest
    
def timeshift_cplx(mest,D):
    #time shift a Fourier transform by some integer value D
    #for use on complex python arrays
    #warning! Use this only on full range Fourier transforms...
    N=len(mest)
    for k in range(N):
        W=-1j*2*pi*k*(-D)/N
        mest[k]=mest[k]*exp(W)
    return mest  

def exifft(M,DATLEN=2049):
    #extends the data to full frequency range and perform ifft
    #for use on non-complex arrays output by YMAEDA_TOOLS
    M=M[:,0]+M[:,1]*1j #combine real and imag parts together
    #for 2**12=4096 frequency points, [1:2047]=conj([4095:2049]) for 2047 partners
    #points 0 and 2048 are the 2 points without complex conjugate partners.
    M2=M[1:DATLEN-1]
    M2=flipud(M2) #flip the order due to [1:2047]=conj([4095:2049])
    M2=conj(M2) #remember to take the complex conjugate!!!!!!
    #M2=conj(M2[::-1]) #[1:2047]=conj([4095:2049])
    M2=hstack([M,M2])
    return idft(M2)

def exifft_cplx(M,DATLEN=2049):
    #extends the data to full frequency range and perform ifft
    #for use on complex python arrays
    M2=M[1:DATLEN-1]
    M2=flipud(M2) #flip the order due to [1:2047]=conj([4095:2049])
    M2=conj(M2) #remember to take the complex conjugate!!!!!!
    #M2=conj(M2[::-1]) #[1:2047]=conj([4095:2049])
    M2=hstack([M,M2])
    return idft(M2)

def exifft_timeshift(M,D,DATLEN=2049):
    #extend and inverse Fourier transform for non-complex matrices
    #as the output by winv from YMAEDA_TOOLS only contains the first half
    #of the frequency range up to Nyquist frequency, the data has to be
    #extended into the complex conjugate half before ifft is performed.
    M=M[:,0]+M[:,1]*1j #combine real and imag parts together to complex array
    M2=M[1:DATLEN-1]
    M2=flipud(M2) #flip the order due to [1:2047]=conj([4095:2049])
    M2=conj(M2) #remember to take the complex conjugate!!!!!!
    M2=hstack([M,M2])
    for k in range(len(M2)):
        W=-1j*2*pi*k*(-D)/len(M2)
        M2[k]=M2[k]*exp(W)
    return idft(M2)

def exifft_cplx_timeshift(M,D,DATLEN=2049):
    #extend and inverse Fourier transform for complex arrays
    M2=M[1:DATLEN-1]
    M2=flipud(M2) #flip the order due to [1:2047]=conj([4095:2049])
    M2=conj(M2) #remember to take the complex conjugate!!!!!!
    M2=hstack([M,M2])
    for k in range(len(M2)):
        W=-1j*2*pi*k*(-D)/len(M2)
        M2[k]=M2[k]*exp(W)
    return idft(M2)

#WAVEFORM INVERSION FUNCTIONS##################################################

def svdinv(G,d):
    #calculate the inverse of G using SVD decomposition...
    Ug,sg,VgT=linalg.svd(G) #V.T is returned! NOT V!!!
    rUg,cUg=shape(Ug)
    Sg=zeros([rUg,len(sg)])
    Sg[:len(sg),:len(sg)]=diag(sg) #create proper matrix of S
    Vg=VgT.T #create proper matrix of V
    Up=Ug[:,:len(sg)] #extract the first p columns of U
    Vp=Vg[:,:len(sg)] #extract the first p columns of V
    #this is in the unique form used in YMAEDA_TOOLS:
    Ginv=dot(dot(Vp,linalg.inv(diag(sg))),Up.T)
    #my own inversion calculations as a sanity check:
    mestimate=dot(Ginv,d)
    return mestimate

def lstsqinv(G,d,w=0):
    #calculate the moment tensor using least squares inversion, with 
    #water level regularization, for each frequency step. The inversion
    #should be repeated for all frequency steps until the end.
    #w = water level regularization parameter
    nrows,ncols=shape(G)
    #as G is in the output format of YMAEDA_TOOLS, we need to first
    #convert G to the standard complex number format a + ib:
    GG=G[0:int(nrows/2)]
    GG=GG[:,0:int(ncols/2)]-GG[:,int(ncols/2):]*1j
    #since we are dealing with an overdetermined problem here, use least squares:
    GINV=dot(conj(GG.T),GG)	
    #.T is faster than transpose()
    #print GINV
    #use water level regularization?
    if w>0:
        if 0<abs(GINV)<=w:
            GINV=w*GINV/abs(GINV)
        elif GINV==0:
            GINV=w
    GINV=GINV**-1*conj(GG.T)
    mestlstsq=dot(GINV,d[0:int(len(d)/2)]+d[int(len(d)/2):]*1j)
    return real(mestlstsq)[0],imag(mestlstsq)[0]

def gtg_magnitude(main_dir,w=0,DATLEN=2049):
    #ginv=gtg_magnitude(main_dir,0,2049)
    #obtains the magnitude of GINV=inv(G.T G) during least squares inversion
    ginv=zeros(DATLEN)
    for i in range(DATLEN):
        G=read_G(main_dir,i)
        nrows,ncols=shape(G)
        #convert G into standard complex number format: a + ib
        GG=G[0:int(nrows/2)]
        GG=GG[:,0:int(ncols/2)]-GG[:,int(ncols/2):]*1j
        GINV=dot(conj(GG.T),GG)	
        if w>0:
            if 0<abs(GINV)<=w:
                GINV=w*GINV/abs(GINV)
            elif GINV==0:
                GINV=w
        ginv[i]=real(GINV) #this is completely real, remove the 0j part
    return ginv

def winv_lstsq(main_dir,w=0,DATLEN=2049,dt=0.1,N_DATA=6,N_MODEL=2):
    #read and load data, conduct inversion using least squares
    #w = water level regularization parameter
    M=zeros([DATLEN,N_MODEL]) #YMAEDA_TOOLS calculated m
    for i in range(DATLEN):
        #the inversion is carried out for each frequency step and must be
        #repeated for all values. Unfortunately this means for each frequency
        #step new data must be loaded and processed which might slow the
        #inversion process down.
        d=read_dobs(main_dir,i)
        G=read_G(main_dir,i)
        M[i,:]=lstsqinv(G,d,w)
    return M
    
def winv_svd(main_dir,DATLEN=2049,dt=0.1,N_DATA=6,N_MODEL=2):
    #read and load data, conduct inversion using SVD
    M=zeros([DATLEN,N_MODEL]) #YMAEDA_TOOLS calculated m
    for i in range(DATLEN):
        d=read_dobs(main_dir,i)
        G=read_G(main_dir,i)
        M[i,:]=svdinv(G,d)
    return M
  
#READ SEISMOMETER POLEZERO FUNCTIONS##########################################    

def read_pzfile(pzfilename="/Volumes/MAC Backup/ymaeda_tools_mac/winv/share/polezero/tri120p"):
    f=open(pzfilename,'r')
    X=f.readlines()
    f.close()
    NPOLES=int(X[0][6])
    POLES=zeros(NPOLES,'complex')
    for i in range(NPOLES):
        buf=X[i+1]
        buf=buf.strip().split()
        POLES[i]=float(buf[0])+float(buf[1])*1j
    NZEROS=int(X[NPOLES+1][6])
    ZEROS=zeros(NZEROS,'complex')
    for i in range(NZEROS):
        buf=X[i+NPOLES+2]
        buf=buf.strip().split()
        if buf[0]=='CONSTANT':
            break
        else:
            ZEROS[i]=float(buf[0])+float(buf[1])*1j
    CONSTANT=float(X[-1].strip().split()[1])
    return POLES, ZEROS, CONSTANT    

def plot_seismometer_resp(POLES,ZEROS,CONSTANT,w0=0,w=100,dw=0.01):  
    s=arange(w0,w,dw)*1j #this is in rad/s and not 1/s
    H=ones(len(s))
    for i in range(len(ZEROS)):
        H=H*(s-ZEROS[i])
    #multiple the poles:
    for i in range(len(POLES)):
        H=H/(s-POLES[i])
    H=H*CONSTANT
    pha=angle(H)
    amp=abs(H)
    mag=20*log10(amp)
    s=s/(2*pi) #convert from rad/s to 1/s 
    plt.subplot(2,1,1)
    plt.semilogx(imag(s),mag)
    plt.ylabel('Amplitude')
    plt.grid('on')
    plt.subplot(2,1,2)
    plt.semilogx(imag(s),pha/pi*180)
    plt.ylabel('Phase')
    plt.grid('on')
    plt.xlabel('Frequency (Hz)')
    plt.show()
    
def seismometer_resp(POLES,ZEROS,CONSTANT,w0=0,w1=100,dw=0.01):  
    s=arange(w0,w1,dw)*1j #this is in rad/s and not 1/s
    H=ones(len(s))
    for i in range(len(ZEROS)):
        H=H*(s-ZEROS[i])
    for i in range(len(POLES)):
        H=H/(s-POLES[i])
    H=H*CONSTANT
    #imag(s) is returned in rad/s and not Hz!!!!!
    return imag(s),H

def remove_resp(POLES,ZEROS,CONSTANT,f,X,NO_DC=True):
    #input the frequency f in Hz! The program will convert to rad/s
    #This script removes the seismometer response from a Fourier-transformed
    #time series. The first element X[0] is assumed to be zero such that the
    #time series has no Direct Current (f=0) component.
    #Note that POLES and ZEROS are in rad/s. Ensure that calculations do not 
    #mix values given in Hz and those given in rad/s!!!!!!!!!!!!!!!!!!!!!!!
    s=f*2*pi*1j #convert frequency to rad/s
    H=ones(len(s))
    for i in range(len(ZEROS)):
        H=H*(s-ZEROS[i])
    for i in range(len(POLES)):
        H=H/(s-POLES[i])
    H=H*CONSTANT
    Y=zeros(len(s),complex)
    Y[1:]=X[1:]/H[1:]
    if NO_DC==False:
        Y[0]=X[0]
    return Y
        
#SOURCE TIME FUNCTIONS########################################################

def create_timefunc_pow5(tp=1.0,size=121,t0=0.0,dt=0.1,Nintegral=0):
    t=array([t0+dt*i for i in range(size)])
    ft=10*(t/tp)**3-15*(t/tp)**4+6*(t/tp)**5
    ft[t>tp]=1
    return t,ft    
    
def create_timefunc_pow33(tp=1.0,size=121,t0=0.0,dt=0.1,Nintegral=0):
    t=array([t0+dt*i for i in range(size)])
    ft=-64.0*tp**-6*(t**5*(t-tp)**3)
    ft[t>tp]=0
    return t,ft

def create_timefunc_pow34(tp=1.0,size=121,t0=0.0,dt=0.1,Nintegral=0):
    t=array([t0+dt*i for i in range(size)])
    ft=(7.0/3.0)**3*(7.0/4.0)**4*(t/tp)**3*(1-t/tp)**4
    ft[t>tp]=0
    return t,ft 

def create_timefunc_cos(tp=1.0,size=121,t0=0.0,dt=0.1,Nintegral=0):
    t=array([t0+dt*i for i in range(size)])
    ft=0.5*(1-cos(2.0*pi*t/tp))
    ft[t>tp]=0
    return t,ft

