#commands to check the green's functions calculated by YMAEDA_TOOLS.
#execute ymaeda_tools.py first!

STARTTIME=time.time()

df=0.002441406 #frequency step size used by YMAEDA_TOOLS
f=arange(0,df*2049,df) #frequency half space
F=arange(0,df*4096,df) #frequency full space

GREENS_FUNCTION='pow34'

STATION="SMW_UD_SMALL"
#HDD_LOC="/media/yumi/INVERSION/" #ubuntu
HDD_LOC="/media/yumi/Backup/"
#HDD_LOC="/Volumes/Mac Backup/kirishima_invert/" #mac air
"""
SMN: -11175, -119878, 1317
SMW: -12295, -120893, 1110
LP:  -10900, -121100, 1000
"""
if GREENS_FUNCTION=='pow5':
    print('pow5 chosen...')
    snapshot_dir=HDD_LOC+"GFDIR1/"+STATION+"/PML/snapshot"
    T,FT=create_timefunc_pow5(tp=1,size=121,t0=0,dt=0.1)
    #using a time function of higher resolution might lead to better results?
    #T,FT=create_timefunc_pow5(tp=1,size=1001,t0=0,dt=0.001)
    #PAD_BUFFER=zeros(len(F)-len(T))
    PAD_BUFFER=zeros(400)
    FTw=dft(hstack([PAD_BUFFER,FT,ones(len(F)-len(PAD_BUFFER)-len(FT))]));FT1=FTw[0:len(f)]
    #instead of creating and calculating TF and FT1, load the output values?
    #T,FT=read_stfunseq2(stfun_dir)
    #F,FTw=read_stfunspecseq2(stfun_dir)
    #f=F[0:2049]
    #FT1=FTw[0:2049]
elif GREENS_FUNCTION=='pow34':
    print('pow34 chosen...')
    snapshot_dir=HDD_LOC+"GFDIR0/"+STATION+"/PML/snapshot"
    T,FT=create_timefunc_pow34(tp=1,size=121,t0=0,dt=0.1)
    #T,FT=create_timefunc_pow34(tp=1,size=1001,t0=0,dt=0.001)
    FTw=dft(hstack([FT,zeros(len(F)-len(FT))]));FT1=FTw[0:len(f)]
elif GREENS_FUNCTION=='cos':
    print('cos chosen...')
    snapshot_dir=HDD_LOC+"GFDIR2/"+STATION+"/PML/snapshot"
    T,FT=create_timefunc_cos(tp=1,size=121,t0=0,dt=0.1)
    #T,FT=create_timefunc_cos(tp=1,size=1001,t0=0,dt=0.001)
    FTw=dft(hstack([FT,zeros(len(F)-len(FT))]));FT1=FTw[0:len(f)]    

print(time.time()-STARTTIME)
               
N,x0,dx=read_snapshot_params(snapshot_file=snapshot_dir+'/source.Fx.t0.0000.3db')
idx,idy,idz=snapshot_stnloc(N,x0,dx,-10900, -121100, 1000)
X,Y,Z=snapshot_XYZ(N,x0,dx)
           
t,g=read_snapshot_loc3D_fast(snapshot_dir=snapshot_dir,X=idx,Y=idy,Z=idz)
print(time.time()-STARTTIME)

if GREENS_FUNCTION=='pow34':
    G0=dft(hstack([g[:,0],zeros(len(F)-len(g[:,0]))]))*(t[2]-t[1])
    G1=dft(hstack([g[:,1],zeros(len(F)-len(g[:,0]))]))*(t[2]-t[1])
    G2=dft(hstack([g[:,2],zeros(len(F)-len(g[:,0]))]))*(t[2]-t[1])
elif GREENS_FUNCTION=='pow5':
    G0=dft(hstack([g[:,0],zeros(len(F)-len(g[:,0]))]))*(t[2]-t[1])
    G1=dft(hstack([g[:,1],zeros(len(F)-len(g[:,1]))]))*(t[2]-t[1])
    G2=dft(hstack([g[:,2],zeros(len(F)-len(g[:,2]))]))*(t[2]-t[1])
elif GREENS_FUNCTION=='cos':
    G0=dft(hstack([g[:,0],zeros(len(F)-len(g[:,0]))]))*(t[2]-t[1])
    G1=dft(hstack([g[:,1],zeros(len(F)-len(g[:,1]))]))*(t[2]-t[1])
    G2=dft(hstack([g[:,2],zeros(len(F)-len(g[:,2]))]))*(t[2]-t[1])
    
G0=G0[0:len(f)];G1=G1[0:len(f)];G2=G2[0:len(f)]

print(time.time()-STARTTIME)

G00=G0/FT1;G10=G1/FT1;G20=G2/FT1
g00=real(exifft_cplx(G00))/(t[2]-t[1])
g10=real(exifft_cplx(G10))/(t[2]-t[1])
g20=real(exifft_cplx(G20))/(t[2]-t[1])

print(time.time()-STARTTIME)

#g[:,i] should equal the numerical convolution of g00[:121] and FT
if GREENS_FUNCTION=='pow34':
    plt.subplot(3,1,1)
    plt.plot(t,convolve(g00[:len(t)],FT)[:len(t)],'k');plt.plot(t,g[:,0],'r.')
    plt.subplot(3,1,2)
    plt.plot(t,convolve(g10[:len(t)],FT)[:len(t)],'k');plt.plot(t,g[:,1],'r.')
    plt.subplot(3,1,3)
    plt.plot(t,convolve(g20[:len(t)],FT)[:len(t)],'k');plt.plot(t,g[:,2],'r.')
    plt.show()
elif GREENS_FUNCTION=='cos':
    plt.subplot(3,1,1)
    plt.plot(t[1:],convolve(g00,FT)[4096:],'k');plt.plot(t,g[:,0],'r.')
    plt.subplot(3,1,2)
    plt.plot(t[1:],convolve(g10,FT)[4096:],'k');plt.plot(t,g[:,1],'r.')
    plt.subplot(3,1,3)
    plt.plot(t[1:],convolve(g20,FT)[4096:],'k');plt.plot(t,g[:,2],'r.')
    plt.show()    
elif GREENS_FUNCTION=='pow5':
    plt.subplot(3,1,1)
    plt.plot(t[:-1],convolve(g00[:len(t)],FT)[len(T):],'k');plt.plot(t,g[:,0],'r.')
    plt.subplot(3,1,2)
    plt.plot(t[:-1],convolve(g10[:len(t)],FT)[len(T):],'k');plt.plot(t,g[:,1],'r.')
    plt.subplot(3,1,3)
    plt.plot(t[:-1],convolve(g20[:len(t)],FT)[len(T):],'k');plt.plot(t,g[:,2],'r.')
    plt.show()
