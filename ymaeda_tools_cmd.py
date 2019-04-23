#commands to run the functions defined in ymaeda_tools.py
#execute ymaeda_tools.py first!

STARTTIME=time.time()

main_dir='/home/yumi/kirishima_invert/inversion_results_dump_new/'
main_dir=main_dir+'inversion_result_residuals117/t60_p100/x-10900y-121100z1000/'

#reads the M.seq1 output from YMAEDA_TOOLS
mt=read_Mseq1(main_dir)

#loads the G and d data and performs inversion using least squares method
mest=winv_lstsq(main_dir,w=0)
m=exifft_timeshift(mest,1000)/0.1 #remember the 1/dt factor
mr=real(m)

dt=0.1
t=arange(0,dt*len(mt),dt)

print(time.time()-STARTTIME)

#plot and check both results
plt.plot(t,mt,'k');plt.plot(t,mr,'r')
plt.xlabel('Time (s)')
plt.ylabel('Seismic Moment (Nm)')
plt.show()