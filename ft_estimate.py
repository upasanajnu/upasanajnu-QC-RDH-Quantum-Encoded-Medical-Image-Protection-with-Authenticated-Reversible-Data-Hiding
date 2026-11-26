"""Order-of-magnitude surface-code estimate for the QC-RDH circuits (Fowler et al. 2012 scaling).
p_L(d)=0.1*(p/p_th)^((d+1)/2), p=1e-3, p_th=1e-2, 1 us code cycle; one T gate per logical step of d cycles
(sequential T consumption); error budget 1% per run over n_L*T logical qubit-steps."""
import json, math
cfg={'(2,2)':dict(nL=31,T=9.4e6,copies=1.51e7),
     '(1,8)':dict(nL=35,T=3.1e6,copies=3.42e6),
     'RR':   dict(nL=27,T=2.1e6,copies=3.42e6)}
p,pth,cyc=1e-3,1e-2,1e-6
out={}
for k,c in cfg.items():
    d=3
    while c['nL']*c['T']*0.1*(p/pth)**((d+1)/2)>0.01: d+=2
    data=2*d*d*c['nL']
    run=c['T']*d*cyc
    out[k]=dict(d=d,data_qubits=data,seconds_per_run=run,
                total_years_serial=run*c['copies']/3.156e7,
                devices_for_one_day=math.ceil(run*c['copies']/86400))
    print(k,out[k])
json.dump(out,open('results/ft_estimate.json','w'),indent=1)
