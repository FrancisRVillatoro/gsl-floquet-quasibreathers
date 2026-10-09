"""Targeted convergence audit for the suppressed-radiation point used in the CSF manuscript.

Recomputes the normalized radiated loss P*T/E0 at (b,Omega,a)=(0.5,0.8,1.25)
under one-factor changes of timestep, spatial resolution, harmonic-balance truncation,
box size, and sponge strength. It also recomputes the two branch fits for a_c with a
stricter configuration. Runtime is a few minutes on a multicore workstation.
"""
import os, time, csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import gsl_floquet as gf
from results_io import result_path
from qb_newton import HarmonicBalance, sg_breather_harmonics, small_amplitude_coefficients
from kg_spectral import SpectralKG

B, OMEGA = 0.5, 0.8

def measure(a,K,L,N,h,x_s,width,sigma0):
    d=gf.DressedGSL(a,B); k1,k3,_=small_amplitude_coefficients(d); omega=OMEGA*np.sqrt(k1)
    hb=HarmonicBalance(L,N,d.F,d.dF,K=K,kappa=k1)
    U,info=hb.solve(np.sqrt(k1/k3)*sg_breather_harmonics(np.sqrt(k1)*hb.x,OMEGA,K),omega,tol=1e-11,maxit=25)
    if not info['converged']:
        raise RuntimeError(f'HB failed at a={a}')
    kg=SpectralKG(L,N,d.F,d.G,kappa=k1,sponge=dict(x_s=x_s,width=width,sigma0=sigma0),G_ref=0.)
    u0,v0=hb.initial_data(U,omega); kg.set_initial(u0,v0); E0=kg.energy(); T=2*np.pi/omega
    rec=[]
    kg.run_until(h,120*T,observer=lambda kg,u:rec.append((kg.t,kg.absorbed)),every=max(1,int(round(10/h))))
    rec=np.asarray(rec); sel=rec[:,0]>60*T
    P=float(np.polyfit(rec[sel,0]/T,rec[sel,1]/E0,1)[0])
    bal=float((kg.energy()+kg.absorbed-E0)/E0)
    return P,float(info['residual']),bal

def worker(row):
    name,a,K,L,N,h,xs,W,s0=row; P,res,bal=measure(a,K,L,N,h,xs,W,s0)
    return name,a,K,L,N,h,xs,W,s0,P,res,bal

def main():
    configs=[
      ('base',1.25,10,140.,1024,.1,45.,22.,.8),
      ('h_half',1.25,10,140.,1024,.05,45.,22.,.8),
      ('N_double',1.25,10,140.,2048,.05,45.,22.,.8),
      ('K14',1.25,14,140.,1024,.05,45.,22.,.8),
      ('L180',1.25,10,180.,1280,.05,60.,28.,.8),
      ('sponge_half',1.25,10,140.,1024,.05,45.,22.,.4),
    ]
    with ProcessPoolExecutor(max_workers=3) as ex:
        rows=list(ex.map(worker,configs))
    with open(result_path('convergence_minimum_a1p25.csv'),'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['case','K','L','N','h','x_s','width','sigma0','P_per_period_over_E0','hb_residual','energy_balance_relative'])
        for r in rows:
            w.writerow([r[0], *r[2:]])
    avals=[1.22,1.23,1.24,1.245,1.26,1.27,1.28]
    strict=[('strict',a,14,180.,1280,.05,60.,28.,.8) for a in avals]
    with ProcessPoolExecutor(max_workers=3) as ex:
        sr=list(ex.map(worker,strict))
    aa=np.array([r[1] for r in sr]); P=np.array([r[9] for r in sr]); s=np.sqrt(P)
    pl=np.polyfit(aa[aa<=1.245],s[aa<=1.245],1)
    pr=np.polyfit(aa[aa>=1.26],s[aa>=1.26],1)
    acl=-pl[1]/pl[0]; acr=-pr[1]/pr[0]
    print('strict a_c left/right/mid/halfspread:',acl,acr,.5*(acl+acr),.5*abs(acl-acr))
    with open(result_path('convergence_ac_strict.csv'),'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['a','P_per_period_over_E0','sqrtP','hb_residual','energy_balance_relative'])
        for r in sr:
            w.writerow([r[1], r[9], np.sqrt(r[9]), r[10], r[11]])
if __name__=='__main__': main()
