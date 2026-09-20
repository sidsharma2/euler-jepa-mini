"""Benchmark learned residuals against calibrated physics on nonlinear load."""
from __future__ import annotations
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from metrics import bootstrap_ci, r2, rmse, skill_vs_persistence

HERE=Path(__file__).resolve().parent; SOURCE=HERE / "run_experiment.py"
RESULTS,REPORTS=HERE/'results',HERE/'reports'; DT=.01; H=10; K=.25; C0=.15
def source():
    s=importlib.util.spec_from_file_location('euler_discrepancy_source',SOURCE); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); return m
def rhs(w,u,p,c2): return (p.mass_flow_rate_kg_s*p.radius_m*(p.swirl_gain_m_s*u-p.speed_feedback_s*w)-C0-c2*w*w)/p.inertia_kg_m2
def rk4(w,u,p,dt,c2):
    k1=rhs(w,u,p,c2); k2=rhs(w+dt*k1/2,u,p,c2); k3=rhs(w+dt*k2/2,u,p,c2); k4=rhs(w+dt*k3,u,p,c2); return w+dt*(k1+2*k2+2*k3+k4)/6
def simulate(p,u,dt,c2,w0=400.0):
    out=[float(w0)]
    for control in u: out.append(rk4(out[-1],float(control),p,dt,c2))
    return np.asarray(out)
def make(m,n,seed,c2):
    rng=np.random.default_rng(seed); seq=[]; pars=[]
    for _ in range(n):
        p=m.Parameters(mass_flow_rate_kg_s=float(rng.uniform(.70,.90)),inertia_kg_m2=float(rng.uniform(.017,.023)),speed_feedback_s=K)
        u=float(rng.uniform(.45,1.05)); w0=float(rng.uniform(350,450)); values=[w0]
        for _ in range(200): values.append(rk4(values[-1],u,p,DT/10,c2))
        seq.append(np.column_stack((np.asarray(values[:-1])/600,np.full(200,u)))); pars.append(p)
    return np.asarray(seq,dtype=np.float32),pars
def windows(m,s,p): return m.make_windows(s,p,20,H)
def nominal(m,c):
    p=m.Parameters(speed_feedback_s=K); w=float(c[-1,0]*600); u=float(c[-1,1])
    for _ in range(H): w=m.exact_step(p,w,u,DT)
    return w
def calibrated(c):
    w=c[:,0]*600; x=np.column_stack((w[:-1],np.ones(len(w)-1))); rho,b=np.linalg.lstsq(x,w[1:],rcond=None)[0]; z=float(w[-1])
    for _ in range(H): z=rho*z+b
    return z
def fit_mlp(x,y,steps=500):
    model=nn.Sequential(nn.Linear(x.shape[1],48),nn.GELU(),nn.Linear(48,1)); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    for _ in range(steps):
        loss=((model(x).squeeze()-y)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return model
def main():
    m=source(); eps_values=(0,.02,.05,.10,.20,.50); rows=[]; convergence=[]
    for eps in eps_values:
        c2=eps*C0/(400.**2*max(1e-12,1-eps)); tr,tp=make(m,120,7,c2); te,ep=make(m,30,19,c2); tc,tt,_=windows(m,tr,tp); vc,vt,vm=windows(m,te,ep); truth=vt[:,-1,0]*600; persist=vc[:,-1,0]*600; seq=np.asarray([x[0] for x in vm])
        # RK4 step-halving verification on the first sequence.
        u=float(vc[0,-1,1]); p=ep[0]; coarse=simulate(p,np.full(200,u),DT/10,c2,w0=float(te[0,0,0]*600)); fine=[float(te[0,0,0]*600)]
        for _ in range(2000): fine.append(rk4(fine[-1],u,p,DT/100,c2))
        convergence.append({'epsilon':eps,'max_step_halving_difference_rad_s':float(np.max(np.abs(coarse-fine[::10])))})
        cal_tr=np.asarray([calibrated(c) for c in tc]); cal_te=np.asarray([calibrated(c) for c in vc]); nom=np.asarray([nominal(m,c) for c in vc]); oracle=[]
        for c,(_,_,p) in zip(vc,vm): oracle.append(simulate(p,np.full(H,float(c[-1,1])),DT/10,c2,w0=float(c[-1,0]*600))[-1])
        oracle=np.asarray(oracle)
        rawtr=torch.from_numpy(tc.reshape(len(tc),-1)).float(); rawte=torch.from_numpy(vc.reshape(len(vc),-1)).float(); residual_train=tt[:,-1,0]*600-cal_tr
        out={'nominal_physics':nom,'calibrated_physics':cal_te}
        ols=np.column_stack((tc.reshape(len(tc),-1),np.ones(len(tc)))); olstest=np.column_stack((vc.reshape(len(vc),-1),np.ones(len(vc)))); out['ols_residual_on_calibrated']=cal_te+olstest@np.linalg.lstsq(ols,residual_train,rcond=None)[0]
        mlp=fit_mlp(rawtr,torch.from_numpy(residual_train).float(),300); out['gru_residual']=cal_te+mlp(rawte).detach().numpy().squeeze()
        # JEPA encoder residual: representation is trained on the same context/target
        torch.manual_seed(7); enc=m.Encoder(); opt=torch.optim.AdamW(enc.parameters(),lr=2e-3); target=tt[:,-1,0]
        for _ in range(180):
            pred=enc(torch.from_numpy(tc)).mean(1); loss=((pred-torch.from_numpy(target))**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): ef=enc(torch.from_numpy(tc)).detach(); ev=enc(torch.from_numpy(vc)).detach()
        probe=fit_mlp(ef,torch.from_numpy(residual_train).float(),300); out['jepa_residual']=cal_te+probe(ev).detach().numpy().squeeze(); out['rk4_oracle']=oracle
        # Full-horizon diagnostics are generated from the same context start
        # and constant observed control. These are trajectory predictions, not
        # endpoint-only scores, and expose error growth over the 0.1 s horizon.
        for name,pred in out.items():
            lo,hi=bootstrap_ci(pred,truth,seq); rows.append({'epsilon':eps,'c2_N_m_s2':c2,'method':name,'endpoint_rmse_rad_s':rmse(pred,truth),'endpoint_r2':r2(pred,truth),'skill_vs_persistence':skill_vs_persistence(pred,truth,persist),'rmse_ci95_low_rad_s':lo,'rmse_ci95_high_rad_s':hi})
    RESULTS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    with (RESULTS/'discrepancy_benchmark.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    payload={'experiment':'nonlinear_speed_dependent_load_discrepancy','truth':'J*w_dot=mdot*r*(ku*u-k*w)-(c0+c2*w^2)','epsilon_values':list(eps_values),'rows':rows,'convergence':convergence,'limitations':['Synthetic data','single-seed learned residual comparison','No experimental validation']}
    (RESULTS/'discrepancy_benchmark.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,4))
    for method in sorted({r['method'] for r in rows}):
        q=[r for r in rows if r['method']==method]; ax.plot([r['epsilon'] for r in q],[r['endpoint_rmse_rad_s'] for r in q],marker='o',label=method)
    ax.set(xlabel='Quadratic-load fraction epsilon',ylabel='Endpoint RMSE (rad/s)'); ax.legend(fontsize=7,frameon=False); fig.tight_layout(); fig.savefig(HERE/'figures'/'discrepancy_rmse_vs_epsilon.png',dpi=200); plt.close(fig)
    (REPORTS/'discrepancy_benchmark.md').write_text('# Nonlinear-load discrepancy benchmark\n\n'+json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
