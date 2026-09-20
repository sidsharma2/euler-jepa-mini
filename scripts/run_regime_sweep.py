"""Sweep the dimensionless forecast horizon beta*h and compare baselines."""
from __future__ import annotations
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from metrics import bootstrap_ci, r2, rmse, skill_vs_persistence

SCRIPT_DIR=Path(__file__).resolve().parent; HERE=SCRIPT_DIR.parent; SOURCE=SCRIPT_DIR / "run_experiment.py"
RESULTS,FIGURES,REPORTS=HERE/'results',HERE/'figures',HERE/'reports'; DT=.01

def source():
    spec=importlib.util.spec_from_file_location('euler_regime_source',SOURCE); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m

def dataset(m,n,seed,k,steps=200):
    rng=np.random.default_rng(seed); seq=[]; ps=[]
    for _ in range(n):
        p=m.Parameters(float(rng.uniform(.70,.90)),.25,.02,8.0,k,.15); u=float(rng.uniform(.45,1.05)); w=m.simulate(p,np.full(steps,u,dtype=np.float32),DT,float(rng.uniform(350,450)))
        seq.append(np.column_stack((w[:-1]/600,np.full(steps,u)))); ps.append(p)
    return np.asarray(seq,dtype=np.float32),ps

def windows(m,s,p,h): return m.make_windows(s,p,20,h)
def exact(m,c,p,h):
    w=float(c[-1,0]*600); u=float(c[-1,1])
    for _ in range(h): w=m.exact_step(p,w,u,DT)
    return w
def evaluate(m,k,h):
    tr,tp=dataset(m,120,7,k,200); te,ep=dataset(m,30,19,k,200); tc,tt,tm=windows(m,tr,tp,h); vc,vt,vm=windows(m,te,ep,h); truth=vt[:,-1,0]*600; persist=vc[:,-1,0]*600; prev=vc[:,-2,0]*600
    ftr=np.column_stack((tc.reshape(len(tc),-1),np.ones(len(tc)))); fte=np.column_stack((vc.reshape(len(vc),-1),np.ones(len(vc))))
    out={'persistence':persist,'two_point_extrapolation':persist+h*(persist-prev),'ols_raw_context':fte@np.linalg.lstsq(ftr,tt[:,-1,0]*600,rcond=None)[0],'least_squares_sysid':[],'nominal_analytical':np.asarray([exact(m,c,m.Parameters(),h) for c,(_,_,_) in zip(vc,vm)]),'true_parameter_oracle':np.asarray([exact(m,c,p,h) for c,(_,_,p) in zip(vc,vm)])}
    for c in vc:
        w=c[:,0]*600; x=np.column_stack((w[:-1],np.ones(len(w)-1))); rho,b=np.linalg.lstsq(x,w[1:],rcond=None)[0]; z=float(w[-1])
        for _ in range(h): z=rho*z+b
        out['least_squares_sysid'].append(z)
    out['least_squares_sysid']=np.asarray(out['least_squares_sysid'])
    # The JEPA row is intentionally evaluated with the same endpoint probe as
    # the original corrected study, while the baselines use the same windows.
    torch_seed = 7
    import torch
    torch.manual_seed(torch_seed)
    context_encoder, _, _ = m.train_jepa(tc, tt, epochs=180)
    with torch.no_grad():
        train_embeddings = context_encoder(torch.from_numpy(tc).to(m.DEVICE))
        test_embeddings = context_encoder(torch.from_numpy(vc).to(m.DEVICE))
        train_labels = torch.from_numpy(tt[:, -1, 0]).to(m.DEVICE)
    probe = m.fit_probe(train_embeddings.detach(), train_labels)
    with torch.no_grad():
        out['jepa_context_probe'] = probe(test_embeddings).squeeze(-1).cpu().numpy() * 600.0
    rows=[]
    for name,pred in out.items(): rows.append({'beta_h':k*0.8*.25/.02*h*DT,'horizon_steps':h,'predictor':name,'rmse_rad_s':rmse(pred,truth),'r2':r2(pred,truth),'skill_vs_persistence':skill_vs_persistence(pred,truth,persist)})
    for row in rows:
        low, high = bootstrap_ci(out[row['predictor']], truth, np.asarray([x[0] for x in vm]))
        row['rmse_ci95_low_rad_s'] = low
        row['rmse_ci95_high_rad_s'] = high
    return rows

def main():
    m=source(); points=[(.01,10),(.05,10),(.25,10),(1.0,10),(3.0,10)]; rows=[]
    for k,h in points: rows.extend(evaluate(m,k,h))
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    with (RESULTS/'regime_sweep.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    payload={'experiment':'physical_regime_beta_h_sweep','points':[{'speed_feedback_m_per_rad':k,'horizon_steps':h,'beta_h':k*.8*.25/.02*h*DT} for k,h in points],'rows':rows,'note':'k_omega is varied while horizon remains 10 samples; k_omega=0.25 m/rad is the physically motivated reference scale.'}
    (RESULTS/'regime_sweep.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    names=sorted({r['predictor'] for r in rows}); fig,ax=plt.subplots(figsize=(7,4));
    for name in names:
        q=[r for r in rows if r['predictor']==name]; ax.plot([r['beta_h'] for r in q],[r['rmse_rad_s'] for r in q],marker='o',label=name)
    ax.set_xscale('log'); ax.set(xlabel='Dimensionless forecast horizon beta h',ylabel='Endpoint RMSE (rad/s)'); ax.legend(fontsize=7,frameon=False); fig.tight_layout(); fig.savefig(FIGURES/'regime_sweep.png',dpi=200); plt.close(fig)
    (FIGURES/'regime_sweep.tex').write_text('% Editable PGFPlots data source: results/regime_sweep.csv\n\\begin{tikzpicture}\\begin{axis}[xmode=log,xlabel={$\\beta h$},ylabel={Endpoint RMSE (rad/s)}]\\addplot table [x=beta_h,y=rmse_rad_s,col sep=comma] {../results/regime_sweep.csv};\\end{axis}\\end{tikzpicture}\n',encoding='utf-8')
    (REPORTS/'regime_sweep.md').write_text('# Physical-regime sweep\n\n'+json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
