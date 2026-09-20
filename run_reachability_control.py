"""Reachability-aware control audit for the affine Euler rotor."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; RESULTS=HERE/'results'; FIGURES=HERE/'figures'; REPORTS=HERE/'reports'
MDOT=.8; R=.25; J=.02; KU=8.; K=.01; TAU=.15; DT=.01; UMIN=.45; UMAX=1.05; W0=400.; T=.4
ALPHA=MDOT*R*KU/J; BETA=MDOT*R*K/J; GAMMA=TAU/J
def endpoint(u):
    eq=(ALPHA*u-GAMMA)/BETA
    return eq+(W0-eq)*np.exp(-BETA*T)
def trajectory(controls):
    w=[W0]
    for u in controls: w.append((ALPHA*u-GAMMA)/BETA+(w[-1]-(ALPHA*u-GAMMA)/BETA)*np.exp(-BETA*DT))
    return np.asarray(w)
def terms(w,controls,target):
    return {'terminal_error_rad_s':float(abs(w[-1]-target)),'tracking_ISE_rad2_s':float(np.sum((w[1:]-target)**2)*DT),'control_effort':float(np.mean(np.asarray(controls)**2)),'control_rate_rms':float(np.sqrt(np.mean(np.diff(np.r_[controls[0],controls])**2))),'saturation_fraction':float(np.mean((np.asarray(controls)==UMIN)|(np.asarray(controls)==UMAX))),'constraint_activity_fraction':0.0}
def main():
    upper=float(endpoint(UMAX)); lower=float(endpoint(UMIN)); rng=np.random.default_rng(7); random_controls=rng.uniform(UMIN,UMAX,size=40); random_w=trajectory(random_controls)
    rows=[]
    for target in (430.,405.,412.):
        optimal_u=float(np.clip(UMIN+(target-lower)/(upper-lower)*(UMAX-UMIN),UMIN,UMAX)); opt=trajectory(np.full(40,optimal_u)); random=terms(random_w,random_controls,target); best=terms(opt,np.full(40,optimal_u),target)
        for method,data in [('analytical_optimum',best),('random_shooting',random)]: rows.append({'target_omega_rad_s':target,'method':method,**data})
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    with (RESULTS/'reachability_control.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    payload={'experiment':'reachability_aware_control_redo','reachable_terminal_set_rad_s':[lower,upper],'target_430_rad_s':430.,'infeasibility_gap_rad_s':430-upper,'random_shooting_terminal_rad_s':float(random_w[-1]),'random_search_gap_vs_upper_rad_s':upper-float(random_w[-1]),'rows':rows,'constraint_note':'The original |omega-550|-150 constraint is inactive over this trajectory.'}
    (RESULTS/'reachability_control.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); (REPORTS/'reachability_control.md').write_text('# Reachability-aware control\n\n'+json.dumps(payload,indent=2),encoding='utf-8')
    (FIGURES/'reachability_pareto.tex').write_text('% Pareto data are in results/reachability_control.csv; each row reports terminal error, effort, and rate.\n',encoding='utf-8'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
