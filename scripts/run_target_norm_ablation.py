"""Test target normalization, loss choice, and EMA schedule independently."""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from metrics import bootstrap_ci, r2, skill_vs_persistence, rmse

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR.parent
SOURCE = SCRIPT_DIR / "run_experiment.py"
RESULTS, REPORTS = HERE / "results", HERE / "reports"


def load_source():
    spec = importlib.util.spec_from_file_location("euler_source_norm", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def data(module):
    def make(n, seed):
        rng = np.random.default_rng(seed); sequences=[]; params=[]
        for _ in range(n):
            p=module.Parameters(float(rng.uniform(.70,.90)), .25, .02, 8.0, float(rng.uniform(.0085,.0115)), .15)
            u=float(rng.uniform(.45,1.05)); omega=module.simulate(p,np.full(200,u,dtype=np.float32),.01,float(rng.uniform(350,450)))
            sequences.append(np.column_stack((omega[:-1]/600,np.full(200,u)))); params.append(p)
        return np.asarray(sequences,dtype=np.float32),params
    tr,tp=make(120,7); te,ep=make(30,19)
    return module.make_windows(tr,tp), module.make_windows(te,ep)


def rank(embedding):
    centered=embedding.detach().cpu().numpy()-embedding.detach().cpu().numpy().mean(0,keepdims=True)
    values=np.linalg.svd(centered,compute_uv=False); values=values[values>1e-10]
    return float((values.sum()**2)/(np.square(values).sum())) if values.size else 0.0


def main():
    module=load_source(); (tc,tt,_),(vc,vt,vm)=data(module)
    torch.set_num_threads(1)
    device=module.DEVICE; train_x=torch.from_numpy(tc).to(device); train_y=torch.from_numpy(tt).to(device); test_x=torch.from_numpy(vc).to(device)
    truth=vt[:,-1,0].astype(float)*600; persistence=vc[:,-1,0].astype(float)*600; seq=np.asarray([m[0] for m in vm])
    rows=[]; history=[]
    for normalize,l1,ema_schedule in itertools.product((False,True),(False,True),(False,True)):
        label=f"norm_{int(normalize)}_loss_{'l1' if l1 else 'l2'}_ema_{'scheduled' if ema_schedule else 'fixed'}"
        for seed in (7,17,27):
            seed_all(seed); context=module.Encoder().to(device); target=module.Encoder().to(device); target.load_state_dict(context.state_dict()); predictor=module.Predictor().to(device)
            optimizer=torch.optim.AdamW(list(context.parameters())+list(predictor.parameters()),lr=2e-3)
            for step in range(1,1441):
                h=context(train_x)
                with torch.no_grad(): target_h=target(train_y)
                if normalize: target_h=F.layer_norm(target_h,(target_h.size(-1),))
                pred=predictor(h); loss=(pred-target_h).abs().mean() if l1 else ((pred-target_h)**2).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                ema=0.998+(1.0-0.998)*step/1440 if ema_schedule else .99
                with torch.no_grad():
                    for tp,cp in zip(target.parameters(),context.parameters()): tp.mul_(ema).add_(cp,alpha=1-ema)
                if step%20==0:
                    with torch.no_grad():
                        emb=context(test_x); std=emb.std(0).cpu().numpy(); history.append({'configuration':label,'seed':seed,'gradient_step':step,'train_latent_loss':float(loss.cpu()),'embedding_std_mean':float(std.mean()),'embedding_std_min':float(std.min()),'embedding_std_max':float(std.max()),'effective_rank':rank(emb)})
            with torch.no_grad():
                tr_emb=context(train_x); te_emb=context(test_x); train_label=torch.from_numpy(tt[:,-1,0]).to(device); test_label=torch.from_numpy(vt[:,-1,0]).to(device)
            probe=module.fit_probe(tr_emb.detach(),train_label)
            with torch.no_grad(): prediction=probe(te_emb).squeeze(-1).cpu().numpy()*600
            low,high=bootstrap_ci(prediction,truth,seq); rows.append({'configuration':label,'seed':seed,'target_normalization':normalize,'loss':'l1' if l1 else 'l2','ema':'scheduled' if ema_schedule else 'fixed','endpoint_rmse_rad_s':rmse(prediction,truth),'endpoint_r2':r2(prediction,truth),'skill_vs_persistence':skill_vs_persistence(prediction,truth,persistence),'rmse_ci95_low_rad_s':low,'rmse_ci95_high_rad_s':high,'effective_rank':rank(te_emb)})
    RESULTS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    for path,items in [(RESULTS/'target_norm_ablation.csv',rows),(RESULTS/'target_norm_ablation_history.csv',history)]:
        with path.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=items[0].keys()); w.writeheader(); w.writerows(items)
    payload={'experiment':'target_feature_normalization_ablation','gradient_steps':1440,'seeds':[7,17,27],'configurations':8,'rows':rows,'history_rows':len(history),'interpretation':'Effective rank and endpoint metrics are reported for all cells; no cell was selected using the test set.'}
    (RESULTS/'target_norm_ablation.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); (REPORTS/'target_norm_ablation.md').write_text('# Target-normalization ablation\n\n'+json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2))


if __name__=='__main__': main()
