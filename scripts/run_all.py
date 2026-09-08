"""Regenerate the revised manuscript's numerical results from one model."""
from pathlib import Path
import sys, json, hashlib, platform
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.model import simulate
from src.statistics import summary, paired, holm, exact_impartial_probability

GRID=[round(i/10,1) for i in range(11)]
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
CACHE={}
EVENTS={}


def run(pi0, **kw):
    key=json.dumps(dict(pi0=pi0,**kw),sort_keys=True)
    if key not in CACHE:
        CACHE[key]=simulate(pi0,**kw)
    return CACHE[key]


def save(name,obj):
    (OUT/f'{name}.json').write_text(json.dumps(obj,indent=2)+'\n')
    print('Saved',name,flush=True)


def main():
    main_results={};pairs=[]
    traces={};distributions={}
    for learner in ['naive','static']:
        for bot in ['halluc','factual']:
            label=f'{learner}_{bot}'
            main_results[label]={}
            for p in GRID:
                a=run(p,learner=learner,bot=bot,alpha=0.)
                b=run(p,learner=learner,bot=bot,alpha=.02)
                comparison=paired(a,b);pairs.append(comparison)
                main_results[label][str(p)]={'fixed':summary(a),'coupled':summary(b),'comparison':comparison}
                EVENTS[f'main_{label}_{p}_fixed']=a.events
                EVENTS[f'main_{label}_{p}_coupled']=b.events
                if learner=='naive' and bot=='halluc':
                    traces[str(p)]={'pi':b.pi_traces.tolist(),'belief':b.belief_traces.tolist()}
                    distributions[str(p)]=b.final_pi.tolist()
                    crossed=b.first_crossing[b.first_crossing>=0]
                    main_results[label][str(p)]['crossing']={
                        'n_by_60':int(np.count_nonzero((b.first_crossing>=0)&(b.first_crossing<=60))),
                        'n_by_100':len(crossed),
                        'median_among_crossers':float(np.median(crossed)) if len(crossed) else None,
                        'population_half_round':int(np.sort(crossed)[4999]) if len(crossed)>=5000 else None}
            print('Computed',label,flush=True)
    holm(pairs);save('main',main_results);save('traces',traces);save('final_pi',distributions)
    save('random_hallucination',{str(p):summary(run(p,bot='random',alpha=0.)) for p in GRID})

    adaptive={};pairs=[]
    for bot in ['halluc','factual']:
        adaptive[bot]={}
        for p in GRID:
            static=run(p,learner='static',bot=bot,alpha=.02)
            row={'static':summary(static)}
            for sigma in [.02,.05,.1,.2]:
                b=run(p,learner='diffusion',bot=bot,sigma=sigma,alpha=.02)
                c=paired(static,b);pairs.append(c)
                row[str(sigma)]={'summary':summary(b),'vs_static':c}
                EVENTS[f'adaptive_{bot}_{p}_{sigma}']=b.events
            for learner in ['aware','oracle']:
                a=run(p,learner=learner,bot=bot,alpha=0.)
                b=run(p,learner=learner,bot=bot,alpha=.02)
                row[learner]={'fixed':summary(a),'coupled':summary(b),'comparison':paired(a,b)}
                EVENTS[f'{learner}_{bot}_{p}_fixed']=a.events
                EVENTS[f'{learner}_{bot}_{p}_coupled']=b.events
            adaptive[bot][str(p)]=row
        print('Computed adaptive',bot,flush=True)
    holm(pairs)
    holm([row[learner]['comparison'] for rows in adaptive.values() for row in rows.values() for learner in ['aware','oracle']])
    save('adaptive',adaptive)

    forms={
        'linear':{}, 'asymmetric_1.5_0.5':dict(rule='asymmetric'),
        'asymmetric_2_0.5':dict(rule='asymmetric',positive=2.),
        'logistic':dict(rule='logistic'),
        'momentum_0.5':dict(rule='momentum'),
        'momentum_0.8':dict(rule='momentum',momentum=.8),
        'decay_0.01':dict(rule='decay'),
        'decay_0.05':dict(rule='decay',decay=.05),
        'threshold':dict(rule='threshold')}
    sensitivity={'forms':{},'sweep':{},'graded':{},'near_zero_logistic':{}}
    pairs=[]
    for p in [0.,.3]:
        a=run(p,learner='naive',bot='halluc',alpha=0.)
        row={}
        for name,kw in forms.items():
            b=run(p,**kw);c=paired(a,b);pairs.append(c)
            row[name]={'summary':summary(b),'vs_fixed':c,'parameters':kw}
        sensitivity['forms'][str(p)]=row
    holm(pairs)
    for name in ['linear','logistic','momentum_0.5','asymmetric_1.5_0.5']:
        sensitivity['sweep'][name]={str(a):summary(run(.3,alpha=a,**forms[name])) for a in [.005,.01,.02,.03,.05]}
    pairs=[]
    for p in GRID:
        a=run(p,learner='naive',bot='halluc',alpha=0.)
        b=run(p,signal='graded');c=paired(a,b);pairs.append(c)
        sensitivity['graded'][str(p)]={'fixed':summary(a),'graded':summary(b),'comparison':c}
        EVENTS[f'graded_{p}']=b.events
    holm(pairs)
    for p in [0.,1e-6,.001,.01]:
        sensitivity['near_zero_logistic'][str(p)]=summary(run(p,rule='logistic'))
    save('sensitivity',sensitivity)

    grid_checks={}
    for bot in ['halluc','factual']:
        for learner,kw in [('static',{}),('diffusion',{'sigma':.05}),('aware',{})]:
            key=f'{bot}_{learner}'
            grid_checks[key]={str(bins):summary(run(.3,bot=bot,learner=learner,bins=bins,**kw)) for bins in [21,51,101]}
    save('grid_checks',grid_checks)

    mitigations={};pairs=[]
    interventions={
        'none':{},'rate_cap':{'cap':.005},'exploration':{'exploration':.1},
        'smoothing':{'ema':.1},'cap_exploration':{'cap':.005,'exploration':.1},
        'combined':{'cap':.005,'exploration':.1,'ema':.1}}
    for p in GRID:
        mitigations[str(p)]={}
        for label,kw in interventions.items():
            a=run(p,alpha=0.,**kw);b=run(p,alpha=.02,**kw)
            c=paired(a,b);pairs.append(c)
            mitigations[str(p)][label]={'fixed':summary(a),'coupled':summary(b),'comparison':c}
            EVENTS[f'mitigation_{p}_{label}_fixed']=a.events
            EVENTS[f'mitigation_{p}_{label}_coupled']=b.events
    holm(pairs);save('mitigations',mitigations)

    controls={}
    for p in [0.,.3,.5]:
        # Donor conversations are independent of all recipient random inputs.
        donor=simulate(p,seed=4242,record=True)
        b=run(p,learner='naive',bot='halluc',alpha=.02)
        rng=np.random.default_rng(987)
        shuffled=np.stack([row[rng.permutation(len(row))] for row in donor.signals])
        fair=np.where(rng.random((100,10000))<.5,1.,-1.)
        conditions={
            'donor_replay':simulate(p,schedule=donor.schedule),
            'donor_mean':simulate(p,schedule=donor.schedule.mean(axis=1)),
            'shuffled_signals':simulate(p,external_signals=shuffled),
            'fair_signals':simulate(p,external_signals=fair)}
        controls[str(p)]={'coupled':summary(b),'donor':summary(donor)}
        for name,c in conditions.items():
            controls[str(p)][name]={'summary':summary(c),'vs_coupled':paired(b,c)}
    holm([row[name]['vs_coupled'] for row in controls.values() for name in ['donor_replay','donor_mean','shuffled_signals','fair_signals']])
    save('feedback_controls',controls)
    # Event arrays retain the pairing needed to verify every principal comparison.
    np.savez_compressed(OUT/'events.npz',**EVENTS)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['src','scripts'] for p in sorted((ROOT/folder).glob('*.py'))}
    save('manifest',{'version':'3.0.0-review','seed':42,'donor_seed':4242,'control_seed':987,
                     'n':10000,'turns':100,'alpha':.02,'pi_grid':GRID,
                     'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                     'source_sha256':hashes,'pairing':'six uniforms per turn; same seed within each comparison',
                     'multiple_comparisons':'Holm within main (44), diffusion (88), exact-rule/oracle (44), forms (18), graded (11), mitigations (66), feedback controls (12).',
                     'exact_impartial_probability':exact_impartial_probability(),
                     'selection':'All settings and the exact-rule benchmarks are reported; no setting is labelled optimal.'})


if __name__=='__main__':
    main()
