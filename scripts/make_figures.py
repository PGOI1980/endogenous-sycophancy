"""Draw every manuscript figure from the revised saved results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'figures';F.mkdir(exist_ok=True)
def read(name):return json.loads((ROOT/'results'/f'{name}.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':9,
                     'axes.labelsize':8,'legend.fontsize':6.5,'lines.linewidth':1.2,
                     'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
P=np.arange(11)/10
def save(fig,name):
    fig.savefig(F/f'{name}.png',dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(F/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)
def rate(rows,key):return np.array([rows[str(float(p))][key]['rate']*100 for p in P])
def labels(ax):
    ax.set_xlabel(r'Initial sycophancy rate ($\pi_0$)')
    ax.set_ylabel('Threshold crossings (%)')
    ax.set_xlim(-.03,1.03)

def main():
    m=read('main');a=read('adaptive');s=read('sensitivity');mti=read('mitigations')
    fig,axes=plt.subplots(1,3,figsize=(6.4,2.35))
    for i,ax in enumerate(axes):
        ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
        for y,lab in [(.8,'Human'),(.42,'AI')]:
            ax.add_patch(Circle((.5,y),.13,fill=False,lw=.8));ax.text(.5,y,lab,ha='center',va='center')
        ax.annotate('',(.45,.67),(.45,.55),arrowprops={'arrowstyle':'->','lw':1.1})
        ax.annotate('',(.55,.55),(.55,.67),arrowprops={'arrowstyle':'->','lw':1.1})
        ax.text(.5,.99,['Initial response policy','Positive net agreement','Higher response propensity'][i],ha='center',va='top',fontsize=8)
        xx=np.linspace(.22,.8,40);yy=.04+.15/(1+np.exp(-12*((xx-.22)/.58-.65+i*.22)))
        ax.plot(xx,yy,color='black');ax.text(.13,.13,r'$\pi$',fontsize=9)
        ax.text(.5,.24,[r'$\pi\approx\pi_0$',r'$\pi$ rises',r'$\pi$ approaches 1'][i],ha='center')
    fig.tight_layout();save(fig,'figure1')
    fig,axes=plt.subplots(1,2,figsize=(6.4,2.8))
    for i,ax in enumerate(axes):
        ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
        positions={'Opinion':(.28,.8),'Bot response':(.76,.5),'Belief update':(.28,.2),r'$\pi$':(.12,.5)}
        for lab,(x,y) in positions.items():ax.text(x,y,lab,ha='center',va='center',bbox={'boxstyle':'round,pad=.3','fc':'white','ec':'black','lw':.7})
        for p,q in [('Opinion','Bot response'),('Bot response','Belief update'),('Belief update','Opinion'),(r'$\pi$','Bot response')]:
            x,y=positions[p];xx,yy=positions[q];ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='->',mutation_scale=10,shrinkA=25,shrinkB=27,lw=.8))
        if i:
            ax.add_patch(FancyArrowPatch((.64,.45),(.14,.45),arrowstyle='->',mutation_scale=10,connectionstyle='arc3,rad=.6',lw=1.2))
            ax.text(.47,.68,'Agreement signal',ha='center',fontsize=7,bbox={'facecolor':'white','edgecolor':'none','pad':1})
        ax.set_title([r'Fixed $\pi$',r'Agreement-dependent $\pi$'][i])
    fig.tight_layout();save(fig,'figure2')
    fig,axes=plt.subplots(2,2,figsize=(6.4,5.5))
    for ax,label,title in zip(axes.flat,m,['(A) Naive user, hallucinating bot','(B) Naive user, factual bot','(C) Static informed user, hallucinating bot','(D) Static informed user, factual bot']):
        rows=m[label]
        for key,style in [('fixed','o-'),('coupled','s-')]:ax.plot(P,rate(rows,key),style,color='black',mfc='white' if key=='coupled' else 'black',ms=3,label=key.capitalize())
        if label=='naive_halluc':
            rnd=read('random_hallucination');ax.plot(P,[rnd[str(float(p))]['rate']*100 for p in P],'--',color='.45',label='Random hallucination')
        labels(ax);ax.set_title(title,fontsize=8);ax.legend()
    fig.tight_layout();save(fig,'figure3')
    traces=read('traces')['0.0'];fig,axes=plt.subplots(1,2,figsize=(6.4,2.9))
    for ax,key,title in zip(axes,['pi','belief'],['(A) Response propensity','(B) Belief in the true state']):
        ax.plot(np.array(traces[key]),lw=.65,alpha=.48,color='black');ax.set_xlabel('Conversational round');ax.set_title(title)
        ax.set_ylim(-.02,1.02);ax.set_ylabel(r'$\pi(t)$' if key=='pi' else '$P(H=1)$')
    axes[1].axhline(.01,color='.5',ls=':',lw=.8)
    fig.tight_layout();save(fig,'figure4')
    fig,axes=plt.subplots(1,2,figsize=(6.4,3.2))
    for ax,bot in zip(axes,['halluc','factual']):
        rows=a[bot]
        ax.plot(P,[rows[str(float(p))]['static']['rate']*100 for p in P],'o-',color='black',ms=3,label='Static')
        for sigma,marker in zip([.02,.05,.1,.2],['s','^','D','v']):
            ax.plot(P,[rows[str(float(p))][str(sigma)]['summary']['rate']*100 for p in P],marker+'-',ms=3,mfc='white',label=f'Diffusion {sigma}',color=str(.1+sigma*2))
        ax.plot(P,[rows[str(float(p))]['aware']['coupled']['rate']*100 for p in P],'x--',color='black',ms=4,label='Known feedback rule')
        labels(ax);ax.set_title('Hallucinating bot' if bot=='halluc' else 'Factual bot')
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',ncol=3,fontsize=6.5,frameon=False)
    fig.tight_layout(rect=(0,.19,1,1));save(fig,'figure5')
    fig,ax=plt.subplots(figsize=(6.4,3.2))
    rows=m['naive_halluc']
    for key,marker in [('fixed','o'),('coupled','s')]:
        y=rate(rows,key);ci=np.array([rows[str(float(p))][key]['wilson'] for p in P])*100
        ax.plot(P,y,marker+'-',ms=3,color='black',mfc='white' if key=='coupled' else 'black',label=key.capitalize())
        ax.fill_between(P,ci[:,0],ci[:,1],color='.3' if key=='fixed' else '.7',alpha=.25)
    labels(ax);ax.legend();fig.tight_layout();save(fig,'figure7')
    dist=read('final_pi');fig,axes=plt.subplots(2,3,figsize=(6.4,4.1))
    for ax,p in zip(axes.flat,[0.,.1,.2,.3,.5,.7]):
        values=np.array(dist[str(p)]);ax.hist(values,bins=np.linspace(0,1,26),weights=np.ones(len(values))*100/len(values),color='.65',edgecolor='black',lw=.3)
        ax.axvline(values.mean(),color='black',ls='--',lw=.8);ax.axvline(np.median(values),color='black',ls=':',lw=.8)
        ax.set_title(fr'$\pi_0={p}$; mean {values.mean():.3f}',fontsize=8);ax.set_xlabel(r'Final $\pi$');ax.set_ylabel('Conversations (%)')
    fig.tight_layout();save(fig,'figure8')
    fig,axes=plt.subplots(1,2,figsize=(6.4,3.25))
    for idx,(name,rows) in enumerate(s['sweep'].items()):
        alphas=[float(k) for k in rows];axes[0].plot(alphas,[v['rate']*100 for v in rows.values()],'-'+['o','s','^','D'][idx],ms=3,color='black',mfc='white' if idx else 'black',label=name.replace('_',' '))
    axes[0].axhline(m['naive_halluc']['0.3']['fixed']['rate']*100,color='.3',ls=':');axes[0].set_xlabel(r'Coupling strength ($\alpha$)');axes[0].set_ylabel('Threshold crossings (%)');axes[0].legend(fontsize=5.5)
    rows=s['forms']['0.0'];names=list(rows)
    axes[1].barh(range(len(names)),[rows[n]['summary']['rate']*100 for n in names],color='.65',edgecolor='black',lw=.5)
    axes[1].set_yticks(range(len(names)),[n.replace('_',' ') for n in names],fontsize=6)
    axes[1].axvline(m['naive_halluc']['0.0']['fixed']['rate']*100,color='black',ls=':');axes[1].set_xlabel('Threshold crossings (%)')
    axes[0].set_title(r'(A) Initial $\pi=0.3$');axes[1].set_title(r'(B) Initial $\pi=0$')
    fig.tight_layout();save(fig,'figure9')
    fig,axes=plt.subplots(1,2,figsize=(6.4,3.2))
    names=['none','rate_cap','exploration','smoothing','cap_exploration','combined']
    ticks=['None','Cap','Explore','Smooth','Cap +\nexplore','Combined'];x=np.arange(len(names))
    for key,offset,colour in [('fixed',-.18,'.85'),('coupled',.18,'.35')]:
        y=[mti['0.3'][n][key]['rate']*100 for n in names]
        axes[0].bar(x+offset,y,.35,color=colour,edgecolor='black',lw=.5,label=key.capitalize())
    axes[0].set_xticks(x,ticks,fontsize=6,rotation=35);axes[0].set_ylabel('Threshold crossings (%)');axes[0].legend();axes[0].set_title(r'(A) Matched interventions, $\pi_0=0.3$')
    for name,key,sty in [('none','fixed','o-'),('none','coupled','s-'),('combined','fixed','o--'),('combined','coupled','s--')]:
        axes[1].plot(P,[mti[str(float(p))][name][key]['rate']*100 for p in P],sty,ms=3,label=f'{name}, {key}',mfc='white',color='black')
    labels(axes[1]);axes[1].legend(fontsize=5.5);axes[1].set_title('(B) Combined versus matched baseline')
    fig.tight_layout();save(fig,'figure6')
    ctrl=read('feedback_controls');fig,ax=plt.subplots(figsize=(6.4,3.0))
    names=['coupled','donor_replay','donor_mean','shuffled_signals','fair_signals'];x=np.arange(3)
    for i,name in enumerate(names):
        y=[(ctrl[str(p)][name] if name=='coupled' else ctrl[str(p)][name]['summary'])['rate']*100 for p in [0.,.3,.5]]
        ax.bar(x+(i-2)*.15,y,.14,label=name.replace('_',' '),color=str(.1+i*.17),edgecolor='black',lw=.4)
    ax.set_xticks(x,['0.0','0.3','0.5']);ax.set_xlabel('Initial sycophancy rate');ax.set_ylabel('Threshold crossings (%)');ax.legend(ncol=2,fontsize=6)
    fig.tight_layout();save(fig,'figureS1')
    print('Saved nine manuscript figures and supplementary Figure S1')


if __name__=='__main__':main()
