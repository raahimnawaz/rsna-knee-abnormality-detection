"""The error budget behind docs/REVIEW-2026-10-12.md: where the five-fold model loses, and why.

    python fusion/error_budget.py [out_dir]

Three tables, from files already on disk (the five folds' held-out and gold predictions, the public report
readers, Stage 0's series table, the header probe). Gold-58 is REPORTED here, never selected on.
  1. per finding: held-out against the reference, and gold with each finding's share of the deficit
  2. report readers against the image model on gold, with a paired bootstrap
  3. held-out macro by scanner, field strength, report language and series count; sampled-slice spacing per plane
"""
import sys, subprocess, tempfile
from pathlib import Path
OUT = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
Path(OUT).mkdir(parents=True, exist_ok=True)
PARTS = {}
PARTS['per_finding'] = r'''
import pandas as pd, numpy as np, glob, sys
from sklearn.metrics import roc_auc_score
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
def auc(y,p):
    y=np.asarray(y); p=np.asarray(p); m=~np.isnan(y)&~np.isnan(p)
    y=y[m]; p=p[m]
    return np.nan if len(set(y))<2 else roc_auc_score(y,p)
R='fusion/'
# ---- OOF five folds
oof=pd.concat([pd.read_csv(R+f'runs_cnxs_cons2_{t}/runs_convnext_small/cv_last/oof_all.csv') for t in ('f0','f12','f34')]).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
print('OOF studies',len(oof))
ref=pd.read_csv('data/public_llm_labels/lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv').drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
tr=pd.read_csv('data/train.csv').set_index('StudyInstanceUID')
gold=tr.dropna(subset=L)
ids=[u for u in oof.index if u in ref.index and u not in gold.index]
Y=(ref.loc[ids,L].apply(pd.to_numeric,errors='coerce')>0.5).astype(float); P=oof.loc[ids,L]
tgt=pd.read_csv('data/student/targets_consensus_v2.csv').set_index('StudyInstanceUID')
print('\n== five-fold OOF vs the reference reader (n=%d)'%len(ids))
row=[]
for c in L:
    row.append((c,auc(Y[c],P[c]),Y[c].mean(), auc((tgt.loc[ids,c]>0.5).astype(float).where(tgt.loc[ids,c].notna()),P[c]), auc(Y[c].where(tgt.loc[ids,c].notna()), tgt.loc[ids,c])))
d=pd.DataFrame(row,columns=['finding','oof_vs_ref','ref_prev','oof_vs_own_target','target_vs_ref']).set_index('finding')
d['model/label']=(d.oof_vs_ref-0.5)/(d.target_vs_ref-0.5)
print(d.round(3).to_string()); print('MACRO',d.oof_vs_ref.mean().round(4), d.target_vs_ref.mean().round(4))
# ---- gold
def gload(p): return pd.read_csv(p).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID').loc[gold.index,L]
folds=[gload(R+'runs_cnxs_cons2_f0/runs_convnext_small/fold0/oof_gold.csv')]
for t,ks in (('f12',(1,2)),('f34',(3,4))):
    for k in ks: folds.append(gload(R+f'runs_cnxs_cons2_{t}/runs_convnext_small/fold{k}/oof_gold.csv'))
ens=sum(folds)/5
teach=gload(R+'runs_coatnet_oai_teacher_pa/runs_coatnet/oof_gold.csv')
G=(gold[L]>0).astype(int)
print('\ngold raw value counts (first 3 findings):'); 
for c in L[:3]: print(c, gold[c].value_counts().to_dict())
rk=lambda x: x.rank()/len(x)
blend=ens.apply(rk)+teach.apply(rk)
rng=np.random.default_rng(0)
def boot(p,c,n=2000):
    y=G[c].values; v=[]
    for _ in range(n):
        k=rng.integers(0,len(y),len(y))
        if len(set(y[k]))>1: v.append(roc_auc_score(y[k],p[c].values[k]))
    return np.std(v)
rows=[]
for c in L:
    rows.append((c,int(G[c].sum()),auc(G[c],folds[0][c]),auc(G[c],ens[c]),boot(ens,c),auc(G[c],teach[c]),auc(G[c],blend[c])))
g=pd.DataFrame(rows,columns=['finding','n_pos','fold0','five_fold','se','teacher','5f+teacher']).set_index('finding')
g['deficit_share']=(1-g.five_fold)/(1-g.five_fold).sum()
print('\n== gold-58, report only'); print(g.round(3).to_string()); print('MACRO',g[['fold0','five_fold','teacher','5f+teacher']].mean().round(4).to_dict())
g.to_csv(sys.argv[1]+'/gold_per_finding.csv'); d.to_csv(sys.argv[1]+'/oof_per_finding.csv'); ens.to_csv(sys.argv[1]+'/gold_ens.csv')

'''
PARTS['report_vs_image'] = r'''
import sys, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'labeling')
import build_consensus_targets as B
L=B.LABELS
val,adr=B.load_readers(Path('data/public_llm_labels'))
tr=pd.read_csv('data/train.csv').set_index('StudyInstanceUID'); gold=tr.dropna(subset=L); G=(gold[L]>0).astype(int)
ens=pd.read_csv(sys.argv[1]+'/gold_ens.csv').set_index('StudyInstanceUID').loc[gold.index]
def a(y,p):
    m=p.notna(); 
    return np.nan if m.sum()<40 or y[m].nunique()<2 else roc_auc_score(y[m],p[m])
cov={k:int(v.reindex(gold.index)[L].notna().all(axis=1).sum()) for k,v in val.items()}
print('gold rows fully read, per reader:',cov)
keep=[k for k in val if cov[k]>=50]
tab=pd.DataFrame({k:[a(G[c],val[k].reindex(gold.index)[c]) for c in L] for k in keep},index=L)
rk=lambda s: s.rank(pct=True)
cons=pd.concat([val[k].reindex(gold.index)[L].apply(rk) for k in keep]).groupby(level=0).mean().loc[gold.index]
tab['report_consensus']=[a(G[c],cons[c]) for c in L]
tab['best_report']=tab[keep].max(axis=1)
tab['our_5fold']=[a(G[c],ens[c]) for c in L]
both=cons.apply(rk)+ens.apply(rk)
tab['report+image']=[a(G[c],both[c]) for c in L]
tab['image_minus_report']=tab.our_5fold-tab.report_consensus
tab['n_pos']=G.sum().values
print(tab.round(3).to_string()); print('MACRO',tab.drop(columns='n_pos').mean().round(4).to_dict())
# paired bootstrap of image - report consensus per finding
rng=np.random.default_rng(0); out={}
for c in L:
    y=G[c].values; v=[]
    for _ in range(3000):
        k=rng.integers(0,58,58)
        if len(set(y[k]))>1: v.append(roc_auc_score(y[k],ens[c].values[k])-roc_auc_score(y[k],cons[c].values[k]))
    out[c]=(np.mean(v),np.std(v),np.mean(np.array(v)>0))
print('\npaired (image - report) on gold: mean, sd, P(>0)'); 
for c in L: print(f'{c:18s} {out[c][0]:+.3f} ± {out[c][1]:.3f}  P {out[c][2]:.2f}')
tab.to_csv(sys.argv[1]+'/gold_report_vs_image.csv')

'''
PARTS['subgroups'] = r'''
import pandas as pd, numpy as np, re
from sklearn.metrics import roc_auc_score
L=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
R='fusion/'
oof=pd.concat([pd.read_csv(R+f'runs_cnxs_cons2_{t}/runs_convnext_small/cv_last/oof_all.csv') for t in ('f0','f12','f34')]).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
ref=pd.read_csv('data/public_llm_labels/lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv').drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
tr=pd.read_csv('data/train.csv').set_index('StudyInstanceUID')
ids=[u for u in oof.index if u in ref.index]
Y=(ref.loc[ids,L].apply(pd.to_numeric,errors='coerce')>0.5).astype(int); P=oof.loc[ids,L]
h=pd.read_parquet('data/external/metadata_probe/headers.parquet')
h['fs']=pd.to_numeric(h.MagneticFieldStrength,errors='coerce').round(1)
h['ns']=h.n_slices
st=h.groupby('StudyInstanceUID').agg(man=('Manufacturer',lambda s:s.mode().iat[0] if len(s.mode()) else 'NA'),fs=('fs','median'),nser=('SeriesInstanceUID','nunique'),has3d=('MRAcquisitionType',lambda s:(s=='3D').any()),maxsl=('ns','max'),sex=('PatientSex','first'))
s0=pd.read_csv('data/stage0/stage0_series.csv')
pl=s0.groupby(['study','plane']).size().unstack(fill_value=0)
st=st.join(pl).reindex(ids)
rep=tr.Report.reindex(ids).fillna('')
def lang(t):
    t=' '+t.lower()+' '
    sc={'en':len(re.findall(r' (the|of|is|there|with|no|and) ',t)),'es':len(re.findall(r' (de|la|el|del|rotura|derrame|sin|con|rodilla) ',t)),'nl':len(re.findall(r' (de|het|een|van|geen|met|knie|scheur) ',t)),'pt':len(re.findall(r' (do|da|com|sem|não|joelho|rotura|derrame) ',t)),'de':len(re.findall(r' (der|die|das|und|kein|keine|mit|des) ',t)),'fr':len(re.findall(r' (le|les|du|des|pas|avec|genou|sans) ',t)),'it':len(re.findall(r' (del|della|il|di|con|ginocchio|lesione|non) ',t)),'tr':len(re.findall(r' (ve|ile|izlenmektedir|normal|menisküs|diz|bir) ',t))}
    k=max(sc,key=sc.get); return k if sc[k]>=3 else 'other'
st['lang']=rep.map(lang).values; st['replen']=rep.str.len().values
def macro(m):
    v=[]
    for c in L:
        y=Y.loc[m,c]; 
        if y.nunique()>1 and y.sum()>=8: v.append(roc_auc_score(y,P.loc[m,c]))
    return np.mean(v) if len(v)>=10 else np.nan, len(v)
def table(col,bins=None,minn=120):
    g=st[col] if bins is None else pd.cut(st[col],bins)
    rows=[]
    for k,idx in g.groupby(g,observed=True).groups.items():
        if len(idx)>=minn:
            m,nv=macro(idx); rows.append((str(k),len(idx),round(m,4) if m==m else None,nv))
    print(f'-- by {col}'); print(pd.DataFrame(rows,columns=[col,'n','macro_vs_ref','n_findings']).sort_values('n',ascending=False).to_string(index=False))
print('all',len(ids),round(macro(pd.Index(ids))[0],4))
table('man'); table('fs'); table('lang'); table('nser',[0,3,4,5,6,8,30]); table('Coronal',[-1,0,1,2,9]); table('Axial',[-1,0,1,9]); table('Sagittal',[-1,1,2,9]); table('has3d'); table('replen',[0,200,400,700,1200,99999])
# per-finding by field strength and by language (weak findings)
for col in ('fs','lang'):
    print(f'-- per finding by {col} (groups n>=400)')
    out={}
    for k,idx in st.groupby(col).groups.items():
        if len(idx)>=400: out[f'{k} (n={len(idx)})']=[roc_auc_score(Y.loc[idx,c],P.loc[idx,c]) if Y.loc[idx,c].sum()>=8 else np.nan for c in L]
    print(pd.DataFrame(out,index=L).round(3).to_string())


'''

SPACING = r"""
import pandas as pd, numpy as np
d=pd.read_csv('data/stage0/stage0_series.csv'); d=d[d.error.isna()]
d['sp']=d.spacing_between_mm.fillna(d.slice_thickness_mm)
rows=[]
for (st,pl),g in d[d.n_files<=80].groupby(['study','plane']):
    g2=g if pl=='Axial' or not (g.fluid==1).any() else g[g.fluid==1]
    r=g2.sort_values('n_files',ascending=False).iloc[0]; rows.append((pl,r.n_files,r.sp))
t=pd.DataFrame(rows,columns=['plane','n','sp'])
print('-- sampled-slice spacing, Stage B slots')
for pl,slot in (('Sagittal',26),('Coronal',16),('Axial',14)):
    x=t[t.plane==pl]; mm=((x.n-1)*0.96/(slot-1)).clip(lower=1)*x.sp
    print(f'{pl:9s} slot {slot:2d}: series median {x.n.median():.0f} slices | kept {np.minimum(slot/x.n,1).median():.0%} | every {mm.median():.1f} mm | a window spans {2*mm.median():.1f} mm')
"""
if __name__ == "__main__":
    for name in ("per_finding", "report_vs_image", "subgroups"):
        print(f"\n########## {name}", flush=True)
        subprocess.run([sys.executable, "-c", PARTS[name], OUT], check=True)
    subprocess.run([sys.executable, "-c", SPACING], check=True)
