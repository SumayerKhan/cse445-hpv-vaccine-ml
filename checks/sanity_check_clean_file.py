"""Independent sanity check of data/hpv_clean.csv against the raw file s003.csv.
Run from the project folder:  .venv/bin/python checks/sanity_check_clean_file.py
It does not reuse any code from notebooks/01_clean.ipynb.
"""
import pandas as pd, numpy as np, itertools, warnings
warnings.filterwarnings('ignore')
raw=pd.read_csv('data/raw/s003.csv');c=pd.read_csv('data/hpv_clean.csv')
P=lambda ok,msg:print(('PASS  ' if ok else 'FAIL  ')+msg)
print('raw',raw.shape,'clean',c.shape)
# 1 row accounting, done independently
r=raw.copy()
no_t=r.vac_accept.isna()|r.vac_if_gov.isna()
r1=r[~no_t]
dup=r1.drop(columns=['sl']).duplicated()
r2=r1[~dup].reset_index(drop=True)
print('rows without a target:',int(no_t.sum()),'| duplicate rows:',int(dup.sum()),'| left:',len(r2))
P(len(r2)==len(c),'1. row count after dropping no-target rows and the duplicate = clean file')
# 2 alignment: untouched columns equal row by row
same=[k for k in ['parent','age','hcw','family_member','children_0','daugther_0','hpv_heard','cancer_hear','pap_done','cu1','cu2','cu3','cu4','vul1','ben3','bar5'] if k in c.columns]
def same_col(k):
    if pd.api.types.is_numeric_dtype(c[k]):
        x=pd.to_numeric(r2[k],errors='coerce');return bool(((x==c[k])|(x.isna()&c[k].isna())).all())
    return bool((r2[k].astype(str).str.strip().str.lower().values==c[k].astype(str).str.strip().str.lower().values).all())
al={k:same_col(k) for k in same}
P(all(al.values()),'2. clean rows are the raw rows in the same order (checked on %d untouched columns)'%len(same)+('' if all(al.values()) else ' -> '+str({k:v for k,v in al.items() if not v})))
# 3 targets re-derived from raw
va=r2.vac_accept.str.strip().str.lower();vg=r2.vac_if_gov.str.strip().str.lower()
t3=np.where(va=='yes','Will vaccinate',np.where(vg=='yes','Only if free','Undecided or no'))
P((t3==c.target_3class.values).all(),'3. target_3class equals my own re-derivation from vac_accept and vac_if_gov, all rows')
t2=np.where(r2.accept.str.strip()=='Yes','Accept','Not accept')
P((t2==c.target_2class.values).all(),'4. target_2class equals the authors\' accept column, all rows')
print('     yes then not-yes-if-free (kept as Yes):',int(((va=='yes')&(vg!='yes')).sum()))
# 5 leakage: can any single feature almost give the target?
feats=[k for k in c.columns if not k.startswith(('target_','meta_'))]
worst=[]
for k in feats:
    s=c[k].astype(str) if not pd.api.types.is_numeric_dtype(c[k]) else pd.qcut(c[k],10,duplicates='drop').astype(str)
    acc=pd.crosstab(s,c.target_3class).max(axis=1).sum()/len(c);worst.append((acc,k))
worst.sort(reverse=True)
print('     best single-feature rule accuracy:',[(k,round(a,3)) for a,k in worst[:4]],'| baseline',round((c.target_3class=='Will vaccinate').mean(),3))
P(worst[0][0]<0.85,'5. no single feature gives the target away')
left=[k for k in ['accept','vac_accept','vac_if_gov','sl','div','dist','division','district','epi2','rel_imp','checkup','eligible','vul','sev','ben','bar','cue','vul_cat','sev_cat','ben_cat','bar_cat','age_cat','income_cat','occup2','marital2','routine_health2','routine_health3','family_type'] if k in c.columns]
P(not left,'6. no leakage, copy or regrouped column is left'+(' -> '+str(left) if left else ''))
# 7 skip logic
gates={'hpv_heard':[k for k in c.columns if k.startswith('hpv_') and k not in('hpv_heard','hpv_cervical','hpv_transmit','hpv_sexual','hpv_blood','hpv_pubtoilet','hpv_mtc')],
       'vac_hear':[k for k in c.columns if k.startswith('vac_') and k not in('vac_hear','vac_reduce_risk')],
       'cancer_hear':[k for k in c.columns if k.startswith('cancer_') and k not in('cancer_hear','cancer_female')],
       'canvac_hear':[k for k in c.columns if k.startswith('canvac_') and k!='canvac_hear']}
bad=0;lost=0
for g,cols in gates.items():
    for k in cols:
        bad+=int(((c[g]=='No')&(c[k]!='No')).sum())
        rr=r2[k].astype(str).str.strip();m=r2[k].notna().values
        lost+=int((rr[m].str.lower().values!=c[k][m].astype(str).str.lower().values).sum())
P(bad==0,'7. a parent who never heard of the topic has "No" in every "where did you hear" column (violations: %d)'%bad)
P(lost==0,'8. no real answer was overwritten when the empty cells were filled (changed real answers: %d)'%lost)
# 9 text hygiene
issues=[]
for k in c.select_dtypes(exclude='number').columns:
    v=c[k].dropna().unique()
    low=pd.Series(v).str.strip().str.lower()
    if low.duplicated().any(): issues.append((k,'same value written two ways',list(v)))
    if any(x!=x.strip() for x in v): issues.append((k,'space at start or end',[x for x in v if x!=x.strip()]))
P(not issues,'9. no value written two ways, no stray spaces'+('' if not issues else ' -> '+str(issues[:4])))
lik=sorted(set(itertools.chain.from_iterable(c[k].dropna().unique() for k in c.columns if k[:3] in('vul','sev','ben','bar'))))
P(len(lik)==5,'10. the 20 belief features use one set of five answers: '+str(lik))
# 11 numbers
num=c.select_dtypes('number');print(num.describe().loc[['min','50%','max']].round(1).to_string())
P((c.daughter_age_youngest.dropna()<=c.daughter_age_oldest.dropna()).all(),'11. youngest daughter age <= oldest daughter age')
P(c.age.between(18,80).all(),'12. parent age in a possible range')
# 13 duplicates / constants
P(c.duplicated().sum()==0,'13. no duplicate rows')
const=[k for k in feats if c[k].nunique(dropna=True)<=1];P(not const,'14. no constant feature'+(' -> '+str(const) if const else ''))
twins=[(a,b) for a,b in itertools.combinations(feats,2) if c[a].astype(str).equals(c[b].astype(str))];P(not twins,'15. no two features are exact copies'+(' -> '+str(twins) if twins else ''))
# 16 column accounting
dropped=[k for k in raw.columns if k not in c.columns];new=[k for k in c.columns if k not in raw.columns]
print('     raw columns not in clean (%d):'%len(dropped),dropped);print('     new columns (%d):'%len(new),new)
P(len(raw.columns)-len(dropped)+len(new)==c.shape[1],'16. column count adds up: 131 - %d + %d = %d'%(len(dropped),len(new),c.shape[1]))
# 17 rare categories
rare=[(k,v,int(n)) for k in c.select_dtypes(exclude='number').columns if not k.startswith(('meta_','target_')) for v,n in c[k].value_counts().items() if n<10]
print('     categories with fewer than 10 parents:',rare)
# 18 misspellings kept from raw
print('     occupation values:',sorted(c.occupation.dropna().unique()));print('     marital_status values:',sorted(c.marital_status.dropna().unique()))
print('     districts:',c.meta_district.nunique(),sorted(c.meta_district.unique())[:45])
