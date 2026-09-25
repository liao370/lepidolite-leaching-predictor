"""Deterministic feature mapping shared by training and the prediction app."""
import numpy as np
import pandas as pd

ORE=['Li2O','Rb','Cs','SiO2','Al2O3','Fe2O3']
ADDITIVES=['H2SO4','HCl','K2S2O7','KHSO4','FeSO4_7H2O','KOH','CaO','NaCl','CaCl2','SLS','NaOH','CaOH2','NH4_2SO4','Na2SO4','CaSO4','CaCO3','K2SO4','NaHSO4','C']
PROCESS=['Roast_temp','Roast_time','Liquid_solid','Leach_temp','Leach_time']
RAW_FEATURES=['Total_ratio']+ORE+ADDITIVES+PROCESS
FAMILIES={
 'Sulfate_sum':['H2SO4','K2S2O7','KHSO4','FeSO4_7H2O','NH4_2SO4','Na2SO4','CaSO4','K2SO4','NaHSO4'],
 'Chloride_sum':['HCl','NaCl','CaCl2'],
 'Alkaline_sum':['KOH','CaO','NaOH','CaOH2','CaCO3'],
 'Calcium_sum':['CaO','CaCl2','CaOH2','CaSO4','CaCO3'],
 'Acid_sum':['H2SO4','HCl','KHSO4','NaHSO4'],
 'Sodium_sum':['NaCl','SLS','NaOH','Na2SO4','NaHSO4'],
 'Potassium_sum':['K2S2O7','KHSO4','KOH','K2SO4']}
ENGINEERED=['Additive_sum','Additive_count','Single_system','Combined_system']+[a+'_fraction' for a in ADDITIVES]+list(FAMILIES)+['Roast_intensity','Leach_intensity','Liquid_time','Li_Al_ratio','Si_Al_ratio','Fe_Al_ratio','Dose_roast','Dose_leach','Roast_leach']
FEATURE_NAMES=RAW_FEATURES+ENGINEERED
assert len(RAW_FEATURES)==31 and len(FEATURE_NAMES)==70

def engineer_features(x):
    d=pd.DataFrame(np.asarray(x,dtype=float),columns=RAW_FEATURES)
    s=d[ADDITIVES].sum(axis=1)
    n=(d[ADDITIVES]>0).sum(axis=1)
    o=d.copy()
    o['Additive_sum']=s;o['Additive_count']=n
    o['Single_system']=(n==1).astype(float);o['Combined_system']=(n>1).astype(float)
    for a in ADDITIVES:o[a+'_fraction']=np.divide(d[a],s,out=np.zeros(len(d)),where=s.to_numpy()!=0)
    for f,cols in FAMILIES.items():o[f]=d[cols].sum(axis=1)
    o['Roast_intensity']=d.Roast_temp*d.Roast_time
    o['Leach_intensity']=d.Leach_temp*d.Leach_time
    o['Liquid_time']=d.Liquid_solid*d.Leach_time
    for p,num in [('Li','Li2O'),('Si','SiO2'),('Fe','Fe2O3')]:o[p+'_Al_ratio']=d[num]/d.Al2O3.clip(lower=1e-6)
    o['Dose_roast']=d.Total_ratio*d.Roast_temp
    o['Dose_leach']=d.Total_ratio*d.Leach_temp
    o['Roast_leach']=d.Roast_temp*d.Leach_temp
    return o[FEATURE_NAMES].to_numpy()
