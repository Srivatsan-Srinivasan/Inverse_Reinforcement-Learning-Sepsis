import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("/Users/linyingzhang/Documents/2017_Fall/CS282/Sepsis_imp.csv")

# Variables to exclude (no good "normal definition", or are actions)
to_exclude = {7,8,9,13,14,19,28,47,48,49,50,51,52,53,57,58}

col_include = [element for i, element in enumerate(df.columns) if i not in to_exclude]

# OneHotEncode SOFA and SIRS
enc = OneHotEncoder()

df_SOFA = pd.DataFrame(enc.fit_transform(df['SOFA'].values.reshape(-1, 1)).toarray(), 
                       columns=["SOFA_"+str(i) for i in range(24)])
df_SIRS = pd.DataFrame(enc.fit_transform(df['SIRS'].values.reshape(-1, 1)).toarray(),
                      columns=["SIRS_"+str(i) for i in range(len(df['SIRS'].unique()))])

# Binary encode other features
df_binary = pd.DataFrame(columns = col_include)
df_binary['bloc'] = df['bloc']
df_binary['icustayid'] = df['icustayid']
df_binary['charttime'] = df['charttime']
df_binary['gender'] = df['gender']
df_binary['age'] = np.where(df['age']<=85,0,1)
df_binary['elixhauser']= np.where(df['elixhauser']<3,0,1)
df_binary['re_admission'] = df['re_admission']
# df_binary['Weight_kg']
df_binary['GCS'] = np.where(df['GCS']>=8,0,1)
df_binary['HR'] = np.where(np.logical_and(df['HR']<=90, df['HR']>=60),0,1)
df_binary['SysBP']  = np.where(df['SysBP']>90,0,1)
# df_binary['MeanBP'] 
# df_binary['DiaBP']  
df_binary['Shock_Index'] = np.where(np.logical_and(df['Shock_Index']<=0.7, df['Shock_Index']>=0.5),0,1)
df_binary['RR'] = np.where(np.logical_and(df['RR']<=20, df['RR']>=12),0,1)
df_binary['SpO2'] = np.where(df['SpO2']>=90,0,1)
df_binary['Temp_C'] = np.where(np.logical_and(df['Temp_C']<=38, df['Temp_C']>=36),0,1)
# df_binary['FiO2_1'] 
df_binary['Potassium'] = np.where(np.logical_and(df['Potassium']<=3.5, df['Potassium']>=5.0),0,1)
df_binary['Sodium'] = np.where(df['Sodium']>120,0,1)
df_binary['Chloride'] = np.where(np.logical_and(df['Chloride']<=106, df['Chloride']>=96),0,1)
df_binary['Glucose'] = np.where(np.logical_and(df['Glucose']<=180, df['Chloride']>=70),0,1)
df_binary['BUN'] = np.where(np.logical_and(df['BUN']<=20, df['BUN']>=7),0,1) 
df_binary['Creatinine'] = np.where(np.logical_and(df['Creatinine']<=1.2, df['Creatinine']>=0.5),0,1) 
df_binary['Magnesium'] = np.where(np.logical_and(df['Magnesium']<=2.5, df['Magnesium']>=1.5),0,1) 
df_binary['Calcium'] = np.where(np.logical_and(df['Calcium']<=10.2, df['Calcium']>=8.5),0,1) 
# df_binary['Ionised_Ca'] 
df_binary['CO2_mEqL'] = np.where(np.logical_and(df['CO2_mEqL']<=29, df['CO2_mEqL']>=23),0,1) 
df_binary['SGOT'] = np.where(np.logical_and(df['SGOT']<=40, df['SGOT']>=5),0,1) 
df_binary['SGPT'] = np.where(np.logical_and(df['SGPT']<=56, df['SGPT']>=7),0,1)
df_binary['Total_bili'] = np.where(np.logical_and(df['Total_bili']<=1.2, df['Total_bili']>=0.1),0,1)
df_binary['Albumin'] = np.where(np.logical_and(df['Albumin']<=5.5, df['Total_bili']>=3.5),0,1)
df_binary['Hb'] = np.where(np.logical_and(df['Hb']<=17.5, df['Total_bili']>=12),0,1)
df_binary['WBC_count'] = np.where(np.logical_and(df['WBC_count']<=11, df['WBC_count']>=4.5),0,1)
df_binary['Platelets_count'] = np.where(np.logical_and(df['Platelets_count']<=450, df['Platelets_count']>=150),0,1)
df_binary['PTT'] = np.where(np.logical_and(df['PTT']<=35, df['PTT']>=25),0,1)
df_binary['PT'] = np.where(np.logical_and(df['PT']<=13.5, df['PT']>=11),0,1)
df_binary['INR'] = np.where(df['INR']<=1.1,0,1)
df_binary['Arterial_pH'] = np.where(np.logical_and(df['Arterial_pH']<=7.45, df['Arterial_pH']>=7.35),0,1)
df_binary['paO2'] = np.where(df['paO2']>=80,0,1)
df_binary['paCO2'] = np.where(df['paCO2']<=45,0,1)
df_binary['Arterial_BE'] = np.where(np.logical_and(df['Arterial_BE']<=4, df['Arterial_BE']>=-4),0,1)
df_binary['Arterial_lactate'] = np.where(df['Arterial_lactate']<=2,0,1)
df_binary['HCO3'] = np.where(np.logical_and(df['HCO3']<=28, df['HCO3']>=22),0,1)
df_binary['PaO2_FiO2'] = np.where(df['PaO2_FiO2']>=300,0,1)
# df_binary['median_dose_vaso'] 
# df_binary['max_dose_vaso'] 
# df_binary['input_total_tev'] 
# df_binary['input_4hourly_tev'] 
# df_binary['output_total'] 
# df_binary['output_4hourly'] 
# df_binary['cumulated_balance_tev'] 
# df_binary['sedation'] = df['sedation'] 
df_binary['mechvent'] = df['mechvent'] 
df_binary['rrt'] = df['rrt']

# Concat binary features with OneHot SOFA and SIRS, and the outcome
df_binary_compl = pd.concat([df_binary, df_SOFA, df_SIRS, df.iloc[:, -2:]], axis=1)
# save binary dataframe
pd.DataFrame.to_csv(df_binary_compl, "Sepsis_binary_1ed.csv")


########################
## Correlation Matrix ##
########################
from string import ascii_letters
import seaborn as sns
import matplotlib.pyplot as plt
# Compute the correlation matrix
corr = df_binary_compl.iloc[:,3:].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.6, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig("Corr_binary_features.png")
