#cleaning code
#  Cleaned data from
#  https://www.kaggle.com/code/hyunseokc/detecting-early-alzheimer-s/notebook
#

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

sns.set()

df = pd.read_csv('./oasis_longitudinal.csv')
print(df.head())

df = df.loc[df['Visit']==1] # use first visit data only because of the analysis we're doing
df = df.reset_index(drop=True) # reset index after filtering first visit data
df['M/F'] = df['M/F'].replace(['F','M'], [0,1]) # M/F column
df['Group'] = df['Group'].replace(['Converted'], ['Demented']) # Target variable
df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1,0]) # Target variable
df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1) # Drop unnecessary columns

##
##  visualzation
##

# bar drawing function
def bar_chart(feature):
    Demented = df[df['Group']==1][feature].value_counts()
    Nondemented = df[df['Group']==0][feature].value_counts()
    df_bar = pd.DataFrame([Demented,Nondemented])
    df_bar.index = ['Demented','Nondemented']
    df_bar.plot(kind='bar',stacked=True, figsize=(8,5))


# Gender  and  Group ( Femal=0, Male=1)
bar_chart('M/F')
plt.xlabel('Group')
plt.ylabel('Number of patients')
plt.legend()
plt.title('Gender and Demented rate')

plt.show()

##
##  data cleanin
##

# Check missing values by each column
print(pd.isnull(df).sum() )
# The column, SES has 8 missing values


##5.A Removing rows with missing values

# Dropped the 8 rows with missing values in the column, SES
df_dropna = df.dropna(axis=0, how='any')
print(pd.isnull(df_dropna).sum())

print(df_dropna['Group'].value_counts())


# 5.B Imputation
#
#Scikit-learn provides package for imputation [6], but we do it manually. Since the SES is a discrete variable, we use median for the imputation.

df["SES"].fillna(df.groupby("EDUC")["SES"].transform("median"), inplace=True)

# I confirm there're no more missing values and all the 150 data were used.
print(pd.isnull(df['SES']).value_counts())



fff = "./py_cleaned_oasis_longitudinal.csv"
df.to_csv(fff , index=False)
print("Cleaned data written to " , fff )


