# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:03:11 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:08:33 2019

@author: user
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#data=pd.read_csv("C:/Users/user/Documents/Python/Customer_Base_Najbolja_201609.csv",encoding='latin1', dtype = {"HANDLING_CHANNEL" : "str", "CUSTOMER_ID" : "str", "SUBSCRIBER_ID" : "str", "MONTHS_TO_MCD_END" : "int",
#                  "TARIF_MODEL" : "str", "MCD_YES_NO" :"str", "REVENUE" : "int", "SUSUBSCRIBER_ACT_DATE" : "str", "CB_DATE" : "str"})                                                                                                        
#Customer Base Najbolja 201609
df1=pd.read_csv("C:/Users/user/Documents/Python/Customer_Base_Najbolja_201609.csv", encoding ='latin')
print(type(df1))

print("DATA COLUMNS")
print(df1.columns)
print("\nDATA INDEX")
print(df1.index)
print("\nDATA HEAD")
print(df1.head(5),"\n")

#Identifying columns with missing values - creating bool series True for NaN values
miss_column_1 = []    
for column_name in df1.columns:  
    bool_series = pd.isnull(df1[column_name])
    print(column_name, "has total", len(bool_series), "values")
    count = 0
    for i in bool_series:
       if (i == True):
          count = count + 1
    if(count != 0):
        miss_column_1.append(column_name)
        print("Warning:", column_name, "has", count, "missing values\n")
print(df1.dtypes,"\n")
print("Colmns with missing values are: ")  
for i in miss_column_1:
      print(i,"\n")
  #Handling identified colums with missing values
for column_name in df1.columns:
    if column_name in miss_column_1:
         print("Handling missing values for column", column_name)
         bool_series = pd.isnull(df1[column_name])
         df_miss = df1[bool_series]
         print(df_miss.head())
         print(len(df_miss))
         df1[column_name] = df1[column_name].fillna(0)
         print(df1.isnull().sum())  
print(df1.dtypes)
#REVENUE: Mjesečni prihod po priključku
print(df1['REVENUE'].describe(),"\n")
#MONTHS_TO_MCD_END: Broj mjeseci do isteka ugovorne obveze
print(df1['MONTHS_TO_MCD_END'].describe(),"\n")
#Broj onih koji imaju ugovorenu obvezu
print(df1['MCD_YES_NO'].value_counts())

#Churn 201610
df2=pd.read_csv("C:/Users/user/Documents/Python/Churn_201610.csv", encoding ='latin')
print(type(df2))
print("DATA COLUMNS")
print(df2.columns)
print("\nDATA INDEX")
print(df2.index)
print("\nDATA HEAD")
print(df2.head(5),"\n")
#Identifying columns with missing values - creating bool series True for NaN values
miss_column_2 = []    
for column_name in df2.columns:  
    bool_series = pd.isnull(df2[column_name])
    count = 0
    for i in bool_series:
       if (i == True):
          count = count + 1
    if(count != 0):
        miss_column_2.append(column_name)
        print("Warning:", column_name, "has", count, "missing values\n")
print(df2.dtypes,"\n")

#Handling identified colums with missing values
df2["PORT_OUT_PROFILE"].fillna("Nema prijenosa broja", inplace = True)  
print("Missing values check")
print(df2.isnull().sum(),"\n")
print("PORT PROFILE VALUE distribution")
print(df2["PORT_OUT_PROFILE"].value_counts())    

df_merged = pd.merge(df1, df2, on='SUBSCRIBER_ID', how='inner')
#df_merged.set_index('SUBSCRIBER_ID', inplace = True)
print(df_merged.columns)
#print(len(df_merged))
print(df_merged.head(5))

#Isključeni priključci prema modelu tarife
print(df_merged.TARIFF_MODEL.unique())
df_tariff = df_merged.groupby(by='TARIFF_MODEL').size()
print(df_tariff)
print(df_merged['TARIFF_MODEL'].value_counts())
#sns.countplot(x='TARIFF_MODEL', data=df_merged, palette='hls')
sns.countplot(y='TARIFF_MODEL', data=df_merged, palette= 'hls')
plt.title('Frequency of churn by Tariff model')
plt.show()
plt.savefig('count_plot')
#Ugovorena obaveza - MCD_YES_NO (Y ako ima ugovornu obvezu, N ako nema)
print(df_merged.MCD_YES_NO.unique())
df_mcd = df_merged.groupby(by='MCD_YES_NO').size()
print(df_mcd)
#print(df_merged['MCD_YES_NOL'].value_counts())
sns.countplot(x='MCD_YES_NO', data=df_merged, palette= 'hls')
plt.title('Frequency of churn by MCD')
plt.show()
plt.savefig('count_plot')

#Ugovorena obaveza - MCD_YES_NO (Y ako ima ugovornu obvezu, N ako nema)
print(df_merged.MCD_YES_NO.unique())
df_mcd = df_merged.groupby(by='MCD_YES_NO').size()
print("Distribucija isključenih priključaka prema ugovorenoj obavezi:")
print(df_mcd)
#print(df_merged['MCD_YES_NOL'].value_counts())
sns.countplot(x='MCD_YES_NO', data=df_merged, palette= 'hls')
plt.title('Frequency of churn by MCD')
plt.show()

#Mjesečni prihod po priključku - REVENUE
min_revenue = df_merged['REVENUE'].min()
max_revenue = df_merged['REVENUE'].max()
print("Minimalni prihod po isključenom priključku:",min_revenue)
print("Maksimalni prihod po isključenom priključku:", max_revenue)
range = (min_revenue, max_revenue) # setting the ranges and no. of intervals
bins = 50
plt.hist(df_merged['REVENUE'], bins, range, color = 'black',
        histtype = 'bar', rwidth = 0.3)
plt.xlabel('REVENUE')
plt.ylabel('No. of churns')
plt.title('Histogram of chruns by revenue')
plt.show()

df_merg_rev_poz = df_merged[df_merged['REVENUE'] > 0 ]
print(df_merg_rev_poz.head(5))
min_rev_poz = df_merg_rev_poz['REVENUE'].min()
print("Minimalni pozitivni prihod po isključenom priključku:",min_rev_poz)
print("Maksimalni prihod po isključenom priključku:", max_revenue)
range = (min_rev_poz, max_revenue) # setting the ranges and no. of intervals
bins = 50
plt.hist(df_merg_rev_poz['REVENUE'], bins, range, color = 'black',
        histtype = 'bar', rwidth = 0.3)
plt.xlabel('REVENUE')
plt.ylabel('No. of churns')
plt.title('Histogram of chruns by revenue')
plt.show()
print("Mean value for revenue is:", df_merg_rev_poz['REVENUE'].mean())

#The interquartile range (IQR), also called the midspread or middle 50%, or technically H-spread,
#is a measure of statistical dispersion, being equal to the difference between 75th and 25th percentiles,
#or between upper and lower quartiles, IQR = Q3 − Q1.
#It is a measure of the dispersion similar to standard deviation or variance, but is much more robust against outliers.
#data_ill_female=data[(data['Illness'] == 'Yes') & (data['Gender'] == 'Female')]

Q1 = df_merg_rev_poz['REVENUE'].quantile(0.25)
print("Q1:", Q1)
Q3 = df_merg_rev_poz['REVENUE'].quantile(0.75)
print("Q3:", Q3)
IQR = Q3 - Q1
print("IQR:", IQR)
sns.boxplot(x=df_merg_rev_poz['REVENUE'])
plt.show()
#Detecting revenue outliers
df_merg_rev_outl = df_merg_rev_poz[((df_merg_rev_poz['REVENUE'] < (Q1 - 1.5 * IQR))|(df_merg_rev_poz['REVENUE'] > (Q3 + 1.5 * IQR)))]
print(df_merg_rev_outl['REVENUE'])
rev_otul_list = df_merg_rev_outl['REVENUE'].tolist()
print(rev_otul_list)
#Removing outliers
df_merg_rev_final = df_merg_rev_poz[~df_merg_rev_poz['REVENUE'].isin(rev_otul_list)]
print(df_merg_rev_final['REVENUE'])

min_rev_fin = df_merg_rev_final['REVENUE'].min()
max_rev_fin = df_merg_rev_final['REVENUE'].max()
print("Minimalni prihod po priključku:",min_rev_fin )
print("Maksimalni prihod po priključku:", max_rev_fin )
range = (min_rev_fin, max_rev_fin) # setting the ranges and no. of intervals
bins = 50
plt.hist(df_merg_rev_final['REVENUE'], bins, range, color = 'black',
        histtype = 'bar', rwidth = 0.3)
plt.xlabel('REVENUE')
plt.ylabel('No. of churns')
plt.title('Histogram of chruns by revenue')
plt.show()

df_rev_count = pd.DataFrame(df_merg_rev_final['REVENUE'].value_counts())
df_rev_count.reset_index(inplace=True)
print(df_rev_count.head(5))
print(df_rev_count.columns)
df_rev_count.rename(columns={'index': 'REVENUE', 'REVENUE': 'REV_COUNT'}, inplace=True)
print(df_rev_count.columns)
df_rev_count_top = df_rev_count.head(5)
min_rev = df_rev_count_top['REVENUE'].min()
max_rev = df_rev_count_top['REVENUE'].max()
print("Minimalni prihod po priključku:",min_rev)
print("Maksimalni prihod po priključku:", max_rev)
range = (min_rev, max_rev) # setting the ranges and no. of intervals
bins = 5
plt.hist(df_rev_count_top['REVENUE'], bins, range, color = 'blue',
        histtype = 'bar', rwidth = 0.3)
plt.xlabel('REVENUE')
plt.ylabel('No. of churns')
plt.title('Histogram of chruns by revenue')
plt.show()

print(df_rev_count_top)
df_rev_count_top.plot(kind='bar',x='REVENUE',y='REV_COUNT',color='red')
plt.show()

#Kanal prodaje koji je zadužen za upravljanje korisnikom - HANDLING_CHANNEL
print(df_merged.HANDLING_CHANNEL.unique())
print(df_merged['HANDLING_CHANNEL'].value_counts())
sns.countplot(y='HANDLING_CHANNEL', data=df_merged, palette= 'hls')
plt.title('Frequency of churn by handling channel')
plt.show()

#Broj mjeseci do isteka ugovorne obveze - MONTHS_TO_MCD_END
print(df_merged.MONTHS_TO_MCD_END.unique())
df_months_end_count = df_merged['MONTHS_TO_MCD_END'].value_counts()
print(df_months_end_count.head(10))

min_months_end = df_merged['MONTHS_TO_MCD_END'].min()
max_months_end = df_merged['MONTHS_TO_MCD_END'].max()
print("Minimalni broj mjeseci do isteka obaveze po isključenom priključku:", min_months_end)
print("Maksimalni broj mjeseci do isteka obaveze po isključenom priključku:", max_months_end)
range = (min_months_end, max_months_end) # setting the ranges and no. of intervals
bins = 5
plt.hist(df_merged['MONTHS_TO_MCD_END'], bins, range, color = 'blue',
        histtype = 'bar', rwidth = 0.3)
plt.xlabel('MONTHS_TO_MCD_END')
plt.ylabel('No. of churns')
plt.title('Histogram of chruns by months to mcd end')
plt.show()

#Priključci koji su ostali aktivni
subsr_churn_list = df2['SUBSCRIBER_ID'].tolist()
df_diff_subsr = df1[~df1['SUBSCRIBER_ID'].isin(subsr_churn_list)]
print(df_diff_subsr.head(5))

#Aktivni priključci prema modelu tarife
print(df_diff_subsr.TARIFF_MODEL.unique())
df_active_tariff = pd.DataFrame(df_diff_subsr['TARIFF_MODEL'].value_counts())
df_active_tariff.reset_index(inplace=True)
print(df_active_tariff.columns)
df_active_tariff.rename(columns={'index': 'TARIFF_MODEL', 'TARIFF_MODEL': 'TARIFF_COUNT'}, inplace=True)
print(df_active_tariff.columns)
print(df_active_tariff)
sns.countplot(y='TARIFF_MODEL', data=df_diff_subsr, palette= 'hls')
plt.title('Frequency of active by tariff model')
plt.show()












