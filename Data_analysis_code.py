# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:48:48 2019

@author: user
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv("C:/Test project/toy_dataset.csv")
data=data.dropna()
print(data.shape)
print(list(data.columns))
df=data.drop(columns='Number') #dropping column Number
data=df

#Categorical variables are: City, Illness, Gender
#data['City'].unique()
#data['Illness'].unique()
#data['Gender'].unique()
print("\nCategorical data description")
print("\nIllness") 
print(data['Illness'].describe())
print("\nCity") 
print(data['City'].describe())
print("\nGender") 
print(data['Gender'].describe())

#percentage of no subscription and subscription data
count_no_ill = len(data[data['Illness']=="No"])
count_ill = len(data[data['Illness']=="Yes"])
pct_of_no_ill = count_no_ill/(count_no_ill+count_ill)
print("\npercentage of no illness data is", pct_of_no_ill*100)
pct_of_ill = count_ill/(count_no_ill+count_ill)
print("percentage of illness data is", pct_of_ill*100)

#Data exploration
print("\nMean value for Age and Income:")
print(data.groupby("Illness").mean())
data['Illness'].value_counts()
sns.countplot(x= 'Illness', data=data, palette='hls')
plt.title('Frequency of Illness') 
plt.show()
plt.savefig('count_plot')

print("\nThe average age and income of resident who got sick and who did not get sick:")
data_ill=data[(data['Illness'] == 'Yes')]
ages_ill = np.array(data_ill['Age'])
income_ill=np.array(data_ill['Income'])
data_no_ill=data[(data['Illness'] != 'Yes')]
ages_no_ill = np.array(data_no_ill['Age'])
income_no_ill=np.array(data_no_ill['Income'])

print("Mean illness age:        ", ages_ill.mean())
print("Mean illness income:     ", income_ill.mean())
print("Mean no illness age:     ", ages_no_ill.mean())
print("Mean no illness income   ", income_no_ill.mean())

#Age sample analysis
ages = np.array(data['Age'])
print("\nMean age:       ", ages.mean())
print("\nStandard deviation:", ages.std())
print("\nMinimum age:    ", ages.min())
print("\nMaximum age:    ", ages.max())
ages_list=ages.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5  
plt.hist(ages_list, bins, range, color = 'grey', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of people') 
plt.title('Age histogram') 
plt.show()

ages_ill = np.array(data_ill['Age'])
print("Mean age:       ", ages_ill.mean())
print("Standard deviation:", ages_ill.std())
print("Minimum age:    ", ages_ill.min())
print("Maximum age:    ", ages_ill.max())
ages_ill_list=ages_ill.tolist()
#plotting age_ill histogram
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(ages_ill_list, bins, range, color = 'red', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of people') 
plt.title('Age-Illness histogram') 
plt.show()


#Gender Analaysis

data_gender=np.array(data['Gender'])
data_gender_list=data_gender.tolist()
male_cnt=data_gender_list.count("Male") #number of male in data"
female_cnt=data_gender_list.count("Female") #number of female in data"
data['Gender'].value_counts()
print(data['Gender'].value_counts())
sns.countplot(x= 'Gender', data=data, palette='hls')
plt.xlabel('Gender') 
plt.ylabel('Count') 
plt.title('Gender histogram') 
plt.show()
plt.savefig('count_gender_plot')

data_ill_gender=np.array(data_ill['Gender'])
data_ill_gender_list=data_ill_gender.tolist()
male_ill_cnt=data_ill_gender_list.count("Male") #number of ill male in sample"
female_ill_cnt=data_ill_gender_list.count("Female") #number of ill female in sample"

data_ill_gender=np.array(data_ill['Gender'])
data_ill_gender_list=data_ill_gender.tolist()
male_ill_cnt=data_ill_gender_list.count("Male") #number of ill male in sample"
female_ill_cnt=data_ill_gender_list.count("Female") #number of ill female in sample"
#data_ill['Gender'].value_counts()

print(data_ill['Gender'].value_counts())
sns.countplot(x= 'Gender', data=data_ill, palette='hls')
plt.xlabel('Gender-Illness') 
plt.ylabel('Count') 
plt.title('Gender-Illness histogram') 
plt.show()
plt.savefig('count_gender_ill_plot')

print("\nNumber of males in data:",male_cnt)
print("\nNumber of ill males in data:", male_ill_cnt)
print("\nNumber of females in data:",female_cnt)
print("\nNumber of ill females in data:", female_ill_cnt)
Ill_male_rate=male_ill_cnt/male_cnt
print("\nPrecentage of ill males in data:", Ill_male_rate*100)
Ill_female_rate=female_ill_cnt/female_cnt
print("\nPrecentage of ill females in data:", Ill_female_rate*100)
if(Ill_male_rate > Ill_female_rate):
    print("\nRate of illness for male is greater than rate of illnes for female")
elif(Ill_male_rate < Ill_female_rate):
    print("\nRate of illness for female is greater than rate of illnes for male")
else:
    print("\nRate of illness for female is equal to rate of illnes for male")
    
#Male Illness-Age Analaysis
data_ill_male=data[(data['Illness'] == 'Yes') & (data['Gender'] == 'Male')]
ages_ill_male = np.array(data_ill_male['Age'])
ages_ill_male_list=ages_ill_male.tolist()
max_ill_m_age = ages_ill_male.max()
min_ill_m_age = ages_ill_male.min()    
range = (min_ill_m_age, max_ill_m_age) # setting the ranges and no. of intervals 
bins = 5
plt.hist(ages_ill_male_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of males') 
plt.title('Histogram of ill males') 
plt.show()

#Female Illness-Age Analaysis
data_ill_female=data[(data['Illness'] == 'Yes') & (data['Gender'] == 'Female')]
ages_ill_female = np.array(data_ill_female['Age'])
ages_ill_female_list=ages_ill_female.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(ages_ill_male_list, bins, range, color = 'yellow', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of females') 
plt.title('Histogram of ill females') 
plt.show()

#Income analysis

income = np.array(data['Income'])
print("\nMean income:       ", income.mean())
print("\nStandard deviation:", income.std())
print("\nMinimum income:    ", income.min())
print("\nMaximum income:  ", income.max())
income_list=income.tolist()
max_income = income.max()
min_income = income.min()    
range = (min_income, max_income) # setting the ranges and no. of intervals 
bins = 50
plt.hist(income_list, bins, range, color = 'black', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('income') 
plt.ylabel('No. of people') 
plt.title('Income histogram') 
plt.show()
#Income-Illness analysis
income_ill = np.array(data_ill['Income'])
print("\nMean income:       ", income_ill.mean())
print("\nStandard deviation:", income_ill.std())
print("\nMinimum income:    ", income_ill.min())
print("\nMaximum income:  ", income_ill.max())
income_ill_list=income_ill.tolist()
range = (min_income, max_income) # setting the ranges and no. of intervals
bins = 50
plt.hist(income_ill_list, bins, range, color = 'red', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('income') 
plt.ylabel('No. of ill people') 
plt.title('Income-Illness histogram') 
plt.show()


#group by summary
print("\nmean values for illness")
print(data_ill.groupby('Illness').mean())
print("\nmean value for illness group by Cityplace as place of residence")
print(data_ill.groupby('City').mean())
print("\nmean valeus for illness groupy by Gender")
print(data_ill.groupby('Gender').mean())

#City Illness Analaysis
data['City'].value_counts()
print(data['City'].value_counts())
sns.countplot(x= 'City', data=data, palette='hls')
plt.xlabel('City') 
plt.ylabel('Count') 
plt.title('Number of residents by the City') 
plt.show()
plt.savefig('count_city_plot')

data_ill['City'].value_counts()
print(data_ill['City'].value_counts())
sns.countplot(x= 'City', data=data_ill, palette='hls')
plt.xlabel('City') 
plt.ylabel('Count') 
plt.title('Number of ill residents by the City') 
plt.show()
plt.savefig('count_city_ill_plot')

data_city=np.array(data['City'])
data_city_list=data_city.tolist()
NY_cnt=data_city_list.count("New York City") #No of  citizen in NYC
LA_cnt=data_city_list.count("Los Angeles")   #No of citizen in LA
Dallas_cnt=data_city_list.count("Dallas")    #No of citizen in Dallas
MV_cnt=data_city_list.count("Mountain View") #No of citizen in MV
Austin_cnt=data_city_list.count("Austin")        #No of citizen in Austin
Boston_cnt=data_city_list.count("Boston")        #No of citizen in Boston
WDC_cnt=data_city_list.count("Washington D.C.")  #No of citizen in WDC
SD_cnt=data_city_list.count("San Diego")         #No of citizen in San Diego

data_ill_city=np.array(data_ill['City'])
data_ill_city_list=data_ill_city.tolist()
NY_ill_cnt=data_ill_city_list.count("New York City") #No ill citizen in NYC
LA_ill_cnt=data_ill_city_list.count("Los Angeles")   #No of ill citizen in LA
Dallas_ill_cnt=data_ill_city_list.count("Dallas")    #No of ill citizen in Dallas
MV_ill_cnt=data_ill_city_list.count("Mountain View") #No of ill citizen in MV
Austin_ill_cnt=data_ill_city_list.count("Austin")    #No  of ill citizen in Austin
Boston_ill_cnt=data_ill_city_list.count("Boston")    #No of ill citizen in Boston
WDC_ill_cnt=data_ill_city_list.count("Washington D.C.")    #No of ill citizen in Boston
SD_ill_cnt=data_ill_city_list.count("San Diego")           #No of ill citizen in San Diego

print("\nRate of illness in NYC:", (NY_ill_cnt/NY_cnt)*100)
print("\nRate of illness in LA:", (LA_ill_cnt/LA_cnt)*100)
print("\nRate of illness in Dallas:", (Dallas_ill_cnt/Dallas_cnt)*100)
print("\nRate of illness in Mountain View:", (MV_ill_cnt/MV_cnt)*100)
print("\nRate of illness in Austin:", (Austin_ill_cnt/Austin_cnt)*100)
print("\nRate of illness in Boston:", (Boston_ill_cnt/Boston_cnt)*100)
print("\nRate of illness in WDC:", (WDC_ill_cnt/WDC_cnt)*100)
print("\nRate of illness in San Diego:", (SD_ill_cnt/SD_cnt)*100)


data_NY_male=data[(data['Gender'] == 'Male') & (data['City'] == 'New York City')]
NY_male = data_NY_male.shape[0] #Number of male in NY
data_NY_ill_male=data[(data['Gender'] == 'Male') & (data['City'] == 'New York City') & (data['Illness'] == 'Yes')]
NY_ill_male = data_NY_ill_male.shape[0]
NY_male_ill_rate = (NY_ill_male/NY_male)*100
print("Rate of Illness for male in NY is:", NY_male_ill_rate)

data_LA_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Los Angeles')]
LA_male = data_LA_male.shape[0] #Number of male in LA
data_LA_ill_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Los Angeles') & (data['Illness'] == 'Yes')]
LA_ill_male = data_LA_ill_male.shape[0]
LA_male_ill_rate = (LA_ill_male/LA_male)*100
print("Rate of Illness for male in LA is:", LA_male_ill_rate)

data_Dallas_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Dallas')]
Dallas_male = data_Dallas_male.shape[0] #Number of male in Dallas
data_Dallas_ill_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Dallas') & (data['Illness'] == 'Yes')]
Dallas_ill_male = data_Dallas_ill_male.shape[0]
Dallas_male_ill_rate = (Dallas_ill_male/Dallas_male)*100
print("Rate of Illness for male in Dallas is:", Dallas_male_ill_rate)

data_MV_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Mountain View')]
MV_male = data_MV_male.shape[0]    #Number of male in Mountain View
data_MV_ill_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Mountain View') & (data['Illness'] == 'Yes')]
MV_ill_male = data_MV_ill_male.shape[0]
MV_male_ill_rate = (MV_ill_male/MV_male)*100
print("Rate of Illness for male in Mountain View is:", MV_male_ill_rate)

data_Austin_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Austin')]
Austin_male = data_Austin_male.shape[0] #Number of male in Austin
data_Austin_ill_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Austin') & (data['Illness'] == 'Yes')]
Austin_ill_male = data_Austin_ill_male.shape[0]
Austin_male_ill_rate = (Austin_ill_male/Austin_male)*100
print("Rate of Illness for male in Austin is:", Austin_male_ill_rate)

data_Boston_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Boston')]
Boston_male = data_Boston_male.shape[0] #Number of male in Boston
data_Boston_ill_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Boston') & (data['Illness'] == 'Yes')]
Boston_ill_male = data_Boston_ill_male.shape[0]
Boston_male_ill_rate = (Boston_ill_male/Boston_male)*100
print("Rate of Illness for male in Boston is:", Boston_male_ill_rate)

data_WDC_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Washington D.C.')]
WDC_male = data_WDC_male.shape[0] #Number of male in WDC
data_WDC_ill_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Washington D.C.') & (data['Illness'] == 'Yes')]
WDC_ill_male = data_WDC_ill_male.shape[0]
WDC_male_ill_rate = (WDC_ill_male/WDC_male)*100
print("Rate of Illness for male in WDC is:", WDC_male_ill_rate)

data_SD_male=data[(data['Gender'] == 'Male') & (data['City'] == 'San Diego')]
SD_male = data_SD_male.shape[0] #Number of male in San Diego
data_SD_ill_male=data[(data['Gender'] == 'Male') & (data['City'] == 'San Diego') & (data['Illness'] == 'Yes')]
SD_ill_male = data_SD_ill_male.shape[0]
SD_male_ill_rate = (SD_ill_male/SD_male)*100
print("Rate of Illness for male in San Diego is:", SD_male_ill_rate)

#Gender-City for Females
city_ill_rate_list=[]  #Create list of illness rate in each city 
data_NY_female=data[(data['Gender'] == 'Female') & (data['City'] == 'New York City')]
NY_female = data_NY_female.shape[0] #Number of female in NY
data_NY_ill_female=data[(data['Gender'] == 'Female') & (data['City'] == 'New York City') & (data['Illness'] == 'Yes')]
NY_ill_female = data_NY_ill_female.shape[0]
NY_female_ill_rate = (NY_ill_female/NY_female)*100
city_ill_rate_list.append(NY_female_ill_rate)
print("\nRate of Illness for female in NY is:", NY_female_ill_rate)

data_LA_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Los Angeles')]
LA_female = data_LA_female.shape[0] #Number of female in LA
data_LA_ill_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Los Angeles') & (data['Illness'] == 'Yes')]
LA_ill_female = data_LA_ill_female.shape[0]
LA_female_ill_rate = (LA_ill_female/LA_female)*100
city_ill_rate_list.append(LA_female_ill_rate)
print("Rate of Illness for female in LA is:", LA_female_ill_rate)

data_Dallas_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Dallas')]
Dallas_female = data_Dallas_female.shape[0] #Number of female in Dallas
data_Dallas_ill_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Dallas') & (data['Illness'] == 'Yes')]
Dallas_ill_female = data_Dallas_ill_female.shape[0]
Dallas_female_ill_rate = (Dallas_ill_female/Dallas_female)*100
city_ill_rate_list.append(Dallas_female_ill_rate)
print("Rate of Illness for female in Dallas is:", Dallas_female_ill_rate)

data_MV_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Mountain View')]
MV_female = data_MV_female.shape[0]    #Number of female in Mountain View
data_MV_ill_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Mountain View') & (data['Illness'] == 'Yes')]
MV_ill_female = data_MV_ill_female.shape[0]
MV_female_ill_rate = (MV_ill_female/MV_female)*100
city_ill_rate_list.append(MV_female_ill_rate)
print("Rate of Illness for female in Mountain View is:", MV_female_ill_rate)

data_Austin_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Austin')]
Austin_female = data_Austin_female.shape[0] #Number of female in Austin
data_Austin_ill_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Austin') & (data['Illness'] == 'Yes')]
Austin_ill_female = data_Austin_ill_female.shape[0]
Austin_female_ill_rate = (Austin_ill_female/Austin_female)*100
city_ill_rate_list.append(Austin_female_ill_rate)
print("Rate of Illness for female in Austin is:", Austin_female_ill_rate)

data_Boston_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Boston')]
Boston_female = data_Boston_female.shape[0] #Number of female in Boston
data_Boston_ill_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Boston') & (data['Illness'] == 'Yes')]
Boston_ill_female = data_Boston_ill_female.shape[0]
Boston_female_ill_rate = (Boston_ill_female/Boston_female)*100
city_ill_rate_list.append(Boston_female_ill_rate)
print("Rate of Illness for female in Boston is:", Boston_female_ill_rate)

data_WDC_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Washington D.C.')]
WDC_female = data_WDC_female.shape[0] #Number of female in WDC
data_WDC_ill_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Washington D.C.') & (data['Illness'] == 'Yes')]
WDC_ill_female = data_WDC_ill_female.shape[0]
WDC_female_ill_rate = (WDC_ill_female/WDC_female)*100
city_ill_rate_list.append(WDC_female_ill_rate)
print("Rate of Illness for female in WDC is:", WDC_female_ill_rate)

data_SD_female=data[(data['Gender'] == 'Female') & (data['City'] == 'San Diego')]
SD_female = data_SD_female.shape[0] #Number of female in San Diego
data_SD_ill_female=data[(data['Gender'] == 'Female') & (data['City'] == 'San Diego') & (data['Illness'] == 'Yes')]
SD_ill_female = data_SD_ill_female.shape[0]
SD_female_ill_rate = (SD_ill_female/SD_female)*100
city_ill_rate_list.append(SD_female_ill_rate)
print("Rate of Illness for female in San Diego is:", SD_female_ill_rate)
city_ill_rate_list.sort()
print(city_ill_rate_list)

#Gender-City-Age analaysis
#Male
data_NY_male=data[(data['Gender'] == 'Male') & (data['City'] == 'New York City') & (data['Illness'] == 'Yes')]
data_NY_male_age=np.array(data_NY_male['Age'])
data_NY_male_age_list=data_NY_male_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_NY_male_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of males') 
plt.title('Histogram of ill males in NYC') 
plt.show()
data_NY_male_age_list.sort()
#print(data_NY_male_age_list)

data_LA_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Los Angeles') & (data['Illness'] == 'Yes')]
data_LA_male_age=np.array(data_LA_male['Age'])
data_LA_male_age_list=data_LA_male_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_LA_male_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of males') 
plt.title('Histogram of ill males in LA') 
plt.show()
data_LA_male_age_list.sort()
#print(data_LA_male_age_list)

data_Dallas_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Dallas') & (data['Illness'] == 'Yes')]
data_Dallas_male_age=np.array(data_Dallas_male['Age'])
data_Dallas_male_age_list=data_Dallas_male_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_Dallas_male_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of males') 
plt.title('Histogram of ill males in Dallas') 
plt.show()
data_Dallas_male_age_list.sort()

data_MV_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Mountain View') & (data['Illness'] == 'Yes')]
data_MV_male_age=np.array(data_MV_male['Age'])
data_MV_male_age_list=data_MV_male_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_MV_male_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of males') 
plt.title('Histogram of ill males in Mountain View') 
plt.show()
data_MV_male_age_list.sort()

data_Austin_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Austin') & (data['Illness'] == 'Yes')]
data_Austin_male_age=np.array(data_Austin_male['Age'])
data_Austin_male_age_list=data_Austin_male_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_Austin_male_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of males') 
plt.title('Histogram of ill males in Austin') 
plt.show()
data_Austin_male_age_list.sort()

data_Boston_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Boston') & (data['Illness'] == 'Yes')]
data_Boston_male_age=np.array(data_Boston_male['Age'])
data_Boston_male_age_list=data_Boston_male_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_Boston_male_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of males') 
plt.title('Histogram of ill males in Boston') 
plt.show()
data_Boston_male_age_list.sort()

data_WDC_male=data[(data['Gender'] == 'Male') & (data['City'] == 'Washington D.C.') & (data['Illness'] == 'Yes')]
data_WDC_male_age=np.array(data_WDC_male['Age'])
data_WDC_male_age_list=data_WDC_male_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_WDC_male_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of males') 
plt.title('Histogram of ill males in Washington D.C.') 
plt.show()
data_WDC_male_age_list.sort()

data_SD_male=data[(data['Gender'] == 'Male') & (data['City'] == 'San Diego') & (data['Illness'] == 'Yes')]
data_SD_male_age=np.array(data_SD_male['Age'])
data_SD_male_age_list=data_SD_male_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_SD_male_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of males') 
plt.title('Histogram of ill males in San Diego') 
plt.show()
data_SD_male_age_list.sort()

#Female
data_NY_female=data[(data['Gender'] == 'Female') & (data['City'] == 'New York City') & (data['Illness'] == 'Yes')]
data_NY_female_age=np.array(data_NY_female['Age'])
data_NY_female_age_list=data_NY_female_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_NY_female_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of females') 
plt.title('Histogram of ill females in NYC') 
plt.show()
data_NY_female_age_list.sort()
#print(data_NY_female_age_list)

data_LA_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Los Angeles') & (data['Illness'] == 'Yes')]
data_LA_female_age=np.array(data_LA_female['Age'])
data_LA_female_age_list=data_LA_female_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_LA_female_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of females') 
plt.title('Histogram of ill females in LA') 
plt.show()
data_LA_female_age_list.sort()
#print(data_LA_female_age_list)

data_Dallas_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Dallas') & (data['Illness'] == 'Yes')]
data_Dallas_female_age=np.array(data_Dallas_female['Age'])
data_Dallas_female_age_list=data_Dallas_female_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_Dallas_female_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of females') 
plt.title('Histogram of ill females in Dallas') 
plt.show()
data_Dallas_female_age_list.sort()

data_MV_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Mountain View') & (data['Illness'] == 'Yes')]
data_MV_female_age=np.array(data_MV_female['Age'])
data_MV_female_age_list=data_MV_female_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_MV_female_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of females') 
plt.title('Histogram of ill females in Mountain View') 
plt.show()
data_MV_female_age_list.sort()

data_Austin_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Austin') & (data['Illness'] == 'Yes')]
data_Austin_female_age=np.array(data_Austin_female['Age'])
data_Austin_female_age_list=data_Austin_female_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_Austin_female_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of females') 
plt.title('Histogram of ill females in Austin') 
plt.show()
data_Austin_female_age_list.sort()

data_Boston_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Boston') & (data['Illness'] == 'Yes')]
data_Boston_female_age=np.array(data_Boston_female['Age'])
data_Boston_female_age_list=data_Boston_female_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_Boston_female_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of females') 
plt.title('Histogram of ill females in Boston') 
plt.show()
data_Boston_female_age_list.sort()

data_WDC_female=data[(data['Gender'] == 'Female') & (data['City'] == 'Washington D.C.') & (data['Illness'] == 'Yes')]
data_WDC_female_age=np.array(data_WDC_female['Age'])
data_WDC_female_age_list=data_WDC_female_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_WDC_female_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of females') 
plt.title('Histogram of ill females in Washington D.C.') 
plt.show()
data_WDC_female_age_list.sort()

data_SD_female=data[(data['Gender'] == 'Female') & (data['City'] == 'San Diego') & (data['Illness'] == 'Yes')]
data_SD_female_age=np.array(data_SD_female['Age'])
data_SD_female_age_list=data_SD_female_age.tolist()
range = (24,60) # setting the ranges and no. of intervals 
bins = 5
plt.hist(data_SD_female_age_list, bins, range, color = 'blue', 
        histtype = 'bar', rwidth = 0.3) 
plt.xlabel('age') 
plt.ylabel('No. of females') 
plt.title('Histogram of ill females in San Diego') 
plt.show()
data_SD_female_age_list.sort()













      












