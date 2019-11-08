# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:07:39 2019

@author: user
"""

import pandas as pd 
import dask.dataframe as dd
import os
import csv

# we want to know the exact number of rows
file_to_load = 'C:/Users/user/Desktop/T_Com/Python/data_analyst_dataset1.csv'

with open(file_to_load) as customer_data:

     reader = csv.DictReader(customer_data, delimiter = '\t')
     customer_id = []
     handling_channel = []
     manager_id = []
     nkd2007_opis = []
     customer_location_id = []
     regija = []
     bundle_tariff = []
     technology = []
     service_status = []
     mcd_status = []
     mcd_expiration_date = []
     avg_f2f_minutes = []
     avg_f2m_minutes = []
     broadband_technology = []
     broadband_speed = []
     broadband_package =[]
     avg_bb_usage_gb =[]
     max_available_speed = []
     ftth_availability = []
     mobile_customer_profile = []
     avg_monthly_bill = []
     snapshot_week = []

     list_name = ['customer_id', 'handling_channel', 'manager_id', 'nkd2007_opis', 
               'customer_location_id', 'regija', 'bundle_tariff', 'technology', 
               'service_status', 'mcd_status', 'mcd_expiration_date', 'avg_f2f_minutes', 
               'avg_f2m_minutes', 'broadband_technology', 'broadband_speed', 
               'broadband_package', 'avg_bb_usage_gb', 'max_available_speed', 
               'ftth_availability', 'mobile_customer_profile', 'avg_monthly_bill', 
               'snapshot_week'] 
     list_table = [customer_id, handling_channel, manager_id, nkd2007_opis, 
               customer_location_id, regija, bundle_tariff, technology, 
               service_status, mcd_status, mcd_expiration_date, avg_f2f_minutes, 
               avg_f2m_minutes, broadband_technology, broadband_speed, 
               broadband_package, avg_bb_usage_gb, max_available_speed, 
               ftth_availability, mobile_customer_profile, avg_monthly_bill, 
               snapshot_week]
       
     for row in reader:
         #print(row['CUSTOMER_ID'])
         customer_id.append(row['CUSTOMER_ID'])
         handling_channel.append(row['HANDLING_CHANNEL'])
         manager_id.append(row['MANAGER_ID'])
         nkd2007_opis.append(row['NKD2007_OPIS'])
         customer_location_id.append(row['CUSTOMER_LOCATION_ID'])
         regija.append(row['REGIJA'])
         bundle_tariff.append(row['BUNDLE_TARIFF'])
         technology.append(row['TECHNOLOGY'])
         service_status.append(row['SERVICE_STATUS'])
         mcd_status.append(row['MCD_STATUS'])
         mcd_expiration_date.append(row['MCD_EXPIRATION_DATE'])
         avg_f2f_minutes.append(row['AVG_F2F_MINUTES'])
         avg_f2m_minutes.append(row['AVG_F2M_MINUTES'])
         broadband_technology.append(row['BROADBAND_TECHNOLOGY'])
         broadband_speed.append(row['BROADBAND_SPEED'])
         broadband_package.append(row['BROADBAND_PACKAGE'])
         avg_bb_usage_gb.append(row['AVG_BB_USAGE_GB'])
         max_available_speed.append(row['MAX_AVAILABLE_SPEED'])
         ftth_availability.append(row['FTTH_AVAILABILITY'])
         mobile_customer_profile.append(row['MOBILE_CUSTOMER_PROFILE'])
         avg_monthly_bill.append(row['AVG_MONTHLY_BILL'])
         snapshot_week.append(row['SNAPSHOT_WEEK'])
         
     #list_table = list(list_table)
     print(type(list_table))
        
     for ls in list_name:                  
            print(ls)
     for i in list_table:
            print(len(i))
     #Ispis prvih 5 elemenata stupca
     #n = 0
     print("Prvih 5 elemenata:\n")
     colums_no = len(list_table)
     print(colums_no)
     print("Prvih 5 elemenata liste:")
     for col in list_table:
         br = 0
         for elem in col:
            if (br <= 4):
               print(elem)
               br = br +1
 #Grupiranje po customeru - stvaranje liste zapisa (redak sadrži 
 #podake vezane uz 22 kolone)
    
    
     column_no = len(list_table)
     print("\n",column_no)
     rec_no = len(customer_id)
     print(rec_no)
    
     cust_rec = []
     i = 0
     while (i <= 1000):   
           tmp_list = []
           tmp_list.append(customer_id[i])
           tmp_list.append(handling_channel[i])
           tmp_list.append(manager_id[i])
           tmp_list.append(nkd2007_opis[i])
           tmp_list.append(customer_location_id[i])
           tmp_list.append(regija[i])
           tmp_list.append(bundle_tariff[i])
           tmp_list.append(technology[i])
           tmp_list.append(service_status[i])
           tmp_list.append(mcd_status[i])
           tmp_list.append(mcd_expiration_date[i])
           tmp_list.append(avg_f2f_minutes[i])
           tmp_list.append(avg_f2m_minutes[i])
           tmp_list.append(broadband_technology[i])
           tmp_list.append(broadband_speed[i])
           tmp_list.append(broadband_package[i])
           tmp_list.append(avg_bb_usage_gb[i])
           tmp_list.append(max_available_speed[i])
           tmp_list.append(ftth_availability[i])
           tmp_list.append(mobile_customer_profile[i])
           tmp_list.append(avg_monthly_bill[i])
           tmp_list.append(snapshot_week[i])
           cust_rec.append(tmp_list)
           i = i + 1
     #Isšis elemeneata - print(cust_rec)
     print(cust_rec[0])
     print(cust_rec[1])
     print(cust_rec[500])

#print(customer_id[0]   
#Definicija procedure koja vraća uniue elemente neke liste
def unique(list1): 
    # intilize a null list 
   unique_list = []  
    # traverse for all elements 
   for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
  



 