#Import libararies and packages essential

import os
import sys
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from sklearn.model_selection import train_test_split
#Load the dataset
import pandas as pd



class prediction():

    def main(self,name):
        data = pd.read_csv(name)

        #Data cleaning and pre processing

        df = []
        df_original = DataFrame(data, columns = ['PRTN_CD','SRC_ID','T1_CEID','T2_CEID','T1_CUST_ID','T2_CUST_ID','EC_CUST_ID','IBM_CNTRY_CD_ISO','T1_CNTRY_CD_ISO','T2_CNTRY_CD_ISO','EC_CNTRY_CD_ISO','ORD_NO','ORD_REAS_CD','SO_DT_T1','INVOC_DT_IBM','INVOC_NO_IBM','INVOC_DT_T1','CHANL_CD','REV_CRCY_CD','REV_AMT_USD_PLAN','REV_AND_COST_DTL_ID'])
        df =DataFrame(data, columns = ['PRTN_CD','SRC_ID','T1_CEID','T2_CEID','T1_CUST_ID','T2_CUST_ID','EC_CUST_ID','IBM_CNTRY_CD_ISO','T1_CNTRY_CD_ISO','T2_CNTRY_CD_ISO','EC_CNTRY_CD_ISO','ORD_NO','ORD_REAS_CD','SO_DT_T1','INVOC_DT_IBM','INVOC_NO_IBM','INVOC_DT_T1','CHANL_CD','REV_CRCY_CD','REV_AMT_USD_PLAN','REV_AND_COST_DTL_ID'])

        df['PRTN_CD'] = data['PRTN_CD']
        df['SRC_ID'] = data['SRC_ID']

        df['T1_CEID'] = data['T1_CEID'].fillna(0)
        df.loc[df['T1_CEID'].str.contains('^NOCEID.*')==True, 'T1_CEID'] = 2
        df['T2_CEID'] = data['T2_CEID'].fillna(0)
        df.loc[df['T2_CEID'].str.contains('^NOCEID.*')==True, 'T2_CEID'] = 2

        df['T1_CUST_ID'] = data.loc[data['T1_CUST_ID'].str.contains('<NULL>'),'T1_CUST_ID'] = 0
        df['T2_CUST_ID'] = data.loc[data['T2_CUST_ID'].str.contains('<NULL>'),'T2_CUST_ID'] = 0
        df['EC_CUST_ID'] = data.loc[data['EC_CUST_ID'].str.contains('<NULL>'),'EC_CUST_ID'] = 0


        df['IBM_CNTRY_CD_ISO'] = data['IBM_CNTRY_CD_ISO']
        df['T1_CNTRY_CD_ISO'] = data['T1_CNTRY_CD_ISO']
        df['T2_CNTRY_CD_ISO'] = data['T2_CNTRY_CD_ISO']
        df['EC_CNTRY_CD_ISO'] = data['EC_CNTRY_CD_ISO']
        df['CHANL_CD'] = data['CHANL_CD']



        df['ORD_NO'] = data['ORD_NO'].fillna(0)
        df['ORD_REAS_CD'] = data['ORD_REAS_CD'].fillna(0)
        df['INVOC_NO_IBM'] = data['INVOC_NO_IBM'].fillna(0)


        df = data.fillna(0)

        catcolumns = df.select_dtypes(['category']).columns
        df[catcolumns] = df[catcolumns].apply(lambda x: x.cat.codes)
        df.head()

        #Data cleaning and pre processing

        df['PRTN_CD'] = df['PRTN_CD'].astype('category').cat.codes
        df['SRC_ID'] = df['SRC_ID'].astype('category').cat.codes
        df['IBM_CNTRY_CD_ISO'] = data['IBM_CNTRY_CD_ISO'].astype('category').cat.codes
        df['T1_CNTRY_CD_ISO'] = data['T1_CNTRY_CD_ISO'].astype('category').cat.codes
        df['T2_CNTRY_CD_ISO'] = data['T2_CNTRY_CD_ISO'].astype('category').cat.codes
        df['EC_CNTRY_CD_ISO'] = data['EC_CNTRY_CD_ISO'].astype('category').cat.codes
        df['REV_CRCY_CD'] = df['REV_CRCY_CD'].astype('category').cat.codes
        df['CHANL_CD'] = df['CHANL_CD'].astype('category').cat.codes
        df.loc[df['T1_CEID'].str.contains('^T1ID.*')==True, 'T1_CEID'] = 1
        df.loc[df['T2_CEID'].str.contains('^T2ID.*')==True, 'T2_CEID'] = 1
        df.loc[df['T1_CEID'].str.contains('NOCEID')==True, 'T1_CEID'] = 2
        df.loc[df['T2_CEID'].str.contains('NOCEID')==True, 'T2_CEID'] = 2


        df.loc[df['T1_CUST_ID'].str.contains('^0.*')==False, 'T1_CUST_ID'] = 1
        df.loc[df['T2_CUST_ID'].str.contains('^0.*')==False, 'T2_CUST_ID'] = 1
        df.loc[df['EC_CUST_ID'].str.contains('^0.*')==False, 'EC_CUST_ID'] = 1
        df.loc[df['REV_AND_COST_DTL_ID'].str.contains('^0.*')==False, 'REV_AND_COST_DTL_ID'] = 1


        df.loc[df['ORD_REAS_CD'].str.contains('^942.*')==False, 'ORD_REAS_CD'] = 1
        df.loc[df['ORD_REAS_CD'] == 0] = 1
        df.loc[df['ORD_REAS_CD'].str.contains('^942.*')==True, 'ORD_REAS_CD'] = 0

        df['ORD_NO']  = data.loc[df['ORD_NO'] != 0] = 1
        df['INVOC_NO_IBM']  = data.loc[df['INVOC_NO_IBM'] != 0] = 1

        df['SO_DT_T1'] = pd.to_datetime(df['SO_DT_T1'])
        df['SO_date_valid'] = 0

        for row in range(len(df)):
            if [[df['SO_DT_T1'][row] >= pd.Timestamp("2021-01-01 00:00:00")] and [df['SO_DT_T1'][row] <= pd.Timestamp("2021-12-31 00:00:00")]]:
                df['SO_date_valid'][row] = 1


        df['SO_date_valid']
        df = df.drop(['SO_DT_T1'],axis=1)
        df = df.drop(['INVOC_DT_IBM'],axis=1)

        df['Anomaly'] = 0
        for row in range(len(df)):
            if df['T1_CEID'][row] == 0 or df['T1_CEID'][row] == 2 or df['EC_CUST_ID'][row] == 0 or df['ORD_REAS_CD'][row] == 0:
                df['Anomaly'][row] = 1
            else:
                df['Anomaly'][row] = 0

        df.to_csv(r'resource/LabelledData.csv',index=False)

        #Split data into training set and testing set

        y = df['Anomaly']
        X = df.drop('Anomaly',axis=1)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

        newdf = df_original
        X_train1,X_test1,y_train1,y_test1 = train_test_split(newdf,y,test_size=0.2,random_state=0)

        from sklearn.naive_bayes import GaussianNB

        #Initialize the GaussianNB model
        GNB = GaussianNB()

        #Fit on the training data
        GNB.fit(X_train,y_train)

        #Make prediction on entire test data
        y_pred = GNB.predict(X_test)
        y_hats = pd.Series(y_pred)

        #Use score method to get accuracy of the model
        score =  GNB.score(X_test,y_test)
        print("Accuracy of GNB classifier:", score)

        predictions = pd.Series(y_pred)

        #Create an empty data frame to contain the predicted anomaly records and then to export to csv later
        resultdata = pd.DataFrame(columns =['PRTN_CD','SRC_ID','T1_CEID','T2_CEID','T1_CUST_ID','T2_CUST_ID','EC_CUST_ID','IBM_CNTRY_CD_ISO','T1_CNTRY_CD_ISO','T2_CNTRY_CD_ISO','EC_CNTRY_CD_ISO','ORD_NO','ORD_REAS_CD','SO_DT_T1','INVOC_DT_IBM','INVOC_NO_IBM','INVOC_DT_T1','CHANL_CD','REV_CRCY_CD','REV_AMT_USD_PLAN','REV_AND_COST_DTL_ID'])
        datalen = len(y_hats)
        for row in range(0,datalen):
            if y_hats[row] == 1:
                resultdata.loc[row] = X_test1.iloc[row,:]
                #print(df_original.iloc[row,:])

        resultdata.to_csv(r'resource/AnomalyPredictions.csv',index=False)