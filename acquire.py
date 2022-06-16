import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_connection(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def zillow_data():
    #we save first before editing so that we still have the original on hand
    filename = "zillow.csv"
    #this is built to save time over time
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        return get_zillow_data()

def get_zillow_data():
    #this sql code is grabbing all of the tables from the sql server
    sql_query = """SELECT 
    *
        FROM
    properties_2017
        LEFT JOIN
    predictions_2017 USING (parcelid)
		LEFT JOIN 
	airconditioningtype USING (airconditioningtypeid)
		LEFT JOIN 
	architecturalstyletype USING (architecturalstyletypeid)
		LEFT JOIN 
	buildingclasstype USING (buildingclasstypeid)
		LEFT JOIN
	heatingorsystemtype USING (heatingorsystemtypeid)
		LEFT JOIN 
	propertylandusetype USING (propertylandusetypeid)
		LEFT JOIN 
	storytype USING (storytypeid)
		LEFT JOIN 
	typeconstructiontype USING (typeconstructiontypeid)
		LEFT JOIN
	unique_properties USING (parcelid)
        WHERE
    properties_2017.latitude IS NOT NULL
        AND properties_2017.longitude IS NOT NULL
        AND transactiondate <= '2017-12-31'
        AND propertylandusetypeid = 261
        OR propertylandusetypeid = 279;"""
    df = pd.read_sql(sql_query, get_connection('zillow'))
    return df

def traintestsplit(df):
    #this function gives us train validate test variabels to model with
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    #80% of value
    #I choose to stratify here to get an even mix of counties to not miss trends and clearly see if the taxvaluedollarcount is in fact evenly set on  counties
    #TL:DR  keeping counties even in sample
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    # 30% of 80% for validate
    return train, validate, test

#this is a copy to trouble shoot code
def takeout_outliers1(df):
         #this code properly sets a data ceiling
    df = df[(df.logerror <= 2.5) & (df.logerror >= -2.5)]
        #this sets the upper threshold to cut off to handle outliers
    df = df[(df.logerror >= 0.03) | (df.logerror <= -0.03)]

    return df

def handle_missing_values(df, prop_required_column, prop_required_row):
    #bring in a dataframe and eliminate colomuns or rows from inputed values
    #these  values will be ready as percentages
    n_required_column = round(df.shape[0] * prop_required_column)
    #for example .9 in prop_required_column would drop columns that have 90 % of their data
    n_required_row = round(df.shape[1] * prop_required_row)
    # same as above for for rows that a re missing values
    df = df.dropna(axis=0, thresh=n_required_row)
    #drops the rows
    df = df.dropna(axis=1, thresh=n_required_column)
    #drops th columns
    return df

def splitmoreways(train, 
               validate, 
               test, 
               columns_to_scale=['propertylandusetypeid', 'bathroomcnt', 'bedroomcnt',
       'calculatedbathnbr', 'calculatedfinishedsquarefeet',
       'finishedsquarefeet12', 'fullbathcnt', 'latitude', 'longitude',
       'lotsizesquarefeet', 'rawcensustractandblock',
        'roomcnt', 'yearbuilt',
       'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'assessmentyear',
       'landtaxvaluedollarcnt', 'taxamount', 'censustractandblock'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    #build out copies of the split data that way we can still access the original dataframe
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    #GET THE THING
    #the choice of scaler
    scaler = MinMaxScaler()
    #this is a simple workhorse scaler

    #FIT THE THING
    scaler.fit(train[columns_to_scale])
    
    #RUN THE THING

    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
    #scaling each one of the split dataframes       
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])

    #If return_scalar is True, the scaler object will be returned as well
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
