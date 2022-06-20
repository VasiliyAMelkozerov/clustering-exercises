import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import env

def acquire():
    database = 'mall_customers'
    url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{database}'
    df = pd.read_sql('SELECT * FROM customers', url, index_col='customer_id')
    return df
