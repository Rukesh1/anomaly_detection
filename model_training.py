import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
import pickle as pkl

df=pd.read_excel("/home/ubuntu/Desktop/e5_anamoly_detection/AD_ DataWorkflowMetrics.xlsx")

#df['time_difference_min']=df.apply(lambda x: time_diff_min(x['time_difference']),axis=1)
df_1=df[['process_name','start_date','end_date','total','success','needs_attention','skipped','failed']]
random_state = np.random.RandomState(42)
model=IsolationForest(n_estimators=50,max_samples='auto',contamination=float(0.1),random_state=random_state)

model.fit(df_1[['total','success','needs_attention','failed','skipped']])
pkl.dump(model, open('anomaly_model.pkl', 'wb'))
print(model)
print('this is branch 3')