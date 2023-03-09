import pickle as pkl
import pandas as pd

df=pd.read_excel("/home/ubuntu/Desktop/e5_anamoly_detection/AD_ DataWorkflowMetrics.xlsx")
filename="/home/ubuntu/Desktop/e5_anamoly_detection/code/anomaly_detection/anomaly_model.pkl"
model = pkl.load(open(filename, 'rb'))

df['anomaly_score'] = model.predict(df[['total','success','needs_attention','failed','skipped']])
df[df['anomaly_score']==-1].to_csv('Anomalies_detected.csv')
