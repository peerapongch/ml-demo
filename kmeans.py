import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# import required libraries for clustering
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

'''
  https://www.kaggle.com/code/ashydv/bank-customer-clustering-k-modes-clustering/notebook
'''

##### Dataset
retail = pd.read_csv('./data/Online_Retail/OnlineRetail.csv', sep=",", encoding="ISO-8859-1", header=0)
retail.head()

##### Preprocessing and feature engineering for RFM analysis
retail = retail.dropna()
retail['CustomerID'] = retail['CustomerID'].astype(str)

# Feature engineering for RFM analysis
# monetary
retail['Amount'] = retail['Quantity']*retail['UnitPrice']
retail = retail[retail['Amount']>0] # only care about spending not refunding
rfm_m = retail.groupby('CustomerID')['Amount'].sum()
rfm_m = rfm_m.reset_index()
# frequency
rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()
rfm_f = rfm_f.reset_index()
rfm_f.columns = ['CustomerID', 'Frequency']
# recency
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],format='%d-%m-%Y %H:%M')
max_date = max(retail['InvoiceDate'])
retail['Diff'] = max_date - retail['InvoiceDate']
rfm_p = retail.groupby('CustomerID')['Diff'].min()
rfm_p = rfm_p.reset_index()
rfm_p['Diff'] = rfm_p['Diff'].dt.days
# merge
rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

# visualise raw rfm data
attributes = ['Amount','Frequency','Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')
plt.savefig('./output/kmeans/kmeans-raw_rfm.png')

# Preparing for kmeans
# Rescaling the attributes
rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
# Instantiate
scaler = StandardScaler()
# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
rfm_df_scaled.head()

# Fitting K means
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    ssd.append(kmeans.inertia_)
plt.figure(0)
plt.plot(ssd)
plt.savefig('./output/kmeans/kmeans-elbo.png')
# Final model with k=3
kmeans = KMeans(n_clusters=3, max_iter=50, random_state=42)
kmeans.fit(rfm_df_scaled)
rfm['Cluster_Id'] = kmeans.labels_

# Visualise results
fig, ax = plt.subplots(1,3,figsize=(15,5))

clusters = [0,1,2]
attr = ['Amount','Frequency','Recency']
# colors = {0:'black',1:'blue',2:'red'}

sns.boxplot(ax=ax[0],data=rfm,x='Cluster_Id',y='Amount')
sns.boxplot(ax=ax[1],data=rfm,x='Cluster_Id',y='Frequency')
sns.boxplot(ax=ax[2],data=rfm,x='Cluster_Id',y='Recency')

ax[0].set_title('Amount')
ax[1].set_title('Frequency')
ax[2].set_title('Recency')

plt.savefig('./output/kmeans/kmeans-rfm_cluster_boxplot.png')
rfm.to_csv('./output/kmeans/kmeans-clusters.csv')