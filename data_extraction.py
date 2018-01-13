import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_data():
    df =  pd.read_csv('PacketLoss_192.170.227.160.csv')
    dests=df['dest'].unique()
    dfs={}
    for d in dests:
        dfs[d] = df[df["dest"] == d].copy()
        dfs[d] = dfs[d][["timestamp","packet_loss"]]
        dfs[d].columns=["timestamp",d]
        dfs[d].index=dfs[d].timestamp
        del dfs[d]["timestamp"]
        dfs[d] = dfs[d].transpose()
        dups = dfs[d].columns.get_duplicates()
        if len(dups)>0:
            print(d)
            print(dups)
            print(dfs[d].columns)
            return
    cdf = pd.concat(dfs.values())
    cdf.to_csv('transformed.csv')
    return cdf

cdf=load_data()
cdf=cdf.iloc[:-6]

#Saving the packet Loss images
plt.clf()
# for i in range(cdf.shape[0]):
#     plt.plot(cdf.iloc[i])
#     plt.xlabel('Time-Stamp')
#     plt.ylabel('Packet-Loss')
#     plt.title('destination: '+str(cdf.index[i]))
#     plt.ylim(0,1)
#     plt.savefig(str(cdf.index[i])+'.png')
#     plt.clf()


#Viewing the correlation between the packet loss of various destination
plt.clf()
chunk=cdf[cdf.columns[:2000]].transpose()
corr=chunk.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
plt.savefig('correlation.png')
