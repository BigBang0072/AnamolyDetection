import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1) #for having consistency while debugging

sampling_rate=1 #in Hz(times per second)
start_date='2018-02-01T00:00:00'
end_date='2018-02-08T00:00:00' #end of the 7thday
max_base_lim=0.5 #the maximum normal packet loss(as used in paper)
num_links=5 #number of different source destination.
#standard_dev_base=np.array([0.007,0.03,0.04,0.05,0.09])

def simpleDataset(num_links,max_base_lim):
    '''Arguments:
        num_links: denoted the total number of destiantion form the current source
        max_base_lim: denote the base signal value on which the normal and anomalous
                        noise will be added.
    '''
    #First generating the base signal. Later noise will be added with varying std and places
    base_val=np.random.uniform(0.1,0.55,num_links);#Tunable
    std_base=np.random.uniform(0.00325,0.02,num_links);#Tunable

    #creating the timestamps for indexing later
    index=np.arange(start_date,end_date,dtype='datetime64')
    timestamps=index.shape[0]
    #generating the time-series
    time_series=np.random.normal(base_val,std_base,size=(timestamps,num_links))
    plotTimeSeries(time_series)

def plotTimeSeries(time_series):
    '''Arguments:
        time-series: this the generated time series at any point of processing
    '''
    links=time_series.shape[1]
    for i in range(links):
        plt.plot(time_series[:,i],label='link'+str(i))
        plt.xlabel('time stamps')
        plt.ylabel('Packet-Loss')
        plt.ylim(0,1)
    plt.legend()
    #plt.savefig('base1.png')
    plt.show()

simpleDataset(num_links,max_base_lim)
