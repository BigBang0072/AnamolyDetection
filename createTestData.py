import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(1) #for having consistency while debugging

sampling_rate=1 #in Hz(times per second)
start_date='2018-02-01T00:00:00'
end_date='2018-02-08T00:00:00' #end of the 7thday
max_base_lim=0.5 #the maximum normal packet loss(as used in paper)
num_links=5 #number of different source destination.
num_anomaly=10 #number of anomaly in each link
#standard_dev_base=np.array([0.007,0.03,0.04,0.05,0.09])

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

            #Now as base signal is created we have to add noise to it
    #select random index from the total timestamp(where to insert anomaly).
    start_offset=0
    offset=15000 #here ~3000 timepoints means one hour(our max anomaly will last 4 hour so for safety)
    anomaly_pos=np.random.randint(start_offset,timestamps-offset,
                                    size=(num_anomaly,num_links))
    #crete radom uniform minute of anomaly to be added.
    #(max four hour and min 30 min) 4hour=4*60=240 min.
    min_anomaly_min=5
    max_anomaly_min=30
    min_minutes_index=(min_anomaly_min)*60 #here 60 is to convet to index which is sampled each second
    max_minutes_index=(max_anomaly_min)*60
    anomaly_min=np.random.randint(min_minutes_index,max_minutes_index,
                                        size=(num_anomaly,num_links))
    #create the new noise and add at the selected pos above.
    #this anomay wont be just a constant shif but overall normal dist with std of var-sigma(2-5)
    for i in range(num_links):
        for j in range(num_anomaly):
            std_val=np.random.randint(2,6) # a random "scalar" bw 2 to 5
            pos=anomaly_pos[j,i]
            to=anomaly_min[j,i]
            time_series[pos:pos+to,i]=np.random.normal(base_val[i],
                                        std_val*std_base[i],size=(to,)) #OK (to,) snapped
    plotTimeSeries(time_series)
    return index,time_series

def correlatedDataset(num_links,max_base_lim):
    '''Arguments:
        num_links: total number of destination from the current source.
        max_base_lim: the maximim base (normal without any anomaly) packet loss that is often seen in ral data.
    '''
        #FOR BASE SIGNAL(above wich noise will be overlayed later)
    base_val=np.random.uniform(0.1,0.55,num_links)
    std_base=np.random.uniform(0.00325,0.02,num_links)


    #Creating the timestamps for indexing
    index=np.arange(start_date,end_date,dtype='datetime64')
    timestamps=index.shape[0]

    #Now creating the actual data which is base-signal (i.e normal characteristic of the signal)
    #Also few of the links will be correlated to reach other both in base signal and anomaly part.
    #the correlation will be given by the co-variance matrix
    covariance_matrix=np.array([[0,0,0,0,0],
                                [0,0,0,0,0], #link 1 and 2 are correlated currently with co-variace=0.5
                                [0,0,0,0,0],
                                [0,0,0,0,0],
                                [0,0,0,0,0]],dtype=np.float64)

    for i in range(num_links):
        covariance_matrix[i,i]=std_base[i] #putting the variance in the digonal
    #Now finally getting multivariate random gaussian time series
    print(base_val)
    print(covariance_matrix)
    time_series=np.random.multivariate_normal(base_val,covariance_matrix,
                                                size=(timestamps))
    # time_series=np.random.normal(base_val,std_base,size=(timestamps,num_links))
    plotTimeSeries(time_series)
    # print(time_series.shape)
        #Now we have to add noise(anomaly) over this base signal

    return index,time_series

def saveToCSV(num_links,index,time_series):
    '''Arguments:
        num_links: total number of different destination
        index: for indexing the rows of data-frame with time
        time_series: numpy array holding the time-series data for all links
    '''
    col=[]
    for i in range(num_links):
        col.append('link'+str(i))
    df=pd.DataFrame(data=time_series,index=index,columns=col)
    df.to_csv('./../Data/time_series3001.csv')
    #df.to_excel('./../Data/time_series.xlsx')

index,time_series=simpleDataset(num_links,max_base_lim)
saveToCSV(num_links,index,time_series)
