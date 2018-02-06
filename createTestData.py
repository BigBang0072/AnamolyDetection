import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(1) #for having consistency while debugging

#CONTROL HYPER-PARAMETERS
#data creation parameters
sampling_rate=1                     #in Hz(times per second)
start_date='2018-02-01T00:00:00'
end_date='2018-02-08T00:00:00'      #end of the 7thday
max_base_lim=0.55                   #the maximum normal packet loss(as used in paper)
num_links=5                         #number of different source destination.

#anomaly parameters
num_anomaly=5           #number of anomaly in each link
min_anomaly_min=1*60    #minimum respose time
max_anomaly_min=4*60    #could be changed

def plotTimeSeries(time_series):
    '''Arguments:
        time-series: this the generated time series at any point of processing
    '''
    links=time_series.shape[1]
    for i in range(links):
        plt.plot(time_series[:,i],label='link'+str(i),alpha=0.8)
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
    base_val=np.random.uniform(0.15,max_base_lim,num_links); #Tunable
    std_base=np.random.uniform(0.00325,0.0105,num_links);     #Tunable(for standard deviation approach,normal deviation around data)
    anomaly_base=np.random.uniform(max_base_lim/8,
                                    max_base_lim/2,(num_links,num_anomaly)) #Tunable for adding noise just by shifting above ceratain val up.
    metadata={} #for storing the useful attributes of data generated for use elsewhere
    metadata['base_val']=base_val
    metadata['std_base']=std_base


    #creating the timestamps for indexing later
    index=np.arange(start_date,end_date,dtype='datetime64')
    timestamps=index.shape[0]
    #generating the time-series
    time_series=np.random.normal(base_val,std_base,size=(timestamps,num_links))
    plotTimeSeries(time_series)

            #Now as base signal is created we have to add noise to it
    #select random index from the total timestamp(where to insert anomaly).
    #all the time steps are in second and indexed that way. So a factor of 60 everywhere
    start_offset=0*60
    offset=60*max_anomaly_min+2#according to max_anomaly. So that we dont cross max index while adding anomaly
    anomaly_pos=np.random.randint(start_offset,timestamps-offset,
                                    size=(num_anomaly,num_links))

    #crete radom uniform minute of anomaly to be added.
    min_minutes_index=(min_anomaly_min)*60 #here 60 is to convet to index which is sampled each second
    max_minutes_index=(max_anomaly_min)*60
    anomaly_min=np.random.randint(min_minutes_index,max_minutes_index,
                                        size=(num_anomaly,num_links))

    metadata['anomaly_pos']=anomaly_pos
    metadata['anomaly_min']=anomaly_min
    #create the new noise and add at the selected pos above.
    #this anomay wont be just a constant shif but overall normal dist with std of var-sigma(2-5)
    for i in range(num_links):
        for j in range(num_anomaly):
            std_val=np.random.randint(2,6) # a random "scalar" bw 2 to 5
            pos=anomaly_pos[j,i]
            to=anomaly_min[j,i]
            #time_series[pos:pos+to,i]=np.random.normal(base_val[i],
                                        #std_val*std_base[i],size=(to,)) #OK (to,) snapped
            time_series[pos:pos+to,i]=anomaly_base[i,j]+time_series[pos:pos+to,i]
    plotTimeSeries(time_series)
    return metadata,index,time_series

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

def saveToCSV(num_links,metadata,index,time_series):
    '''Arguments:
        num_links: total number of different destination
        index: for indexing the rows of data-frame with time
        time_series: numpy array holding the time-series data for all links
    '''
    col=[]
    for i in range(num_links):
        col.append('link'+str(i))
    df=pd.DataFrame(data=time_series,index=index,columns=col)
    #Now save them in same folder. gitignore added
    filename='time_series3001_'+str(num_anomaly)
    df.to_csv(filename+'.csv')
    np.savez(filename+'_metadata',**metadata)
    #df.to_excel('./../Data/time_series.xlsx')

metadata,index,time_series=simpleDataset(num_links,max_base_lim)
saveToCSV(num_links,metadata,index,time_series)
