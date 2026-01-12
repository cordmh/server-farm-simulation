# -*- coding: utf-8 -*-
"""
Python script for COMP9334 Project
T1/2021
Wrtten by Matthew Cord
Submitted 24/4/2021

This script is divided into 4 cells:
Cell 1 (lines 000-023): Import statements
Cell 2 (lines 024-040): Read in ASCII files or use custom parameters?
Cell 3 (lines 041-395): Function definitions
Cell 4 (lines 394-413): Generate output
"""

import time
import random as rdm
from scipy import stats
import numpy as np
import pandas as pd # note v1.2.5 or similar should be used (not v2!)
import matplotlib.pyplot as plt
rdm.seed(93342021)


#%%
# Parameters:
    
_ascii = False      # True if parameters are read in by ASCII files (default)
                    # False if using custom parameters, specified below

if _ascii == True:
    test_number = 4                 # test number
else:
    N = 250                         # no. events in each simulation run
    S = 10                          # no. MSTs to simulate (for each d for each algorithm)
    f = 1.5                         # processing rate of fast server
    m = 50                          # length of transient part (first m events discarded)
    alpha = 0.05                    # size of test for statistical inference
    params = [6.1,0.8,1.1,0.2,3.5]  # statistical parameters


#%%
# Function definitions:

    
def get_inputs(test_number):
    ''' 
    Retrieves inputs for desires test case. Only relevant if _ascii is True.
    '''
    i = str(test_number)
    with open('config\\para_'+i+'.txt') as F:
        data = F.readlines()
        if len(data) == 4:
            random = True
        else:
            random = False
        f = float(data[0])
        algorithm = int(data[1])
        d = float(data[2])
        if random == True:
            T = float(data[3])
            N = 50000
        else:
            T = None
    with open('config\\interarrival_'+i+'.txt') as F:
        data = F.readlines()
        if random == True:
            stat_params = [float(v) for v in data]
        else:
            interarrival_times = [float(v) for v in data]
            N = len(interarrival_times)*2
    with open('config\\service_'+i+'.txt') as F:
        data = F.readlines()
        if random == True:
            stat_params.extend([float(v) for v in data])
        else:
            service_times = [float(v) for v in data]
    if random == True:
        out = [algorithm,d,N,100,f,T,random],stat_params
    else:
        out = [algorithm,d,N,0,f,T,random],[interarrival_times,service_times]
    return out


def queue_assignment(n1,n2,n3,d,f,algorithm):
    '''
    Uses algorithm 1 or 2 to assign job to a server according to Sect 3.2.
    Function returns either 1, 2, or 3 - denoting which server job
    has been assigned to.
    '''
    if algorithm == 1:
        condition = n3-d
    if algorithm == 2:
        condition = n3/f - d
    if n3 != 0 and (min(n1,n2)==0 or min(n1,n2) <= condition):
        if n1 <= n2:
            return 1
        else:
            return 2
    else:
        return 3


def generate_arrival_and_service_times(N,params,random):
    '''
    If random mode (random=True), simulates N interarrival and service times 
    according to Sect 5.1.1, where params is a list of five elements - lambda, 
    uniform dist lower bound, uniform dist upper bound, alpha, and beta.
    If trace model (random=False), generates arrival and service times
    from params argument - params is a list of two elements, the first
    containing interarrival times, the second with the service times of the
    slow server.
    '''
    if random == True:
        lamb = params[0]
        unif_lower = params[1]
        unif_upper = params[2]
        pareto_alpha = params[3]
        pareto_beta = params[4] + 1
        interarrival_times = [rdm.expovariate(lamb)*rdm.uniform(unif_lower,unif_upper) for v in range(N)]
        arrival_times = np.cumsum(interarrival_times)
        service_times = [rdm.paretovariate(pareto_beta)*pareto_alpha for v in range(N)] # for slow servers
    if random == False:
        arrival_times = np.cumsum(params[0]).tolist() + [9999]
        service_times = params[1] + [9999]
        # note a 'dummy number' is added to the end that doesn't affect results
        N = len(service_times)
    data = pd.DataFrame({'Request index': range(1,N+1), 'Arrival time': arrival_times,
                         'Service time': service_times})
    data = data.set_index('Request index')
    return data

    
def compute_mean_service_time(algorithm,d,N,m,f,T,random,params):
    '''
    Applies a load balancing algorithm to simulate events (arrivals and
    departures) of a server farm.

    Parameters
    ----------
    algorithm : {1, 2}
        Which load balancing algorithm to use.
        
    d : int
        Loan balancing algorithm parameter.
        
    N : int or None
        Number of events to be simulated. 
        If int, T must be None.
        If None, must supply a value for T.
        
    T : int, float, or None
        Units of time until program stops
        If not None, N must be None.
        If None, must supply a value for N.
        
    m : int < N
        Number of events to discard (transient removal). 
        
    f : float
        Load balancing algorithm parameter (only used if algorithm = 2).
        
    random: bool
        If True, arrival and service times are simulated. Otherwise, must
        be specified in the params argument.
    
    params: list
        If random is True, this is a list of two elements -- the first with
        the interarrival times, the second with the service times for the
        slow servers.
        If random is False, this is a list of five elements -- lambda, a, b,
        alpha, and beta (where a and b are the lower and upper parameters
        of the continuous uniform distribution)
    
    Returns
    -------
    A list of five elements---
    The first  : the mean service time of the jobs in the system (float)
    The second : arrival & departure times from server 1 (array)
    The third  : arrival & departure times from server 2 (array)
    The fourth : arrival & departure times from server 3 (array)
    The fifth  : data frame containing all events
    '''
    start_time = time.perf_counter() 
    if random == True:
        data = generate_arrival_and_service_times(N, params, random)
    else:
        data = generate_arrival_and_service_times(None, params, random)
    # Creating pandas data frame, df, to store each event:
    cols_1 = ['t', 'Request index (i.a)', 'Service required (i.a)', 'Server assignment (i.a)']
    cols_2 = ['Request index (i.d)', 'Service time (i.d)', 'Time of next arrival']
    cols_3 = ['Service requirement of next arrival', 'Queue 1', 'Queue 2', 'Queue 3']
    cols_4 = ['Job 1', 'Job 2', 'Job 3']
    cols = cols_1 + cols_2 + cols_3 + cols_4
    df = pd.DataFrame(columns = cols, index = range(N*2-2))
    df['Queue 1'] = [[]]*df.shape[0]
    df['Queue 2'] = [[]]*df.shape[0]
    df['Queue 3'] = [[]]*df.shape[0]
    # Initialising df (each row represents an event => 1st row considers 1st arrival):
    df['t'].iat[0] = data['Arrival time'].iat[0]
    df['Request index (i.a)'].iat[0] = data.index[0]
    df['Service required (i.a)'].iat[0] = data['Service time'].iat[0]
    df['Server assignment (i.a)'].iat[0] = 3 
    df['Time of next arrival'].iat[0] = data['Arrival time'].iat[1]
    df['Service requirement of next arrival'].iat[0] = data['Service time'].iat[1]
    df.iloc[0,10+int(df.iloc[0]['Server assignment (i.a)'])] = \
    [df.iloc[0]['Request index (i.a)'],df.iloc[0]['Service required (i.a)']/f]
    # ensure numeric columns in df are float64 type, not object type:
    for col in df.columns[:8]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    i=1
    # While loop continues until final arrival, and involves 3 steps:
    while True:
        # Step 1 of 3: determine if next event is an arrival or departure
        jobs = df['Job 1'].iat[i-1],df['Job 2'].iat[i-1],df['Job 3'].iat[i-1]
        if jobs == (np.nan, np.nan, np.nan):
            event = 'arrival' # always arrival if all servers idle
        elif min([v[1] for v in jobs if type(v)==list]) + \
        df['t'].iat[i-1] >= df['Time of next arrival'].iat[i-1]:
            event = 'arrival' # arrival if next departure occurs after next arrival
        else:
            event = 'departure' # otherwise, departure
        # Step 2 of 3: record values in data frame if arrival
        if event == 'arrival':
            index = int(max(df[df['Request index (i.a)'].notna()]['Request index (i.a)'])+1)
            if _ascii == False:
                if index == len(data): # at last arrival, exit loop
                    break 
            df['Request index (i.a)'].iat[i] = index
            df['t'].iat[i] = data.loc[index]['Arrival time']
            if random == True and _ascii == True:
                if df['t'].iat[i] > T or time.perf_counter() - start_time > 290:
                    print('almost done...')
                    break # exit loop when max units of time reached
            df['Service required (i.a)'].iat[i] = data.loc[index]['Service time']
            # calculating no. jobs at each server:
            N1 = len(df['Queue 1'].iat[i-1])
            if type(df['Job 1'].iat[i-1]) == list:
                N1 += 1
            N2 = len(df['Queue 2'].iat[i-1])
            if type(df['Job 2'].iat[i-1]) == list:
                N2 += 1
            N3 = len(df['Queue 3'].iat[i-1])
            if type(df['Job 3'].iat[i-1]) == list:
                N3 += 1
            df['Server assignment (i.a)'].iat[i] = \
            queue_assignment(N1,N2,N3,d,f,algorithm)
            df['Time of next arrival'].iat[i] = data['Arrival time'].loc[index+1]
            df['Service requirement of next arrival'].iat[i] = data['Service time'].loc[index+1]
            # updating remaining server times:
            time_diff = df['t'].iat[i] - df['t'].iat[i-1]
            for col in ['Job 1', 'Job 2', 'Job 3']:
                if type(df[col].iat[i-1]) == list:
                    df[col].iat[i] = [df[col].iat[i-1][0], df[col].iat[i-1][1] - time_diff]
            # copying down values in queues from previous row (appended if job sent to queue):
            for col in ['Queue 1', 'Queue 2', 'Queue 3']:
                df[col].iat[i] = df[col].iat[i-1]
            # creating new job:
            if df['Server assignment (i.a)'].iat[i] in [1,2]:
                service_time = df['Service required (i.a)'].iat[i]
            if df['Server assignment (i.a)'].iat[i] == 3:
                service_time = df['Service required (i.a)'].iat[i]/f
            if df['Request index (i.a)'].iat[i].is_integer():
                new_job = [int(df['Request index (i.a)'].iat[i]), service_time]
            else:
                new_job = [df['Request index (i.a)'].iat[i], service_time]
            # move job to queue (if server occupied), else to server:
            temp_col_1 = int(10+df['Server assignment (i.a)'].iat[i])
            temp_col_2 = int(7+df['Server assignment (i.a)'].iat[i])
            if type(df.iat[i,temp_col_1]) == list:
                df.iat[i,temp_col_2] = df.iat[i-1,temp_col_2] + [new_job]
            else: # server not occupied => add to server
                df.iat[i,temp_col_1] = new_job
        # Step 3 of 3: record values in data frame if departure
        if event == 'departure':
            job_rows = list(df.iloc[i-1][-3:])
            previous_jobs = sorted([v for v in job_rows if type(v) == list], key=lambda x: x[1])
            index = previous_jobs[0][0]
            time_diff = previous_jobs[0][1]
            df['t'].iat[i] = time_diff + df['t'].iat[i-1]
            df['Request index (i.d)'].iat[i] = index
            df['Service time (i.d)'].iat[i] = df['t'].iat[i] - \
            float(df.loc[df['Request index (i.a)'] == index]['t'].tolist()[0])
            df['Time of next arrival'].iat[i] = df['Time of next arrival'].iat[i-1]
            df['Service requirement of next arrival'].iat[i] = \
            df['Service requirement of next arrival'].iat[i-1]
            server_no = 1+job_rows.index([index,time_diff]) # the server that had a job depart
            df['Server assignment (i.a)'].iat[i] = server_no
            df.iloc[i,-6:] = df.iloc[i-1,-6:] # copying values from previous queues & servers
            time_diff = df['t'].iat[i] - df['t'].iat[i-1]
            for col in ['Job 1', 'Job 2', 'Job 3']:
                if type(df[col].iat[i-1]) == list:
                    df[col].iat[i] = [df[col].iat[i-1][0], df[col].iat[i-1][1] - time_diff]
            if len(df.iat[i,7+server_no]) == 0: 
                df.iat[i,10+server_no] = np.nan
            else: # remove job at front of queue:
                df.iat[i,7+server_no] = df.iat[i-1,7+server_no][1:]
                job_moved_to_server = list(df.iat[i-1,7+server_no][0]) 
                # move job to server:
                if server_no in [1,2]:
                    df.iat[i,10+server_no] = job_moved_to_server 
                if server_no == 3:
                    df.iat[i,10+server_no] = [job_moved_to_server[0],job_moved_to_server[1]]
            if random == False:
                if sum(df['Request index (i.d)'].notna()) == len(params[0]):
                    break # exit loop after all requests have departed
        i+=1
    # generate output data frame containing all events:
    df = df[df['t'].notna()] # remove empty rows at end (if there are any)
    # how many jobs in queue 3 at time of each event:
    # n_jobs_q3 = [len(v) if type(v)==list else 0 for v in df['Queue 3']]
    # generate arrival & departure times from each server
    server = [ [], [], [], [] ]
    for s in [1,2,3]:
        df_temp = df.loc[df['Server assignment (i.a)'] == s]
        ind = df_temp[df_temp['Request index (i.d)'].notna()]['Request index (i.d)'].tolist()
        for i in ind:
            server[s].append([float(df_temp.loc[df_temp['Request index (i.a)'] == i]['t']),
                  float(df_temp.loc[df_temp['Request index (i.d)'] == i]['t'])])
        server[s] = np.array(server[s])
    # calculate mean service time (excluding burn-in sample)
    mst = df['Service time (i.d)'].iloc[m:].mean()
    print(f'MST (algo={algorithm}, d={d}): {mst:.4} seconds')     
    return [mst, server[1], server[2], server[3], df]


def simulated_mst_inference(N:int,S:int,T:int,f:float,m:int,params:list,alpha:float) -> pd.DataFrame:
    '''
    Prints 100(1-alpha)% CIs for the mean service time of both algorithms,
    for various values of d. Each mean service time (MST) is calculated
    by simulating N arrivals, not considering the first m events. S MST's are 
    estimated for each value of d, for both algorithms. This means the 
    simulate_mean_service_time function is called S*maxd*2 times. To plot the 
    CIs, the plot argument must be True.
    '''
    # Generate S*2*D array to store MSTs
    d_list = [0,0.3,0.7,1,1.5,2,99]
    mean_service_time_array = np.zeros((S,2,len(d_list)))
    for _algorithm in range(2):
        for _d in enumerate(d_list):
            for simulation in range(S):
                algorithm= _algorithm + 1
                mean_service_time_array[simulation, _algorithm, _d[0]] = \
                compute_mean_service_time(algorithm, _d[1], N, m, f, None, True, params)[0]
    # Calculate average MST, std error of average MST and CI of each MST...
    # ... for the output of the first algorithm:
    average_algo_1 = [np.mean(mean_service_time_array[:,0,_d[0]]) for _d in enumerate(d_list)]
    std_err_algo_1 = [np.std(mean_service_time_array[:,0,_d[0]],ddof=1) for _d in enumerate(d_list)]
    confint_algo_1 = [[average_algo_1[i]-stats.t.ppf(1-alpha/2,S-1)*std_err_algo_1[i], \
                       average_algo_1[i]+stats.t.ppf(1-alpha/2,S-1)*std_err_algo_1[i]] \
                       for i in range(len(d_list))]
    confint_algo_1_rounded = [[round(v[0],4),round(v[1],4)] for v in confint_algo_1]
    # ... and the second algorithm:
    average_algo_2 = [np.mean(mean_service_time_array[:,1,_d[0]]) for _d in enumerate(d_list)]
    std_err_algo_2 = [np.std(mean_service_time_array[:,1,_d[0]],ddof=1) for _d in enumerate(d_list)]
    confint_algo_2 = [[average_algo_2[i]-stats.t.ppf(1-alpha/2,S-1)*std_err_algo_2[i], \
                       average_algo_2[i]+stats.t.ppf(1-alpha/2,S-1)*std_err_algo_2[i]] \
                       for i in range(len(d_list))]
    confint_algo_2_rounded = [[round(v[0],4),round(v[1],4)] for v in confint_algo_2]
    # Print and return each MST confidence interval:
    confidence_intervals = pd.DataFrame({'Algorithm': [1]*len(d_list)+[2]*len(d_list),
                           'd': d_list*2,'MST CI': confint_algo_1_rounded+confint_algo_2_rounded})
    print(f'\n{100*(1-alpha)}% Confidence Intervals for estimated mean service time:')
    print(confidence_intervals.to_string(index=False))
    return confidence_intervals


def analyse_confidence_intervals(ci_output):
    '''Given a data frame containing the confidence intervals of each estimated
    MST for each {d,algorithm} pair, plots the results'''
    x_ticks = ('d=0', 'd=0.3', 'd=0.7', 'd=1', 'd=1.5', 'd=2', 'd=inf')
    x_1 = np.arange(1, 8)
    x_2 = x_1 + 0.25
    # generate lists of means and interval widths:
    means_algo_1 = [sum(ci_output.loc[ci_output['Algorithm'] == 1]['MST CI'].iat[i])/2 for i in range(len(x_ticks))]
    means_algo_2 = [sum(ci_output.loc[ci_output['Algorithm'] == 1]['MST CI'].iat[i])/2 for i in range(len(x_ticks))]
    temp = ci_output.loc[ci_output['Algorithm'] == 1]['MST CI'].iloc[5]
    ci_half_width_algo_1 = (temp[1] - temp[0])/2
    temp = ci_output.loc[ci_output['Algorithm'] == 2]['MST CI'].iloc[5]
    ci_half_width_algo_2 = (temp[1] - temp[0])/2
    plt.figure(figsize=(1.6*5,5))
    # t-tests: t.test(a,b), where a and b are the length-S vectors of simulated MSTs
    # plot error bars:
    plt.errorbar(x=x_1, y=means_algo_1, yerr=ci_half_width_algo_1, color='blue', capsize=3,
                 linestyle='None',marker='s', markersize=7, mfc='blue', mec='blue')
    plt.errorbar(x=x_2, y=means_algo_2, yerr=ci_half_width_algo_2, color='orange', capsize=3,
                 linestyle='None',marker='s', markersize=7, mfc='orange', mec='orange')
    plt.xticks(x_1, x_ticks)
    plt.ylabel('Mean Service Time')
    plt.legend(['Algorithm 1', 'Algorithm 2'], loc=0, frameon=True)
    plt.tight_layout()
    plt.savefig('./output/MST plot.png')


#%%
# Generate output:
    
# If _ascii is True, read in input files and write required output to output folder:

if _ascii == True:
    inputs = get_inputs(test_number)
    parameters = inputs[0]+[inputs[1]]
    out = compute_mean_service_time(*parameters)
    i = str(test_number)
    np.savetxt('./output/mrt_'+i+'.txt', np.array([out[0]]), fmt='%.4f')
    np.savetxt('./output/s1_dep_'+i+'.txt', out[1], fmt='%.4f')
    np.savetxt('./output/s2_dep_'+i+'.txt', out[2], fmt='%.4f')
    np.savetxt('./output/s3_dep_'+i+'.txt', out[3], fmt='%.4f')

# If _ascii is False, generate CIs for each d for each algorithm:
    
if _ascii == False:
    output = simulated_mst_inference(N,S,None,f,m,params,alpha)
    plot = analyse_confidence_intervals(output)
        




















