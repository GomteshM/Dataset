"""
PYTHON IMPLEMENTATION OF STEP COUNTER ALGORITHM


Algorithm based on oxford java step counter
(https://github.com/Oxford-step-counter/Java-Step-Counter)

Modifications:
    Added plotting to compare steps data available in csv file and detected steps data
    
sample output of program is present at the end as comments

"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Class datapoint contains acceleration magnitude and time value in ms
class DataPoint:
    '''
    Class DataPoint
    getter methods: getTime(), getMagnitude()
    setter methods: setTime(time), setMagnitude(mag)
    '''
    def __init__(self, mag, time):
        self.mag = mag
        self.time = time
    
    def __str__(self):
        ret = " Acceleration magnitude is {} and time in ms is {} ".format(self.mag, self.time)
        return ret
    
    def getTime(self):
        return self.time
    
    def setTime(self, time):
        self.time = time
        
    def getMagnitude(self):
        return self.mag
    
    def setMagnitude(self, mag):
        self.mag = mag


# 1. Preprocessing stage
def PreProcessStage(rawData, skipInterpolation = True, samplingPeriod = 80):
    '''
    First stage of step detection
      This stage is responsible for computing the magnitude of the triaxial 
      accelerometry signal and ensuring a constant sampling frequency by means 
      of linear interpolation.
    Performs the operations involved in preprocessing stage and
    Returns the output of preprocessing stage
    
    Parameters
    ----------
    rawData : DataFrame
        pandas dataframe containing accelerometer values viz. acc_x, acc_y, acc_z 
        and corresponding timestamp of sample in nanoseconds
    skipInterpolation : bool, optional
        The default is True.
        should we interpolate datapoints depending on the sampling period
    samplingPeriod : float, optional
        The default is 80.
        sampling period is in milli second to interpolate values

    Returns
    -------
    ppData : list
        the output list  of preprocessing stage which contains DataPoints.
    '''
    rawData["time"] = rawData["time"]*1e-6 # convert time from ns to ms
    ppData = [] # Output of preprocessing stage
    if(skipInterpolation):
        # If skip interpolation is true then iterate over the raw data
        # and append each acceleration magnitude and timevalue to the output list
        dataLen = len(rawData["time"])
        for idx in range(dataLen):
            # acceleration magnitude
            acc_mag = ((rawData["acc_x"][idx] ** 2 + rawData["acc_y"][idx] ** 2 + rawData["acc_z"][idx] ** 2) ** .5)
            cur_time = rawData["time"][idx]
            dp = DataPoint(acc_mag, cur_time)
            ppData.append(dp)
    else:
        # Interpolate data points
        dataLen = len(rawData["time"])
        if(dataLen > 2):
            # append first point into the list
            acc_mag = ((rawData["acc_x"][0] ** 2 + rawData["acc_y"][0] ** 2 + rawData["acc_z"][0] ** 2) ** .5)
            cur_time = rawData["time"][0]
            dp = DataPoint(acc_mag, cur_time)
            ppData.append(dp)
            
            for idx in range(1, dataLen):
                # acceleration magnitude at current index
                acc_mag = ((rawData["acc_x"][idx] ** 2 + rawData["acc_y"][idx] ** 2 + rawData["acc_z"][idx] ** 2) ** .5)
                # Previous time value and acceleration magnitude of datapoint
                t_prev = ppData[-1].getTime()
                a_prev = ppData[-1].getMagnitude()
                tSampleGap = (rawData["time"][idx] - t_prev) #Total millisecond gap
                numPoints = 0 # number of points for interpolation
                if( tSampleGap >= samplingPeriod):
                    numPoints = int((tSampleGap - 1)/ samplingPeriod)
                if( numPoints > 0):
                    for i in range(numPoints):
                        t = ppData[-1].getTime() + samplingPeriod #milli seconds
                        if(t < rawData['time'][idx]):
                            # Interpolate data point and append into output list
                            a_mag_new = (a_prev + ((acc_mag - a_prev) / (rawData['time'][idx] - t_prev)) * (t - t_prev))
                            dp = DataPoint(a_mag_new, t)
                            ppData.append(dp)
    
    # dp = DataPoint(0, 0)
    # ppData.append(dp) # Last data point as 0,0
    return ppData


# 2. Filter stage
def FilterStage(ppData, filterLength = 13, filterSTD = 0.35):
    '''
    Second stage of oxford step counting algorithm
    In order to reduce the noise level, oxford step algorithm implements a 
    finite impulse response (FIR) low-pass filter

    Parameters
    ----------
    ppData : list
        This ppData list is a output of preprocessing stage which contains DataPoints
    filterLength : int, optional
        The default is 13.
        length of window for a filter
    filterSTD : float, optional
        The default is 0.35.
        std dev for generating filter coefficients

    Returns
    -------
    smoothData : list
        smoothened data.

    '''
    midPoint = int(filterLength/2)
    # Generate filter coefficients
    filterVals = GenerateFilterCoef(filterLength, filterSTD)
    filterSum = sum(filterVals)
    inputQueue = ppData[:] # Shallow copy
    smoothData = [] # output of filter stage
    active = True
    window = [] # List containing datapoint values
    
    while(active):        
        window.append(inputQueue.pop(0))
        if(len(inputQueue) == 0):
            active = False
            # # Special handling for final data point.
            # dp = DataPoint(0, 0)
            # smoothData.append(dp)
            # continue
        
        if(len(window) == filterLength):
            temp = [v1*v2.getMagnitude() for v1,v2 in zip(filterVals, window)]
            acc_new_mag = sum(temp)/filterSum
            dp = DataPoint(acc_new_mag, window[midPoint].getTime())
            smoothData.append(dp)
            window.pop(0) # Remove the oldest element in the list.
            
    return smoothData

def GenerateFilterCoef(filterLength = 13, filterSTD = 0.35):
    '''
    Generate the filter coefficients based on the filter length and std dev

    Parameters
    ----------
    filterLength : int, optional
        length of filter. The default is 13.
    filterSTD : float, optional
        std dev. The default is 0.35.

    Returns
    -------
    FIR_Vals : list
        filter coefficients.

    '''
    FIR_Vals = [ math.pow(math.e, -0.5*math.pow((i - (filterLength - 1) / 2) / (filterSTD * (filterLength - 1) / 2), 2)) for i in range(filterLength)]
    return FIR_Vals


# 3. Scoring stage
def ScoringStage(smoothData, windowSize = 35):
    '''
    Third stage of oxford step counting algorithm
    The function of the scoring stage is to evaluate the peakiness of a given 
    sample. The result of this stage should increase the magnitude of any 
    peaks, making them more evident for the subsequent peak detection.

    Parameters
    ----------
    smoothData : list
        list containing smoothened datapoint values
    windowSize : int, optional
        window size for score peak calculation. The default is 35.

    Returns
    -------
    peakScoreData : list
        output of scoring stage.

    '''
    midPoint = int(windowSize/2) # Mid point of window
    inputQueue = smoothData[:] # Shallow copy
    peakScoreData = []    
    window = [] # List containing magnitude values
    active = True
    
    while(active):
        window.append(inputQueue.pop(0))
        if(len(inputQueue) == 0):
            active = False
            # dp = DataPoint(0, 0)
            # peakScoreData.append(dp)
            # continue
            
        if(len(window) == windowSize):
            diffLeft = 0
            diffRight = 0
            # calculate diffleft and diffright based on the algorithm
            for i in range(midPoint):
                diffLeft += window[midPoint].getMagnitude() - window[i].getMagnitude();
            for J in range(midPoint, windowSize):
                diffRight += window[midPoint].getMagnitude() - window[J].getMagnitude();
        
            # Calculate the score and append to the output list
            score = (diffRight + diffLeft) / (windowSize - 1)
            dp = DataPoint(score, window[midPoint].getTime())
            peakScoreData.append(dp)
            # Pop out the oldest point from the window
            window.pop(0)
        
    return peakScoreData


# 4. Detection stage
def DetectionStage(peakScoreData, threshold = 1.2):
    '''
    Fourth stage of oxford step counting algorithm
    This stage identifies potential candidate peaks to be associated with a 
    step by statistically detecting outliers. 
    As the algorithm processes the signal, it keeps track of a running mean 
    and standard deviation. These two quantities are used to determine 
    whether any given sample is an outlier.

    Parameters
    ----------
    peakScoreData : list
        list containing peakiness values.
    threshold : float, optional
        detection threshold. The default is 1.2.

    Returns
    -------
    outputQueue : list
        output list containing DataPoints.

    '''
    inputQueue = peakScoreData[:] # Shallow copy
    outputQueue = []
    # initial parameters
    active = True
    count = 0
    acc_mean = 0
    acc_std = 0
    while(active):
        dp = inputQueue.pop(0)
        if(len(inputQueue) == 0):
            active = False
            # dp = DataPoint(0, 0)
            # outputQueue.append(dp)
            # continue
        count +=1
        o_mean = acc_mean
        
        # Update calculations of mean and std deviation
        if(count == 1):
            acc_mean = dp.getMagnitude()
            acc_std = 0
        elif(count == 2):
            acc_mean = (acc_mean + dp.getMagnitude())/2
            acc_std = (((dp.getMagnitude() - acc_mean)**2 + (o_mean - acc_mean)**2 ) ** .5)/2            
        else:
            acc_mean = (dp.getMagnitude() + (count - 1)*acc_mean)/count
            acc_std = (((count - 2) * (acc_std**2)/(count-1)) + (o_mean - acc_mean)**2 + ((dp.getMagnitude() - acc_mean) ** 2)/count)**.5
        
        # Once we have enough data points to have a reasonable mean/standard deviation, start detecting
        if(count > 15):
            if ((dp.getMagnitude() - acc_mean) > acc_std * threshold):
                # This is peak
                outputQueue.append(dp)
            
    return outputQueue


# 5. Post processing stage
def PostProcessStage(peakData, timeThreshold=200):
    '''
    handles false positives from the detection stage by having a sliding 
    window of fixed size t_window and selecting the higher peak within the window

    Parameters
    ----------
    peakData : list
        this list is output of detection stage.
    timeThreshold : float/int, optional
        The default is 200.
        time in millisecond

    Returns
    -------
    steps : int
        number of steps detected by algorithm
    outputQueue : list
        list of datapoints for which step is detected.

    '''
    steps = 0 # number of steps detected
    inputQueue = peakData[:]
    outputQueue = []
    current = peakData[0]
    active = True
    while(active):
        dp = inputQueue.pop(0)
        if(len(inputQueue) == 0):
            active = False
            # dp = DataPoint(0, 0)
            # End of stage
            # continue
        
        if ((dp.getTime() - current.getTime()) > timeThreshold):
            # If the time difference exceeds the threshold, we have a confirmed step
            current = dp
            steps += 1
            outputQueue.append(dp)
        else:
            if (dp.getMagnitude() > current.getMagnitude()):
                # Keep the point with the largest magnitude.
                current = dp
    
    return steps, outputQueue



def readCSVFile(fPath):
    '''
    read the csv file present at current path

    Parameters
    ----------
    fPath : string
        csv file path.

    Returns
    -------
    raw_DF : DataFrame
        pandas dataframe which contains all csv data.

    '''
    # timestamp (nanosencods since boot), accx, accy, accz, steps as detected by algo, 
    # steps as detected by ground truth device, steps as detected by the hardware step counter    
    colNames = ['time' , 'acc_x', 'acc_y', 'acc_z', 'steps_algo', 'steps_GTD', 'steps_counter']
    raw_DF = pd.read_csv(fPath, header=None, names= colNames, engine='python')
    return raw_DF

def RunAlgo(rawData, skipInterpolation = True, samplingPeriod = 80, \
            SKIPFILTER = True, filterLength = 13, filterSTD = 0.35, \
            windowSize = 35, threshold = 1.2, timeThreshold = 200):
    '''
    Implement the oxford java step counter algorithm

    Parameters
    ----------
    rawData : DataFrame
        input data required for preprocessing stage.
    skipInterpolation : bool, optional
        interpolate datapoints condition. The default is True.
    samplingPeriod : float, optional
        Time period to interpolate data points. The default is 80 millisecond.
    SKIPFILTER : bool, optional
        wheather filter stage should be executed or not. The default is True.
    filterLength : int, optional
        length of filter window. The default is 13.
    filterSTD : float, optional
        std dev for generating filter coefficients. The default is 0.35.
    windowSize : int, optional
        length of window in scoring stage. The default is 35.
    threshold : float, optional
        threshold required for detection stage. The default is 1.2.
    timeThreshold : float/int, optional
        time in millisecond, used to detect steps. The default is 200.

    Returns
    -------
    steps : int
        number of steps.
    detectedStepsList : list
        datapoints for which step is detected.

    '''
    ppData = PreProcessStage(rawData) # skipInterpolation, samplingPeriod
    if(not SKIPFILTER):
        smoothData = FilterStage(ppData) # filterLength, filterSTD
    else:
        smoothData = ppData
    peakScoreData = ScoringStage(smoothData, windowSize)
    peakData = DetectionStage(peakScoreData, threshold)
    steps, detectedStepsList = PostProcessStage(peakData, timeThreshold)
    #print(steps)
    return steps, detectedStepsList


def AlgoPlot(d):
    '''
    Plot the graph representing steps detected by current algorithm and steps
    already existing in csv data file

    Parameters
    ----------
    d : dataframe
        pandas dataframe containing "steps_algo" , "steps_GTD" & "steps_new_algo"
        columns

    Returns
    -------
    plt : matplot
        the plot.

    '''
    # plot:
    stAlgo = [0]
    stGTD = [0]
    for i in range(1, len(d)):
        stAlgo.append(d["steps_algo"][i] - d["steps_algo"][i-1])
        stGTD.append(d["steps_GTD"][i] - d["steps_GTD"][i-1])
        
    stAlgo = np.array(stAlgo)
    stGTD = np.array(stGTD)
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True, figsize=(30, 5))
    fig.suptitle('Steps detected by different tech')
    # Plot steps by Ground Truth Device
    ax1.plot(d["time"], stGTD, 'r')
    ax1.set_ylabel('GTD')
    ax1.set_title("Steps Detected by Ground Truth Device")
    # Plot steps by algorithm existing csv file data
    ax2.plot(d["time"], stAlgo, 'g')
    ax2.set_ylabel('Algorithm')
    ax2.set_title("Steps Detected by oxford algorithm existing CSV file data")
    # Plot steps by oxford java step counter algorithm
    ax3.plot(d["time"], d["steps_new_algo"], 'b')
    ax3.set_ylabel('New algo')
    ax3.set_title("Steps detected by new oxford java step algorithm")
    plt.xlabel('time in ms')
    plt.show()
    # fig.savefig("OxfordAlgo_CSVFile1.png")
    return plt

def main():
    # csv files present in DataSet-master validation folder
    CSV_Files = ['user1_armband_1506423438471.csv', 'user1_backpocket_1506422470497.csv', 'user1_bag_1506423095164.csv', \
                'user1_frontpocket_1506422223341.csv', 'user1_hand_1506421989895.csv', 'user1_neckpouch_1506422851785.csv', \
                'user2_armband_1506423383401.csv', 'user2_backpocket_1506422483834.csv', 'user2_bag_1506422838474.csv', \
                'user2_frontpocket_1506422217391.csv', 'user2_hand_1506421987098.csv', 'user2_neckpouch_1506423094931.csv']

# =============================================================================
    # Sample: step detection for 1st csv file
    folDir = '../validation/' + CSV_Files[4]
    print(folDir)
    d = readCSVFile(folDir) # read the file
    # run the algorithm
    steps, d1 = RunAlgo(rawData=d, threshold = 1.2, windowSize = 35, timeThreshold = 200, SKIPFILTER = False)
    op_str = ("steps detected by current algorithm : {}" + \
        "\n from data in current csv files ... \n\t steps detected by GTD: {}" + \
        "\n\t steps detected by algorithm {}").format(steps, max(d["steps_GTD"]), max(d["steps_algo"]))            
    print(op_str)
    # txtFileObj = open("OxfordAlgo_Results.txt","a")
    # txtFileObj.write(op_str)
    # txtFileObj.close()
    
    # Plotting data
    indexList = [d[d["time"] == i.getTime()].index[0] for i in d1]
    d["steps_new_algo"] = 0
    for idx in indexList:
        d.at[idx,"steps_new_algo"]=1 
    
    AlgoPlot(d)
# =============================================================================
        
# =============================================================================
    # Step detection for all csv files
    # txtFileObj = open("OxfordAlgo_Results.txt","a")
    for fp in CSV_Files:
        folDir = '../validation/' + fp
        d = readCSVFile(folDir)
        steps, d1 = RunAlgo(rawData=d, threshold = 1.2, windowSize = 35, timeThreshold = 200, SKIPFILTER = False)
        op_str = ("\n\nFile name : {} \nsteps detected by current algorithm : {}" + \
        "\n from data in current csv files ... \n\t steps detected by GTD: {}" + \
            "\n\t steps detected by algorithm {}").format(fp, steps, max(d["steps_GTD"]), max(d["steps_algo"]))
        print(op_str)
        # txtFileObj.write(op_str)
    # txtFileObj.close()
# =============================================================================

# =============================================================================
# =============================================================================
#     # Step detection for different values of threshold and window size
#     th_values = np.arange(1, 1.41, 0.05)
#     wd_values = np.arange(3, 52, 8)
#     # txtFileObj = open("OxfordAlgo_Results.txt","a")
#     for fp in CSV_Files:
#         folDir = '../validation/' + fp
#         d = readCSVFile(folDir)
#         for i in th_values:
#             for j in wd_values:
#                 temp1 = d[:]
#                 temp2 = d[:]
#                 steps1,d1 = RunAlgo(rawData = temp1, threshold = i, windowSize = j, timeThreshold=200, SKIPFILTER = True)
#                 steps2,d2 = RunAlgo(rawData = temp2, threshold = i, windowSize = j, timeThreshold=200, SKIPFILTER = False)
#                 op = ("\n\nFile name : {} \nSteps detected for motion threshold {} and scoring window size {} " + \
#                     "time threshold 200 ms \n\t skip filter value as TRUE are :  {} " + \
#                     "\n\t skip filter value as FALSE are : {}").format(fp, i, j, steps1, steps2)
#                 print(op)
#                 # txtFileObj.write(op)
#     # txtFileObj.close()
# =============================================================================
# =============================================================================

    
if __name__ == '__main__':
    main()
    
    
# =============================================================================
# 
# File name : user1_armband_1506423438471.csv 
# steps detected by current algorithm : 315
#  from data in current csv files ... 
# 	 steps detected by GTD: 335
# 	 steps detected by algorithm 316
# 
# File name : user1_backpocket_1506422470497.csv 
# steps detected by current algorithm : 504
#  from data in current csv files ... 
# 	 steps detected by GTD: 343
# 	 steps detected by algorithm 501
# 
# File name : user1_bag_1506423095164.csv 
# steps detected by current algorithm : 328
#  from data in current csv files ... 
# 	 steps detected by GTD: 346
# 	 steps detected by algorithm 321
# 
# File name : user1_frontpocket_1506422223341.csv 
# steps detected by current algorithm : 347
#  from data in current csv files ... 
# 	 steps detected by GTD: 327
# 	 steps detected by algorithm 338
# 
# File name : user1_hand_1506421989895.csv 
# steps detected by current algorithm : 311
#  from data in current csv files ... 
# 	 steps detected by GTD: 326
# 	 steps detected by algorithm 311
# 
# File name : user1_neckpouch_1506422851785.csv 
# steps detected by current algorithm : 345
#  from data in current csv files ... 
# 	 steps detected by GTD: 346
# 	 steps detected by algorithm 344
# 
# File name : user2_armband_1506423383401.csv 
# steps detected by current algorithm : 337
#  from data in current csv files ... 
# 	 steps detected by GTD: 343
# 	 steps detected by algorithm 339
# 
# File name : user2_backpocket_1506422483834.csv 
# steps detected by current algorithm : 349
#  from data in current csv files ... 
# 	 steps detected by GTD: 337
# 	 steps detected by algorithm 345
# 
# File name : user2_bag_1506422838474.csv 
# steps detected by current algorithm : 383
#  from data in current csv files ... 
# 	 steps detected by GTD: 361
# 	 steps detected by algorithm 376
# 
# File name : user2_frontpocket_1506422217391.csv 
# steps detected by current algorithm : 339
#  from data in current csv files ... 
# 	 steps detected by GTD: 343
# 	 steps detected by algorithm 340
# 
# File name : user2_hand_1506421987098.csv 
# steps detected by current algorithm : 341
#  from data in current csv files ... 
# 	 steps detected by GTD: 340
# 	 steps detected by algorithm 341
# 
# File name : user2_neckpouch_1506423094931.csv 
# steps detected by current algorithm : 348
#  from data in current csv files ... 
# 	 steps detected by GTD: 360
# 	 steps detected by algorithm 348
# =============================================================================
