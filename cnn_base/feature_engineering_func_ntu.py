import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import vectorize
import json
import time
np.random.seed(32113)

def feature_engineering_CNN(df, out_data_path, category, sample=60000):
    '''
    function:
    - aggregates 2 user defined functions that prepares dataframe for CNN modeling.
    - it also prints out how long it takes to run.

    input:
    - df = dataframe that was converted from raw_data json file
    - category = used to name output pickle file
    - sample = number of datapoints included in the final dataframe. 


    output:
    - pickled dataframe that will be used for CNN modeling (1176 features)
    - each row represents 42 by 28 pixel image
    file name: "./data/{}.pkl".format(category)
    '''

    start_time = time.time()
    #runs CNN feature engineering functions
    df_1 = CNN_feat_eng_pt1(df)
    df_2 = CNN_feat_eng_pt2(df_1)

    df_2.index = range(len(df_2))
    random_ind = np.random.choice(list(df_2.index), sample, replace=False)
    df_2 = df_2.loc[list(random_ind)]

    df_2.index = df_2['key_id']
    df_2.to_pickle(out_data_path + "{}.pkl".format(category))
    print("--- %s seconds ---" % (time.time() - start_time))
    return df_2




def _array_normalizer(array1,Xmin,Xmax,array_min):
    '''
    function:
        - normalize X,Y array by range of X
        - used in feature_eng_pt2
    input:
        array1 = array that you want to normalize (1D array or list)
        Xmin = minimum value of your X array (int)
        Xmax = maximum value of your X array (int)
        array_min = minimum value of array1

    output:
        normalized array of array1
    '''
    return (np.array(array1)-np.array([array_min]*len(array1)))/float(Xmax-Xmin)

def _radian_direction(dy,dx):
    '''
    function:
        - based on given dy and dx it calculates direction in radian.
        - used in feature_eng_pt3
    input:
        dy = change in y
        dx = change in x

    output:
        returns radian value (0 to 6.28)
    '''
    if dy < 0.0 and dx > 0.0:
        return (2*np.pi + np.arctan(dy/dx))
    elif dy >=0.0 and dx > 0.0:
        return (np.arctan(dy/dx))
    else:
        return np.pi + np.arctan(dy/dx)


def _value_from_stroke(stroke_length,percentage,xperstroke):
    '''
    function:
        - generates list of equally spaced x,y or time using input values.
        - used in feature_eng_pt5
        - for example: if your stroke_length is 60 and percentage is 0.4,
        it will create a list of indexes that equally spaced
        40(=0.4*100) datapoints from 60 data points.
        Using this list of index, it will create a list of X,Y, or time.
    input:
        stroke_length = length of the stroke
        percentage = number of data points of the stroke/total number of data points
        xperstroke =  data points in a stroke
                        (the list of data points should be in chronological order)
    output:
        list of data point (x,y,time) that
        return np.linspace array which represents index of datapoints in each stroke

    '''
    idxs = np.around(np.linspace(0,stroke_length-1,int(np.around(percentage*100))))
    return [xperstroke[int(ind)] for ind in idxs]




##############################################################################
#                       functions for CNN (neural networks)                  #
##############################################################################



def CNN_feat_eng_pt1(df):
    '''
    function:
        this function prepares features that are needed for CNN_feat_eng_pt2.
        codes are similar to feature_eng_pt1 and feature_eng_pt2.
        for time efficiency reason I created this function.

        - generates following features:
            total_number_of_datapoints = total number of datapoints
                                                 exist in an image [int]
            X = normalized X Ranges between 0 to 1 [list]
            Y = Y values normalized using X. Ranges between 0 and 1.5 [list]
            Ymax = maximum value of Y [int]
            time = list of time [list]
              * note: X,Y,time should have same length.
            total_time_drawing = total amount of time when user was drawing [int]

        - Filtering applied:
          1: filtered out data where recognize == 0
          2: filtered out data where stroke_number is greater than 15
          3: filtered out data where final time is greater than 20000
          4: randomly selecting 60000 rows from existing data.
             This will balance out number of datapoints per each drawing topic.
             seed(32113) is used.
    input:
        df = dataframe. raw data json converted to pd.dataframe
    output:
        df_cf = new dataframe that contains additional features needed for CNN

    '''

    # create feature "stroke_number"
    df['stroke_number']=df['drawing'].str.len()
    b_loon = {True: 1, False:0}
    df['recognized'] = df['recognized'].map(b_loon)
    df_cf = df[(df['recognized']==1) & (df['stroke_number'] <= 15)]
    df_cf['final_time'] = [df_cf.loc[i,'drawing'][df_cf.stroke_number[i]-1][2][-1] for i in df_cf.index]


     # process:
    # 1. make a list or int
    # 2. store contents of 1. in a new dictionary
    # 3. make new column in your dataframe with 2. dictionary

    X = {}
    Y = {}
    Ymax ={}
    time = {}
    ttnum_dp = {}
    sumtimeps = {}

    for i in df_cf.index:
        num = df_cf.loc[i,'stroke_number']
        #store X,Y,time of the stroke in a temp list
        Xt = [df_cf.loc[i,'drawing'][stroke][0] for stroke in range(num)]
        Yt = [df_cf.loc[i,'drawing'][stroke][1] for stroke in range(num)]
        tt = [df_cf.loc[i,'drawing'][stroke][2] for stroke in range(num)]

        # calculate the difference between final and initial time of a stroke
        Tdifftemp = [(df_cf.loc[i,'drawing'][stroke][2][-1] - df_cf.loc[i,'drawing'][stroke][2][0])\
                     for stroke in range(num)]

        # normalizing X and Y
        Xtemp = [item for stroke in Xt for item in stroke]
        Ytemp = [item for stroke in Yt for item in stroke]
        time[i] = [item for stroke in tt for item in stroke]

        #normalizing X and Y
        Xmintemp = np.min(Xtemp)-10
        Xmaxtemp = np.max(Xtemp)+10
        Ymintemp = np.min(Ytemp)-10
        Xnorm = _array_normalizer(Xtemp, Xmintemp,Xmaxtemp,Xmintemp)
        Ynorm = _array_normalizer(Ytemp, Xmintemp,Xmaxtemp,Ymintemp)
        Ymax[i] = np.max(Ynorm)
        X[i] = Xnorm
        Y[i] = Ynorm
        ttnum_dp[i] = len(Ynorm)
        sumtimeps[i] = sum(Tdifftemp)
    # create new features
    df_cf['total_number_of_datapoints'] = pd.Series(ttnum_dp)
    df_cf['Ymax'] = pd.Series(Ymax)
    df_cf['time'] = pd.Series(time)
    df_cf['total_time_drawing'] = pd.Series(sumtimeps)
    df_cf['X'] = pd.Series(X)
    df_cf['Y'] = pd.Series(Y)
    df_cf = df_cf[df_cf['Ymax']<=1.5]
    df_cf = df_cf[df_cf['final_time']<=20000]
    return df_cf


def CNN_feat_eng_pt2(df_cf):

    '''
    function:
        this function is used to reformat input df to create CNN ready dataframe
        - generating a dataframe that will contains 1176 features per image
        1176 = 42(Y axis) * 28 (X axis)
        - it will also contain word and countrycode features

    input:
        df_cf = output dataframe from CNN_feat_eng_pt1
        category = string. type of topic for instance, cat. [str]
    output:
        no output. the function saves final data frame as a pickle file

    '''
    orig_index = df_cf.index
    df_cf.index = range(len(df_cf))
    image_pile = np.zeros((len(df_cf),1176))
    for ind in df_cf.index:
        image = np.zeros((42,28))
        xarray = np.around(np.array(df_cf.loc[ind,'X'])*28)
        yarray = np.around(np.array(df_cf.loc[ind,'Y'])*42/float(df_cf.loc[ind,'Ymax']))
        xarray[xarray>=28.] = 27
        yarray[yarray>=42.] = 41
        for item in range(len(xarray)):
            image[int(np.around(yarray[item])),int(np.around(xarray[item]))] = df_cf.loc[ind,'time'][item]
        image_pile[ind] = image.reshape(1,1176)
    #return pd.DataFrame(image_pile, index = orig_index)
    df_final = pd.DataFrame(image_pile, index = orig_index)
    df_cf_country = df_cf['countrycode']
    df_cf_word = df_cf['word']
    df_cf_keyid = df_cf['key_id']
    df_cf_country.index = orig_index
    df_cf_word.index = orig_index
    df_cf_keyid.index = orig_index
    return pd.concat([df_final,df_cf_country,df_cf_word,df_cf_keyid], axis=1)
    #df_final.to_pickle("./data/{}_15.pkl".format(category))



##############################################################################
#                               other functions                              #
##############################################################################



def load_json(filename):
    '''
    Function:
        - opens json file and store information in a pandas dataframe
        - also prints out aggregated df with counts of picture by countrycode
    Input:
        1. filename/path ex: ./data/filename.json
    Output:
        1. new dataframe containing json info
    '''
    df = pd.read_json(filename, lines=True)
    test = df.groupby(df['countrycode']).count()
    print(test.sort(columns='drawing',ascending=False).head(15))
    return df

def pic_viewer(df_cf, _id):

    '''
    Function:
        - If X and Y columns exist in your dataframe, you can use this function
                            to view drawing with specific id.
        - run this after running CNN_feat_eng_pt1 or feature_eng_pt2
    Input:
        1. dataframe df_cf
        2. object id _id
    Output:
        1. scatter plot of x and y
    '''
    plt.scatter(df_cf.X[_id],df_cf.Y[_id])
    plt.gca().invert_yaxis()
