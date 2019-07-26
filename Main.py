# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:02:10 2017

@author: Payam Bagheri
"""
# A look at the data
mdata.head(10)

#############################################
# Near 52 week High and Low
""" hilo52w is a function that provides the highest High and the lowest Low for 
each year and ticker. 
Inputs: 
year: A string which is the desired year
ticker: A string which is the desired ticker name
"""
def hilo52w(year,ticker):
    ydata = mdata[mdata.DateTime.apply(lambda x: x[0:4]) == year]
    hi_lo_52w = pd.DataFrame({'high_52w': ydata.groupby('Ticker')['High'].max(),
                              'low_52w': ydata.groupby('Ticker')['LOW'].min() })
    return hi_lo_52w.loc[ticker]

# examples of how to use the function
hilo52w('2017','AA')
hilo52w('2017','A')


#############################################
# calculating the day number for each day in the data
first_day = datetime.date(2017,1,1)
Dates = mdata.DateTime.apply(lambda x: datetime.date(int(x[0:4]),int(x[5:7]),int(x[8:10])))
delta = Dates - first_day
mdata['Days'] = delta.apply(lambda x: x.days)
mdata.head()

#############################################
# Calclating gaps and gap percentages
"""
day_diff is calculated which is the difference between close and open for
each day. Then based on this day sign is identified: 1 for a positive day and 
0 for a negative day. day_diff and day_sign are added as two new columns to the
dataset. 
""" 
mdata['day_diff'] = mdata.Close-mdata.Open
mdata['day_sign'] = mdata['day_diff']
mdata['day_sign'][mdata['day_diff'] >= 0] = 1
mdata['day_sign'][mdata['day_diff'] < 0] = 0
mdata.head()

################
"""
G_minus is defined as the high of the next day minus the low of the previous day.
Whereever this is negative, there is a negative gap. G_Plus is defined as the
low of the next day minus the high of the prevous day. Whenever this is positive
there is a positive gap. These are added as new columns to the dataset.
"""
mdata['Tic_shift'] = mdata.Ticker.shift(-1)
mdata['Hi_shift'] = mdata.High.shift(-1)
mdata['Lo_shift'] = mdata.LOW.shift(-1)
mdata['G_minus'] = mdata.Hi_shift[mdata.Ticker == mdata.Tic_shift] - mdata.LOW[mdata.Ticker == mdata.Tic_shift]
mdata['G_plus'] = mdata.Lo_shift[mdata.Ticker == mdata.Tic_shift] - mdata.High[mdata.Ticker == mdata.Tic_shift]
mdata.drop(['Tic_shift','Hi_shift','Lo_shift'], axis=1, inplace=True)
mdata.head(10)


"""
Gap_minus_pct is defined as G_minus (negative gap) divided by the low of the 
previous day times 100. This provides the size of a negative gap in percerage.
Gap_plus_pct is defined as G_plus (positive gap) divided by the high of the 
previous day times 100. This provides the size of a positive gap in percerage.
These are added as new columns to the dataset. 
"""
mdata['Gap_minus_pct'] = (mdata.G_minus/(mdata.LOW+0.00001))*100
mdata['Gap_plus_pct'] = (mdata.G_plus/(mdata.High+0.00001))*100
mdata.head()

#############################################
# Calculating volume percetages based on the moving average

"""
A list called mov_avg is created that contains the moving average of the volume
that is going to be used for each day, using the 10 previous days. This is done
separately for each Ticker. A column named Volume_MA is added to the dataset
that contains these moving averages. Then, Volume_pct is calculated (and added
to the dataset) as the percentage of the volume for each day relative to the
moving average.
"""
marr = np.array(mdata)
mov_avg = []
for i in mdata.Ticker.unique():
    roll = list(pd.Series(marr[:,6][marr[:,0] == i]).rolling(window=10,min_periods=1).mean())
    mov_avg.extend(roll)

mdata['Volume_MA'] = mov_avg
mdata['Volume_pct'] = (mdata.Volume/(mdata.Volume_MA+0.00001))*100
mdata.head()


#############################################
# Applying thresholds to find interesting gaps
"""
This function applies criteria on positive gap percentage, Volume_pct, and price.
Input: 
gapplus_thr: threshold of the Gap_plus_pct which should be a positive percentage
volume_thr: threshold of the Volume_pct as a positive percetage
price_thr: threshold of the price as a positive amount
Output:
A dataframe which is the main dataset filtered by the thresholds
"""
def pos_gapdays(gapplus_thr, volume_thr, price_thr):
    posgap_days = mdata[['Ticker', 'DateTime', 'Open', 'High', 'LOW', \
    'Close', 'Volume', 'G_plus', 'Gap_plus_pct', \
    'Volume_pct', 'day_sign', 'Days']][(mdata.Gap_plus_pct > gapplus_thr) & \
    (mdata.Volume_pct > volume_thr) & (mdata.Close > price_thr)]
    return posgap_days

"""
This function applies criteria on negative gap percentage, Volume_pct, and price.
Input: 
gapminus_thr: threshold of the Gap_plus_pct which should be a negative percentage
volume_thr: threshold of the Volume_pct as a positive percetage
price_thr: threshold of the price as a positive amount
Output:
A dataframe which is the main dataset filtered by the thresholds
"""
def neg_gapdays(gapminus_thr, volume_thr, price_thr):
    neggap_days = mdata[['Ticker', 'DateTime', 'Open', 'High', 'LOW', \
    'Close', 'Volume', 'G_minus', 'Gap_minus_pct', \
    'Volume_pct', 'day_sign', 'Days']][(mdata.Gap_minus_pct < gapminus_thr) & \
    (mdata.Volume_pct > volume_thr) & (mdata.Close > price_thr)]
    return neggap_days

"""
Examples of using the above functions. The results are stored in new dataframes
called pos_gap_days and neg_gap_days
"""
pos_gap_days = pos_gapdays(3,500,5)
pos_gap_days.head()

neg_gap_days = neg_gapdays(-3,500,5)
neg_gap_days.head()


#############################################
# Inspecting days before and after a gap
"""
This function provides indeces for the days before and after a gap day.
Input:
df: The dataframe containing the gap days
nday: the number of days before and after the gap that we are interested in
Output:
index_rep: the gap day index repeated nday number of times
index_aft: the day index for nday days after the gap day
index_bef: the day index for nday days before the gap day
"""
def bef_aft_gap(df, ndays):
    index_rep = [x for x in df.index for i in range(ndays)]
    index_aft = [x+i for x in df.index for i in range(1,ndays+1)]
    index_bef = [x-i for x in df.index for i in range(1,ndays+1)]
    return (index_rep, index_aft, index_bef)

################
"""
Here we Store the gap day indeces and the indeces for n=3 days before and after. Then
using these indeces to create dataframes called after_neg_gap, and before_neg_gap 
that contain day_sign for the positive and negative gap days and the day_signs 
for n=3 days before and after the gap. 
"""
################
# Positive gaps
pos_gap_index_rep, pos_gap_index_aft, pos_gap_index_bef = bef_aft_gap(pos_gap_days, 3)

bef_aft_pos_gap = pd.DataFrame({'day_index': pos_gap_index_rep,\
'day_Ticker': list(mdata.Ticker.loc[pos_gap_index_rep]),\
'day_sign': list(mdata.day_sign.loc[pos_gap_index_rep]),\
'before_index': pos_gap_index_bef,                            
'before_signs': list(mdata.day_sign.loc[pos_gap_index_bef]),\
'before_Ticker': list(mdata.Ticker.loc[pos_gap_index_bef]),\
'after_index': pos_gap_index_aft,\
'after_signs': list(mdata.day_sign.loc[pos_gap_index_aft]),\
'after_Ticker': list(mdata.Ticker.loc[pos_gap_index_aft])})
    
"""
It may happen that a few indeces away from a gap day we arrive at a different
Ticker, we replace such cases with NaN.
"""
bef_aft_pos_gap[(bef_aft_pos_gap.before_Ticker != bef_aft_pos_gap.day_Ticker) \
| (bef_aft_pos_gap.after_Ticker != bef_aft_pos_gap.day_Ticker)] = np.nan
"""
However we will need the indeces to be not NaN for later calculations:
"""               
bef_aft_pos_gap.day_index, bef_aft_pos_gap.before_index, bef_aft_pos_gap.after_index\
= (pos_gap_index_rep, pos_gap_index_bef, pos_gap_index_aft)

bef_aft_pos_gap.isnull().any()

#bef_aft_pos_gap[after_pos_gap.isnull().any(axis=1)]
bef_aft_pos_gap.drop(['after_Ticker', 'before_Ticker'], axis=1, inplace=True)
bef_aft_pos_gap.head(30)


################
bef_aft_pos_gap.groupby('day_index')['after_signs'].sum()
bef_aft_pos_gap.groupby('day_index')['before_signs'].sum()



################
# anegative gaps
neg_gap_index_rep, neg_gap_index_aft, neg_gap_index_bef = bef_aft_gap(neg_gap_days, 3)

bef_aft_neg_gap = pd.DataFrame({'day_index': neg_gap_index_rep,\
'day_Ticker': list(mdata.Ticker.loc[neg_gap_index_rep]),\
'day_sign': list(mdata.day_sign.loc[neg_gap_index_rep]),\
'before_index': neg_gap_index_bef,                            
'before_signs': list(mdata.day_sign.loc[neg_gap_index_bef]),\
'before_Ticker': list(mdata.Ticker.loc[neg_gap_index_bef]),\
'after_index': neg_gap_index_aft,\
'after_signs': list(mdata.day_sign.loc[neg_gap_index_aft]),\
'after_Ticker': list(mdata.Ticker.loc[neg_gap_index_aft])})
    
"""
It may happen that a few indeces away from a gap day we arrive at a different
Ticker, we replace such cases with NaN.
"""
bef_aft_neg_gap[(bef_aft_neg_gap.before_Ticker != bef_aft_neg_gap.day_Ticker) \
| (bef_aft_neg_gap.after_Ticker != bef_aft_neg_gap.day_Ticker)] = np.nan
"""
However we will need the indeces to be not NaN for later calculations:
"""
bef_aft_neg_gap.day_index, bef_aft_neg_gap.before_index, bef_aft_neg_gap.after_index\
= (neg_gap_index_rep, neg_gap_index_bef, neg_gap_index_aft)

bef_aft_neg_gap.isnull().any()

bef_aft_neg_gap.drop(['after_Ticker', 'before_Ticker'], axis=1, inplace=True)
bef_aft_neg_gap.head(30)


################
bef_aft_neg_gap.groupby('day_index')['after_signs'].sum()
bef_aft_neg_gap.groupby('day_index')['before_signs'].sum()


################
"""
This fucntion provides the information about the nth day before or afer a gap day.
Input:
df: dataframe containg the information about days prior to and after the gap days
n_bef: a number indicating the nth day before the gap days
n_aft: a number indicating the mth day after the gap days
Output:
Two dataframes containing information about the nth (or mth) day before (or after) 
the gap days
"""
def nth_day_bef_aft(df, n_bef, m_aft):
    bef = pd.DataFrame({'nth_bef_sign':[list(df.before_signs[df.before_index == x-n_bef])[0] \
    for x in sorted(list(set(df.day_index)))],\
    'day_index': [x for x in sorted(list(set(df.day_index)))],\
    'day_sign': [list(df.day_sign[df.before_index == x-n_bef])[0] \
    for x in sorted(list(set(df.day_index)))]})
    aft = pd.DataFrame({'mth_aft_sign':[list(df.after_signs[df.after_index == x+m_aft])[0] \
    for x in sorted(list(set(df.day_index)))],\
    'day_index': [x for x in sorted(list(set(df.day_index)))],\
    'day_sign': [list(df.day_sign[df.after_index == x+m_aft])[0] \
    for x in sorted(list(set(df.day_index)))]})
    return (bef, aft)

# using the above function for positive gaps
nth_day_before_pos_gap, mth_day_after_pos_gap = nth_day_bef_aft(bef_aft_pos_gap, 1,1)

nth_day_before_pos_gap.head()
mth_day_after_pos_gap.head()

# using the above function for negative gaps
nth_day_before_neg_gap, mth_day_after_neg_gap = nth_day_bef_aft(bef_aft_neg_gap, 1,1)

nth_day_before_neg_gap.head()
mth_day_after_neg_gap.head()


#############################################
# Gap fill probability
pos_gap_days.head()
mdata.head()

"""
num_pos_gap_fills is defined as an empty list. Search is performed through all
days a after a postive gap to see if the gap is filled. In case it is
filled, 1 is added to the list, other 0 is added.
"""
num_pos_gap_fills = []
for g in  pos_gap_days.index:
    if len(mdata[mdata.Ticker == pos_gap_days.Ticker.loc[g]]\
    [(mdata.LOW <= pos_gap_days.High.loc[g]) & (mdata.Days > \
    pos_gap_days.Days.loc[g])]):
        num_pos_gap_fills.append(1)
    else:
        num_pos_gap_fills.append(0)

"""
Probability of gap fill is calculated as the numer of all the filled gaps 
divided by the total number of gaps.
"""
gap_pos_fill_prob = sum(num_pos_gap_fills)/len(num_pos_gap_fills); gap_pos_fill_prob


"""
Likewise for negative gaps.
"""
neg_gap_days.head()

num_neg_gap_fills = []
for g in  neg_gap_days.index:
    if len(mdata[mdata.Ticker == neg_gap_days.Ticker.loc[g]]\
    [(mdata.High >= neg_gap_days.LOW.loc[g]) & (mdata.Days > \
    neg_gap_days.Days.loc[g])]):
        num_neg_gap_fills.append(1)
    else:
        num_neg_gap_fills.append(0)

gap_neg_fill_prob = sum(num_neg_gap_fills)/len(num_neg_gap_fills); gap_neg_fill_prob


#############################################
# Slope of the treand after a gap day
################
# Positive gaps
def after_gap(df, ndays):
    index_repeat = [x for x in df.index for i in range(ndays)]
    index_after = [x+i for x in df.index for i in range(1,ndays+1)]
    return (index_repeat, index_after)


pos_index_repeat, pos_index_after = after_gap(pos_gap_days, 30)

after_posit_gap = pd.DataFrame({'day_index': pos_index_repeat,\
'day_Ticker': list(mdata.Ticker.loc[pos_index_repeat]),\
'after_days': list(mdata.Days.loc[pos_index_after]),\
'after_index': pos_index_after,\
'after_low': list(mdata.LOW.loc[pos_index_repeat]),\
'after_high': list(mdata.High.loc[pos_index_after]),\
'after_med': list((mdata.LOW.loc[pos_index_after]+mdata.High.loc[pos_index_after])/2),\
'after_Ticker': list(mdata.Ticker.loc[pos_index_after])})
    
after_posit_gap.head()
"""
It may happen that a few indeces away from a gap day we arrive at a different
Ticker, we replace such cases with NaN.
"""
after_posit_gap[(after_posit_gap.after_Ticker != after_posit_gap.day_Ticker)] = np.nan
"""
However we will need the indeces to be not NaN for later calculations:
"""               
after_posit_gap.day_index, after_posit_gap.after_index = (pos_index_repeat, pos_index_after)

after_posit_gap.isnull().any()

after_posit_gap.drop(['after_Ticker'], axis=1, inplace=True)
after_posit_gap.head()

################
# Example of 30-days-after behavior and the linear fit
# try 1438, 1492, 17766, 46891
aft = after_posit_gap[after_posit_gap.day_index == 1438].dropna(axis=0, how='any')
aft.head()
aft.shape[0]
aft['ones'] = np.ones(aft.shape[0])
linreg = linmod.LinearRegression()
linreg.fit(aft[['ones','after_days']], aft.after_med)

linreg.coef_
pred = linreg.predict(aft[['ones','after_days']])

# the plot
plt.figure(figsize=(20, 12.34))
plt.tick_params(labelsize=40, length=15, width = 2, pad = 10)
plt.xlabel('Days', fontsize=35, labelpad=20)
plt.ylabel('Day Average', fontsize=35, labelpad=20)
plt.scatter(aft.after_days, aft.after_med, s=100)
plt.plot(aft.after_days,pred)
plt.title('Ticker_name: ' + aft.day_Ticker.iloc[0], fontsize=40,  y=1.03)
################

# Generating the slopes for 30-days after all positive gaps
after_slope = []
num_aft_days = []
for i in pos_gap_days.index:
    aft = after_posit_gap[after_posit_gap.day_index == i].dropna(axis=0, how='any')
    aft['ones'] = np.ones(aft.shape[0])
    linreg = linmod.LinearRegression()
    linreg.fit(aft[['ones','after_days']], aft.after_med)
    #linreg.intercept_
    after_slope.append(linreg.coef_[1])
    num_aft_days.append(aft.shape[0])
    

# Adding slopes to pos_gap_days dataframe
pos_gap_days['after_slope'] = after_slope
pos_gap_days['num_aft_days'] = num_aft_days

pos_gap_days.head()


################
# Negative gaps
def after_gap(df, ndays):
    index_repeat = [x for x in df.index for i in range(ndays)]
    index_after = [x+i for x in df.index for i in range(1,ndays+1)]
    return (index_repeat, index_after)


neg_index_repeat, neg_index_after = after_gap(neg_gap_days, 30)

after_negat_gap = pd.DataFrame({'day_index': neg_index_repeat,\
'day_Ticker': list(mdata.Ticker.loc[neg_index_repeat]),\
'after_days': list(mdata.Days.loc[neg_index_after]),\
'after_index': neg_index_after,\
'after_low': list(mdata.LOW.loc[neg_index_repeat]),\
'after_high': list(mdata.High.loc[neg_index_after]),\
'after_med': list((mdata.LOW.loc[neg_index_after]+mdata.High.loc[neg_index_after])/2),\
'after_Ticker': list(mdata.Ticker.loc[neg_index_after])})
    
after_negat_gap.head()
"""
It may happen that a few indeces away from a gap day we arrive at a different
Ticker, we replace such cases with NaN.
"""
after_negat_gap[(after_negat_gap.after_Ticker != after_negat_gap.day_Ticker)] = np.nan
"""
However we will need the indeces to be not NaN for later calculations:
"""               
after_negat_gap.day_index, after_negat_gap.after_index = (neg_index_repeat, neg_index_after)

after_negat_gap.isnull().any()

after_negat_gap.drop(['after_Ticker'], axis=1, inplace=True)
after_negat_gap.head()

################

# Generating the slopes for 30-days after all negative gaps
after_slope = []
num_aft_days = []
for i in neg_gap_days.index:
    aft = after_negat_gap[after_negat_gap.day_index == i].dropna(axis=0, how='any')
    aft['ones'] = np.ones(aft.shape[0])
    linreg = linmod.LinearRegression()
    linreg.fit(aft[['ones','after_days']], aft.after_med)
    #linreg.intercept_
    after_slope.append(linreg.coef_[1])
    num_aft_days.append(aft.shape[0])
    

# Adding slopes to neg_gap_days dataframe
neg_gap_days['after_slope'] = after_slope
neg_gap_days['num_aft_days'] = num_aft_days

neg_gap_days.head()


#############################################
"""
This function can be used to output any dataframe as a csv.
Input:
df: the name of the dataframe to be stored as a csv
csv_name: a string to be used as the name of the csv
"""
def csv_out(df, csv_name):
    df.to_csv(dir_path + '/output/' + csv_name + '.csv')

# Example of the usage of the above functionS
csv_out(neg_gap_days, 'neg_gap')
