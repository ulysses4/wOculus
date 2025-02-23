import platform
import sys

if platform.system() == 'Linux':
    sys.path.append("/home/ulysses/GitHub/Functions")
    save_path = "/home/ulysses/GitHub/ZC/"
    save_fixings = "/home/ulysses/GitHub/ZC/ZC_fixings_daily/"
    save_spots = "/home/ulysses/GitHub/ZC/ZC_spots/"
    save_fixings_cum = "/home/ulysses/GitHub/ZC/ZC_fixings_cumulative/"
    BBG_path = "/home/ulysses/GitHub/BBG/"
elif platform.system() == 'Windows':
    sys.path.append("C:\\Users\\igorc_\\OneDrive\\Documents\\US_Inflation\\GitHub\\Functions")
    save_path = "C:\\Users\\igorc_\\OneDrive\\Documents\\US_Inflation\\Github\\ZC\\"
    save_fixings = "C:\\Users\\igorc_\\OneDrive\\Documents\\US_Inflation\\Github\\ZC\\ZC_fixings_daily\\"
    save_spots = "C:\\Users\\igorc_\\OneDrive\\Documents\\US_Inflation\\Github\\ZC\\ZC_spots\\"
    save_fixings_cum = "C:\\Users\\igorc_\\OneDrive\\Documents\\US_Inflation\\Github\\ZC\\ZC_fixings_cumulative\\"
    BBG_path = "C:\\Users\\igorc_\\OneDrive\\Documents\\US_Inflation\\Github\\BBG\\"

import functions_TIPS
import get_calendar

import pandas as pd
from datetime import date
from datetime import datetime
import numpy as np
from os import listdir
from os.path import isfile, join
import re

df_ZC = pd.read_csv(BBG_path + "df_ZC_test2.csv", index_col=0, parse_dates=True)
df_ZC.index = df_ZC.index.date

df_seas = pd.read_csv(save_path + "BLS_seasonals.csv")
df_seas = df_seas.set_index("Month")
df_seas.index.name = None

df_sf = pd.read_csv(save_path + "sFactors.csv", index_col=0)
df_sf = df_sf.T
df_sf = pd.concat([df_sf.iloc[-1].to_frame().T, df_sf])
df_sf = np.log(df_sf).diff().dropna()*100
df_sf = df_sf.reset_index(drop=True)
df_sf.index = df_sf.index + 1

df_seas = df_sf

#generate the starting index value
#calculate the 1y1y value
#calculate the two fixings using that 1y1y value as the growth rate
# ... between the two fixings + seasonality

# from there, use seasonality + growth rate to hit 2y mark

def get_1y1y(trade_dt, df_ZC):
    """returns the 1y1y value for a given date"""
    rate_01 = df_ZC.loc[trade_dt].at["USSWIT1"]
    rate_02 = df_ZC.loc[trade_dt].at["USSWIT2"]

    rate_1y1y = ((pow(1+rate_02/100.0, 2) / (1+rate_01/100.0)) - 1) * 100

    return rate_1y1y

def get_CPI_term(settle_CPI, rate, term):
    """returns the CPI term"""

    return settle_CPI*pow(1+rate/100.0, term)

def generate_fixings_subcurve_short(trade_dt, df_seas, date_list,\
        ending_CPI, rate):
    """walks back the curve from the 1y point to fill shorter fixings"""

    fixing_subcurve = []
    shortest_CPI = ending_CPI

    for date in reversed(date_list):
        date = (date + pd.DateOffset(months=1)).date()
        seas = df_seas.loc[date.month,\
                min(int(df_seas.columns[-1]),trade_dt.year-1)]
        prior_CPI = shortest_CPI / np.exp(seas/100.0 + rate/12)
        #print shortest_CPI, rate, date, prior_CPI
        fixing_subcurve.insert(0,round(prior_CPI,5))
        shortest_CPI = prior_CPI

    return fixing_subcurve


def generate_fixings_subcurve(trade_dt, df_seas, date_list, \
        starting_CPI, CPIII_term, rate_guess):
    """fills in the CPI fixings to corresponding dates of a curve section"""

    #print "starting CPI: " + str(starting_CPI)
    #print "Original rate guess: " + str(rate_guess)

    while True:
        fixing_subcurve = []
        prior_CPI = starting_CPI

        for date in date_list:
            seas = df_seas.loc[date.month,\
                    min(int(df_seas.columns[-1]),trade_dt.year-1)]
            next_CPI = prior_CPI * np.exp(seas/100.0 + rate_guess/100.0/12)
            fixing_subcurve.append(round(next_CPI,5))
            prior_CPI = next_CPI

        #calc term

        CPI_1 = fixing_subcurve[-2]
        CPI_2 = fixing_subcurve[-1]

        settle_dt = get_calendar.add_bdays(trade_dt, 2)
        day_1 = settle_dt.day
        day_2 = (settle_dt + pd.offsets.MonthEnd(0)).day

        CPIII_guess = round(CPI_1 + (day_1 - 1.0) / day_2 * (CPI_2 - CPI_1), 5)

        curve_sec_yrs = len(date_list)/12

        error_adj = np.log(CPIII_term/CPIII_guess)/curve_sec_yrs
        rate_guess = rate_guess+error_adj*100.0
        """
        print "CPIII_term:" + str(CPIII_term) +\
                " CPIII_guess:" + str(CPIII_guess) +\
                " new rt_guess:" + str(round(rate_guess,5))
                #"CPIs :" + str(CPI_1) + " " + str(CPI_2)
        """
        if abs(CPIII_guess - CPIII_term) < 0.001:
            break

    return fixing_subcurve

def expand_ZC_tenors(trade_dt, df, knot_pts, spot_seeds):
    """returns ZC curve with all the tenors filled out so fwds are smooth"""

    df.rename(columns=lambda x: int(re.search(r'\d+', x).group()), inplace=True)
    df = df[df.index == trade_dt]
    df = df[df.columns.intersection(knot_pts)].transpose()
    df.rename(columns={trade_dt: "knot_pts"}, inplace=True)

    df_temp = pd.DataFrame(spot_seeds,\
                index=[a for a in range(1,31)], columns=["flex_pts"])

    df2 = pd.merge(df, df_temp, left_index=True, right_index=True, how="outer")
    df2["spot"] = df2["knot_pts"].fillna(df2["flex_pts"])

    def fwd_rt(spot_1, spot_2, year_1, year_2):
        return (((1 + spot_2/100) ** year_2 / (1 + spot_1/100) ** year_1) ** \
                (1 / (year_2 - year_1)) - 1) * 100

    #calc 'fwd' column
    for i in range(1, len(df2)):
        year_1 = df2.index[i - 1]
        year_2 = df2.index[i]
        spot_1 = df2.loc[year_1, "spot"]
        spot_2 = df2.loc[year_2, "spot"]
        df2.loc[year_2, "fwd"] = fwd_rt(spot_1, spot_2, year_1, year_2)

    #calc distance from prior / to next knot points
    for i in range(1, len(df2)):
        df2.loc[i, "prior_knot"] = np.nan
        if pd.isna(df2.loc[i, "knot_pts"]):
            if pd.isna(df2.loc[i - 1, "prior_knot"]):
                df2.loc[i, "prior_knot"] = 1
            else:
                df2.loc[i, "prior_knot"] = df2.loc[i - 1, "prior_knot"] + 1
            count = 0
            for j in range(i + 1, len(df2) + 1):
                count += 1
                if not pd.isna(df2.loc[j, "knot_pts"]):
                    df2.loc[i, "next_knot"] = count
                    break

    #calc 'fwd_t' column, which is the target forward
    for i in range(1, len(df2)):
        if not pd.isna(df2.loc[i, "next_knot"]):
            df2.loc[i, "fwd_t"] = (df2.loc[i + df2.loc[i, "next_knot"], "fwd"] - \
            df2.loc[i - df2.loc[i, "prior_knot"], "fwd"]) / \
            (df2.loc[i, "next_knot"] + df2.loc[i, "prior_knot"]) * \
            df2.loc[i, "prior_knot"] + df2.loc[i - df2.loc[i, "prior_knot"], "fwd"]

    #calc 'spot_t' column, which is the target spot
    for i in range(1, len(df2)):
        if not pd.isna(df2.loc[i, "fwd_t"]):
            df2.loc[i, "spot_t"] = ((((1 + df2.loc[i - 1, "spot"] / 100) ** (i - 1)) * \
            (1 + df2.loc[i, "fwd_t"] / 100)) ** (1 / i) - 1) * 100

    #calc the cummulative sum of the error
    df2["error"] = abs(df2["spot"] - df2["spot_t"])

    #cacl 'adj' column, which is the adjustments to spot needed
    for i in range(1, len(df2)):
        if not pd.isna(df2.loc[i, "spot_t"]):
            df2.loc[i, "spot"] = df2.loc[i, "spot_t"]

    return df2["spot"].tolist(), df2["error"].sum(), df2


def iterate_smooth_fwds(trade_dt, df, knot_pts):
    """returns spot rates for a smooth fwds curve based on knot points"""
    spot_seeds = [2 for a in range(1, 31)]

    error = 10
    while error > .0005:
        #print(f"ZC curve iteration error at {error}")
        spot_seeds, error, a = expand_ZC_tenors(trade_dt, df.copy(), knot_pts, spot_seeds)

    return spot_seeds

def generate_fixing_curve(trade_dt, df_ZC, df_seas, knot_pts):
    """generates the fixing curve from a ZC curve"""

    final_dates = []
    final_fixings = []

    #################################
    #STEP 1: solve for the 1y fixings
    #################################
    rate_1y1y = np.log(1.0 + get_1y1y(trade_dt, df_ZC)/100.0)

    settle_dt = get_calendar.add_bdays(trade_dt, 2)
    settle_CPI = functions_TIPS.TIPS_CPIII("df_CPURNSA.csv", settle_dt)

    #get 1y fixings
    rate_01 = df_ZC.loc[trade_dt].at["USSWIT1"]
    CPIII_term = round(get_CPI_term(settle_CPI, rate_01, 1), 5) 

    month_1 = (settle_dt + pd.offsets.MonthEnd(-4+13) +\
            pd.offsets.Day(1)).date()
    month_2 = (settle_dt + pd.offsets.MonthEnd(-3+13) +\
            pd.offsets.Day(1)).date()

    day_1 = settle_dt.day
    day_2 = (settle_dt + pd.offsets.MonthEnd(0)).day

    seas = df_seas.loc[month_2.month,\
                    min(int(df_seas.columns[-1]),trade_dt.year-1)]

    #seas = round(df_seas.loc[df_seas["Month"] == month_2.month,\
    #        "seas"].iloc[0], 5)

    CPI_1 = CPIII_term / (1+(day_1-1.0)/day_2 * (np.exp(seas/100.0 + \
            rate_1y1y/12.0) - 1.0))
    CPI_2 = CPI_1 * np.exp(seas/100.0 + rate_1y1y/12.0)

    '''
    return settle_CPI, CPIII_term, month_1, month_2, day_1, day_2, seas,\
            rate_1y1y, CPI_1, CPI_2
    '''
    #append the 1y dates / fixings to the final lists
    #final_dates = [month_1] + [month_2]
    #final_fixings = [CPI_1] + [CPI_2]

    ################################################################
    #STEP 2: bootstrap the fixings for the shorter part of the curve
    ################################################################
    """NOTE: the key assumption here is the shorter fixings are
             also growing at the 1y1y rate, which is the best we can
             do b/c we need to have some fixings show up for pricing
             TIPS on ASW.
    """

    short_CPI_dates = []

    short_CPI_start = (settle_dt + pd.offsets.MonthEnd(-2) +\
            pd.offsets.Day(1)).date()

    while short_CPI_start < month_1:
        short_CPI_dates.append(short_CPI_start)
        short_CPI_start = (short_CPI_start + pd.DateOffset(months=1)).date()

    fixing_subcurve_short = generate_fixings_subcurve_short(\
            trade_dt, df_seas, short_CPI_dates, CPI_1, rate_1y1y)

    final_dates = short_CPI_dates + [month_1] + [month_2]
    final_fixings = fixing_subcurve_short + [round(CPI_1,5)] + [round(CPI_2,5)]

    ###############################################################
    #STEP 3: bootstrap the fixings for the longer part of the curve
    ###############################################################

    ZC_tenor_list = list(range(2,31)) #will cycle thru every year from 2-30
    prior_term = 1
    prior_last_date = final_dates[-1]
    prior_last_CPI = final_fixings[-1]

    #get a smooth fwd curve
    #knot_pts = [1,2,3,4,5,6,7,8,9,10,12,15,20,25,30] #standard matching BGC
    smooth_fwd_spots = iterate_smooth_fwds(trade_dt, df_ZC, knot_pts)

    for tenor in ZC_tenor_list:
        #print "running: " + str(tenor)
        #rate = df_ZC.loc[trade_dt].at["USSWIT"+str(tenor)]
        rate = smooth_fwd_spots[tenor-1]
        CPIII_term = round(get_CPI_term(settle_CPI, rate, tenor), 5)
        curve_section_years = tenor - prior_term
        new_last_date = (prior_last_date + pd.DateOffset(months=12*\
                curve_section_years)).date()
        curve_section_dates = []
        while prior_last_date < new_last_date:
            new_date = (prior_last_date + pd.DateOffset(months=1)).date()
            curve_section_dates.append(new_date)
            prior_last_date = new_date

        #print curve_section_dates

        fixing_subcurve = generate_fixings_subcurve(trade_dt, df_seas,\
            curve_section_dates, prior_last_CPI, CPIII_term, rate)

        #print fixing_subcurve

        final_dates = final_dates + curve_section_dates
        final_fixings = final_fixings + fixing_subcurve

        prior_term = tenor
        prior_last_date = new_last_date
        prior_last_CPI = fixing_subcurve[-1]

    df = pd.DataFrame({"date": final_dates, trade_dt: final_fixings})
    df = df.set_index("date")
    df.index.name = None

    return df, pd.DataFrame({trade_dt: smooth_fwd_spots}, index=range(1,31))

if __name__ == "__main__":

    file_list = [f[-12:-4] for f in listdir(save_fixings)]
    file_list2 = [f for f in listdir(save_fixings_cum)]
    file_list3 = [f for f in listdir(save_spots)]

    if not file_list2:
        df_cum = pd.DataFrame()
    else:
        df_cum = pd.read_csv(save_fixings_cum + "CPI_fixings_cum.csv", index_col=0, parse_dates=True)
        df_cum.index = df_cum.index.date
        df_cum.rename(columns=lambda x: datetime.strptime(x, "%Y-%m-%d").date(), inplace=True)

    if not file_list3:
        df_spots = pd.DataFrame()
    else:
        df_spots = pd.read_csv(save_spots + "CPI_spots.csv", index_col=0, parse_dates=True)
        df_spots.index = df_spots.index.date
        df_spots.rename(columns=lambda x: int(x), inplace=True)

    knot_pts = [1,2,3,4,5,6,7,8,9,10,12,15,20,25,30] #standard matching BGC

    for run_date in df_ZC.index:
        if run_date.strftime("%Y%m%d") in file_list:
            print("Skipping ZC curve for: " + run_date.strftime("%Y%m%d"))
        else:
            try:
                print("Building ZC curve for: " + run_date.strftime("%Y%m%d"))
                df, df_spots_new = generate_fixing_curve(run_date, df_ZC, df_seas, knot_pts)
                df.to_csv(save_fixings + "CPI_fixings_"+ run_date.strftime("%Y%m%d")+".csv")
                df_cum = pd.concat([df_cum, df.transpose()], axis=0)
                df_spots = pd.concat([df_spots, df_spots_new.round(5).transpose()], axis=0)
            except:
                print("FAILED ZC curve for: " + run_date.strftime("%Y%m%d"))

    df_cum.sort_index().to_csv(save_fixings_cum + "CPI_fixings_cum.csv")
    df_spots.sort_index().to_csv(save_spots + "CPI_spots.csv")
