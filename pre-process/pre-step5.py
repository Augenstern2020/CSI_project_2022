"""
modify the header into a more readable one
"""

import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# default path ..\\DATA\\xxxx
os.chdir(os.path.realpath(__file__) + "\\..")
train_path_list = os.listdir("DATA\\pre2")
os.chdir("DATA\\pre2")


data = pd.read_csv(r'20210725Ppre2.csv')

colname = 0
for j in range(6):
    for k in range(30):
        head = "L" + str(j) + "S" + str(k) + "_"

        # mean
        m = head + "mean"
        data = data.rename(columns={str(colname): m})
        colname += 1

        # variance
        v = head + "variance"
        data = data.rename(columns={str(colname): v})
        colname += 1

        # standard deviation
        sd = head + "standard_deviation"
        data = data.rename(columns={str(colname): sd})
        colname += 1

        # skewness
        s = head + "skewness"
        data = data.rename(columns={str(colname): s})
        colname += 1

        # kurtosis
        ku = head + "kurtosis"
        data = data.rename(columns={str(colname): ku})
        colname += 1

        # Inter - quartileRange
        iqr = head + "inter_quartile_range"
        data = data.rename(columns={str(colname): iqr})
        colname += 1

        # entropy
        e = head + "entropy"
        data = data.rename(columns={str(colname): e})
        colname += 1


data.info()

savepath = r'D:\Datasets\pre3blah\20210725Ppre3.csv'
# savepath = r'D:\Datasets\pre3\20210708pre3.csv'
data.to_csv(savepath, encoding='utf-8', index=False)