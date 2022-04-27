"""
merge files
"""
import os
import pandas as pd
import re


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# simplified version of extract_from_filename()
def file_check_SN(filename: str):

    searobj = re.search("\d+th", filename)
    violation_num = 0
    count = 0
    start = False
    type_num = -1

    if searobj:
        type = searobj.group().replace("th", "")
        type_num = int(type)

    return type_num


# default path is ..\\DATA\\xxxx
os.chdir(os.path.realpath(__file__) + "\\..")
train_path_list = os.listdir("DATA\\20210724P")
os.chdir("DATA\\20210724P")

dflist = []

# file traversal
for file in train_path_list:
    sn = file_check_SN(file)
    data = pd.read_csv(file)
    data.insert(0, 'th', sn)
    dflist.append(data)

df_final = pd.concat(dflist)

df_final.info()

savepath = r'D:\Datasets\pre2\20210725Ppre2.csv'
df_final.to_csv(savepath, encoding='utf-8', index=False)