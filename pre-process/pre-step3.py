"""
main pipeline of pre-process
"""
from hampel import *
import os
import pandas as pd
import re
import pywt
from numpy.fft import fft
import scipy.stats


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# extract useful info from the seat pattern in file name
def extract_from_filename(filename: str):

    searobj = re.search("_\d+_", filename)
    violation_num = 0
    count = 0
    start = False
    type_num = 0

    if searobj:
        type = searobj.group().replace("_", "")
        type_num = int(type, 2)
        # print(type_num)

        setBits = [bit for bit in type if bit == '1']
        for bit in type:
            if bit == '1' and start == False:
                count = 1
                start = True
            elif bit == '1' and start == True:
                count += 1
            else:
                start = False
                if count > 1:
                    violation_num += count
                count = 0
        if count > 1:
            violation_num += count

    else:
        setBits = []

    return (type_num, len(setBits), violation_num)


# sgn is used in wavelet_noising
def sgn(num):
    if num > 0.0:
        return 1.0
    elif num == 0.0:
        return 0.0
    else:
        return -1.0


# attention attention attention
# this is the correct version of wavelet transform
# the old one has critical flaws thus it worked in a ridiculous way
# please delete the old one asap
def wavelet_noising(new_list):
    data = new_list
    w = pywt.Wavelet('sym4')
    [ca3, cd3, cd2, cd1] = pywt.wavedec(data, w, level=3)  # 分解波

    length1 = len(cd1)
    length0 = len(data)

    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0 ), math.e))
    usecoeffs = []
    usecoeffs.append(ca3)  # append objects

    # our threshold method
    # one of the most crucial parts of our methodology
    a = 0.5
    # a = 0

    for k in range(length1):
        if abs(cd1[k]) >= lamda:
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if abs(cd2[k]) >= lamda:
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if abs(cd3[k]) >= lamda:
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0

    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)
    return recoeffs


# read all files under a certain path
os.chdir(os.path.realpath(__file__) + "\\..")
train_path_list = os.listdir("DATA\\day2")
os.chdir("DATA\\day2")


# since .csv has only raw data, we need to make a header for it
# an easy solution is to add headers to dataframe after get the data from .csv
head = []
head.append("time")  # the first column is time
# A - amplitude
for linknumber in range(6):
    for subcarrier in range(30):
        temp = 'A' + str(linknumber + 1) + '_' + str(subcarrier + 1)
        head.append(temp)
# P - phase
for linknumber in range(6):
    for subcarrier in range(30):
        temp = 'P' + str(linknumber + 1) + '_' + str(subcarrier + 1)
        head.append(temp)

# time to do some basic data augmentation

line = []  # a row in csv
linelist = []  # the info in csv
window = 500  # window size
step = 10  # step length


fileNO = 0
# search each file under a certain path
for file in train_path_list:
    # if fileNO == 0: # to stop the program
        # use the file name to extract some informatino
        file_extend_info = extract_from_filename(file)
        # print(file_extend_info)

        # read csv
        data = pd.read_csv(file, names=head)

        # init matrix
        Amplitudes = np.zeros((len(data), 6, 30))
        Phases = np.zeros((len(data), 6, 30))

        # import info from csv to matrix
        # length x link number x subcarrier number
        for linknumber in range(6):
            for subcarrier in range(30):
                temp = 'A' + str(linknumber + 1) + '_' + str(subcarrier + 1)
                temp1 = 'P' + str(linknumber + 1) + '_' + str(subcarrier + 1)
                for i in range(len(data)):
                    Amplitudes[i, linknumber, subcarrier] = data[temp][i]
                    Phases[i, linknumber, subcarrier] = data[temp1][i]

        # csi_data = []  # not useful anymore
        # process the links one by one (all 6)
        for link in range(6):
            Amplitudes[:, link, :]

            """
            plt.subplot(311);
            plt.title('original');
            plt.plot(Amplitudes[:, link, :])  # raw sig
            # """

            for i in range(30):
                csi_temp = Amplitudes[:, link,  i]
                res = hampel(csi_temp)                  # 去除异常值
                Amplitudes[:, link,  i] = res

            """
            plt.subplot(312);
            plt.title('after hampel')
            plt.plot(Amplitudes[:, link, :])
            # """

            for j in range(30):
                tempwave = []
                for i in range(len(Amplitudes)):
                    tempwave.append(Amplitudes[i, link, j])
                tempwave = wavelet_noising(tempwave)        # 小波阈值去噪
                for i in range(len(Amplitudes)):
                    Amplitudes[i, link, j] = tempwave[i]

            """
            plt.subplot(313);
            plt.title('after DWT')
            plt.plot(Amplitudes[:, link, :])
            plt.show()
            # """



        # print(csi_data)
        # now all 6 links are completely processed

        linelist = []
        for i in range(200, 901 - window, step):

            line = []
            for j in range(6):
                for k in range(30):

                    tempcsi = Amplitudes[i:i+window, j, k]

                    print()
                    print()
                    print(file)
                    print(str(i) + " - " + str(i + window))
                    print("link " + str(j))
                    print("subcarrier " + str(k))

                    mean1 = np.mean(tempcsi)  # mean
                    line.append(mean1)
                    print(mean1)

                    var1 = np.var(tempcsi)  # variance
                    line.append(var1)

                    std1 = np.std(tempcsi)  # standard deviation
                    line.append(std1)

                    s = pd.Series(tempcsi)

                    skew1 = s.skew()  # skewness
                    line.append(skew1)

                    kurt1 = s.kurt()  # kurtosis
                    line.append(kurt1)

                    lower_q = np.quantile(tempcsi, 0.25, interpolation='lower')  # lower q
                    higher_q = np.quantile(tempcsi, 0.75, interpolation='higher')  # higher q
                    int_r = higher_q - lower_q  # int r
                    line.append(int_r)

                    # fft
                    X = fft(tempcsi)
                    # plt.plot(X)
                    # plt.show()
                    N = len(X)
                    # print("N", N)
                    X = X / N
                    n = np.arange(N)
                    # get the sampling rate
                    sr = 1 / 0.01
                    T = N / sr
                    freq = n / T
                    # Get the one-sided spectrum
                    n_oneside = N // 2
                    # get the one side frequency
                    f_oneside = freq[:n_oneside]

                    # x - f_oneside,
                    # y - np.abs(X[:n_oneside])

                    ent = scipy.stats.entropy(np.abs(X[:n_oneside]))
                    # print(ent)
                    line.append(ent)

            linelist.append(line)

        tempdf = pd.DataFrame(linelist)
        tempdf.insert(0, 'people_num', file_extend_info[1])
        tempdf.insert(0, 'bin_type', file_extend_info[0])
        if file_extend_info[2] > 0:
            is_violating = 1
        else:
            is_violating = 0
        tempdf.insert(0, 'is_violating', is_violating)
        tempdf.insert(0, 'violation_num', file_extend_info[2])
        tempdf.info()
        #savepath = r'D:\Datasets\BlahBlah\pre1_' + file
        #savepath = r'D:\Datasets\20210724bbbP\pre1_' + file
        #tempdf.to_csv(savepath, encoding='utf-8', index=False)
    # fileNO+=1

    # 保存到 DATA\\20210724P



