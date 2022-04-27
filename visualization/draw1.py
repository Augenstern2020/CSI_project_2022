import matplotlib.pyplot as plt
from wifilib import *
from clean import cleanCsi
from hampel import *
import os
import pandas as pd
import re
from scipy import signal
from sklearn.decomposition import PCA
from pywt import wavedec
import pywt
from numpy.fft import fft, ifft
from scipy.signal import savgol_filter
import scipy.stats


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 从文件名的二进制串读出座位信息
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


def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0


# 正确的小波变换滤波函数 原来的那个小波代码有严重的问题不要再用了
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
    usecoeffs.append(ca3)  # 向列表末尾添加对象

    #软硬阈值折中的方法
    a = 0.5
    #a = 0

    for k in range(length1):
        if (abs(cd1[k]) >= lamda):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0

    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)
    return recoeffs


# 读取一个路径下所有的.dat文件，默认路径设为..\\DATA\\xxxx
os.chdir(os.path.realpath(__file__) + "\\..")
train_path_list = os.listdir("DATA\\day2")
os.chdir("DATA\\day2")


# 因为.csv里面存的是纯数据没有表头 需要自己做一个表头
# 在从csv读取数据之后可以为dataframe增加这个表头 便于后面操作
head = []
head.append("time")  # 第一列是时间
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


line = []  # csv中的一行信息
linelist = []  # csv中的信息
window = 500  # 每个分片/滑动窗口的大小
step = 10  # 窗口滑动距离


fileNO = 0
# 在路径下的文件夹里一个文件一个文件的找
for file in train_path_list:
    # 111
    if fileNO == 111:
        # 通过文件名提取一部分信息
        file_extend_info = extract_from_filename(file)
        # print(file_extend_info)

        # 读csv内容
        data = pd.read_csv(file, names=head)

        # 矩阵初始化
        Amplitudes = np.zeros((len(data), 6, 30))
        Phases = np.zeros((len(data), 6, 30))

        # 把csv中的信息读取到矩阵中
        # length x 链路编号 x 子载波编号
        for linknumber in range(6):
            for subcarrier in range(30):
                temp = 'A' + str(linknumber + 1) + '_' + str(subcarrier + 1)
                temp1 = 'P' + str(linknumber + 1) + '_' + str(subcarrier + 1)
                for i in range(len(data)):
                    Amplitudes[i, linknumber, subcarrier] = data[temp][i]
                    Phases[i, linknumber, subcarrier] = data[temp1][i]

        csi_data = []  # 用来存当前文件读取出来的信息
        # 六条链路 一条链路一条链路的处理
        for link in range(1):
            Amplitudes[:, link, :]

            # """
            #plt.subplot(121);
            #plt.title('original');
            plt.plot(Amplitudes[0:1000, link, 0:5], color='red')  # 原始信号
            # """

            for i in range(30):
                csi_temp = Amplitudes[:, link,  i]
                res = hampel(csi_temp)
                Amplitudes[:, link,  i] = res

            # """
            #plt.subplot(122);
            #plt.title('after hampel')
            plt.plot(Amplitudes[0:1000, link, 0:5], color='black')
            plt.show()
            # """




            exit(0)




    fileNO+=1




