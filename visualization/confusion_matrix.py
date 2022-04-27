from sklearn.metrics import confusion_matrix  # 生成混淆矩阵的函数
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
'''
首先是从结果文件中读取预测标签与真实标签，然后将读取的标签信息传入python内置的混淆矩阵矩阵函数confusion_matrix(真实标签,
预测标签)中计算得到混淆矩阵，之后调用自己实现的混淆矩阵可视化函数plot_confusion_matrix()即可实现可视化。
三个参数分别是混淆矩阵归一化值，总的类别标签集合，可是化图的标题
'''

def plot_confusion_matrix(cm, labels_name, title):
    np.set_printoptions(precision=2)
    # print(cm)
    plt.rc('font', family='Times New Roman')
    #colors = ['blue']

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # 在特定的窗口上显示图像
    #plt.plot(cm)
    plt.title(title,fontsize=14)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))

    font={'style':'normal','weight':'black'}
    thresh = cm.max() / 2.
    for i in range(len(cm)):
        for j in range(len(cm)):
            #plt.annotate(cm[i, j], xy=(i, j), horizontalalignment='center', verticalalignment='center')
            plt.text(j, i, cm[i][j],horizontalalignment="center", verticalalignment='center',fontsize=12,fontdict=font,
                     color="white" if cm[i, j] >= thresh else "black")

    plt.xticks(num_local, labels_name, rotation=45, fontsize=12)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, fontsize=12)  # 将标签印在y轴坐标上
    plt.ylabel('True',fontsize=12)
    plt.xlabel('Predicted',fontsize=12)
    # show confusion matrix
    #plt.savefig('./fig/' + title + '.png', format='png')
    plt.show()
#gt = []
#pre = []
#cm = confusion_matrix(gt, pre)  # 计算混淆矩阵
cm=np.array( [[1200   ,29  , 10 ,  21   , 0  ,  0 ,   0 ,   0   , 0  ,  0],
 [ 492  ,736 ,   0   , 0  ,  0   ,11   , 0   , 0  ,  0 ,   0],
 [  72  , 10 ,1139  ,  4   ,26  ,  9   , 0   , 0  ,  0  ,  0],
 [  93  , 60 ,  67 ,1036   , 4  ,  0  ,  0  ,  0   , 0   , 0],
 [  24  ,  3 , 341   , 0 , 767  ,125  ,  0    ,0  ,  0 ,   0],
 [  45  , 37  , 43 ,  19 ,  77, 1039   , 0  ,  0  ,  0,    0],
 [  36  , 42 ,   0 ,  16 ,   0    ,0, 1159  ,  0  ,  7  ,  0],
 [   0  , 19 ,   0  ,  0 ,   0  ,  0 ,   0 ,1239  ,  0 ,   2],
 [   0  ,  0 ,   0 
