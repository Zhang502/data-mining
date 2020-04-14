#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.mlab as mlab 
import matplotlib.pyplot as plt 
from scipy.stats import norm
import numpy as np



names=['0', '1', '2', '3', '4','5','6','7','8','9','10']
dataset = pd.read_csv("magic04.data",header=0, names=names) #读取csv数据
# print(dataset)
# print(dataset.describe())
dataset
data=dataset[names[:-1]]
data


# In[2]:





# In[31]:





# In[30]:





# In[31]:


class CCovMat(object):
    '''计算多维度样本集的协方差矩阵
    '''
    def __init__(self, samples):
        #样本集shpae=(m,n)，m是样本总数，n是样本的特征个数
        self.samples = samples
        self.covmat1 = [] #保存方法1求得的协方差矩阵
        self.covmat2 = [] #保存方法2求得的协方差矩阵
        
        #用方法1计算协方差矩阵
        self._calc_covmat1()
        #用方法2计算协方差矩阵
        self._calc_covmat2()
        
    def _covariance(self, X, Y):
        '''
        计算两个等长向量的协方差convariance
        '''
        n = np.shape(X)[0]
        X, Y = np.array(X), np.array(Y)
        meanX, meanY = np.mean(X), np.mean(Y)
        #按照协方差公式计算协方差，Note:分母一定是n-1
        cov = sum(np.multiply(X-meanX, Y-meanY))/(n-1)
        return cov
        
    def _calc_covmat1(self):
        '''
        方法1：根据协方差公式和协方差矩阵的概念计算协方差矩阵
        '''
        S = self.samples #样本集
        na = np.shape(S)[1] #特征attr总数
        self.covmat1 = np.full((na, na), fill_value=0.) #保存协方差矩阵
        for i in range(na):
            for j in range(na):
                self.covmat1[i,j] = self._covariance(S[:,i], S[:,j])
        return self.covmat1
        
    def _calc_covmat2(self):
        '''
        方法2：先样本集中心化再求协方差矩阵
        '''
        S = self.samples #样本集
        ns = np.shape(S)[0] #样例总数
        mean = np.array([np.mean(attr) for attr in S.T]) #样本集的特征均值
        print('Q1：multivariate mean vector. :\n',mean)
        centrS = S - mean ##样本集的中心化
        print('样本集的中心化(每个元素将去当前维度特征的均值):\n', centrS)
        
        #求协方差矩阵
        self.covmat2 = np.dot(centrS.T, centrS)/(ns - 1)
        return self.covmat2
        
    def CovMat1(self):
        return self.covmat1
        
    def CovMat2(self):
        return self.covmat2
        
if __name__=='__main__':
    
    samples = np.array(data)
    cm = CCovMat(samples)
    
    print('样本集:\n', samples)
    print('Q2：Compute the sample covariance matrix as inner products between the columns of the centered data matrix.:\n', cm.CovMat1())
    print('Q3: Compute the sample covariance matrix as outer product between the centered data points. :\n', cm.CovMat1())


# In[34]:


#中心化后计算相关系数
import pandas as pd
S = data #样本集

mean = np.array([np.mean(S) ]) #样本集的特征均值
centrS = S - mean ##样本集的中心化
#print(data.corr()) # 任意两列的相关系数，相关系数矩阵 
#print(data.corr()[u'0']) #只显示“0”与其他属性的相关系数
print('Q4: \n',centrS[u'0'].corr(centrS[u'1'])) #计算“0”与“1”的相关系数


# In[14]:


#0和1绘制散点图

x='0'
y='1'

#绘制散点图
dataset.plot(x, y, kind='scatter')


# In[20]:


#0的概率密度函数

x = data.iloc[:,0] #提取第一列的sepal-length变量 
mu =np.mean(x) #计算均值 
sigma =np.std(x) 
mu,sigma

num_bins = 50 #直方图柱子的数量 
n, bins, patches = plt.hist(x, num_bins,normed=1, facecolor='blue', alpha=0.5) 
#直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象 


y=norm.pdf(bins,mu,sigma)

plt.plot(bins, y, 'r--') #绘制y的曲线 
plt.xlabel('0') #绘制x轴 
plt.ylabel('Probability') #绘制y轴 
plt.title(r'attribute 1 : $\mu=5.8433$,$\sigma=0.8253$')
plt.subplots_adjust(left=0.15)#左边距 
plt.show()


# In[22]:


#计算矩阵各列的方差。
import numpy as np
np.var(data, axis = 0) # 计算矩阵每一列的方差
#4方差最小，9方差最大


# In[35]:


#计算各列之间的协方差
result = []
for i in range(len(data.columns)):
    for j in range(i+1, len(data.columns)):
        result.append(data[data.columns[i]].cov(data[data.columns[j]]))

print('max = ', max(result))
print('min = ', min(result))


# In[ ]:




