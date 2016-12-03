# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:00:28 2016

@author: XuLiu
"""
import numpy
from sklearn import preprocessing
import pandas


def loadDataSet(fileName):   #读取文件，txt文档
    numFeat = len(open(fileName).readline().split('\t')) #自动检测特征的数目
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):      
            lineArr.append(curLine[i])
        dataMat.append(lineArr)
    return dataMat

def is_num_by_except(num):#判断字符类型,数值型为True
    try:
        float(num)
        return True       
    except ValueError:
        return False

def seperate_num_str(data):#将连续型和离散型的数据分开
    dataMatrix=numpy.array(data)
    val=pandas.DataFrame(data,dtype=object,columns=range(numpy.shape(data)[1]))
    df=val.convert_objects(convert_numeric=True)
    a=df.count()
    dataconti=[]
    datastring=[]
    for i in range(numpy.shape(data)[1]):
        if a[i]==numpy.shape(data)[0]:
            if is_num_by_except(dataMatrix[0,i])==True:
                dataconti.append(dataMatrix[:,i])
            if is_num_by_except(dataMatrix[0,i])==False:
                datastring.append(dataMatrix[:,i])
        if a[i]!=numpy.shape(data)[0]:
            datastring.append(dataMatrix[:,i])
    datacontifloat=[]
    continum=numpy.shape(dataconti)[0]
    for i in range(continum):
        datacontifloat.append(list(map(float, dataconti[i])))
    return datacontifloat,datastring

def vector(datastring):
    #字符型转化为因子
    le = preprocessing.LabelEncoder()
    datastringarr = numpy.array(datastring)
    labeldatain=[]
    for i in range(numpy.shape(datastringarr)[0]):#离散型变量数
        le.fit(datastringarr[i])
        labeldatain.append(le.transform(datastringarr[i]))
    labeldataT=numpy.array(labeldatain).T
    return labeldataT 
    
    
def inspectvector(string,vector,n):
#infostring为原始的离散型变量取值，infovector为因子表，n为要验证的变量在infostring第几列（python计数）
    attr= list(set(string[n]))
    vect=[]
    for i in range(numpy.shape(attr)[0]):
        a=[]
        for m in range(2):
            a.append("a")
        for j in range(numpy.shape(string[n])[0]):
            if attr[i]==string[n][j]:
                a[0]=attr[i]
                a[1]=vector[j,n-1]
        vect.append(a)
    return vect
    
conn_sku=numpy.array(loadDataSet('/Users/XuLiu/Desktop/data/connect_apriori_V'))
sku_info=numpy.array(loadDataSet('/Users/XuLiu/Desktop/data/sku_info_v'))

#将连续型和字符型的数据分出来
infoconti,infostring=seperate_num_str(sku_info)

#字符型转化为因子
infovector=vector(infostring[1:18])

#查看因子代表的含义
vector=numpy.array(inspectvector(infostring,infovector,17))

f=open('/Users/XuLiu/Desktop/data/vector.txt','w')
for i in vector:
    k=' '.join([str(j) for j in i])
    f.write(k+"\t")
f.close()              



#将转化为数字的变量转化为哑变量
enc = preprocessing.OneHotEncoder()
enc.fit(infovector)
infodump=list((enc.transform(infovector).toarray()).T)


#连续型和处理好的离散型变量拼接
skudump=[]
skudump.append(list(sku_info[:,0]))

for i in range(numpy.shape(infodump)[0]):
    skudump.append(list(infodump[i]))

for i in range(numpy.shape(infoconti)[0]):
    skudump.append(list(infoconti[i]))
#最终用于匹配属性的表    
skuDump=(numpy.array(skudump)).T
#最终格式
formatfnl=[]
n=numpy.shape(skuDump)[1]-1
for l in range(numpy.shape(conn_sku)[0]):
    a=[]
    for m in range(n*2):
        a.append("a")
    for k in range(numpy.shape(skuDump)[0]):
        if conn_sku[l,0]==skuDump[k,0]:
            a[0:n]=skuDump[k,1:n+1]
        if conn_sku[l,1]==skuDump[k,0]:
            a[n:n*2]=skuDump[k,1:n+1]
    formatfnl.append(a)
#导出
Formatfnl=numpy.array(formatfnl)
f=open('/Users/XuLiu/Desktop/data/Conn_Bayes_v.txt','w')
for i in Formatfnl:
    k=' '.join([str(j) for j in i])
    f.write(k+"\t")
f.close()