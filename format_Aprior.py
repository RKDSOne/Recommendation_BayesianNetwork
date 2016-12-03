# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 17:43:39 2016

@author: XuLiu
"""

import numpy 
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
    
#读文件
dataMat=loadDataSet('/Users/XuLiu/Desktop/data/Sale_validation')
dataMat.sort(key=lambda l:(l[0],l[1]))

dataraw=numpy.array(dataMat)
dataNodup=dataraw

#删除用户名相等且购买产品编号相等的重复项
for i in range(numpy.shape(dataNodup)[0]-1,0,-1):
    if dataNodup[i-1,0]==dataNodup[i,0] and dataNodup[i-1,1]==dataNodup[i,1]:
        dataNodup = numpy.delete(dataNodup, (i), axis=0)

#计数
la=[]#用户表
for i in range(numpy.shape(dataNodup)[0]):
    la.append(dataNodup[i,0])

count=[]#用户出现次数的计数，即购买商品数
for i in range(numpy.shape(la)[0]):
    count.append(la.count(la[i]))


#删除仅出现一次的人
dataRe=dataNodup
for i in range(numpy.shape(dataNodup)[0]-1,-1,-1):
    if count[i]==1:
        dataRe=numpy.delete(dataRe,(i),axis=0)
        count=numpy.delete(count,(i),axis=0)

#计算rank 
rank=[1]
a=1
for n in range(1,numpy.shape(dataRe)[0]):
    if dataRe[n-1,0]==dataRe[n,0] :
        a+=1
    else:
        a=1
    rank.append(a)
    
#更改格式     
maxrank=max(rank)
DataFormat= list(set(dataRe[:,0]))#用户ID的unique
DataFormat.sort(key=lambda l:(l[0]))
    
formatfnl=[]   
    
for l in range(numpy.shape(DataFormat)[0]):
    a=[]
    for m in range(maxrank):
        a.append("a")
    for k in range(numpy.shape(dataRe)[0]):
        if DataFormat[l]==dataRe[k,0]:
            a[rank[k]-1]=dataRe[k,1]
    formatfnl.append(a)
              
Formatfnl=numpy.array(formatfnl)      


f=open('/Users/XuLiu/Desktop/data/Sales_Apriori_V.txt','w')
for i in Formatfnl:
    k=' '.join([str(j) for j in i])
    f.write(k+"\n")
f.close()











    
    