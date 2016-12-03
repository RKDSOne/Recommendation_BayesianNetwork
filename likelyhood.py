# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:54:39 2016

@author: LIUXU410
"""

import numpy 
from sklearn import preprocessing
from scipy import stats
import pandas
import math

numpy.random.seed(1994)
def loadDataSet(fileName):   #读取文件，txt文档
    numFeat = len(open(fileName).readline().split('\t')) #自动检测特征的数目
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):      
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
    return dataMat
    
def loadStringSet(fileName):   #读取文件，txt文档
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

    
    
def likelyhood_alltoall(data):
    likelyhood=pandas.DataFrame(columns=['likely','x','y'])
#计算全部的相似度
    for j in range(numpy.shape(data)[0]-1):
        for k in range(1,numpy.shape(data)[0]-j):
            numerator=0
            for i in range(numpy.shape(data)[1]):
                numerator=numerator+data[j,i]*data[j+k,i]
            addx=0
            for i in range(numpy.shape(data)[1]):
                addx=addx+data[j,i]*data[j,i]
            addy=0
            for i in range(numpy.shape(data)[1]):
                addy=addy+data[j+k,i]*data[j+k,i]
            denominator=math.sqrt(addx)*math.sqrt(addy)
            likely=round(numerator/denominator,3)
            row = pandas.DataFrame([dict(likely=likely, x=j, y=j+k), ])
            likelyhood = likelyhood.append(row, ignore_index=True)
    return likelyhood

 
def likelyhood_onetoone(data1,data2):
    likelyhood=pandas.DataFrame(columns=['likely','row'])
    for j in range(numpy.shape(data1)[0]):
        numerator=0
        for i in range(numpy.shape(data1)[1]):
            numerator=numerator+data1[j,i]*data2[j,i]
        addx=0
        for i in range(numpy.shape(data1)[1]):
            addx=addx+data1[j,i]*data1[j,i]
        addy=0
        for i in range(numpy.shape(data2)[1]):
            addy=addy+data2[j,i]*data2[j,i]    
        denominator=math.sqrt(addx)*math.sqrt(addy)
        likely=round(numerator/denominator,3)
        onerow = pandas.DataFrame([dict(likely=likely, row=j), ])
        likelyhood = likelyhood.append(onerow, ignore_index=True)            
    return likelyhood


def likelyhood_onetomany(data1,data2,sku):#data1预测出的属性值，data2为商品库的属性值，sku为单独一列
    likelyhood=pandas.DataFrame(columns=['likely','x','y'])
#计算全部的相似度
    for j in range(numpy.shape(data1)[0]):
        for k in range(numpy.shape(data2)[0]):
            numerator=0
            for i in range(numpy.shape(data1)[1]):
                numerator=numerator+data1[j,i]*data2[k,i]
            addx=0
            for i in range(numpy.shape(data1)[1]):
                addx=addx+data1[j,i]*data1[j,i]
            addy=0
            for i in range(numpy.shape(data2)[1]):
                addy=addy+data2[k,i]*data2[k,i]
            denominator=math.sqrt(addx)*math.sqrt(addy)
            likely=round(numerator/denominator,3)
            row = pandas.DataFrame([dict(likely=likely, x=j, y=sku[k,0]),])
            likelyhood = likelyhood.append(row, ignore_index=True)    
    return likelyhood



#第一件商品的预测信息
Bayes_1=loadDataSet('/Users/XuLiu/Desktop/data/Bayes_1')
#训练集商品的属性dump信息
Bayes_real=numpy.array(loadDataSet('/Users/XuLiu/Desktop/data/attr_dump_real'))
#训练集商品得到的属性信息
Bayes_pred=numpy.array(loadDataSet('/Users/XuLiu/Desktop/data/attr_dump_pred'))
#所有商品的sku
sku=numpy.array(loadStringSet('/Users/XuLiu/Desktop/data/skulist'))
#所有商品的属性dump信息
skuinfo_dump=numpy.array(loadDataSet('/Users/XuLiu/Desktop/data/skuinfo_dump'))
#训练集预测的dump信息
Bayes_pred_1=numpy.array(loadDataSet('/Users/XuLiu/Desktop/data/Bayes_pred_1'))
#测试集预测的dump信息
pred_valid=numpy.array(loadDataSet('/Users/XuLiu/Desktop/data/pred_valid'))


#计算相似度
likely_1=likelyhood_alltoall(numpy.array(Bayes_1))

#一对一相似度
likely_learning=likelyhood_onetoone(Bayes_real,Bayes_pred)
avglikely=sum(likely_learning['likely'])/numpy.shape(Bayes_real)[0]
#一对多相似度
likely_match=likelyhood_onetomany(Bayes_pred_1,skuinfo_dump,sku)
likely_match.describe()


a=likelyhood.sort(columns=["x","likely"],ascending=[1,0])

a.to_csv('/Users/XuLiu/Desktop/data/recomm_validation', encoding='utf-8', index=False)





likelyhood=pandas.DataFrame(columns=['likely','x','y'])
#计算全部的相似度
for j in range(numpy.shape(pred_valid)[0]):
    for k in range(numpy.shape(skuinfo_dump)[0]):
        numerator=0
        for i in range(numpy.shape(pred_valid)[1]):
            numerator=numerator+pred_valid[j,i]*skuinfo_dump[k,i]
        addx=0
        for i in range(numpy.shape(pred_valid)[1]):
            addx=addx+pred_valid[j,i]*pred_valid[j,i]
        addy=0
        for i in range(numpy.shape(skuinfo_dump)[1]):
            addy=addy+skuinfo_dump[k,i]*skuinfo_dump[k,i]
        denominator=math.sqrt(addx)*math.sqrt(addy)
        likely=round(numerator/denominator,3)
        row = pandas.DataFrame([dict(likely=likely, x=j, y=sku[k,0]),])
        likelyhood = likelyhood.append(row, ignore_index=True)  


















