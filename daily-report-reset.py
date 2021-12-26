from _datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
matplotlib.use('Agg')

#柱状图顶点数据统计
def mark(nums,ax):
    sum =0
    for x,y in enumerate(nums):
        sum = sum+y
        ax.text(x,y,'%d'%y, fontsize=18, va='baseline',ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', ec='r',lw=1 ,alpha=0.9))
    return sum

#饼状图内标签，百分比化真实值
def sums(value):
    def sumint(pct):
        total = sum(value)
        val = int(round(pct*total/100.0))
        return '{v:d}'.format(v=val)
    return sumint

def getFileList(path):
    for filename in os.listdir(path):
        file = os.path.join(path,filename)    
    return file

def TimePlot():

    now = datetime.now()
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.width', 1000)
    
    path=r'.\IPCCIS'
    file = getFileList(path)
    data = pd.read_excel(file)
    print(data['Company name'])
    filter = data['Company name'].isin(['HUAWEI DEVICE HONG KONG CO LIMITED','Huawei device Hong Kong co limited','Sitel GmbH'])
    data0 = data[filter]
    data1 = data[~filter]

    dataA = data0.copy()
    dataA['Start_date_dt'] = pd.to_datetime(dataA.loc[:,'Start date'], format='%d/%m/%Y %H:%M')
    dataA['Age_td'] = now - dataA['Start_date_dt']
    dataA['Age_days'] = dataA['Age_td'].map(lambda x: x.days)
    d0_age_groups = pd.cut(dataA['Age_days'], bins=[0, 1, 3, 7, 14, 28, np.inf], include_lowest=True)
    d0_res = dataA.groupby(d0_age_groups).size()

    dataB = data1.copy()
    dataB['Start_date_dt'] = pd.to_datetime(dataB.loc[:,'Start date'], format='%d/%m/%Y %H:%M')
    dataB['Age_td'] = now - dataB['Start_date_dt']
    dataB['Age_days'] = dataB['Age_td'].map(lambda x: x.days)
    d1_age_groups = pd.cut(dataB['Age_days'], bins=[0, 1, 3, 7, 14, 28, np.inf], include_lowest=True)
    d1_res = dataB.groupby(d1_age_groups).size()
    
    
    path=r'.\SAT'
    file = getFileList(path)
    data2 = pd.read_excel(file)
    d2_res = data2.groupby('Proc. priority').size()

    
    # 画图
    # fig, axes = plt.subplots(1, 3, constrained_layout=True, figsize=(24, 10))
    fig=plt.figure(constrained_layout=False,figsize=(24, 10))
    gs=fig.add_gridspec(ncols=27,nrows=10)
    axes1=fig.add_subplot(gs[:,0:6])
    axes2=fig.add_subplot(gs[:,7:13]) 
    axes3=fig.add_subplot(gs[:,14:20])
    sizes=[]
    sizes.append(mark(d0_res,axes1))
    sizes.append(mark(d1_res,axes2))
    sizes.append(mark(d2_res,axes3))
    d0_res.plot(ax=axes1, kind='bar', title=('ipcc ' + str(now)), fontsize=18)
    d1_res.plot(ax=axes2, kind='bar', title=('csd ' + str(now)), fontsize=18)
    d2_res.plot(ax=axes3, kind='bar', title=('sat ' + str(now)), fontsize=18)
    #饼图
    axes4=fig.add_subplot(gs[:,21:27])
    labels=['IPCC','IS','SAT']
    explode=(0,0,0)
    colors=['coral','limegreen','gold']
    plt.pie(sizes,explode=explode,labels=labels,colors=colors,shadow=True,autopct=sums(sizes),startangle=90,textprops={'fontsize': 20, 'color': 'b'}) 
    plt.axis('equal')
    plt.title('NUM-Pie',fontsize=18) 
    plt.legend(loc='lower center',fontsize=15)
    fig.autofmt_xdate() #横坐标旋转
    fig.axes[0].title.set_size(18)
    fig.axes[1].title.set_size(18)
    fig.axes[2].title.set_size(18)
    
    plt.savefig(r'C:\Users\Administrator\Desktop\csd\static\plot.png')
    plt.savefig(r'C:\Users\Administrator\Desktop\HelloCSD\static\plot.png')

if __name__ == '__main__':
    TimePlot()