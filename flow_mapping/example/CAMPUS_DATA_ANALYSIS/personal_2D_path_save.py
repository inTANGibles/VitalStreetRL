import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from collections import Counter
from pandas._libs.tslibs.timedeltas import Timedelta
from tqdm import tqdm

#导入WiFi数据
def import_data():
    data_path=r"D:\Architecture\AAA-Master\24Fall\ARI\CAMPUS_DATA_ANALYSIS\final_data\2_prob20231226.json"
    with open(data_path,'r',encoding='utf-8') as f:
        data = pd.read_json(f,lines=True)#lines用于传入多行数据
    wifi_df = pd.DataFrame(data)
    return wifi_df
#导入最短路径数据
def import_shortest_path_data():
    data_path=r"D:\Architecture\AAA-Master\24Fall\ARI\CAMPUS_DATA_ANALYSIS\devices_shortest_path.json"
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)

    path_df = pd.DataFrame([(key,v[0],v[1]) for key ,v in data.items()],columns=['path','shortest distance','passing points'])

    return path_df
#导入点坐标数据
def import_node_axis_data():
    data_path=r"D:\Architecture\AAA-Master\24Fall\ARI\CAMPUS_DATA_ANALYSIS\points_axis.json"
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)#lines用于传入多行数据

    axis_df = pd.DataFrame(data).T

    return axis_df
#导入设备坐标数据
def import_devices_axis_data():
    data_path=r"D:\Architecture\AAA-Master\24Fall\ARI\CAMPUS_DATA_ANALYSIS\devices_axis.json"
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    devices_axis_df = pd.DataFrame([(k,v[0],v[1]) for k,v in data.items()],columns=['devices_number','x_axis','y_axis'])

    return devices_axis_df

#映射次数
def process_input_number(num):
    #1-3次
    if 1 <= num <= 3:
        return num-0.5
    #3-5次
    elif 3 < num <= 5:
        return 2.5 + (num - 3) / 4
    #5-8次
    elif 5 < num <= 8:
        return 3 + (num - 5) / 6
    #>8
    else :
        return 4
#映射停留   
def process_input_number2(num):
    #1min
    if 0 <= num <= 60:
        return num
    #5min
    elif 60 < num <= 300:
        return 60 + (num - 100) / 1.5
    #10min
    elif 300 < num <= 600:
        return 180 + (num - 100) / 4.2
    #>10min
    else :
        return 500
#总体轨迹图
def total_path(wifi_df):
    
    save_path = r"D:\Architecture\AAA-Master\24Fall\ARI\CAMPUS_DATA_ANALYSIS\12.26"
    str_mac_tuple = wifi_df["m"].apply(lambda x : tuple(x)).unique()
    print(len(str_mac_tuple))
    
    for i, str_mac in enumerate(tqdm(str_mac_tuple[:10], desc='Saving Plots', unit='plot')):

        fig = plt.figure(figsize=(2.24,2.24))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        ax = fig.add_subplot(1, 1, 1,aspect='equal')
        mac = list(str_mac)

        match_df = wifi_df[wifi_df['m'].apply(lambda x:x == mac)]
        match_df['t'] = pd.to_datetime(match_df['t'],unit='s')
        t_sort_df = match_df.sort_values(by='t')

        #核心绘图函数
        personal_path(ax,t_sort_df,path_df,axis_df,devices_axis_df,mac)

        #隐藏轴数据
        plt.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #保存图片
        save_file = f'{save_path}\\path_{i}.png'
        
        plt.savefig(save_file)
        plt.close(fig)
    

#单MAC地址轨迹
def personal_path(ax,time_df,path_df,axis_df,devices_axis_df,mac):

    #获取时间t列表
    time_list=time_df['t'].tolist()
    #取mac途径的探针号列表
    a_passlist = []
    time_df['a'].apply(lambda x :a_passlist.append(x))
    a_and_time_list=list(zip(a_passlist,time_list))

        
    devided_path_list = []
    #依次取出两个相邻的数据组成新列表进行比较
    for i in range(len(a_and_time_list)-1):
        a1 = a_and_time_list[i][0]
        a2 = a_and_time_list[i+1][0]
        cur_path = tuple(sorted((a1,a2)))#将路径元组转化为排序后的元组
        devided_path_list.append(cur_path)
        
        if a1 == a2 :
            #如果a1=a1，则以t2-t1作为在a1点的大小
            t1= a_and_time_list[i][1]
            t2= a_and_time_list[i+1][1]
            stay_time = Timedelta(t2-t1).total_seconds()
            a1_axis_x=devices_axis_df[devices_axis_df["devices_number"] == str(a1)]["x_axis"]
            a1_axis_y=devices_axis_df[devices_axis_df["devices_number"] == str(a1)]["y_axis"]
            circle_radius = process_input_number2(stay_time)
            ax.scatter(a1_axis_x,a1_axis_y,s=circle_radius,c='red',alpha=0.2)          
    #根据path_passtime_dict绘制轨迹
    counter_dict = Counter(devided_path_list)
    for path,pass_times in counter_dict.items():
        match_path_row = path_df[path_df['path'] == str(path)]
        passing_series = match_path_row['passing points']
        #遍历series获取每段轨迹的途径点坐标
        for index,passing_list in passing_series.items():
            #获取每段轨迹途径点的坐标数据
            x_axis_list=[]
            y_axis_list=[]
            for pass_point in passing_list:
                #dataframe.at(row,column),取出途径点的x，y
                x_axis_list.append(axis_df.at[str(pass_point),0])
                y_axis_list.append(axis_df.at[str(pass_point),1])
                #绘制单个mac的单独lable的轨迹
                line_width = process_input_number(pass_times)
                ax.plot(x_axis_list, y_axis_list,marker = 'o',ms=0,linewidth = line_width,color='red')

    
    plt.xlim(0, 60)  # x轴范围从0到5
    plt.ylim(0, 60)  # y轴范围从0到35

 
if __name__ == '__main__':
    #导入本地wifi文件
    wifi_df = import_data()
    #设备之间最短路径数据
    path_df=import_shortest_path_data()
    #导入所有坐标数据
    axis_df=import_node_axis_data()
    #导入探针坐标数据
    devices_axis_df = import_devices_axis_data()
    

    #总体轨迹图
    total_path(wifi_df)






   
    




    