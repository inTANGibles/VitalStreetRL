import pandas as pd
import matplotlib.pyplot as plt

import json
from datetime import datetime 
from collections import Counter
from pandas._libs.tslibs.timedeltas import Timedelta
from mpl_toolkits.mplot3d import Axes3D

#导入WiFi数据
def import_data():
    data_path=r"D:\python code\CAMPUS_DATA_ANALYSIS\final_data\1_prob20231225.json"
    with open(data_path,'r',encoding='utf-8') as f:
        data = pd.read_json(f,lines=True)#lines用于传入多行数据
    wifi_df = pd.DataFrame(data)
    return wifi_df
#导入最短路径数据
def import_shortest_path_data():
    data_path=r"D:\python code\CAMPUS_DATA_ANALYSIS\devices_shortest_path.json"
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    #以dataframe形式表示数据
    path_df = pd.DataFrame([(key,v[0],v[1]) for key ,v in data.items()],columns=['path','shortest distance','passing points'])

    return path_df
#导入点坐标数据
def import_node_axis_data():
    data_path=r"D:\python code\CAMPUS_DATA_ANALYSIS\points_axis.json"
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)#lines用于传入多行数据
    #以dataframe形式表示数据
    axis_df = pd.DataFrame(data).T

    return axis_df
#导入设备坐标数据
def import_devices_axis_data():
    data_path=r"D:\python code\CAMPUS_DATA_ANALYSIS\devices_axis.json"
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)#lines用于传入多行数据
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

#个人三维轨迹图
def personal_3D_path (wifi_df,devices_axis_df):
    fig = plt.figure(figsize=(9, 9))
    #为探针数据按照时间分段
    z_list=[7,10,14,18,24]
    lables = ['morning','noon','afternoon','evening']
    z_axis_list=[7,10,14,18]
    time_range = pd.to_datetime(["2023-12-25 07:00:00","2023-12-25 10:00:00","2023-12-25 14:00:00","2023-12-25 18:00:00","2023-12-25 23:59:59"])
    wifi_df['t'] =pd.to_datetime(wifi_df['t'],unit='s')
    wifi_df['time_range_category'] = pd.cut(wifi_df['t'],bins=time_range,labels=lables,right=False)
    #获取所有的mac元组，每次取出一个进行绘图操作
    str_mac_tuple = wifi_df["m"].apply(lambda x : tuple(x)).unique()
    
    i = 1
    for str_mac in str_mac_tuple[1000:1030]:
        ax = fig.add_subplot(5, 6, i, projection='3d')
        mac = list(str_mac)
        
        for lable,z_axis in dict(zip(lables,z_axis_list)).items():
            
            mac_match_df = wifi_df[wifi_df['m'].apply(lambda x:x == mac)]
            t_sort_df = mac_match_df.sort_values(by='t')
            lable_df = t_sort_df[t_sort_df['time_range_category'] == lable]
            #核心绘图函数
            personal_path(lable_df,path_df,axis_df,devices_axis_df,ax,z_axis,mac,lable)
            
            #获取不同lable的a_last,以最后一个a点绘制垂直线
            a_passlist = []
            lable_df['a'].apply(lambda x :a_passlist.append(x))
            if a_passlist :
                a_last = a_passlist[0]
                a_last_x = devices_axis_df[devices_axis_df["devices_number"] == str(a_last)]["x_axis"].iloc[0]
                a_last_y = devices_axis_df[devices_axis_df["devices_number"] == str(a_last)]["y_axis"].iloc[0]
                ax.plot([a_last_x,a_last_x],[a_last_y,a_last_y],zs=[z_axis,z_axis-4],color='red',linewidth=1)
            else: continue
        i+=1
        plt.title('One Mac 3D Path')
        
    #显示图
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    plt.tight_layout()
    plt.show()

#单MAC地址轨迹
def personal_path(lable_df,path_df,axis_df,devices_axis_df,ax,z_axis,mac,lable):
    #取mac途径的探针号列表
    time_list=lable_df['t'].tolist()
    a_passlist = []
    lable_df['a'].apply(lambda x :a_passlist.append(x))
    a_and_time_list=list(zip(a_passlist,time_list))
    
    devided_path_list = []#获取段路径列表
    for i in range(len(a_and_time_list)-1):
        a1 = a_and_time_list[i][0]
        a2 = a_and_time_list[i+1][0]
        cur_path = tuple(sorted((a1,a2)))#将路径元组转化为排序后的元组
        devided_path_list.append(cur_path)
        
        if a1 == a2 :
            #如果a1=a1，则以t2-t1作为在a1点的大小 
            t1 = a_and_time_list[i][1]
            t2 = a_and_time_list[i+1][1]
            stay_time = Timedelta(t2-t1).total_seconds()
            a1_axis_x = devices_axis_df[devices_axis_df["devices_number"] == str(a1)]["x_axis"]
            a1_axis_y = devices_axis_df[devices_axis_df["devices_number"] == str(a1)]["y_axis"]
            circle_radius = process_input_number2(stay_time)
            ax.scatter(a1_axis_x,a1_axis_y,z_axis,s=circle_radius,c='r',alpha=0.2)
            
    #根据path_passtime_dict绘制轨迹
    counter_dict = Counter(devided_path_list)
    # print(f"当前{mac}的{lable}时间段的counter—path:{counter_dict}") 
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
                ax.plot(x_axis_list, y_axis_list,z_axis,marker = 'o',ms=0,linewidth = line_width,color='red')

    #添加轴标
    ax.set_xlabel('X Coor')
    ax.set_ylabel('Y Coor')
    ax.set_zlabel('Time')
    # plt.xlim(0, 60)  
    # plt.ylim(0, 60)  
    
 
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
    personal_3D_path(wifi_df,devices_axis_df)




   
    




    