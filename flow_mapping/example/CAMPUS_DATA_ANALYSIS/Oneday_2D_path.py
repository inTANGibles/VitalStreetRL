import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime
from collections import Counter
from pandas._libs.tslibs.timedeltas import Timedelta
from tqdm import tqdm
import os

#导入WiFi数据
def import_data(data_path):

    data_path=rf"{data_path}"
    with open(data_path,'r',encoding='utf-8') as f:
        data = pd.read_json(f,lines=True)#lines用于传入多行数据
    wifi_df = pd.DataFrame(data)

    return wifi_df
#导入最短路径数据
def import_shortest_path_data():
    data_path=r"D:\python code\CAMPUS_DATA_ANALYSIS\devices_shortest_path.json"
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)

    path_df = pd.DataFrame([(key,v[0],v[1]) for key ,v in data.items()],columns=['path','shortest distance','passing points'])

    return path_df
#导入点坐标数据
def import_node_axis_data():
    data_path=r"D:\python code\CAMPUS_DATA_ANALYSIS\points_axis.json"
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)#lines用于传入多行数据

    axis_df = pd.DataFrame(data).T

    return axis_df
#导入设备坐标数据
def import_devices_axis_data():
    data_path=r"D:\python code\CAMPUS_DATA_ANALYSIS\devices_axis.json"
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    devices_axis_df = pd.DataFrame([(k,v[0],v[1]) for k,v in data.items()],columns=['devices_number','x_axis','y_axis'])

    return devices_axis_df

#映射次数
def process_input_number(num):
    #1-30次
    if 1 <= num <= 10:
        return 1
    #30-50次
    elif 10 < num <= 50:
        return 1 + (num - 10) / 80
    #50-80次
    elif 50 < num <= 100:
        return 1.5 + (num - 50) / 100
     #50-80次
    elif 100 < num <= 300:
        return 2 + (num - 100) / 100
     #50-80次
    elif 300 < num <= 500:
        return 3 + (num - 300) / 200
     #50-80次
    elif 500 < num <= 800:
        return 4 + (num - 500) / 300
    #50-80次
    elif 800 < num <= 1000:
        return 5 + (num - 800) / 200
    #>800
    else :
        return 6
#设备数据数量  
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
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1,aspect='equal')
    #按时间段取出 start time <df[df['t']]<end time
    wifi_df['t'] = pd.to_datetime(wifi_df['t'],unit='s')
    t_sort_df = wifi_df.sort_values(by='t')
    start_time = str(t_sort_df['t'].iloc[0])
    end_time = str(t_sort_df['t'].iloc[-1])

    time_df = t_sort_df
    #核心绘图函数
    personal_path(ax,time_df,path_df,axis_df,devices_axis_df)
   
    fig.suptitle(f'2023-12-22 Totoal Path Plot')
    # plt.show()
    save_path = r"D:\postgraduate study\互动建筑设计\东吴校园数据挖掘\最终图纸\Total path"
    save_file = f'{save_path}\\{start_time[0:11]}.png'
    plt.savefig(save_file)
    plt.close(fig)
    print(f'save {start_time[0:11]}fig successfully')

def personal_path(ax,time_df,path_df,axis_df,devices_axis_df):

    #获取时间t列表
    time_list=time_df['t'].tolist()
    #取mac途径的探针号列表
    a_passlist = []
    time_df['a'].apply(lambda x :a_passlist.append(x))
    a_and_time_list=list(zip(a_passlist,time_list))
    devices_number=set(a_passlist)

    #添加途径探针号码
    for device in list(devices_number):
        x=devices_axis_df[devices_axis_df["devices_number"] == str(device)]["x_axis"]
        y=devices_axis_df[devices_axis_df["devices_number"] == str(device)]["y_axis"]
        plt.text(x,y,device,fontsize = 7)
        
    #根据获取的a的数据数量绘制点大小
    a_counter = Counter(a_passlist)
    for a,counter in a_counter.items():
        a_axis_x=devices_axis_df[devices_axis_df["devices_number"] == str(a)]["x_axis"]
        a_axis_y=devices_axis_df[devices_axis_df["devices_number"] == str(a)]["y_axis"]
        # circle_radius = process_input_number2(counter)
        ax.scatter(a_axis_x,a_axis_y,s=counter/5,c='r',alpha=1)

    devided_path_list = []    
    for i in range(len(a_and_time_list)-1):
        a1 = a_and_time_list[i][0]
        a2 = a_and_time_list[i+1][0]
        if a1 != a2:
            cur_path = tuple(sorted((a1,a2)))#将路径元组转化为排序后的元组
            devided_path_list.append(cur_path)
        else:continue
            
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
                x_axis_list.append(axis_df.at[str(pass_point),0])
                y_axis_list.append(axis_df.at[str(pass_point),1])
                #绘制单个mac的轨迹
                line_width = process_input_number(pass_times)
                ax.plot(x_axis_list, y_axis_list,marker = 'o',ms=0,linewidth = line_width,color='red')

    # #添加轴标
    ax.set_xlabel('X (Relative Coordinates)')
    ax.set_ylabel('Y (Relative Coordinates)')     
    # #叠合地图
    img_path = r"D:\python code\CAMPUS_DATA_ANALYSIS\总平面图2.jpg"
    x_min, x_max = 1.8, 50
    y_min, y_max = 3, 60
    img = Image.open(img_path)
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], aspect='auto', alpha=0.5)
 

 
if __name__ == '__main__':
    # g = os.walk(r'D:\\python code\\CAMPUS_DATA_ANALYSIS\\final_data')
    # for path,dir_list,file_list in g:
    #     for file in file_list:
    #         data_path = os.path.join(path,file)
    #         data_path=rf"{data_path}"

    #导入本地wifi文件
    data_path = r'D:\python code\CAMPUS_DATA_ANALYSIS\final_data\5_prob20231222.json'
    wifi_df = import_data(data_path)
    wifi_df['t'] = pd.to_datetime(wifi_df['t'],unit='s')
    t_sort_df = wifi_df.sort_values(by='t')
    print(t_sort_df['t'].iloc[0])
    print(t_sort_df['t'].iloc[-1])
    print(len(wifi_df))

    #设备之间最短路径数据
    path_df=import_shortest_path_data()
    #导入所有坐标数据
    axis_df=import_node_axis_data()
    #导入探针坐标数据
    devices_axis_df = import_devices_axis_data()

    #总体轨迹图
    total_path(wifi_df)






   
    




    