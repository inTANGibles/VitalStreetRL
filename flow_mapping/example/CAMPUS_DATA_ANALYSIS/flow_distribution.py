import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime
from collections import Counter
from pandas._libs.tslibs.timedeltas import Timedelta
import matplotlib.patches as patches
import numpy as np
import os

#导入WiFi数据
def import_data(data_path):

    data_path=rf"{data_path}"
    with open(data_path,'r',encoding='utf-8') as f:
        data = pd.read_json(f,lines=True)#lines用于传入多行数据
    wifi_df = pd.DataFrame(data)

    return wifi_df

#导入设备坐标数据
def import_devices_axis_data():
    data_path=r"D:\python code\CAMPUS_DATA_ANALYSIS\devices_axis.json"
    with open(data_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    devices_axis_df = pd.DataFrame([(k,v[0],v[1]) for k,v in data.items()],columns=['devices_number','x_axis','y_axis'])

    return devices_axis_df

def pass_plott(wifi_df,devices_axis_df,day):
    #流量分布饼图
    fig = plt.figure(figsize=(8,9.5))
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')

    #获取所有的mac数量，以及单个探针的mac数量
    str_mac_tuple = wifi_df["m"].apply(lambda x : tuple(x)).unique()
    anchor_mac_set_counts = wifi_df.groupby('a')['m'].apply(lambda x: len(set(tuple(mac) for mac in x)))
    anchor_ratio_dict={}
    for anchor,number in dict(anchor_mac_set_counts).items():
        anchor_ratio_dict[anchor] = ((number/len(str_mac_tuple)).round(2))*360

    #取有数据的探针的坐标点list
    x_list = []
    y_list = []
    a_unique_list= np.sort(wifi_df['a'].unique())
    for device_num in a_unique_list:

        x_list.append(devices_axis_df[devices_axis_df["devices_number"] == str(device_num)].iloc[0,1])
        y_list.append(devices_axis_df[devices_axis_df["devices_number"] == str(device_num)].iloc[0,2])

    def min_max_scaling(data, min_range, max_range):
        min_val = min(data)
        max_val = max(data)
        scaled_data = [(x - min_val) / (max_val - min_val) * (max_range - min_range) + min_range for x in data]
        return scaled_data
        
    #根据探针获取的mac数量作为半径大小
    for x,y,num in zip(x_list,y_list,a_unique_list):
        ax.text(x-1,y-1,num,fontsize = 8,color='black')
    anchor_mac_counts_list = list(dict(anchor_mac_set_counts).values())
    radius_list = min_max_scaling(anchor_mac_counts_list,min_range=1,max_range=5)

    #添加饼图
    anchor_ratio_list=list(anchor_ratio_dict.values())
    for i in range(len(x_list)):
        wedge = patches.Wedge(center=(x_list[i],y_list[i]),r=radius_list[i],theta1=0,theta2=anchor_ratio_list[i],width=radius_list[i]/2,color='red',alpha = 0.5)
        wedge2 = patches.Wedge(center=(x_list[i],y_list[i]),r=radius_list[i],theta1=anchor_ratio_list[i],theta2=360,width=radius_list[i]/2,color='black',alpha=0.3)

        ax.text(x_list[i]-1,y_list[i]+0.2,s = (anchor_ratio_list[i]*0.8/360).round(2),fontsize = 10,color = 'red')
        ax.add_patch(wedge)
        ax.add_patch(wedge2)
        
    # #叠合地图
    img_path = r"D:\python code\CAMPUS_DATA_ANALYSIS\总平面图2.jpg"
    x_min, x_max = 1.8, 50
    y_min, y_max = 3, 60
    img = Image.open(img_path)
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], aspect='auto', alpha=0.5)
    ax.set_xlabel('X coordination')
    ax.set_ylabel('Y cooordination')

    save_path = r"D:\postgraduate study\互动建筑设计\东吴校园数据挖掘\最终图纸\经停率"
    save_file = f'{save_path}\\{day}.png'
    plt.savefig(save_file)
    plt.close(fig)
    print(f'save {day}fig successfully')

if __name__ == '__main__':
    g = os.walk(r'D:\\python code\\CAMPUS_DATA_ANALYSIS\\final_data')
    for path,dir_list,file_list in g:
        for file in file_list:
            data_path = os.path.join(path,file)
            data_path=rf"{data_path}"
            #导入本地wifi文件
            wifi_df = import_data(data_path)
            wifi_df['t'] = pd.to_datetime(wifi_df['t'],unit='s')
            t_sort_df = wifi_df.sort_values(by='t')

            day = t_sort_df['t'].iloc[0]
            day = str(day)[0:10]

            devices_axis_df = import_devices_axis_data()

            pass_plott(wifi_df,devices_axis_df,day)