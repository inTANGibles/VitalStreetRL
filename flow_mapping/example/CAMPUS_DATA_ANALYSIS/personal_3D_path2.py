import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime 
from pandas._libs.tslibs.timedeltas import Timedelta
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

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

#个人三维轨迹图
def personal_3D_path (wifi_df,devices_axis_df):

    fig = plt.figure(figsize=(25,20))
    str_mac_tuple = wifi_df["m"].apply(lambda x : tuple(x)).unique()
    print(len(str_mac_tuple))

    i = 1
    for str_mac in str_mac_tuple[1000:1036]:
        ax = fig.add_subplot(6, 6, i, projection='3d')
        mac = list(str_mac)

        match_df = wifi_df[wifi_df['m'].apply(lambda x:x == mac)]
        match_df['t'] = pd.to_datetime(match_df['t'],unit='s')
        t_sort_df = match_df.sort_values(by='t')

        #核心绘图函数
        personal_path(t_sort_df,path_df,axis_df,devices_axis_df,ax)

        i += 1
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.tight_layout()
    plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
    # plt.axis('off')
    # plt.show()
    save_path = r"D:\postgraduate study\互动建筑设计\东吴校园数据挖掘\最终图纸\聚类"
    save_file = f'{save_path}\\444.png'
    plt.savefig(save_file, dpi=300,format="png")
    print('success')
    

#单MAC地址轨迹
def personal_path(t_sort_df,path_df,axis_df,devices_axis_df,ax):
    #取mac途径的探针号列表
    time_list = t_sort_df['t'].tolist()
    a_passlist = []
    t_sort_df['a'].apply(lambda x :a_passlist.append(x))
    a_and_time_list = list(zip(a_passlist,time_list))

    z_axis = 6
    for i in range(len(a_and_time_list)-1):
        a1 = a_and_time_list[i][0]
        a2 = a_and_time_list[i+1][0]
        path = tuple((a1,a2))#将路径元组转化为排序后的元组
        z_axis_change = 0
        if a1 == a2 :

            #如果a1=a1，则以t2-t1作为z坐标上升的高度
            t1 = a_and_time_list[i][1]
            t2 = a_and_time_list[i+1][1]
            stay_time = Timedelta(t2-t1).total_seconds()
            z_axis_change = stay_time/3600
            a1_axis_x = devices_axis_df[devices_axis_df["devices_number"] == str(a1)]["x_axis"]
            a1_axis_y = devices_axis_df[devices_axis_df["devices_number"] == str(a1)]["y_axis"]
            ax.plot([a1_axis_x.iloc[0], a1_axis_x.iloc[0]], [a1_axis_y.iloc[0], a1_axis_y.iloc[0]], [z_axis, z_axis + z_axis_change], marker='o', ms=0, color='red', linewidth=1)
            
        z_axis = z_axis_change+z_axis

        if a1 != a2 :
            #如果a1 != a2，则高度不变，绘制（a1 to a2）的轨迹
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
                    ax.plot(x_axis_list, y_axis_list,z_axis,marker = 'o',ms=0,linewidth = 1,color='red')
        
    #添加轴标
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticks([])
    # x_min, x_max = 1.8, 50
    # y_min, y_max = 3, 60
    # ax.set_zlim(6, 23)
    # plt.xlim(x_min, x_max)  
    # plt.ylim(y_min, y_max)

  
    
 
if __name__ == '__main__':
    #导入本地wifi文件
    wifi_df = import_data()
 
    #设备之间最短路径数据
    path_df = import_shortest_path_data()
  
    #导入所有坐标数据
    axis_df = import_node_axis_data()
 
    #导入探针坐标数据
    devices_axis_df = import_devices_axis_data()
    
    #总体轨迹图
    personal_3D_path(wifi_df,devices_axis_df)




   
    




    