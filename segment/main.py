import cv2
import time
import os
import copy
import network_big_apex.network_deeplab_upernet_aspp_psp_ab_swin_skips_1288 as network
import pandas as pd
from function.custom_transforms_mine import *
from function.segmentation_optimization import seg_opt
from function.calcu_DHI_512 import calcu_DHI
from function.calcu_signal_strength import calcu_Sigs
from function.quantitative_analysis import quantitative_analysis
from function.shiyan_jihe_signal_mean_std_plot_function import scatter_mean_std
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
time_start = time.time()

class DualCompose_img:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


image_only_transform = DualCompose_img([
    ToTensor_img(),
    Normalize_img()
])


def clahe_cv(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    output_cv = cv2.merge([b, g, r])
    return output_cv

model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

model = model_map['deeplabv3plus_resnet101'](num_classes=14, output_stride=16)
model = torch.nn.DataParallel(model)
model.to(device)
# load model weights
model_weight_path = "./weights_big_apex/deeplab_upernet_aspp_psp_ab_swin_skips_1288/deeplab_upe" \
                    "rnet_aspp_psp_ab_swin_skips_1288_0.0003.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
model.eval()
data_input_path = "./input/data_input"
results_output_path = "./output"
# base_dir = './input/baseline_range'
# SI_dir = os.path.join(base_dir, 'SI')
# data_SI_L2L3_excel_name = os.path.join(SI_dir, 'SI_L2L3_trend' + '.xlsx')
# SI_L2L3 = pd.read_excel(data_SI_L2L3_excel_name, 'SI trend', usecols=[0, 1])
# print(SI_L2L3.info())
quantitative_analysis_results_output_name = 'quantitative_analysis_results' + '.xlsx'

dirList = os.listdir(data_input_path)

with torch.no_grad():
    for dir in dirList:
        data_input_dir = os.path.join(data_input_path, dir)
        data_output_path = os.path.join(results_output_path, dir)
        img_list = os.listdir(data_input_dir)
        for im_name in img_list:
            im_name_no_suffix = (im_name.split('.')[0]).split('-')[-1]
            input_age = int(im_name_no_suffix[0:2])
            input_sex = int(im_name_no_suffix[3])
            im_path = os.path.join(data_input_dir, im_name)
            print('processing ' + str(im_path) + '.'*20)
            input = cv2.imread(im_path)
            out_cv = clahe_cv(input)
            input_img = image_only_transform(out_cv)
            input_img = torch.unsqueeze(input_img, 0)
            pred_img = model(input_img)
            output = torch.nn.Softmax2d()(pred_img)
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            output_seg_opt = output.clone()

            output_seg_opt = torch.squeeze(output_seg_opt).cpu()
            output_seg_opt = output_seg_opt.numpy()

            # output_seg_opt = torch.squeeze(output_seg_opt).numpy()
            output = seg_opt(output_seg_opt)

            jihe_parameter = []  # 初始化存储计算结果的列表
            time_calcu_DHI_bf = time.time()  # 记录开始计算DHI的时间
            DHI, DWR, HD, HV, point_big_h, point_big_w, point_fenge_h_big, point_fenge_w_big = calcu_DHI(
                output)  # 计算DHI、DWR、HD、HV等参数
            jihe_parameter.append(HD)  # 将HD参数添加到jihe_parameter列表中
            jihe_parameter.append(DHI)  # 将DHI参数添加到jihe_parameter列表中
            jihe_parameter.append(DWR)  # 将DWR参数添加到jihe_parameter列表中
            time_calcu_DHI_af = time.time()  # 记录结束计算DHI的时间
            time_calcu_DHI = time_calcu_DHI_af - time_calcu_DHI_bf  # 计算计算DHI所用的时间

            point_big_h = np.array(point_big_h)  # 将point_big_h转换为NumPy数组
            point_big_h = point_big_h.flatten()  # 将point_big_h数组展平为一维数组
            point_big_w = np.array(point_big_w)  # 将point_big_w转换为NumPy数组
            point_big_w = point_big_w.flatten()  # 将point_big_w数组展平为一维数组
            point_input_pic = copy.deepcopy(input)  # 复制输入图像
            print(point_big_h, point_big_w)
            point_size = 1  # 圆点的大小
            point_color = (0, 0, 255)  # 圆点的颜色(BGR格式)
            thickness = 4  # 圆点的粗细

            for p in range(len(point_big_h)):
                point = (point_big_w[p], point_big_h[p])  # 圆点的坐标
                cv2.circle(point_input_pic, point, point_size, point_color, thickness)  # 在point_input_pic图像上绘制圆点

            # point_fenge_h_big = np.array(point_fenge_h_big)  # 将point_fenge_h_big转换为NumPy数组
            # point_fenge_h_big = point_fenge_h_big.flatten()  # 将point_fenge_h_big数组展平为一维数组
            # point_fenge_w_big = np.array(point_fenge_w_big)  # 将point_fenge_w_big转换为NumPy数组
            # point_fenge_w_big = point_fenge_w_big.flatten()  # 将point_fenge_w_big数组展平为一维数组
            # point_size = 1  # 圆点的大小
            # point_color = (0, 0, 0)  # 圆点的颜色(BGR格式)
            # thickness = 4  # 圆点的粗细

            # for s in range(len(point_fenge_w_big)):
            #     point = (point_fenge_w_big[s], point_fenge_h_big[s])  # 圆点的坐标
            #     cv2.circle(point_input_pic, point, point_size, point_color, thickness)  # 在point_input_pic图像上绘制圆点

            cv2.imshow('point_input_pic', point_input_pic)  # 在窗口中显示point_input_pic图像
            cv2.imwrite(os.path.join(data_output_path, "point_detect.JPG"),
                        point_input_pic)  # 将point_input_pic图像保存到文件中

            SI_parameter = []  # 初始化存储计算结果的列表
            time_calcu_Sigs_bf = time.time()  # 记录开始计算Sigs的时间
            inputs_Sigs = input  # 将input赋值给inputs_Sigs
            output_Sigs = output.copy()  # 复制output并赋值给output_Sigs
            SI_big_final, disc_si_dif_final = calcu_Sigs(inputs_Sigs, output_Sigs)  # 计算Sigs参数
            SI_parameter.append(disc_si_dif_final)  # 将disc_si_dif_final参数添加到SI_parameter列表中
            time_calcu_Sigs_af = time.time()  # 记录结束计算Sigs的时间
            time_calcu_Sigs = time_calcu_Sigs_af - time_calcu_Sigs_bf  # 计算计算Sigs所用的时间

            scatter_mean_std(data_output_path, input_sex, input_age, disc_si_dif_final, HD, DHI,
                             DWR)  # 绘制散点图并计算均值和标准差

            quantitative_results = quantitative_analysis(disc_si_dif_final, HD, DHI, DWR, input_sex)  # 进行定量分析

            data_jihe_parameter = pd.DataFrame(jihe_parameter)  # 将jihe_parameter列表转换为DataFrame对象
            data_SI_parameter = pd.DataFrame(SI_parameter)  # 将SI_parameter列表转换为DataFrame对象
            data_quantitative_results = pd.DataFrame(quantitative_results)  # 将quantitative_results列表转换为DataFrame对象

            quantitative_analysis_results_output_name_path = os.path.join(data_output_path, str(
                im_name.split('.')[0]) + quantitative_analysis_results_output_name)  # 设置定量分析结果的输出路径和文件名
            writer = pd.ExcelWriter(quantitative_analysis_results_output_name_path)  # 创建一个Excel文件
            data_jihe_parameter.to_excel(writer, 'jihe_parameter',
                                         float_format='%.5f')  # 将jihe_parameter的数据写入名为'jihe_parameter'的工作表中
            data_SI_parameter.to_excel(writer, 'SI_parameter',
                                       float_format='%.5f')  # 将SI_parameter的数据写入名为'SI_parameter'的工作表中
            data_quantitative_results.to_excel(writer, 'quantitative_results',float_format='%.5f')  # 将quantitative_results的数据写入名为'quantitative_results'的工作表中
            writer._save()  # 保存Excel文件
            writer.close()  # 关闭Excel文件

