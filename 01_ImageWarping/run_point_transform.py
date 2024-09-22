import cv2
import numpy as np
import gradio as gr
from scipy.linalg import solve
from scipy.interpolate import Rbf

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(im, psrc, pdst, alpha=1000, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    """
    warped_image = np.array(image)
    warped_image_new = warped_image
    ### FILL: 基于MLS or RBF 实现 image warping

    # 获取图像尺寸
    h, w = warped_image.shape[:2]

    # 分离控制点的 x 和 y 坐标
    src_x, src_y = source_pts[:, 0], source_pts[:, 1]
    dst_x, dst_y = target_pts[:, 0], target_pts[:, 1]

    #delta_tempx = dst_x[:, None] - dst_x[None, :]
    delta_tempx = src_x[:, None] - src_x[None, :]
    delta_x=np.square(delta_tempx)

    #delta_tempy = dst_y[:, None] - dst_y[None, :]
    delta_tempy = src_y[:, None] - src_y[None, :]
    delta_y = np.square(delta_tempy)
    delta=delta_x + delta_y + alpha

    ones_matrix = np.ones_like(delta)
    B=ones_matrix / delta

    #A1 = np.linalg.inv(B) @ target_pts
    A = np.linalg.solve(B, target_pts)

    row_indices = np.arange(h).reshape(-1, 1)  # 行索引矩阵
    col_indices = np.arange(w)  # 列索引矩阵
    coordinates = np.stack(np.meshgrid(row_indices, col_indices, indexing='ij'), axis=-1)

    n_p = src_x.shape[0]
    f_x = coordinates
    for i in range(n_p):
        temp_i = coordinates - source_pts[i, :]
        square_sum = np.sum(np.square(temp_i), axis=-1) + alpha
        ones_matrix_i = np.ones_like(square_sum)
        B_i = ones_matrix_i / square_sum
        temp = B_i[:, :, np.newaxis] * A[i, :]
        f_x = f_x + B_i[:, :, np.newaxis] * A[i, :]
    f_x = np.round(f_x).astype(int)


    f_x[:, :, 0] = np.where((f_x[:, :, 0] < 0) | (f_x[:, :, 0] >= h), 0, f_x[:, :, 0])

    # 针对第二个维度（w）的越界处理
    f_x[:, :, 1] = np.where((f_x[:, :, 1] < 0) | (f_x[:, :, 1] >= h), 0, f_x[:, :, 1])


    for i in range(h):
        for j in range(w):
            a1 = f_x[i, j, :]

            #print (a1)
            a2 = warped_image_new[a1[0], a1[1], :]
            a3 = warped_image[i, j, :]
            warped_image_new[a1, :] = warped_image[i, j, :]




    rows = f_x[..., 0]  # 取出 f_x 中的所有行索引 (h*w)
    cols = f_x[..., 1]  # 取出 f_x 中的所有列索引 (h*w)

    # 利用行列索引从 image 中提取相应的 RGB 值
    #warped_image = image[rows, cols]

    return warped_image_new
    """
    # 获取图像的尺寸
    h, w, dim = im.shape
    n = len(psrc)

    # 初始化变形后的图像为白色
    im3 = np.ones((h, w, dim), dtype=np.uint8) * 255

    if n == 0:
        return im3

    # 交换 psrc 和 pdst 的列
    psrc = psrc[:, [1, 0]]
    pdst = pdst[:, [1, 0]]

    d = np.sum(np.linalg.norm(pdst - psrc, axis=1) ** 2)

    # 计算 D 矩阵
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = 1 / (np.sum((pdst[i] - pdst[j]) ** 2) + d)

    # 计算 B 矩阵
    B = np.linalg.solve(D, psrc - pdst)

    # 创建 meshgrid
    key1, key2 = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
    p1 = np.zeros((h, w))
    p2 = np.zeros((h, w))

    for k in range(n):
        q = 1 / ((key1 - pdst[k, 0]) ** 2 + (key2 - pdst[k, 1]) ** 2 + d)
        p1 += B[k, 0] * q
        p2 += B[k, 1] * q

    p1 = np.round(p1 + key1).astype(int)
    p2 = np.round(p2 + key2).astype(int)

    # 确保索引在有效范围内
    p1 = np.clip(p1, 1, h)
    p2 = np.clip(p2, 1, w)

    # 映射图像
    for i in range(h):
        for j in range(w):
            if 1 <= p1[i, j] <= h and 1 <= p2[i, j] <= w:
                im3[i, j, :] = im[p1[i, j] - 1, p2[i, j] - 1, :]
            else:
                im3[i, j, :] = [0, 0, 0]
    im3 = np.transpose(im3, (1, 0, 2))
    return im3

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image( source="upload",label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
