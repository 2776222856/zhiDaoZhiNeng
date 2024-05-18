import numpy as np  # 导入numpy库
import cv2  # 导入cv2库

# 定义矩形边界的区域编码
LEFT = 1  # 左边界编码0001
RIGHT = 2  # 右边界编码0010
BOTTOM = 4  # 下边界编码0100
TOP = 8  # 上边界编码1000

# 定义矩形边界坐标
xl = 0  # 设定左边界
xr = 5  # 设定右边界
yb = 0  # 设定下边界
yt = 5  # 设定上边界

clipped_img = np.full((50, 80, 3), 255, dtype=np.uint8)  # 建立一个500×800的白色画布


# 定义主函数
def main():
    img = np.full((50, 80, 3), 255, dtype=np.uint8)  # 创建一个500x800的白色图像
    cv2.rectangle(img, (0, 5), (5, 0), 255)  # 在图像上绘制一个蓝色矩形，左上角坐标是(0, 5)，右下角坐标是(5, 0)，表示剪裁窗口

    # 在图像上绘制几条直线，颜色为红色，不同的直线使用不同的起点和终点坐标
    cv2.line(img, (3, 6), (-1,-1), (0, 0, 255))  # 从(3, 6)到(-1, -1)画一条红色直线
    cv2.line(img, (-1, 7), (6, 6), (0, 0, 255))  # 从(-1,7)到(6,6)画一条红色直线

    # 对上述绘制的直线应用Cohen-Sutherland剪裁算法，只保留落在剪裁窗口内的线段部分
    CohenSutherland(3, 6, -1, -1)  # 调用CohenSutherland算法处理直线1
    CohenSutherland(-1, 7, 6, 6)  # 调用CohenSutherland算法处理直线2


    cv2.rectangle(clipped_img,  (0, 5), (5, 0), 255)  # 在剪裁后的图像上再次绘制矩形，表示剪裁窗口，应当使用与原图相同的颜色参数

    canvas = np.hstack((img, clipped_img))  # 将原图和剪裁后的图像并排合成一个新的图像以便观察对比

    # 窗口显示图形
    cv2.imwrite('out1.jpg', canvas)  # 生成一张图片
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # 显示窗口
    cv2.resizeWindow('image', 80, 50)  # 定义窗口大小
    cv2.imshow('image', canvas)  # 显示图像
    k = cv2.waitKey(0)  # 等待键盘输入，为毫秒级  0表示一直等待
    if k == 27:  # 键盘上Esc键的键值  按下就会退出
        cv2.destroyAllWindows()


# 定义编码子程序
def encode(x, y):
    c = 0  # 初始化编码为0，表示点位于剪裁窗口内部
    if x < xl:  # 如果点的x坐标小于窗口的左边界
        c = c | LEFT  # 使用位或操作，将编码设置为LEFT，表示点在窗口左侧
    if x > xr:  # 如果点的x坐标大于窗口的右边界
        c = c | RIGHT  # 使用位或操作，将编码设置为RIGHT，表示点在窗口右侧
    if y < yb:  # 如果点的y坐标小于窗口的下边界
        c = c | BOTTOM  # 使用位或操作，将编码设置为BOTTOM，表示点在窗口下方
    if y > yt:  # 如果点的y坐标大于窗口的上边界
        c = c | TOP  # 使用位或操作，将编码设置为TOP，表示点在窗口上方
    return c  # 返回计算得到的编码


# 定义Cohen-Sutherland算法子程序
def CohenSutherland(x1, y1, x2, y2):
    code1 = encode(x1, y1)  # 对线段起点进行编码
    code2 = encode(x2, y2)  # 对线段终点进行编码
    outcode = code1  # 初始化outcode为code1，假设起点在窗口外
    x, y = 0, 0  # 初始化交点坐标
    area = False  # 标记线段是否至少部分在窗口内
    while True:  # 使用循环直到找到解决方案或确定线段完全在窗口外
        if (code2 | code1) == 0:  # 如果code1和code2的或运算结果为0，表示两点均在窗口内
            area = True
            break
        if (code1 & code2) != 0:  # 如果code1和code2的与运算不为0，表示两点均在同一边界之外
            break
        if code1 == 0:  # 如果起点在窗口内，交换起点和终点的编码
            outcode = code2

        # 下面四个条件判断是找出线段与窗口边界的交点
        if (LEFT & outcode) != 0:  # 如果线段的一个端点在窗口左边界之外
            x = xl  # 交点的x坐标设置为窗口左边界的x坐标
            y = y1 + (y2 - y1) * (xl - x1) / (x2 - x1)  # 计算交点的y坐标，基于线性插值公式，利用相似三角形的原理
        elif (RIGHT & outcode) != 0:  # 如果线段的一个端点在窗口右边界之外
            x = xr  # 交点的x坐标设置为窗口右边界的x坐标
            y = y1 + (y2 - y1) * (xr - x1) / (x2 - x1)  # 基于线性插值公式计算交点的y坐标
        elif (BOTTOM & outcode) != 0:  # 如果线段的一个端点在窗口下边界之外
            y = yb  # 交点的y坐标设置为窗口下边界的y坐标
            x = x1 + (x2 - x1) * (yb - y1) / (y2 - y1)  # 基于线性插值公式计算交点的x坐标
        elif (TOP & outcode) != 0:  # 如果线段的一个端点在窗口上边界之外
            y = yt  # 交点的y坐标设置为窗口上边界的y坐标
            x = x1 + (x2 - x1) * (yt - y1) / (y2 - y1)  # 通过线性插值公式计算交点的x坐标
        x = int(x)  # 将交点x坐标转换为整型
        y = int(y)  # 将交点y坐标转换为整型
        # 更新线段的起点或终点为交点，并重新编码
        if outcode == code1:  # 如果正在处理的是线段的第一个端点
            x1 = x  # 更新线段起点的x坐标为交点的x坐标
            y1 = y  # 更新线段起点的y坐标为交点的y坐标
            code1 = encode(x, y)  # 重新计算更新后的起点坐标相对于剪裁窗口的位置编码
        else:  # 如果正在处理的是线段的第二个端点
            x2 = x  # 更新线段终点的x坐标为交点的x坐标
            y2 = y  # 更新线段终点的y坐标为交点的y坐标
            code2 = encode(x, y)  # 重新计算更新后的终点坐标相对于剪裁窗口的位置编码
    if area == True:  # 如果线段至少部分在窗口内
        cv2.line(clipped_img, (x1, y1), (x2, y2), (0, 0, 255))  # 在剪裁图像上绘制线段
    return


if __name__ == '__main__':
    main()