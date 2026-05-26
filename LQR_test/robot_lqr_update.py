import numpy as np
import math

# 本程序提供基于matlab的LQR整定与多项式拟合
# 通过阅读 robot.xml 提取的参数
# 对于腿长变化 通过在 ./robot_vmc_test/vmc_verification.py 使用位置环+逆运动学分析
# 暂定腿长为 0.09m-0.17m

LQR_K_RIGHT_DEFAULT = np.array([
    -3.544430, -0.268816, -0.653742, -0.841154,  0.925709,  0.116405, 
     3.469316,  0.190430,  0.477351,  0.556240,  2.168246,  0.181568
])

LQR_K_LEFT_DEFAULT = np.array([
    -3.544430, -0.268816, -0.653742, -0.841154,  0.925709,  0.116405, 
     3.469316,  0.190430,  0.477351,  0.556240,  2.168246,  0.181568
])

POLY_COEFFICIENT = np.array([
    [             635.852855,            -133.635363,             -50.923516,               0.387987],
    [            1967.300994,           -1494.178758,             344.621817,             -13.691384],
    [             112.968771,             -29.467594,              -8.645454,               0.276152],
    [             302.569026,            -241.898339,              57.949493,              -2.113770],
    [             313.900380,            -111.142936,               4.229136,              -1.153871],
    [             906.946695,            -576.201838,             119.175430,              -5.726742],
    [             397.606417,            -145.981712,               8.759654,              -1.773861],
    [            1426.561369,            -835.355111,             161.339682,              -7.470983],
    [             214.431806,             -85.604193,              -5.658215,               6.843320],
    [            -809.213632,             229.580360,              26.905963,               7.722442],
    [               6.319481,              -2.298235,              -1.392479,               0.830905],
    [            -144.708159,              57.133242,              -3.309786,               1.121277],
])

class lqr_controller:
    def __init__(self, dt):
        """
        初始化 LQR 控制器参数
        """
        self.dt = dt
        self.x_set = 0.0
        self.v_set = 0.0
        
        # 机器人状态
        self.x_b_set=0.0
        self.phi = 0.0          # 机体俯仰角 (Pitch)
        self.d_phi = 0.0        # 机体俯仰角速度
        self.theta = 0.0        # 摆杆与机体的相对夹角
        self.d_theta = 0.0      # 角速度
        self.w = 0.0            # 车轮转速
        self.v_b = 0.0          # 机体线速度估计
        self.v_b_whole = 0.0    # 整体平均线速度
        self.x_b = 0.0          # 机体位移积分
        
        # LQR K 矩阵 (实时多项式计算得出)
        self.K = np.zeros(12)

    def states_update(self, leg_flag, five_links, pitch, gyro_pitch, motor_pos, motor_speed, wheel_radius):
        """
        更新用于 LQR 控制的状态变量
        :param leg_flag: 0 = 右腿, 1 = 左腿
        :param five_links: 对应腿的 five_links_param 实例，提供 theta, d_theta, L_0, d_L_0 等运动学计算结果
        :param pitch: IMU测量的 Pitch
        :param gyro_pitch: IMU的 Pitch 角速度
        :param motor_speed: 车轮电机转速 (rad/s)
        """
        self.phi = pitch
        self.d_phi = gyro_pitch
        self.theta = five_links.theta
        self.d_theta = five_links.d_theta
        
        # 处理左右电机安装反向的问题：保证往同一方向运动时参数标志一致
        if leg_flag == 0:  # right
            self.w = -motor_speed
            self.x_b = -motor_pos*wheel_radius
        elif leg_flag == 1:  # left
            self.w = motor_speed
            # self.v_b = -self.w * wheel_radius - five_links.L_0 * self.d_theta * math.cos(self.theta) - five_links.d_L_0 * math.sin(self.theta)
            self.x_b = motor_pos*wheel_radius
            
        # 依据轮子半径0.06m及五连杆虚拟腿长进行前进位移和速度估计

        # self.v_b = self.w * wheel_radius
        self.v_b = self.w * wheel_radius          
        self.v_b_whole = self.v_b

        # self.x_b += self.v_b_whole * self.dt
        
    def calc_k_matrix_from_poly(self, L_0):
        """
        基于当前虚拟腿长 L_0 的多项式拟合系数计算连续变化的 LQR-K 矩阵
        """
        for i in range(12):
            self.K[i] = self._poly_calc(POLY_COEFFICIENT[i], L_0)
        return self.K
        
    def _poly_calc(self, coe, length):
        """
        计算多项式: coe[0]*L^3 + coe[1]*L^2 + coe[2]*L + coe[3]
        使用霍纳法则(Horner's method): ((coe[0]*L + coe[1])*L + coe[2])*L + coe[3]
        """
        return ((coe[0] * length + coe[1]) * length + coe[2]) * length + coe[3]
