# 提供正向运动学+VMC
import numpy as np
import math
class five_links_param:
    def __init__(self,dt):
        # up link
        self.l_1 = 0.075
        self.l_4 = 0.075
        # down link
        self.l_2 = 0.14
        self.l_3 = 0.14
        # fixed link
        self.l_5 = 0.08
        self.dt = dt
    def forward_kinematics_cal(self, leg_flag,motor1_angle, motor2_angle, pitch, gyro_pitch):
        # 1. 角度量纲转换与零点调整 (确保 motor_angle 单位是 rad)
        if leg_flag == 0: # right
            self.phi_1 = motor1_angle + np.pi / 2.0
            self.phi_4 = motor2_angle + np.pi / 2.0
        elif leg_flag == 1: # left
            self.phi_1 = -motor1_angle + np.pi / 2.0
            self.phi_4 = -motor2_angle + np.pi / 2.0

        # 2. 正向运动学几何计算
        self.y_D = self.l_4 * math.sin(self.phi_4)
        self.x_D = self.l_5 + self.l_4 * math.cos(self.phi_4)
        
        self.y_B = self.l_1 * math.sin(self.phi_1)
        self.x_B = self.l_1 * math.cos(self.phi_1)
        
        self.A_0 = 2.0 * self.l_2 * (self.x_D - self.x_B)
        self.B_0 = 2.0 * self.l_2 * (self.y_D - self.y_B)
        self.l_BD = math.sqrt((self.x_D - self.x_B)**2 + (self.y_D - self.y_B)**2)
        self.C_0 = self.l_2**2 + self.l_BD**2 - self.l_3**2

        # 3. 增加 NaN 安全保护的开方计算
        sqrt_arg = self.A_0**2 + self.B_0**2 - self.C_0**2
        if sqrt_arg < 0.0: 
            sqrt_arg = 0.0 # 防止机械死点或噪声导致开方负数
        sqrt_val = math.sqrt(sqrt_arg)

        # 使用 math.atan2，分母为 0 没关系，它内部有处理
        self.phi_2 = 2.0 * math.atan2(self.B_0 + sqrt_val, self.A_0 + self.C_0)
        self.phi_3 = math.atan2(self.y_B - self.y_D + self.l_2 * math.sin(self.phi_2),
                                self.x_B - self.x_D + self.l_2 * math.cos(self.phi_2))
        
        self.x_C = self.l_1 * math.cos(self.phi_1) + self.l_2 * math.cos(self.phi_2)
        self.y_C = self.l_1 * math.sin(self.phi_1) + self.l_2 * math.sin(self.phi_2)

        # 计算虚拟腿长 L_0 和相对角度 phi_0
        self.L_0 = math.sqrt((self.x_C - self.l_5 / 2.0)**2 + self.y_C**2)
        self.last_L_0 = self.L_0
        self.d_L_0 = (self.L_0 - self.last_L_0) / self.dt

        self.phi_0 = math.atan2(self.y_C, (self.x_C - self.l_5 / 2.0)) 
        self.last_phi_0 = self.phi_0
        # 使用传入的 dt 计算微分，避免硬编码
        if self.dt > 0.0:
            self.d_phi_0 = (self.phi_0 - self.last_phi_0) / self.dt
        else:
            self.d_phi_0 = 0.0
            
        # 4. 状态变量解算（修正单位及正负号对齐）
        # 转换陀螺仪单位为 rad/s，此处直接使用传入的 gyro_pitch
        gyro_rad = gyro_pitch

        # 相对角度 alpha：摆杆相对于机体垂直线的偏角（顺时针为正）
        self.alpha = math.pi / 2.0 - self.phi_0
        self.last_alpha = self.alpha
        self.d_alpha = (self.alpha - self.last_alpha) / self.dt

        self.theta = self.alpha -  pitch
        self.d_theta = -self.d_phi_0 - gyro_rad
        
    def VMC_torque_cal(self, F_0, T_p):
        # 根据五连杆雅可比矩阵将极坐标力 [F_0, T_p] 转换到关节空间力矩 [T1, T2] 髋关节
        
        # NaN保护: 若产生共线奇点sin(phi3 - phi2) 会趋于0，给定一个极小下限防止除零错误
        denom = math.sin(self.phi_3 - self.phi_2)
        if math.isclose(denom, 0.0, abs_tol=1e-6):
            denom = 1e-6 * (1 if denom >= 0 else -1)
            
        L0_safe = self.L_0 if self.L_0 > 1e-6 else 1e-6

        J11 = (self.l_1 * math.sin(self.phi_0 - self.phi_3) * math.sin(self.phi_1 - self.phi_2)) / denom
        J12 = (self.l_1 * math.cos(self.phi_0 - self.phi_3) * math.sin(self.phi_1 - self.phi_2)) / (L0_safe * denom)
        J21 = (self.l_4 * math.sin(self.phi_0 - self.phi_2) * math.sin(self.phi_3 - self.phi_4)) / denom
        J22 = (self.l_4 * math.cos(self.phi_0 - self.phi_2) * math.sin(self.phi_3 - self.phi_4)) / (L0_safe * denom)
        
        T1 = J11 * F_0 + J12 * T_p
        T2 = J21 * F_0 + J22 * T_p
        
        T1 = np.clip(T1, -1.8, 1.8)
        T2 = np.clip(T2, -1.8, 1.8)
        
        # 符号适配 (如果不同侧电机安装相反则做对应反转)
        return T1, T2


