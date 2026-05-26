# 利用位置控制去验证电机零点的标注，VMC腿长控制的计算 
# 运行启动时候第一视角是右边的腿
# 首先这里的电机零点都是数值向下的 
import time
import math

import mujoco
import mujoco.viewer
import numpy as np
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

        # 若是首次运行，初始化上一时刻状态避免导数突变
        if not hasattr(self, 'last_L_0'):
            self.last_L_0 = math.sqrt((self.x_C - self.l_5 / 2.0)**2 + self.y_C**2)
            self.last_phi_0 = math.atan2(self.y_C, (self.x_C - self.l_5 / 2.0))
            self.last_alpha = math.pi / 2.0 - self.last_phi_0

        # 计算虚拟腿长 L_0 和相对角度 phi_0
        self.L_0 = math.sqrt((self.x_C - self.l_5 / 2.0)**2 + self.y_C**2)
        self.d_L_0 = (self.L_0 - self.last_L_0) / self.dt
        self.last_L_0 = self.L_0
        
        self.phi_0 = math.atan2(self.y_C, (self.x_C - self.l_5 / 2.0)) 
        
        # 使用传入的 dt 计算微分，避免硬编码
        if self.dt > 0.0:
            self.d_phi_0 = (self.phi_0 - self.last_phi_0) / self.dt
        else:
            self.d_phi_0 = 0.0
            
        self.last_phi_0 = self.phi_0
        
        # 4. 状态变量解算（修正单位及正负号对齐）
        # 转换陀螺仪单位为 rad/s，此处直接使用传入的 gyro_pitch
        gyro_rad = gyro_pitch

        # 相对角度 alpha：摆杆相对于机体垂直线的偏角（顺时针为正）
        self.alpha = math.pi / 2.0 - self.phi_0
        self.d_alpha = (self.alpha - self.last_alpha) / self.dt
        self.last_alpha = self.alpha
            
        # (验证正逆向纯运动学时不需要计算速度和 IMU)
        self.theta = self.alpha -  pitch
        self.d_theta = -self.d_phi_0 - gyro_rad

    def inverse_kinematics_cal_right(self,target_L0, target_phi0):
        # 1. 根据期望的长度和角度，算出足端坐标 (x_C, y_C)
        x_C = self.l_5 / 2.0 + target_L0 * math.cos(target_phi0)
        y_C = target_L0 * math.sin(target_phi0)

        # 2. 逆解算电机1（后侧，坐标 0,0）
        L_OC = math.sqrt(x_C**2 + y_C**2)
        if L_OC > (self.l_1 + self.l_2) or L_OC < abs(self.l_1 - self.l_2):
            return None, None # 超出工作空间范围
            
        eta_1 = math.atan2(y_C, x_C)
        psi_1 = math.acos((self.l_1**2 + L_OC**2 - self.l_2**2) / (2.0 * self.l_1 * L_OC))
        phi_1_target = eta_1 + psi_1  # 取外侧解（膝盖向后）

        # 3. 逆解算电机2（前侧，坐标 l_5,0）
        x_C2 = x_C - self.l_5
        L_O2C = math.sqrt(x_C2**2 + y_C**2)
        if L_O2C > (self.l_4 + self.l_3) or L_O2C < abs(self.l_4 - self.l_3):
            return None, None
            
        eta_2 = math.atan2(y_C, x_C2)
        psi_2 = math.acos((self.l_4**2 + L_O2C**2 - self.l_3**2) / (2.0 * self.l_4 * L_O2C))
        phi_4_target = eta_2 - psi_2  # 取外侧解（膝盖向前）

        # 4. 转换回电机命令系：减去 PI/2 的零点偏置
        motor1_cmd = phi_1_target
        motor2_cmd = phi_4_target
        
        return motor1_cmd, motor2_cmd
    
    def inverse_kinematics_cal_left(self,target_L0, target_phi0):
        # 1. 根据期望的长度和角度，算出足端坐标 (x_C, y_C)
        x_C = self.l_5 / 2.0 + target_L0 * math.cos(target_phi0)
        y_C = target_L0 * math.sin(target_phi0)

        # 2. 逆解算电机1（后侧，坐标 0,0）
        L_OC = math.sqrt(x_C**2 + y_C**2)
        if L_OC > (self.l_1 + self.l_2) or L_OC < abs(self.l_1 - self.l_2):
            return None, None # 超出工作空间范围
            
        eta_1 = math.atan2(y_C, x_C)
        psi_1 = math.acos((self.l_1**2 + L_OC**2 - self.l_2**2) / (2.0 * self.l_1 * L_OC))
        phi_1_target = eta_1 + psi_1  # 取外侧解（膝盖向后）

        # 3. 逆解算电机2（前侧，坐标 l_5,0）
        x_C2 = x_C - self.l_5
        L_O2C = math.sqrt(x_C2**2 + y_C**2)
        if L_O2C > (self.l_4 + self.l_3) or L_O2C < abs(self.l_4 - self.l_3):
            return None, None
            
        eta_2 = math.atan2(y_C, x_C2)
        psi_2 = math.acos((self.l_4**2 + L_O2C**2 - self.l_3**2) / (2.0 * self.l_4 * L_O2C))
        phi_4_target = eta_2 - psi_2  # 取外侧解（膝盖向前）

        # 4. 转换回电机命令系：减去 PI/2 的零点偏置
        motor1_cmd = phi_1_target
        motor2_cmd = phi_4_target
        
        return motor1_cmd, motor2_cmd

def robot_info_update(d):
    info = {}
    
    # 1. 提取六个电机位置 (rad) 和速度 (rad/s)
    info['pos'] = {
        'right1': d.sensor("right1_pos").data[0],
        'right2': d.sensor("right2_pos").data[0],
        'rightwheel': d.sensor("rightwheel_pos").data[0],
        'left1': d.sensor("left1_pos").data[0],
        'left2': d.sensor("left2_pos").data[0],
        'leftwheel': d.sensor("leftwheel_pos").data[0]
    }
    info['vel'] = {
        'right1': d.sensor("right1_vel").data[0],
        'right2': d.sensor("right2_vel").data[0],
        'rightwheel': d.sensor("rightwheel_vel").data[0],
        'left1': d.sensor("left1_vel").data[0],
        'left2': d.sensor("left2_vel").data[0],
        'leftwheel': d.sensor("leftwheel_vel").data[0]
    }
    
    # 2. 读取 IMU 四元数并转为欧拉角 (rad)
    w, x, y, z = d.sensor("orientation").data
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))  # 防止运算出界产生 NaN
    pitch = math.asin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    
    info['euler'] = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
    
    # 3. 读取 y 轴陀螺仪 (Pitch 相对应的角速度)
    info['gyro_pitch'] = d.sensor("angular_vel").data[1]
    
    return info

if __name__ == "__main__":
    five_links_right = five_links_param(dt=0.002)
    five_links_left = five_links_param(dt=0.002)
    m = mujoco.MjModel.from_xml_path('robot_pos_ctrl_for_vmc.xml')
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
        offset1 = -np.pi/2
        offset2 = -np.pi/2
        offset3 = np.pi/2
        offset4 = np.pi/2
        while viewer.is_running():
            # 正运动学验证

            # d.ctrl[0] = offset1 + 4*np.pi/6 # 竖直朝下，零点为顺时针转动90度
            # mujoco.mj_step(m, d) 
            # d.ctrl[1] = offset2 + np.pi/6  # 竖直朝下，零点为顺时针转动90度
            # mujoco.mj_step(m, d)


            # d.ctrl[2] = (offset3 - 4*np.pi/6)# 竖直朝下，零点本来为顺时针转动90度，但是对侧安装，所以需要逆时针转动90度，同时需要将转动角度逆向
            # mujoco.mj_step(m, d)
            # d.ctrl[3] = (offset4 - np.pi/6)  # 竖直朝下，零点本来为顺时针转动90度，但是对侧安装，所以需要逆时针转动90度，同时需要将转动角度逆向
            # mujoco.mj_step(m, d)  
            # d.ctrl[0] = offset1 + 3*np.pi/6 # 竖直朝下，零点为顺时针转动90度
            # mujoco.mj_step(m, d) 
            # d.ctrl[1] = offset2 + 3*np.pi/6  # 竖直朝下，零点为顺时针转动90度
            # mujoco.mj_step(m, d)


            # d.ctrl[2] = (offset3 - 3*np.pi/6)# 竖直朝下，零点本来为顺时针转动90度，但是对侧安装，所以需要逆时针转动90度，同时需要将转动角度逆向
            # mujoco.mj_step(m, d)
            # d.ctrl[3] = (offset4 - 3*np.pi/6)  # 竖直朝下，零点本来为顺时针转动90度，但是对侧安装，所以需要逆时针转动90度，同时需要将转动角度逆向
            # mujoco.mj_step(m, d)  
            # 逆运动学验证
            # cmd1,cmd2 = five_links_right.inverse_kinematics_cal_right(0.1,np.pi/2);
            five_links_right.forward_kinematics_cal(0,0,0,0,0)
            print(five_links_right.phi_0)
            # print(cmd1) # 1.446144432196033
            # print(cmd2) # -1.446144432196033
            five_links_right.phi_1 = np.pi/2 - np.pi/2  # 这里的0是竖直向下的，但是这个0在定义中是pi/2的位置，所以需要固定 减去-pi/2 因为这里即是编码器返回值，又会直接驱动电机运动，在VMC计算中我们需要注意                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
            d.ctrl[0] = five_links_right.phi_1
            mujoco.mj_step(m, d)
            five_links_right.phi_4 = -np.pi/2
            d.ctrl[1] = five_links_right.phi_4
            mujoco.mj_step(m, d)

            # cmd3,cmd4 = five_links_left.inverse_kinematics_cal_left(0.1,np.pi/2);
            five_links_left.forward_kinematics_cal(1,0,0,0,0)
            print(five_links_left.phi_0)
            # print(cmd3) # -1.446144432196033
            # print(cmd4) # 1.446144432196033
            five_links_left.phi_1 = -(np.pi - np.pi/2)
            d.ctrl[2] = five_links_left.phi_1
            mujoco.mj_step(m, d)
            five_links_left.phi_4 = -(0-np.pi/2)
            d.ctrl[3] = five_links_left.phi_4 
            mujoco.mj_step(m, d)  

            viewer.sync()


 