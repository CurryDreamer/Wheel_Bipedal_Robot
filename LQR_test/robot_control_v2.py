import time
import math

import pygame
import mujoco
import mujoco.viewer
import numpy as np

from robot_info_update import robot_info_update
from robot_lqr_update import lqr_controller
from robot_vmc_cal import five_links_param

class PIDController:
    def __init__(self, kp, ki, kd, max_out, max_i, feedforward=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.max_i = max_i
        self.feedforward = feedforward
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, current, target):
        error = target - current
        self.integral += error
        self.integral = np.clip(self.integral, -self.max_i, self.max_i)
        derivative = (error - self.prev_error)
        self.prev_error = error
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative + self.feedforward
        return np.clip(output, -self.max_out, self.max_out)

if __name__ == "__main__":
    dt = 0.002
    wheel_radius = 0.06
    target_length = 0.1
    target_v = 0.0
    current_v = 0.0 
    target_w = 0.0
    current_w =0.0
    target_yaw = 0.0
    five_links_right = five_links_param(dt=dt)
    five_links_left = five_links_param(dt=dt)
    
    lqr_right = lqr_controller(dt=dt)
    lqr_left = lqr_controller(dt=dt)
    
    # YAW PID: kp=0.5, ki=0.0, kd=1.0, max_out=2.0
    left_yaw_pid = PIDController(0.5, 0.0, 1.0, 2.0, 0.0, feedforward=0.0)
    right_yaw_pid = PIDController(0.5, 0.0, 1.0, 2.0, 0.0, feedforward=0.0)
    
    # L0 PID: kp=100.0, ki=0.0, kd=1200.0, max_out=50.0
    left_L0_pid = PIDController(500.0, 0.0, 5.0, 150.0, 0.0, feedforward=0.0)
    right_L0_pid = PIDController(500.0, 0.0, 5.0, 150.0, 0.0, feedforward=0.0)
    
    m = mujoco.MjModel.from_xml_path('robot.xml')
    d = mujoco.MjData(m)

    paused = False
    control_enabled = False

    # ---------------- Pygame 按键监听配置 ----------------
    pygame.init()
    # 创建一个小面板捕获焦点
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Robot Control (Focus Here!)")
    font = pygame.font.SysFont(None, 32)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # --- 处理 Pygame 事件 ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    viewer.close()
                    pygame.quit()
                    exit(0)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_RETURN:
                        control_enabled = not control_enabled
                        print(f"当前控制状态: {'开启 (ON)' if control_enabled else '关闭 (OFF)'}")
                    elif event.key in [pygame.K_EQUALS, pygame.K_PLUS]:
                        target_length = float(np.clip(target_length + 0.01, 0.10, 0.14))
                        print(f"目标腿长增加，当前腿长为: {target_length:.3f}")
                    elif event.key == pygame.K_MINUS:
                        target_length = float(np.clip(target_length - 0.01, 0.10, 0.14))
                        print(f"目标腿长减少，当前腿长为: {target_length:.3f}")

            # 直接查询键盘物理按键状态，无视操作系统的按键重复策略引起的顿挫
            keys_pressed = pygame.key.get_pressed()
            if keys_pressed[pygame.K_UP]:
                target_v = 0.6
            elif keys_pressed[pygame.K_DOWN]:
                target_v = -0.6
            else:
                target_v = 0.0
                
            if keys_pressed[pygame.K_LEFT]:
                target_w = 2.  # 期望的偏航角速度(rad/s)
            elif keys_pressed[pygame.K_RIGHT]:
                target_w = -2.
            else:
                target_w = 0.0
                
            # print(target_w)
            # 刷新控制面板UI
            screen.fill((30, 30, 30))
            text1 = font.render(f"Control (ENTER) : {'ON' if control_enabled else 'OFF'}", True, (255, 255, 255))
            text2 = font.render(f"Target V (UP/DOWN): {target_v:.2f} | Cur: {current_v:.2f}", True, (255, 255, 255))
            text3 = font.render(f"Leg Len (+/-)     : {target_length:.3f}", True, (255, 255, 255))
            text4 = font.render(f"Paused (SPACE)    : {paused}", True, (255, 255, 255))
            screen.blit(text1, (20, 30))
            screen.blit(text2, (20, 80))
            screen.blit(text3, (20, 130))
            screen.blit(text4, (20, 180))
            pygame.display.flip()
            
            if paused:
                viewer.sync()
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                continue

            # 获取统一传感器信息
            info = robot_info_update(d)
            
            pitch = info['euler']['pitch']
            pitch = -pitch
            yaw = info['euler']['yaw']
            
            gyro_pitch = info['gyro_pitch']
            gyro_pitch = -gyro_pitch
            gyro_yaw = info['gyro_yaw']  # 用于自转闭环

            # --- 共模提取 (去除转向差模对 LQR 的干扰) ---
            motor_speed_left = info['vel']['leftwheel']
            motor_speed_right = info['vel']['rightwheel']
            motor_pos_left = info['pos']['leftwheel']
            motor_pos_right = info['pos']['rightwheel']
            
            # 由于左右轮电机安装方向相反（在 robot_lqr_update.py 中右轮多乘了一个负号）
            # 所以真实的物理前进方向速度是：左轮为正是前进，右轮为负是前进
            # 共模速度 (用来更新 LQR 的 v_b_whole)
            common_motor_speed = (motor_speed_left - motor_speed_right) / 2.0
            # 共模位置 (用来更新 LQR 的 x_b)
            common_motor_pos = (motor_pos_left - motor_pos_right) / 2.0

            # 线性速度爬坡 (通过限制最大加速度来平滑目标速度)
            velocity_ramp_rate = 0.3 
            max_step_size = abs(dt * velocity_ramp_rate)
            full_step = target_v - current_v
            step = np.clip(full_step, -max_step_size, max_step_size)
            current_v += step
            
            if abs(current_v) < 0.001 and target_v == 0:
                current_v = 0.0
                # 静止驻车时，不再积分 x_b_set，让系统根据位移误差原地锁止以抵抗外力
            else:
                # 运动状态下，直接读取当前共模位移，消除因为差速自转积累的左右轮异向误差
                # 注意：因为 states_update 内部对右腿(leg_flag=0)的 x_b 会乘负号： self.x_b = -motor_pos*wheel_radius
                # 我们传入 right_states_update 的是 -common_motor_pos，意味着它内部存的 x_b 就是 +common_motor_pos
                # 所以右侧的 x_b_set 必须和左侧完全相同（正的 common），否则一运动就会算出巨大的相反位置误差导致起步劈叉！
                lqr_left.x_b_set = (common_motor_pos * wheel_radius)
                lqr_right.x_b_set = (common_motor_pos * wheel_radius) 
            
            lqr_left.v_set = current_v
            lqr_right.v_set = current_v

            # ---------------- 左腿控制 ----------------
            motor1_angle_left = info['pos']['left1'] # 通常 left2 为电机1或对侧, 需根据您系统调整
            motor2_angle_left = info['pos']['left2']

            five_links_left.forward_kinematics_cal(1, motor1_angle_left, motor2_angle_left, pitch, gyro_pitch)
            # 传给 LQR 的不再是原始左轮速度，而是共模平均速度
            lqr_left.states_update(1, five_links_left, pitch, gyro_pitch, common_motor_pos, common_motor_speed, wheel_radius)
            lqr_left.calc_k_matrix_from_poly(five_links_left.L_0)

            # 自转偏航控制 (计算差模力矩)
            left_yaw_ctrl = left_yaw_pid.compute(gyro_yaw, target_w)

            left_wheel_T = (lqr_left.K[0] * (lqr_left.theta - 0.0)
                          + lqr_left.K[2] * (lqr_left.d_theta - 0.0)
                          + lqr_left.K[4] * (lqr_left.x_b - lqr_left.x_b_set)
                          + lqr_left.K[6] * (lqr_left.v_b_whole - lqr_left.v_set)
                          + lqr_left.K[8] * (lqr_left.phi - 0.0)
                          + lqr_left.K[10] * (lqr_left.d_phi - 0.0))
            
            left_F_0_ctrl = left_L0_pid.compute(five_links_left.L_0, target_length)
            left_F_0 = (8 / math.cos(five_links_left.theta) if math.cos(five_links_left.theta) != 0 else 0) + left_F_0_ctrl
            # left_F_0 = left_F_0_ctrl
            left_F_0 = np.clip(left_F_0, -100.0, 100.0)

            left_T_p = (lqr_left.K[1] * (lqr_left.theta - 0.0)
                      + lqr_left.K[3] * (lqr_left.d_theta - 0.0)
                      + lqr_left.K[5] * (lqr_left.x_b - lqr_left.x_b_set)
                      + lqr_left.K[7] * (lqr_left.v_b_whole - lqr_left.v_set)
                      + lqr_left.K[9] * (lqr_left.phi - 0.0)
                      + lqr_left.K[11] * (lqr_left.d_phi - 0.0))
            torque_set_left_0, torque_set_left_1 = five_links_left.VMC_torque_cal(left_F_0, left_T_p)


            # ---------------- 右腿控制 ----------------
            motor1_angle_right = info['pos']['right1']
            motor2_angle_right = info['pos']['right2']
            
            five_links_right.forward_kinematics_cal(0, motor1_angle_right, motor2_angle_right, pitch, gyro_pitch)
            # 传给 LQR 的不再是原始右轮速度，而是共模平均速度（注意符号，为了兼容 robot_lqr_update 里对 leg_flag==0 的取反，这里传负的 common）
            lqr_right.states_update(0, five_links_right, pitch, gyro_pitch, -common_motor_pos, -common_motor_speed, wheel_radius)
            lqr_right.calc_k_matrix_from_poly(five_links_right.L_0)

            right_yaw_ctrl = right_yaw_pid.compute(gyro_yaw, target_w)
            
            right_wheel_T = (lqr_right.K[0] * (lqr_right.theta - 0.0)
                           + lqr_right.K[2] * (lqr_right.d_theta - 0.0)
                           + lqr_right.K[4] * (lqr_right.x_b - lqr_right.x_b_set)
                           + lqr_right.K[6] * (lqr_right.v_b_whole - lqr_right.v_set)
                           + lqr_right.K[8] * (lqr_right.phi - 0.0)
                           + lqr_right.K[10] * (lqr_right.d_phi - 0.0))
                           
            # 引入偏航差速控制扭矩
            left_wheel_T = -left_wheel_T - left_yaw_ctrl
            right_wheel_T = right_wheel_T - right_yaw_ctrl

            left_wheel_T = np.clip(left_wheel_T, -1.5, 1.5)
            right_wheel_T = np.clip(right_wheel_T, -1.5, 1.5)

            right_F_0_ctrl = right_L0_pid.compute(five_links_right.L_0, target_length)
         
            right_F_0 = (8 / math.cos(five_links_right.theta) if math.cos(five_links_right.theta) != 0 else 0) + right_F_0_ctrl
            # right_F_0 = right_F_0_ctrl
            right_F_0 = np.clip(right_F_0, -100.0, 100.0)

            right_T_p = (lqr_right.K[1] * (lqr_right.theta - 0.0)
                       + lqr_right.K[3] * (lqr_right.d_theta - 0.0)
                       + lqr_right.K[5] * (lqr_right.x_b - lqr_right.x_b_set)
                       + lqr_right.K[7] * (lqr_right.v_b_whole - lqr_right.v_set)
                       + lqr_right.K[9] * (lqr_right.phi - 0.0)
                       + lqr_right.K[11] * (lqr_right.d_phi - 0.0))
            # print(lqr_right.d_phi,lqr_left.d_phi)
            torque_set_right_0, torque_set_right_1 = five_links_right.VMC_torque_cal(right_F_0, right_T_p)
            # print(five_links_left.L_0,five_links_right.L_0)
            # print(five_links_left.theta,five_links_right.theta)
            # ---------------- 整体执行执行器下发 ----------------
            if control_enabled:
                d.ctrl[0] = torque_set_right_0 # 右腿phi1对应的关节电机 
                d.ctrl[1] = torque_set_right_1 # 右腿phi4对应的关节电机
                d.ctrl[2] = -torque_set_left_0  # 左腿phi1对应的关节电机 
                d.ctrl[3] = -torque_set_left_1  # 左腿phi4对应的关节电机
                d.ctrl[4] = right_wheel_T       # 右轮毂电机下发时已经是正确叠加的力矩
                d.ctrl[5] = left_wheel_T        # 左轮毂电机在上方已经包含了反向，所以这里直接=即可
            else:
                d.ctrl[0] = torque_set_right_0 
                d.ctrl[1] = torque_set_right_1 
                d.ctrl[2] = -torque_set_left_0  
                d.ctrl[3] = -torque_set_left_1  
                d.ctrl[4] = right_wheel_T      
                d.ctrl[5] = left_wheel_T       

            mujoco.mj_step(m, d)
            viewer.sync()
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
