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
    current_v = 0.0  # 加入速度平滑过渡变量
    target_yaw = 0.0
    five_links_right = five_links_param(dt=dt)
    five_links_left = five_links_param(dt=dt)
    
    lqr_right = lqr_controller(dt=dt)
    lqr_left = lqr_controller(dt=dt)
    
    # YAW PID: kp=0.5, ki=0.0, kd=1.0, max_out=2.0
    left_yaw_pid = PIDController(0.5, 0.0, 1.0, 2.0, 0.0, feedforward=0.0)
    right_yaw_pid = PIDController(0.5, 0.0, 1.0, 2.0, 0.0, feedforward=0.0)
    
    # L0 PID: kp=100.0, ki=0.0, kd=1200.0, max_out=50.0
    left_L0_pid = PIDController(800.0, 0.0, 10.0, 100.0, 0.0, feedforward=0.0)
    right_L0_pid = PIDController(800.0, 0.0, 10.0, 100.0, 0.0, feedforward=0.0)
    
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
                        target_length = float(np.clip(target_length + 0.01, 0.10, 0.17))
                        print(f"目标腿长增加，当前腿长为: {target_length:.3f}")
                    elif event.key == pygame.K_MINUS:
                        target_length = float(np.clip(target_length - 0.01, 0.10, 0.17))
                        print(f"目标腿长减少，当前腿长为: {target_length:.3f}")

            # 直接查询键盘物理按键状态，无视操作系统的按键重复策略引起的顿挫
            keys_pressed = pygame.key.get_pressed()
            if keys_pressed[pygame.K_UP]:
                target_v = 0.3
            elif keys_pressed[pygame.K_DOWN]:
                target_v = -0.3
            else:
                target_v = 0.0

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
            
            # 速度平滑滤波 (一阶低通防止阶跃带来的加减速超调振荡)
            # 加速/减速的调节因子，设为 0.05 (较平滑)
            current_v += (target_v - current_v) * 0.02 
            if abs(current_v) < 0.001 and target_v == 0:
                current_v = 0.0

            # 完美追踪逻辑: 积分平滑后的速度作为目标位置
            lqr_left.x_b_set += current_v * dt
            lqr_right.x_b_set += current_v * dt
            
            lqr_left.v_set = current_v
            lqr_right.v_set = current_v
            pitch = info['euler']['pitch']
            pitch = -pitch
            yaw = info['euler']['yaw']
            gyro_pitch = info['gyro_pitch']
            gyro_pitch = -gyro_pitch

            # ---------------- 左腿控制 ----------------
            motor1_angle_left = info['pos']['left1'] # 通常 left2 为电机1或对侧, 需根据您系统调整
            motor2_angle_left = info['pos']['left2']

            motor_speed_left = info['vel']['leftwheel']
            motor_pos_left = info['pos']['leftwheel']
            five_links_left.forward_kinematics_cal(1, motor1_angle_left, motor2_angle_left, pitch, gyro_pitch)
            # print(five_links_left.phi_1,five_links_left.phi_4)
            lqr_left.states_update(1, five_links_left, pitch, gyro_pitch, motor_speed_left,motor_pos_left,wheel_radius)
            lqr_left.calc_k_matrix_from_poly(five_links_left.L_0)

            left_yaw_ctrl = left_yaw_pid.compute(yaw, 0.0)

            left_wheel_T = (lqr_left.K[0] * (lqr_left.theta - 0.0)
                          + lqr_left.K[2] * (lqr_left.d_theta - 0.0)
                          + lqr_left.K[4] * (lqr_left.x_b - lqr_left.x_b_set)
                          + lqr_left.K[6] * (lqr_left.v_b_whole - lqr_left.v_set)
                          + lqr_left.K[8] * (lqr_left.phi - 0.0)
                          + lqr_left.K[10] * (lqr_left.d_phi - 0.0))
            # left_wheel_T = -left_wheel_T + left_yaw_ctrl
            # left_wheel_T = -left_wheel_T 
            left_wheel_T = np.clip(left_wheel_T, -1.0, 1.0)

            left_F_0_ctrl = left_L0_pid.compute(five_links_left.L_0, target_length)
            # left_F_0 = (13 / math.cos(five_links_left.theta) if math.cos(five_links_left.theta) != 0 else 0) + left_F_0_ctrl
            left_F_0 = left_F_0_ctrl
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
            motor_speed_right = info['vel']['rightwheel']
            motor_pos_right = info['pos']['rightwheel'] 
            five_links_right.forward_kinematics_cal(0, motor1_angle_right, motor2_angle_right, pitch, gyro_pitch)
            lqr_right.states_update(0, five_links_right, pitch, gyro_pitch, motor_speed_right,motor_pos_right,wheel_radius)
            lqr_right.calc_k_matrix_from_poly(five_links_right.L_0)

            right_yaw_ctrl = right_yaw_pid.compute(yaw, 0.0)
            
            right_wheel_T = (lqr_right.K[0] * (lqr_right.theta - 0.0)
                           + lqr_right.K[2] * (lqr_right.d_theta - 0.0)
                           + lqr_right.K[4] * (lqr_right.x_b - lqr_right.x_b_set)
                           + lqr_right.K[6] * (lqr_right.v_b_whole - lqr_right.v_set)
                           + lqr_right.K[8] * (lqr_right.phi - 0.0)
                           + lqr_right.K[10] * (lqr_right.d_phi - 0.0))
                           
            # right_wheel_T = -right_wheel_T + right_yaw_ctrl
            # right_wheel_T = right_wheel_T
            right_wheel_T = np.clip(right_wheel_T, -1.0, 1.0)

            right_F_0_ctrl = right_L0_pid.compute(five_links_right.L_0, target_length)
         
            # right_F_0 = (13 / math.cos(five_links_right.theta) if math.cos(five_links_right.theta) != 0 else 0) + right_F_0_ctrl
            right_F_0 = right_F_0_ctrl
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
                d.ctrl[4] = right_wheel_T      # 右轮毂电机
                d.ctrl[5] = -left_wheel_T       # 左轮毂电机
            else:
                d.ctrl[0] = torque_set_right_0 # 右腿phi1对应的关节电机 
                d.ctrl[1] = torque_set_right_1 # 右腿phi4对应的关节电机
                d.ctrl[2] = -torque_set_left_0  # 左腿phi1对应的关节电机 
                d.ctrl[3] = -torque_set_left_1  # 左腿phi4对应的关节电机
                d.ctrl[4] = right_wheel_T      # 右轮毂电机
                d.ctrl[5] = -left_wheel_T       # 左轮毂电机

            mujoco.mj_step(m, d)
            viewer.sync()
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
