import time
import math
import mujoco
import mujoco.viewer
import numpy as np

from robot_mujoco_sensor_feedback.robot_info_update import robot_info_update
from lqr_update import lqr_controller
from vmc_cal import five_links_param

from remote.xbox_controller import PygameController 
from util.climb_traj import VelocityRampFilter 
from roll_leg_control import RollLegController
from ground_detection import detect_leg_ground_status

class PIDController:
    def __init__(self, kp, ki, kd, max_out, max_i, feedforward=0.0):
        self.kp = kp; self.ki = ki; self.kd = kd; self.max_out = max_out; self.max_i = max_i; self.feedforward = feedforward
        self.integral = 0.0; self.prev_error = 0.0
    def compute(self, current, target):
        error = target - current; self.integral += error
        self.integral = np.clip(self.integral, -self.max_i, self.max_i)
        derivative = (error - self.prev_error); self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative + self.feedforward
        return np.clip(output, -self.max_out, self.max_out)


# ==================== 封装的轮腿控制函数 ====================

def control_left_leg(info, five_links, lqr, yaw_pid, L0_pid, pitch, gyro_pitch, gyro_yaw, 
                     target_w, target_length, common_motor_pos, common_motor_speed, wheel_radius,
                     is_airborne_last_step, ground_detection_enabled, leg_tp_comp=0.0):
    """
    左腿 LQR + VMC 核心控制计算与条件离地保护机制
    :param ground_detection_enabled: 是否开启离地检测（即 teleop.control_enabled）
    """
    motor1_angle = info['pos']['left1'] 
    motor2_angle = info['pos']['left2']

    # 1. 运动学与状态更新
    five_links.forward_kinematics_cal(1, motor1_angle, motor2_angle, pitch, gyro_pitch)
    lqr.states_update(1, five_links, pitch, gyro_pitch, common_motor_pos, common_motor_speed, wheel_radius)
    lqr.calc_k_matrix_from_poly(five_links.L_0)

    # 2. VMC 虚拟力基本计算
    F_0_ctrl = L0_pid.compute(five_links.L_0, target_length)
    F_0 = (4 / math.cos(five_links.theta) if math.cos(five_links.theta) != 0 else 0) + F_0_ctrl
    F_0 = np.clip(F_0, -100.0, 100.0)

    # 预计算常规状况下的 Tp
    T_p = (lqr.K[1] * (lqr.theta - 0.0)
           + lqr.K[3] * (lqr.d_theta - 0.0)
           + lqr.K[5] * (lqr.x_b - lqr.x_b_set)
           + lqr.K[7] * (lqr.v_b_whole - lqr.v_set)
           + lqr.K[9] * (lqr.phi - 0.0)
           + lqr.K[11] * (lqr.d_phi - 0.0))

    # 3. 核心条件判定：只有在 Enter 激活后，才执行离地保护状态切换
    if ground_detection_enabled and (detect_leg_ground_status(five_links, F_0, T_p) == 1 or is_airborne_last_step):
        # 离地保护状态：轮子力矩清零，髋关节切换为空中阻尼姿态维持
        left_wheel_T = 0.0
        T_p = lqr.K[1] * (lqr.theta - 0.0) + lqr.K[3] * (lqr.d_theta - 0.0)
        T_p += leg_tp_comp
        flight_flag = 1
    else:
        # 正常控制状态（未开启 Enter 或者是处于着地状态）：正常的自平衡控制
        yaw_ctrl = yaw_pid.compute(gyro_yaw, target_w)
        wheel_T = (lqr.K[0] * (lqr.theta - 0.0)
                   + lqr.K[2] * (lqr.d_theta - 0.0)
                   + lqr.K[4] * (lqr.x_b - lqr.x_b_set)
                   + lqr.K[6] * (lqr.v_b_whole - lqr.v_set)
                   + lqr.K[8] * (lqr.phi - 0.0)
                   + lqr.K[10] * (lqr.d_phi - 0.0))
        
        left_wheel_T = -wheel_T - yaw_ctrl
        left_wheel_T = np.clip(left_wheel_T, -1.5, 1.5)
        flight_flag = 0

    # 4. 最终 VMC 力矩映射
    torque_set_0, torque_set_1 = five_links.VMC_torque_cal(F_0, T_p)
    
    return -torque_set_0, -torque_set_1, left_wheel_T, flight_flag


def control_right_leg(info, five_links, lqr, yaw_pid, L0_pid, pitch, gyro_pitch, gyro_yaw, 
                      target_w, target_length, common_motor_pos, common_motor_speed, wheel_radius,
                      is_airborne_last_step, ground_detection_enabled, leg_tp_comp=0.0):
    """
    右腿 LQR + VMC 核心控制计算与条件离地保护机制
    """
    motor1_angle = info['pos']['right1']
    motor2_angle = info['pos']['right2']
    
    # 1. 运动学与状态更新
    five_links.forward_kinematics_cal(0, motor1_angle, motor2_angle, pitch, gyro_pitch)
    lqr.states_update(0, five_links, pitch, gyro_pitch, -common_motor_pos, -common_motor_speed, wheel_radius)
    lqr.calc_k_matrix_from_poly(five_links.L_0)

    # 2. VMC 虚拟力基本计算
    F_0_ctrl = L0_pid.compute(five_links.L_0, target_length)
    F_0 = (4 / math.cos(five_links.theta) if math.cos(five_links.theta) != 0 else 0) + F_0_ctrl
    F_0 = np.clip(F_0, -100.0, 100.0)

    # 预计算常规状况下的 Tp
    T_p = (lqr.K[1] * (lqr.theta - 0.0)
           + lqr.K[3] * (lqr.d_theta - 0.0)
           + lqr.K[5] * (lqr.x_b - lqr.x_b_set)
           + lqr.K[7] * (lqr.v_b_whole - lqr.v_set)
           + lqr.K[9] * (lqr.phi - 0.0)
           + lqr.K[11] * (lqr.d_phi - 0.0))
    
    # 3. 核心条件判定
    if ground_detection_enabled and (detect_leg_ground_status(five_links, F_0, T_p) == 1 or is_airborne_last_step):
        right_wheel_T = 0.0
        T_p = lqr.K[1] * (lqr.theta - 0.0) + lqr.K[3] * (lqr.d_theta - 0.0)
        T_p += leg_tp_comp
        flight_flag = 1
    else:
        yaw_ctrl = yaw_pid.compute(gyro_yaw, target_w)
        wheel_T = (lqr.K[0] * (lqr.theta - 0.0)
                   + lqr.K[2] * (lqr.d_theta - 0.0)
                   + lqr.K[4] * (lqr.x_b - lqr.x_b_set)
                   + lqr.K[6] * (lqr.v_b_whole - lqr.v_set)
                   + lqr.K[8] * (lqr.phi - 0.0)
                   + lqr.K[10] * (lqr.d_phi - 0.0))
                   
        right_wheel_T = wheel_T - yaw_ctrl
        right_wheel_T = np.clip(right_wheel_T, -1.5, 1.5)
        flight_flag = 0

    torque_set_0, torque_set_1 = five_links.VMC_torque_cal(F_0, T_p)
    
    return torque_set_0, torque_set_1, right_wheel_T, flight_flag


# ============================================================

if __name__ == "__main__":
    dt = 0.002
    wheel_radius = 0.06
    
    target_length_left = 0.1
    target_length_right = 0.1
    roll = 0.0
    current_v = 0.0 
    
    # 状态标志：记录整车上一周期是否触发了空中保护状态
    robot_is_airborne = False
    
    five_links_right = five_links_param(dt=dt)
    five_links_left = five_links_param(dt=dt)
    lqr_right = lqr_controller(dt=dt)
    lqr_left = lqr_controller(dt=dt)
    
    left_yaw_pid = PIDController(0.5, 0.0, 1.0, 2.0, 0.0, feedforward=0.0)
    right_yaw_pid = PIDController(0.5, 0.0, 1.0, 2.0, 0.0, feedforward=0.0)
    left_L0_pid = PIDController(700.0, 0.0, 5.0, 150.0, 0.0, feedforward=0.0)
    right_L0_pid = PIDController(700.0, 0.0, 5.0, 150.0, 0.0, feedforward=0.0)
    
    m = mujoco.MjModel.from_xml_path('mjcf/robot.xml')
    d = mujoco.MjData(m)

    teleop = PygameController()
    v_filter = VelocityRampFilter(dt=dt, accel_rate=0.8, decel_rate=1.4)
    roll_balancer = RollLegController(kp=1.0, ki=0.0, kd=0.0, max_out=0.14, min_leg=0.10, max_leg=0.14)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            teleop.update(
                current_v=current_v, 
                target_length_left=target_length_left, 
                target_length_right=target_length_right, 
                roll=roll, 
                viewer=viewer
            )
            
            if teleop.paused:
                viewer.sync()
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                continue

            # 获取统一传感器信息
            info = robot_info_update(d)
            
            pitch = -info['euler']['pitch']
            yaw = info['euler']['yaw']
            roll = -info['euler']['roll']
            
            gyro_pitch = -info['gyro_pitch']
            gyro_yaw = info['gyro_yaw']  
            gyro_roll = -info['gyro_roll']
            
            # 使用封装后的 Roll 平衡控制器
            target_length_left, target_length_right = roll_balancer.compute_leg_lengths(
                current_roll=roll,
                target_roll=0.0,
                base_length=teleop.target_length
            )

            # --- 共模提取 ---
            motor_speed_left = info['vel']['leftwheel']
            motor_speed_right = info['vel']['rightwheel']
            motor_pos_left = info['pos']['leftwheel']
            motor_pos_right = info['pos']['rightwheel']
            
            common_motor_speed = (motor_speed_left - motor_speed_right) / 2.0
            common_motor_pos = (motor_pos_left - motor_pos_right) / 2.0

            # ---------------- 离地保护中的：位移、速度期望清零 ----------------
            # 只有在 Enter 开关激活 (teleop.control_enabled 为 True) 并且检测到空中状态时才重置
            if teleop.control_enabled and robot_is_airborne:
                lqr_left.x_b_set = (common_motor_pos * wheel_radius)
                lqr_right.x_b_set = (common_motor_pos * wheel_radius)
                current_v = 0.0
                teleop.target_w = 0.0  
            else:
                # 正常平衡状态（无论是否按 Enter 键，都可以通过爬坡滤波器正常运动平衡）
                current_v = v_filter.update(current_v=current_v, target_v=teleop.target_v)
                if not (current_v == 0.0 and teleop.target_v == 0):
                    lqr_left.x_b_set = (common_motor_pos * wheel_radius)
                    lqr_right.x_b_set = (common_motor_pos * wheel_radius) 
            
            lqr_left.v_set = current_v
            lqr_right.v_set = current_v

            # ---------------- 调用左腿控制函数（传入检测使能锁） ----------------
            ctrl_left_0, ctrl_left_1, left_wheel_T, left_flight = control_left_leg(
                info, five_links_left, lqr_left, left_yaw_pid, left_L0_pid,
                pitch, gyro_pitch, gyro_yaw, teleop.target_w, target_length_left,
                common_motor_pos, common_motor_speed, wheel_radius,
                is_airborne_last_step=robot_is_airborne,
                ground_detection_enabled=teleop.control_enabled, # 按 Enter 开启保护
                leg_tp_comp=0.0
            )

            # ---------------- 调用右腿控制函数（传入检测使能锁） ----------------
            ctrl_right_0, ctrl_right_1, right_wheel_T, right_flight = control_right_leg(
                info, five_links_right, lqr_right, right_yaw_pid, right_L0_pid,
                pitch, gyro_pitch, gyro_yaw, teleop.target_w, target_length_right,
                common_motor_pos, common_motor_speed, wheel_radius,
                is_airborne_last_step=robot_is_airborne,
                ground_detection_enabled=teleop.control_enabled, # 按 Enter 开启保护
                leg_tp_comp=0.0
            )

            # ---------------- 整体离地状态判断汇总 ----------------
            if teleop.control_enabled:
                robot_is_airborne = (left_flight == 1) or (right_flight == 1)
            else:
                robot_is_airborne = False

            # ---------------- 整体执行执行器下发（任何时候都保持自平衡控制输出） ----------------
            # 不受 teleop.control_enabled 的直接拦截，任何时候力矩均可以下发
            d.ctrl[0] = ctrl_right_0 
            d.ctrl[1] = ctrl_right_1 
            d.ctrl[2] = ctrl_left_0  
            d.ctrl[3] = ctrl_left_1  
            d.ctrl[4] = right_wheel_T       
            d.ctrl[5] = left_wheel_T        

            mujoco.mj_step(m, d)
            viewer.sync()
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)