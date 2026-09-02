import time
import math
import mujoco
import mujoco.viewer
import numpy as np
import os
import csv
import atexit
from datetime import datetime

from robot_mujoco_sensor_feedback.robot_info_update import robot_info_update
from lqr_update import lqr_controller
from vmc_cal import five_links_param

from remote.xbox_controller import PygameController 
from util.climb_traj import VelocityRampFilter 
from roll_leg_control import RollLegController
from ground_detection import LegGroundDetector

# 触地 debounce：连续 N 帧检测到着地才确认触地恢复（dt=0.002, 10帧=20ms）
TOUCHDOWN_DEBOUNCE_FRAMES = 5
# 仿真速度倍率：1.0=等速实时，>1.0=加速，<1.0=减速
SIM_SPEED = 1.0
# ROLL 平衡器总开关：False=完全关闭 roll 平衡（对照实验用）。
# 对照实验已确认本机 roll 自稳、高增益平衡器会引发发散，现改用低增益方案并重新启用。
ROLL_BALANCER_ENABLED = True

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
                     is_airborne_last_step, ground_detection_enabled, ground_detector, leg_tp_comp=0.0,
                     touchdown_count=0, vehicle_airborne=False, roll_fast_force=0.0):
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
    # roll 快通道差动力：直接叠加到轴向力上，快速压制 roll 动态（对齐固件 ctrl.F_l = ... + fast_force_l）
    F_0 = 5 + F_0_ctrl + roll_fast_force
    F_0 = np.clip(F_0, -100.0, 100.0)

    # 预计算常规状况下的 Tp
    T_p = (lqr.K[1] * (lqr.theta - 0.0)
           + lqr.K[3] * (lqr.d_theta - 0.0)
           + lqr.K[5] * (lqr.x_b - lqr.x_b_set)
           + lqr.K[7] * (lqr.v_b_whole - lqr.v_set)
           + lqr.K[9] * (lqr.phi - 0.0)
           + lqr.K[11] * (lqr.d_phi - 0.0))

    # 3. 逆VMC：从上一周期实际髋关节力矩反馈反算真实虚拟力，用于离地检测
    T1_fb = info['trq']['left1']
    T2_fb = info['trq']['left2']
    F_actual, T_p_actual = five_links.inverse_VMC_torque_cal(-T1_fb, -T2_fb)

    # 4. 触地 debounce 判定（检测器内部已做低通滤波 + 双阈值迟滞）
    # 传入目标腿长：判定用目标腿长算 FN，避免腿被地形压伸长时 T_p 项被稀释造成误判
    is_airborne_now = ground_detector.update(five_links, F_actual, T_p_actual, target_l0=target_length) == 1

    if is_airborne_now:
        # 检测到离地 → 重置触地计数器，进入空中保护
        touchdown_count = 0
        in_airborne = True
    elif is_airborne_last_step:
        # 上一帧在空中，当前帧 FN>=5.0 → 潜在触地，开始 debounce
        touchdown_count += 1
        if touchdown_count >= TOUCHDOWN_DEBOUNCE_FRAMES:
            # 连续 N 帧着地，确认触地恢复
            touchdown_count = 0
            in_airborne = False
        else:
            # debounce 未完成，维持空中保护
            in_airborne = True
    else:
        # 既未离地也非上一帧在空中 → 正常着地状态
        in_airborne = False

    if ground_detection_enabled and in_airborne:
        # 离地保护状态：轮子力矩清零，髋关节切换为空中阻尼姿态维持
        left_wheel_T = 0.0
        T_p = lqr.K[1] * (lqr.theta - 0.0) + lqr.K[3] * (lqr.d_theta - 0.0)
        T_p += leg_tp_comp
        flight_flag = 1
    else:
        # 正常控制状态（未开启 Enter 或者是处于着地状态）：正常的自平衡控制
        # 整车处于离地保护态时，卸载 yaw：单腿离地会因单侧轮力矩失衡产生 yaw 旋转，
        # yaw_pid 纠偏会给着地腿轮子叠加大 yaw 力矩，抢占唯一扛 pitch 的轮子的力矩预算
        # → 加速 pitch 饱和发散。故此时 yaw_ctrl 置 0，让着地轮专心扛 pitch。
        if ground_detection_enabled and vehicle_airborne:
            yaw_ctrl = 0.0
            yaw_pid.integral = 0.0  # 同时清 yaw 积分，避免离地期间累积、恢复时回抽
        else:
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

    # 5. 最终 VMC 力矩映射
    torque_set_0, torque_set_1 = five_links.VMC_torque_cal(F_0, T_p)
    
    return -torque_set_0, -torque_set_1, left_wheel_T, flight_flag, touchdown_count


def control_right_leg(info, five_links, lqr, yaw_pid, L0_pid, pitch, gyro_pitch, gyro_yaw,
                      target_w, target_length, common_motor_pos, common_motor_speed, wheel_radius,
                      is_airborne_last_step, ground_detection_enabled, ground_detector, leg_tp_comp=0.0,
                      touchdown_count=0, vehicle_airborne=False, roll_fast_force=0.0):
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
    # roll 快通道差动力（右腿取相反符号，由调用方传入已含符号的值）
    F_0 = 5 + F_0_ctrl + roll_fast_force
    F_0 = np.clip(F_0, -100.0, 100.0)

    # 预计算常规状况下的 Tp
    T_p = (lqr.K[1] * (lqr.theta - 0.0)
           + lqr.K[3] * (lqr.d_theta - 0.0)
           + lqr.K[5] * (lqr.x_b - lqr.x_b_set)
           + lqr.K[7] * (lqr.v_b_whole - lqr.v_set)
           + lqr.K[9] * (lqr.phi - 0.0)
           + lqr.K[11] * (lqr.d_phi - 0.0))
    
    # 3. 逆VMC：从上一周期实际髋关节力矩反馈反算真实虚拟力，用于离地检测
    T1_fb = info['trq']['right1']
    T2_fb = info['trq']['right2']
    F_actual, T_p_actual = five_links.inverse_VMC_torque_cal(T1_fb, T2_fb)

    # 4. 触地 debounce 判定（检测器内部已做低通滤波 + 双阈值迟滞）
    # 传入目标腿长：判定用目标腿长算 FN，避免腿被地形压伸长时 T_p 项被稀释造成误判
    is_airborne_now = ground_detector.update(five_links, F_actual, T_p_actual, target_l0=target_length) == 1

    if is_airborne_now:
        touchdown_count = 0
        in_airborne = True
    elif is_airborne_last_step:
        touchdown_count += 1
        if touchdown_count >= TOUCHDOWN_DEBOUNCE_FRAMES:
            touchdown_count = 0
            in_airborne = False
        else:
            in_airborne = True
    else:
        in_airborne = False

    if ground_detection_enabled and in_airborne:
        right_wheel_T = 0.0
        T_p = lqr.K[1] * (lqr.theta - 0.0) + lqr.K[3] * (lqr.d_theta - 0.0)
        T_p += leg_tp_comp
        flight_flag = 1
    else:
        # 整车离地保护态时卸载 yaw（理由同左腿）：让着地轮专心扛 pitch
        if ground_detection_enabled and vehicle_airborne:
            yaw_ctrl = 0.0
            yaw_pid.integral = 0.0
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
    
    return torque_set_0, torque_set_1, right_wheel_T, flight_flag, touchdown_count


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
    # 触地 debounce 计数器（左右腿各自独立）
    touchdown_count_left = 0
    touchdown_count_right = 0
    # 每条腿上一帧的 flight 状态（各自独立做 debounce，避免两腿用整车状态互相重置计数器造成死锁）
    left_flight_last = 0
    right_flight_last = 0
    
    five_links_right = five_links_param(dt=dt)
    five_links_left = five_links_param(dt=dt)
    lqr_right = lqr_controller(dt=dt)
    lqr_left = lqr_controller(dt=dt)

    # 左右腿各自独立的离地检测器（内部维护低通滤波值与迟滞状态）
    ground_detector_left = LegGroundDetector()
    ground_detector_right = LegGroundDetector()
    
    left_yaw_pid = PIDController(0.45, 0.0, 1.0, 2.0, 0.0, feedforward=0.0)
    right_yaw_pid = PIDController(0.45, 0.0, 1.0, 2.0, 0.0, feedforward=0.0)
    left_L0_pid = PIDController(1200.0, 0.0, 20.0, 150.0, 0.0, feedforward=0.0)
    right_L0_pid = PIDController(1200.0, 0.0, 20.0, 150.0, 0.0, feedforward=0.0)
    
    m = mujoco.MjModel.from_xml_path('mjcf/robot.xml')
    d = mujoco.MjData(m)

    teleop = PygameController()
    v_filter = VelocityRampFilter(dt=dt, accel_rate=0.8, decel_rate=1.4)
    roll_balancer = RollLegController(kp=0.0, ki=0.0001, kd=0.0, max_out=0.03, min_leg=0.10, max_leg=0.14, dt=dt,
                                      fast_kp=0.5, fast_rate_damp=1.0, fast_force_limit=16.0,
                                      fast_force_slew=150.0, fast_sign=1.0, slow_deadzone=0.05)

    # ---------------- 离地检测日志 ----------------
    # 记录飞坡/单边桥等场景下的离地检测数据，供离线分析。每帧一行，写到 log/ground_YYYYmmdd_HHMMSS.csv
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"ground_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "t", "torso_z", "pitch", "roll", "gyro_pitch",
        "ctrl_enabled", "robot_airborne",
        # 左腿：fn_raw=用实测L0, fn_raw_tgt=用目标L0, tgt_len=目标腿长
        "L_fn_raw", "L_fn_raw_tgt", "L_fn_lpf", "L_air", "L_flight",
        "L0_left", "L_tgt_len", "theta_left",
        "L_xb", "L_xbset", "L_wheelT", "tc_left",
        # 右腿
        "R_fn_raw", "R_fn_raw_tgt", "R_fn_lpf", "R_air", "R_flight",
        "L0_right", "R_tgt_len", "theta_right",
        "R_xb", "R_xbset", "R_wheelT", "tc_right",
        # roll 快通道（差动轴向力）+ roll 角速度，用于 roll 调参
        "gyro_roll", "roll_ffL", "roll_ffR",
    ])
    log_frame = 0
    print(f"[LOG] 离地检测日志: {log_path}")

    # 兜底：Ctrl+C 或异常退出时 with 块下方的显式 flush/close 不会执行，
    # 靠 atexit 保证日志一定落盘、句柄一定关闭；正常退出路径已提前 close，这里判空跳过。
    def _flush_close_log():
        if not log_file.closed:
            log_file.flush()
            log_file.close()
            print(f"[LOG] 已保存离地检测日志（共 {log_frame} 帧）: {log_path}")
    atexit.register(_flush_close_log)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 相机追踪 torso
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = m.body('torso').id
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 60
        viewer.cam.lookat = [0, 0, 0.1]

        # FPS 监控 + 真实时间驱动物理
        fps_frame_count = 0
        fps_last_time = time.time()
        fps_display_interval = 2.0  # 每2秒打印一次
        sim_time = 0.0              # 仿真累计时间
        last_real_time = time.time()

        while viewer.is_running():
            step_start = time.time()
            # 最外层：遥控器
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

            roll = -info['euler']['roll'] 
            pitch = -info['euler']['pitch']
            yaw = info['euler']['yaw'] 

            gyro_roll = -info['gyro_roll']
            gyro_pitch = -info['gyro_pitch']
            gyro_yaw = info['gyro_yaw']
            # 使用封装后的 Roll 平衡控制器（内建 roll 差模软卸载）
            # 离地腿的 roll 差模修正会被快降慢升地软卸载：悬空腿调腿长纠不了 roll，
            # 硬调只会放大两腿轴向力差、造成 FN 反相震荡→翻倒。软卸载相比硬冻结的好处：
            # 基础腿长(遥控目标)始终生效，不与手动调腿长打架横跳；且左右独立只卸载离地侧。
            # 传上一帧的左右腿离地状态（roll 在循环开头调用，本帧 flight 尚未算出）。
            if ROLL_BALANCER_ENABLED:
                roll_gate_on = teleop.control_enabled
                target_length_left, target_length_right = roll_balancer.compute_leg_lengths(
                    current_roll=roll,
                    target_roll=0.0,
                    base_length=teleop.target_length,
                    left_airborne=(roll_gate_on and left_flight_last == 1),
                    right_airborne=(roll_gate_on and right_flight_last == 1),
                )
                # roll 快通道：直接差动轴向力，快速压制 roll 动态（慢通道只补静态偏置）。
                # 必须在 compute_leg_lengths 之后调用——快通道复用其更新的左右卸载系数 scale。
                roll_fast_force_left, roll_fast_force_right = roll_balancer.compute_fast_force(
                    current_roll=roll,
                    target_roll=0.0,
                    roll_speed=gyro_roll,
                    left_airborne=(roll_gate_on and left_flight_last == 1),
                    right_airborne=(roll_gate_on and right_flight_last == 1),
                )
            else:
                # 对照实验：完全关闭 roll 平衡，两腿同基础腿长、快通道不出力
                target_length_left = teleop.target_length
                target_length_right = teleop.target_length
                roll_fast_force_left = 0.0
                roll_fast_force_right = 0.0

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
            # 传入的参数为
            '''
                info: 电机编码器数据
                five_links: 五连杆参数 正运动学计算
                lqr: 传入的LQR控制器参数
                left_yaw_pid: yaw计算的pd
                left_L0_pid: 腿长计算的pid
                pitch gyro_pitch gyro_yaw: imu数据
                teleop.target_w: 遥控器传入的自转
                target_length_left: 目标腿长
                common_motor_pos: 两边轮子平均位移
                common_motor_speed: 两边轮子平均速度
                wheel_radius: 轮毂半径
                is_airborne_last_step ground_detection_enabled leg_tp_comp: 离地检测相关

            '''
            ctrl_left_0, ctrl_left_1, left_wheel_T, left_flight, touchdown_count_left = control_left_leg(
                info, five_links_left, lqr_left, left_yaw_pid, left_L0_pid,
                pitch, gyro_pitch, gyro_yaw, teleop.target_w, target_length_left,
                common_motor_pos, common_motor_speed, wheel_radius,
                is_airborne_last_step=(left_flight_last == 1),
                ground_detection_enabled=teleop.control_enabled, # 按 Enter 开启保护
                ground_detector=ground_detector_left,
                leg_tp_comp=0.0,
                touchdown_count=touchdown_count_left,
                vehicle_airborne=robot_is_airborne,
                roll_fast_force=roll_fast_force_left
            )

            # ---------------- 调用右腿控制函数（传入检测使能锁） ----------------
            ctrl_right_0, ctrl_right_1, right_wheel_T, right_flight, touchdown_count_right = control_right_leg(
                info, five_links_right, lqr_right, right_yaw_pid, right_L0_pid,
                pitch, gyro_pitch, gyro_yaw, teleop.target_w, target_length_right,
                common_motor_pos, common_motor_speed, wheel_radius,
                is_airborne_last_step=(right_flight_last == 1),
                ground_detection_enabled=teleop.control_enabled, # 按 Enter 开启保护
                ground_detector=ground_detector_right,
                leg_tp_comp=0.0,
                touchdown_count=touchdown_count_right,
                vehicle_airborne=robot_is_airborne,
                roll_fast_force=roll_fast_force_right
            )

            # ---------------- 整体离地状态判断汇总 ----------------
            was_airborne = robot_is_airborne
            if teleop.control_enabled:
                robot_is_airborne = (left_flight == 1) or (right_flight == 1)
            else:
                robot_is_airborne = False
            # 记录每条腿本帧 flight，供下一帧各自独立 debounce 使用
            left_flight_last = left_flight
            right_flight_last = right_flight

            # ---------------- 落地沿：重置位移目标 ----------------
            # 空中期间轮子自由转动，common_motor_pos 会漂移，x_b_set 停在旧值。
            # 落地瞬间（was_airborne True→False）若不重置，位移项 K[4]*(x_b - x_b_set)
            # 会对一个陈旧的大误差猛发力，导致落地后猛冲/失稳。
            # 故在落地沿把 x_b_set 强制重置为当前落地位置，位移误差从 0 重新积累。
            if was_airborne and not robot_is_airborne:
                landing_x_b = common_motor_pos * wheel_radius
                lqr_left.x_b_set = landing_x_b
                lqr_right.x_b_set = landing_x_b

            # ---------------- 写离地检测日志 ----------------
            log_writer.writerow([
                f"{sim_time:.4f}", f"{d.qpos[2]:.4f}", f"{pitch:.4f}", f"{roll:.4f}", f"{gyro_pitch:.4f}",
                int(teleop.control_enabled), int(robot_is_airborne),
                # 左腿
                f"{ground_detector_left.fn_raw:.3f}", f"{ground_detector_left.fn_raw_tgt:.3f}",
                f"{ground_detector_left.fn_filtered:.3f}",
                int(ground_detector_left.airborne), left_flight,
                f"{five_links_left.L_0:.4f}", f"{target_length_left:.4f}", f"{lqr_left.theta:.4f}",
                f"{lqr_left.x_b:.4f}", f"{lqr_left.x_b_set:.4f}", f"{left_wheel_T:.3f}", touchdown_count_left,
                # 右腿
                f"{ground_detector_right.fn_raw:.3f}", f"{ground_detector_right.fn_raw_tgt:.3f}",
                f"{ground_detector_right.fn_filtered:.3f}",
                int(ground_detector_right.airborne), right_flight,
                f"{five_links_right.L_0:.4f}", f"{target_length_right:.4f}", f"{lqr_right.theta:.4f}",
                f"{lqr_right.x_b:.4f}", f"{lqr_right.x_b_set:.4f}", f"{right_wheel_T:.3f}", touchdown_count_right,
                # roll 快通道 + roll 角速度
                f"{gyro_roll:.4f}", f"{roll_fast_force_left:.3f}", f"{roll_fast_force_right:.3f}",
            ])
            log_frame += 1
            if log_frame % 500 == 0:
                log_file.flush()  # 定期落盘，防止意外退出丢数据

            # ---------------- 真实时间驱动物理步进 ----------------
            d.ctrl[0] = ctrl_right_0
            d.ctrl[1] = ctrl_right_1 
            d.ctrl[2] = ctrl_left_0  
            d.ctrl[3] = ctrl_left_1  
            d.ctrl[4] = right_wheel_T       
            d.ctrl[5] = left_wheel_T        

            now = time.time()
            real_elapsed = now - last_real_time
            last_real_time = now
            sim_time += real_elapsed * SIM_SPEED

            steps = 0
            while d.time < sim_time:
                mujoco.mj_step(m, d)
                steps += 1
            viewer.sync()

            # FPS 统计
            fps_frame_count += 1
            elapsed_since_report = now - fps_last_time
            if elapsed_since_report >= fps_display_interval:
                fps = fps_frame_count / elapsed_since_report
                loop_ms = (elapsed_since_report / fps_frame_count) * 1000.0
                print(f"[FPS] {fps:.1f} fps | 每帧 {loop_ms:.1f} ms | 子步 {steps}")
                fps_frame_count = 0
                fps_last_time = now

    # ---------------- 关闭日志文件 ----------------
    log_file.flush()
    log_file.close()
    print(f"[LOG] 已保存离地检测日志（共 {log_frame} 帧）: {log_path}")