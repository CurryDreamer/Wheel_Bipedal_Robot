# 提供mujoco接口中，六个关节电机的位置和速度的信息反馈，六轴imu的信息反馈

import math

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
    info['gyro_yaw'] = d.sensor("angular_vel").data[0]
    info['gyro_pitch'] = d.sensor("angular_vel").data[1]
    info['gyro_roll'] = d.sensor("angular_vel").data[2]

    return info