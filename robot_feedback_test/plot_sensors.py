import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
import collections

def quat2euler(q):
    """
    将四元数 (w, x, y, z) 转换为欧拉角 (Roll, Pitch, Yaw)
    """
    w, x, y, z = q
    
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    return np.array([roll, pitch, yaw])

def main():
    xml_path = "robot.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    pos_sensors = ["right1_pos", "right2_pos", "rightwheel_pos", 
                   "left1_pos", "left2_pos", "leftwheel_pos"]
    vel_sensors = ["right1_vel", "right2_vel", "rightwheel_vel", 
                   "left1_vel", "left2_vel", "leftwheel_vel"]

    # 历史数据长度
    history_size = 500
    
    time_history = collections.deque(maxlen=history_size)
    
    # 字典用于存储历史数据
    pos_data = {name: collections.deque(maxlen=history_size) for name in pos_sensors}
    vel_data = {name: collections.deque(maxlen=history_size) for name in vel_sensors}
    imu_euler = {
        'Roll': collections.deque(maxlen=history_size),
        'Pitch': collections.deque(maxlen=history_size),
        'Yaw': collections.deque(maxlen=history_size)
    }

    # 打开 matplotlib 交互模式
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title('Robot Sensors Telemetry')

    # 初始化绘图线
    lines_pos = {}
    lines_vel = {}
    lines_imu = {}
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i, (pos_name, vel_name) in enumerate(zip(pos_sensors, vel_sensors)):
        line, = axs[0].plot([], [], label=pos_name, color=colors[i])
        lines_pos[pos_name] = line
        
        line_vel, = axs[1].plot([], [], label=vel_name, color=colors[i])
        lines_vel[vel_name] = line_vel

    lines_imu['Roll'], = axs[2].plot([], [], label='Roll', color='r')
    lines_imu['Pitch'], = axs[2].plot([], [], label='Pitch', color='g')
    lines_imu['Yaw'], = axs[2].plot([], [], label='Yaw', color='b')

    axs[0].set_title("Motor Positions (rad)")
    axs[0].legend(loc="upper right", fontsize='small')
    axs[0].grid(True)
    
    axs[1].set_title("Motor Velocities (rad/s)")
    axs[1].legend(loc="upper right", fontsize='small')
    axs[1].grid(True)
    
    axs[2].set_title("IMU Euler Angles (degrees)")
    axs[2].legend(loc="upper right", fontsize='small')
    axs[2].grid(True)

    plt.tight_layout()

    print("开始仿真并打开实时数据监控...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 记录时间
            current_time = data.time
            time_history.append(current_time)
            
            # 记录位置数据
            for name in pos_sensors:
                pos_data[name].append(data.sensor(name).data[0])
                
            # 记录速度数据
            for name in vel_sensors:
                vel_data[name].append(data.sensor(name).data[0])
                
            # 记录IMU欧拉角数据
            quat = data.sensor("orientation").data.copy()
            euler_rad = quat2euler(quat)
            euler_deg = np.degrees(euler_rad)
            
            imu_euler['Roll'].append(euler_deg[0])
            imu_euler['Pitch'].append(euler_deg[1])
            imu_euler['Yaw'].append(euler_deg[2])
            
            # 每隔 100 步更新一次画面以防卡顿 (100 * 0.002 = 0.2s)
            if step_count % 100 == 0 and len(time_history) > 1:
                t_arr = list(time_history)
                
                # 更新坐标轴时间范围
                t_min, t_max = t_arr[0], t_arr[-1]
                for ax in axs:
                    ax.set_xlim(t_min, t_max)
                
                # 更新位置线
                for name in pos_sensors:
                    lines_pos[name].set_data(t_arr, list(pos_data[name]))
                axs[0].relim()
                axs[0].autoscale_view(scalex=False, scaley=True)
                
                # 更新速度线
                for name in vel_sensors:
                    lines_vel[name].set_data(t_arr, list(vel_data[name]))
                axs[1].relim()
                axs[1].autoscale_view(scalex=False, scaley=True)

                # 更新 IMU 线
                lines_imu['Roll'].set_data(t_arr, list(imu_euler['Roll']))
                lines_imu['Pitch'].set_data(t_arr, list(imu_euler['Pitch']))
                lines_imu['Yaw'].set_data(t_arr, list(imu_euler['Yaw']))
                axs[2].relim()
                axs[2].autoscale_view(scalex=False, scaley=True)
                
                # 刷新 matplotlib 界面
                fig.canvas.draw()
                fig.canvas.flush_events()
                
            step_count += 1
            # time.sleep(0.001)

if __name__ == "__main__":
    main()
