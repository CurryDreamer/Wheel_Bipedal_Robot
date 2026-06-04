# Wheeled Bipedal Robot Simulation & Control

本项目是一个基于 **MuJoCo** 物理引擎的轮足双足机器人仿真与控制框架。机器人采用五连杆腿部机构 + 轮毂电机的构型，结合 **LQR**（线性二次型调节器）进行轮式倒立摆平衡控制，以及 **VMC**（虚拟模型控制）进行腿长与姿态控制，并支持通过 Xbox 手柄或键盘进行实时遥控。

## 主要特性

- **物理仿真**：基于 MuJoCo 构建的精确机器人 MJCF 模型及障碍物场景。
- **平衡与运动控制**：LQR 状态反馈控制器实现轮式倒立摆姿态的稳定平衡，支持不同虚拟腿长下 K 矩阵的实时多项式插值。
- **虚拟模型驱动 (VMC)**：利用五连杆雅可比矩阵将极坐标虚拟力（轴向力 + 髋关节力矩）映射到关节空间，实现双腿独立控制。
- **横滚 (Roll) 姿态调节**：通过 PID 差模分配左右腿目标长度，自动保持机身横滚水平。
- **离地检测与保护**：基于虚拟支持力估计判断单腿是否离地，离地时清除轮毂扭矩并切换为空中阻尼姿态维持，防止越障时失控。
- **速度爬坡滤波**：支持独立配置加减速率的线性速度斜坡滤波器，避免速度阶跃导致失稳。
- **偏航 (Yaw) 控制**：通过左右轮差速 PID 闭环实现机身朝向控制。
- **实时手控输入**：支持 Xbox 手柄（左摇杆控制速度、右摇杆控制偏航、LB/RB 调节腿长）和键盘（方向键 + +/-）。
- **传感器数据处理**：IMU 四元数转欧拉角、关节位置/速度反馈、陀螺仪角速度读取。
- **实时遥测绘图**：可选启用 matplotlib 实时绘制电机位置、速度和 IMU 欧拉角曲线。

## 目录结构

```text
wheel_bipedal_robot/
├── LQR_VMC/                        # [核心] LQR + VMC 联合主控制代码
│   ├── robot_control_main.py       #   主程序入口，封装左右腿控制函数，含离地保护逻辑
│   ├── lqr_update.py               #   LQR 控制器：12维状态更新、多项式实时计算 K 矩阵
│   ├── vmc_cal.py                  #   五连杆正运动学 + VMC 雅可比力矩映射
│   ├── roll_leg_control.py         #   横滚平衡控制器（PID 差模分配左右腿长）
│   ├── ground_detection.py         #   单腿离地检测（基于虚拟支持力阈值）
│   ├── mjcf/                       #   MuJoCo 模型文件
│   │   ├── robot.xml               #     机器人 URDF/MJCF 模型
│   │   └── obstacles.xml           #     障碍物场景文件
│   ├── param_cal/                  #   机器人动力学参数计算
│   │   └── robot_param_cal.py      #     从 XML 提取质量/惯量/半径等参数
│   ├── remote/                     #   遥控输入模块
│   │   └── xbox_controller.py      #     封装 PygameController 类（键盘 + Xbox 手柄）
│   ├── robot_mujoco_sensor_feedback/  # 传感器反馈处理
│   │   └── robot_info_update.py    #     统一读取 6 个关节 + IMU 数据
│   └── util/                       #   工具类
│       └── climb_traj.py           #     线性速度爬坡滤波器
|
├── robot_vmc_test/                 # [验证] VMC 五连杆运动学独立验证
│   ├── vmc_verification.py         #   位置环 + 逆运动学分析，验证零点标定与腿长控制
│   └── robot_pos_ctrl_for_vmc.xml  #   位置控制模式的专用模型文件
│
├── robot_feedback_test/            # [验证] 传感器反馈测试
│   ├── plot_sensors.py             #   实时绘制电机位置/速度/IMU 欧拉角的遥测工具
│   ├── quat_euler_trans.py         #   四元数 ↔ 欧拉角转换测试
│   └── robot.xml                   #   MuJoCo 模型文件
│
└── README.md
```

## 环境依赖

本项目基于 **Python 3.12+** 开发，系统为 Ubuntu 22.04。运行前请安装以下依赖：

```bash
pip install mujoco
pip install numpy
pip install scipy
pip install pygame
pip install matplotlib   # 用于遥测绘图（可选）
```

## 快速开始

### 1. 运行主仿真程序

主程序位于 `LQR_VMC` 目录：

```bash
cd LQR_VMC
python robot_control_main.py
```

### 2. 操控说明

程序启动后会打开 MuJoCo 查看器和一个 Pygame 控制面板。

| 操作                     | 键盘                    | Xbox 手柄                    |
|-------------------------|-------------------------|------------------------------|
| 前进 / 后退              | ↑ / ↓ 方向键           | 左摇杆上下                     |
| 左转 / 右转              | ← / → 方向键           | 右摇杆左右                     |
| 增加 / 减少目标腿长       | + / -                   | RB / LB（肩键）               |
| 暂停 / 继续仿真           | 空格键                  | —                            |
| 开启 / 关闭离地检测保护    | 回车键 (Enter)          | 左摇杆按键                      |

**离地检测保护**：系统会实时估计足端虚拟支持力。当检测到某条腿离地时，自动将该侧轮毂力矩清零、髋关节切换为空中阻尼姿态维持模式，防止因轮子悬空误驱动导致姿态失控。

### 3. 修改为自定义同构型机器人
LQR_VMC/param_cal/robot_param_cal.py提供了读取仿真中机器人的质量与惯性参数；
可以通过修改LQR_VMC/mjcf/robot.xml去进行修改质量与转动惯量(此工程的机体转动惯量选择使用MuJoco自动计算出的转动惯量)

### 4. 其他目录简单介绍（可选）
1. 传感器(电机编码器 IMU)可视化

```bash
cd robot_feedback_test
python plot_sensors.py
```

实时绘制 6 个关节电机的位置、速度以及 IMU 欧拉角变化曲线；
查看机体的姿态可以将robot_feedback_test/robot.xml中的freejoint取消注释，固定在天上输入quat进行查看

2. 初期运动学验证

```bash
cd robot_vmc_test
python vmc_verification.py
```

验证五连杆正/逆运动学、电机零点标定和腿长控制范围。

## TODO

- [ ] 实现打滑检测
- [ ] 进行真机验证
