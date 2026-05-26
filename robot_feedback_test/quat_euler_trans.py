import mujoco
import numpy as np

def euler_to_mujoco_quat(roll, pitch, yaw, seq='xyz'):
    """
    将欧拉角转换为 MuJoCo 的 (w, x, y, z) 四元数
    :param seq: 旋转序列，默认为 'xyz' (对应 MuJoCo 默认)
    """
    # 准备一个接收结果的数组 (4维)
    quat = np.zeros(4)
    # 欧拉角需要是弧度
    euler = np.array([roll, pitch, yaw], dtype=np.float64)
    
    # MuJoCo 内置转换函数 (结果直接就是 wxyz)
    # 这里的 seq 对应 MuJoCo 的 compiler 设定，默认通常是 'xyz'
    mujoco.mju_euler2Quat(quat, euler, seq)
    
    return quat

# 测试
r, p, y = 0,np.pi/4, 0
q = euler_to_mujoco_quat(r, p, y)
print(f"MuJoCo Quat (w, x, y, z): {q}")