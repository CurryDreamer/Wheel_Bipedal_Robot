import mujoco
import math
import numpy as np
'''
提供以下计算
机体质量
驱动轮质量
驱动轮转动惯量
驱动轮半径
摆杆质量
摆杆转动惯量
质点偏移
'''


def calculate_robot_params(xml_path):
    m = mujoco.MjModel.from_xml_path(xml_path)
    
    params = {}
    
    # 1. 机体质量 (Torso Mass)
    # 取 torso body 及其子 body (限位块) 中的对应 geom / site 的质量总和
    # 在这个结构中，直接取 torso body 的整体动力学参数 mass 即可，Mujoco 会自动把焊死的刚体累加计算
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    
    torso_mass = m.body_mass[torso_id]

    limit_names = [
        "right_limit1", "right_limit2", "left_limit1", "left_limit2",
    ]
    limit_mass = 0.0
    for name in limit_names:
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        limit_mass += m.body_mass[body_id]

    # 让我们验证一下您的想法：相加下面所有的值
    base_mass = 0.823
    front_mass = 0.824
    back_mass = 0.824
    limit_masses = 0.005 * 4
    calculated_torso_mass =  base_mass + front_mass + back_mass + limit_masses
    
    print(f"机体总质量计算结果:\n  理论累加值 = {calculated_torso_mass:.5f} kg\n  Mujoco 动力学计算值 = {torso_mass+limit_mass:.5f} kg")

    # 2. 驱动轮参数 (单个轮子)
    # 通过给 geom 命名来准确获取
    wheel_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel_geom")
    wheel_radius = m.geom_size[wheel_geom_id][0]
    
    wheel_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wheel") 
    wheel_mass = m.body_mass[wheel_body_id]
    
    # 按照圆柱体绕中心轴旋转的公式: I = 0.5 * m * r^2
    wheel_inertia = 0.5 * wheel_mass * (wheel_radius ** 2)
    
    params['wheel_radius'] = wheel_radius
    params['wheel_mass'] = wheel_mass
    params['wheel_inertia'] = wheel_inertia
    
    print(f"\n驱动轮参数 (单轮):")
    print(f"  质量 (m) = {wheel_mass} kg")
    print(f"  半径 (R) = {wheel_radius:.4f} m")
    print(f"  转动惯量 (I_zz) = {wheel_inertia:.6f} kg·m^2")

    # 3. 机体转动惯量 (Torso Inertia)
    # 获取绕本体系 X, Y, Z 轴的主转动惯量。由于二维倒立摆通常考察前倾后仰的 Pitch 轴，因此最重要的是 Y 轴转动惯量 I_yy
    torso_inertia = m.body_inertia[torso_id]
    params['torso_inertia'] = torso_inertia
    
    print(f"\n机体转动惯量 (Torso Inertia):")
    print(f"  [I_xx, I_yy, I_zz] = [{torso_inertia[0]:.6f}, {torso_inertia[1]:.6f}, {torso_inertia[2]:.6f}] kg·m^2")
    print(f"  (其中 I_yy 即为绕 Pitch 轴的机体转动惯量)")

    # 4. 摆杆质量 (Pendulum Mass)
    # 取全部左右连杆的总和（整机模型通常把左右两条腿作为一根等效摆杆）
    leg_body_names = [
        "right_up_leg_1", "right_down_leg_1", "right_up_leg_2", "right_down_leg_2",
        "left_up_leg_1", "left_down_leg_1", "left_up_leg_2", "left_down_leg_2"
    ]
    pendulum_mass = 0.0
    for name in leg_body_names:
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        pendulum_mass += m.body_mass[body_id]
        
    print(f"\n等效摆杆参数 (Pendulum Params):")
    print(f"  全部 8 个连杆总质量 (m_p) = {pendulum_mass:.6f} kg")
    print(f"  等效摆杆的总质量 (m_p/2) = {pendulum_mass/2:.6f} kg")

    # 5. 质点偏移 (CoM Offset - IMU 到 Torso 的 Z 轴距离)
    # 提取 torso 和 imu 的相对位置。imu 在 xml 中是被绑定在 torso 内的 site
    imu_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
    # site_pos 给出的是该 site 相对于所在 body 本地坐标系的位置偏移
    com_offset_z = abs(m.site_pos[imu_site_id][2])
    params['com_offset'] = com_offset_z
    
    print(f"\n质点偏移 (CoM Offset):")
    print(f"  IMU 到机体中心 Z 轴绝对偏差距离 = {com_offset_z:.4f} m")

def calc_pendulum_inertia(mp, L0):
    """
    计算等效摆杆的转动惯量
    根据用户定义：按照细长杆绕端点旋转模型 I = 1/3 * m * L0^2
    注意 L0 是当前 VMC 虚拟腿长，为动态变化值
    """
    return mp * (L0 ** 2) / 3.0

if __name__ == "__main__":
    calculate_robot_params("robot.xml")
