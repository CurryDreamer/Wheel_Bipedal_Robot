import numpy as np

class VelocityRampFilter:
    def __init__(self, dt=0.002, accel_rate=0.8, decel_rate=1.4):
        """
        初始化线性速度爬坡滤波器
        :param dt: 仿真控制周期
        :param accel_rate: 加速度最大限制 (m/s^2)
        :param decel_rate: 减速度最大限制 (m/s^2)
        """
        self.dt = dt
        self.accel_rate = accel_rate
        self.decel_rate = decel_rate

    def update(self, current_v, target_v):
        """
        根据目标速度和当前速度，计算当前周期滤波后的期望速度
        :param current_v: 当前周期的实际或平滑速度
        :param target_v: 遥控器输入的原始目标速度
        :return: 经过爬坡限制后的当前期望速度
        """
        # 判断当前是加速还是减速
        # 如果目标速度绝对值大于当前速度绝对值，且方向相同（或当前速度接近0），视为加速
        if abs(target_v) > abs(current_v) and target_v * current_v >= 0:
            ramp_rate = self.accel_rate
        else:
            ramp_rate = self.decel_rate

        max_step_size = abs(self.dt * ramp_rate)
        full_step = target_v - current_v
        
        # 限制当前步进大小
        step = np.clip(full_step, -max_step_size, max_step_size)
        out_v = current_v + step
        
        # 死区微调：如果目标速度为 0 且当前速度非常小，直接清零，防止由于浮点数精度原地蠕动
        if abs(out_v) < 0.001 and target_v == 0.0:
            out_v = 0.0
            
        return out_v