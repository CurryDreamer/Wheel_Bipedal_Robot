import numpy as np

class RollLegController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, max_out=0.14, min_leg=0.10, max_leg=0.14):
        """
        初始化横滚角轮腿机身平衡控制器
        :param kp: PID 比例系数
        :param ki: PID 积分系数
        :param kd: PID 微分系数
        :param max_out: PID 输出限制的最大幅值 (单侧腿长的最大修正差额)
        :param min_leg: 物理限制的最小腿长 (m)
        :param max_leg: 物理限制的最大腿长 (m)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.min_leg = min_leg
        self.max_leg = max_leg
        
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_leg_lengths(self, current_roll, target_roll, base_length):
        """
        根据当前机身 Roll 角，计算出分配给左、右两条腿的最优期望长度
        :param current_roll: 传感器测量得到的当前 Roll 弧度 (rad)
        :param target_roll: 期望的机身 Roll 角，通常为 0.0 (rad)
        :param base_length: 遥控器给出的基础期望腿长 (m)
        :return: (target_length_left, target_length_right)
        """
        # 1. 基础 PID 计算
        error = target_roll - current_roll
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        
        # 2. 计算输出修正量并做安全裁剪
        delta_L = self.kp * error + self.ki * self.integral + self.kd * derivative
        delta_L = np.clip(delta_L, -self.max_out, self.max_out)
        
        # 3. 差模分配到左右腿，并限制在机器人的物理限界范围内
        # 当向左倾斜（Roll为正）时，左腿需要变长，右腿需要变短来顶回平衡（具体正负号与传感器定义的坐标系强相关）
        target_length_left = np.clip(base_length - delta_L, self.min_leg, self.max_leg)
        target_length_right = np.clip(base_length + delta_L, self.min_leg, self.max_leg)
        
        return float(target_length_left), float(target_length_right)