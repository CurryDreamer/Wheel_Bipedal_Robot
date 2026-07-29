import numpy as np


class RollLegController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, max_out=0.14, min_leg=0.10, max_leg=0.14,
                 unload_min_scale=0.35, unload_down_rate=20.0, unload_up_rate=4.0, dt=0.001,
                 fast_kp=0.0, fast_rate_damp=3.0, fast_force_limit=16.0, fast_force_slew=150.0,
                 fast_sign=1.0, slow_deadzone=0.05):
        """
        横滚角轮腿机身平衡控制器（快/慢双通道并联，对齐固件架构）。

        - 慢通道（本类原有）：改左右目标腿长差，经腿长位置环慢慢调平静态偏置。响应慢。
        - 快通道（新增）：直接输出差动轴向力 fast_force，叠加到左右腿虚拟轴向力 F_0 上，
          快速压制 roll 动态/晃动。仿真原本只有慢通道，roll 一有扰动只能靠慢改腿长，
          纠不过来就把单腿拉到机械极限翻车；补上快通道后由它扛 roll 动态。

        :param kp/ki/kd: 慢通道（腿长差）PID 系数
        :param max_out: 慢通道单侧腿长最大修正差额
        :param min_leg/max_leg: 腿长物理限位
        :param unload_*: 离地软卸载参数（快慢通道输出都乘卸载系数）
        :param fast_kp: 快通道比例增益（作用于 roll 角误差），保守默认 40（固件 90）
        :param fast_rate_damp: 快通道 roll 角速度阻尼，保守默认 3（固件 5.5）
        :param fast_force_limit: 快通道差动力限幅 (N)
        :param fast_force_slew: 快通道差动力斜率限幅 (N/s)
        :param fast_sign: 快通道力方向符号（+1/-1），与慢通道方向对齐，实测校准
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.min_leg = min_leg
        self.max_leg = max_leg
        # 慢通道死区 (rad)：|roll误差| < slow_deadzone 时慢通道完全不动腿长，
        # 避免追运动诱发、本可轮式自稳的小 roll 偏差而持续拉腿→机体歪→翻车。
        # 只有斜坡等大 roll 才介入做静态倾角适配。快通道不受死区影响（弱阻尼、不累积）。
        self.slow_deadzone = slow_deadzone

        self.integral = 0.0
        self.prev_error = 0.0

        # ---- roll 差模软卸载 ----
        self.unload_min_scale = unload_min_scale
        self.unload_down_rate = unload_down_rate
        self.unload_up_rate = unload_up_rate
        self.dt = dt
        self.scale_left = 1.0
        self.scale_right = 1.0

        # ---- 快通道（差动轴向力）----
        self.fast_kp = fast_kp
        self.fast_rate_damp = fast_rate_damp
        self.fast_force_limit = fast_force_limit
        self.fast_force_slew = fast_force_slew
        self.fast_sign = fast_sign
        self.fast_force_prev = 0.0  # 上一帧快通道力，用于斜率限幅

    def reset(self):
        """重置内部积分、上一帧误差、卸载系数与快通道状态。"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.scale_left = 1.0
        self.scale_right = 1.0
        self.fast_force_prev = 0.0

    def _update_scale(self, scale, unloading):
        """单腿卸载系数的快降慢升平滑过渡。"""
        if unloading:
            target = self.unload_min_scale
            step = self.unload_down_rate * self.dt
        else:
            target = 1.0
            step = self.unload_up_rate * self.dt
        if scale < target:
            return min(target, scale + step)
        else:
            return max(target, scale - step)

    def compute_fast_force(self, current_roll, target_roll, roll_speed,
                           left_airborne=False, right_airborne=False):
        """快通道：计算差动轴向力，叠加到左右腿 F_0 上（左 +f，右 -f）。

        :param current_roll: 当前 roll (rad)
        :param target_roll: 目标 roll (通常 0)
        :param roll_speed: roll 角速度 (rad/s)，提供阻尼
        :param left_airborne/right_airborne: 左右腿离地 → 该侧快通道力软卸载
        :return: (fast_force_left, fast_force_right)  两者符号相反
        """
        error = target_roll - current_roll
        # 比例 + roll 角速度阻尼（阻尼项抵抗 roll 转动）
        f = self.fast_sign * (self.fast_kp * error - self.fast_rate_damp * roll_speed)
        # 限幅
        f = float(np.clip(f, -self.fast_force_limit, self.fast_force_limit))
        # 斜率限幅（防突变冲击）
        max_step = self.fast_force_slew * self.dt
        f = float(np.clip(f, self.fast_force_prev - max_step, self.fast_force_prev + max_step))
        self.fast_force_prev = f
        # 左右各乘卸载系数（复用慢通道更新过的 scale；这里不重复更新，避免一帧更新两次）
        return f * self.scale_left, -f * self.scale_right

    def compute_leg_lengths(self, current_roll, target_roll, base_length,
                            left_airborne=False, right_airborne=False):
        """
        慢通道：根据 roll 角计算左右目标腿长，对离地腿的差模修正做软卸载（快降慢升）。
        （快通道的 fast_force 请另调用 compute_fast_force，本方法只负责腿长差。）

        :return: (target_length_left, target_length_right)
        """
        # 1. 基础 PID 计算（先过死区）
        error = target_roll - current_roll
        # 死区：|误差| < slow_deadzone 时误差归零，慢通道不动腿长；
        # 过阈值后减去死区值，使输出从 0 连续增长不跳变。
        if abs(error) <= self.slow_deadzone:
            error_dz = 0.0
            self.integral = 0.0  # 死区内清积分，避免累积残留
        else:
            error_dz = error - self.slow_deadzone if error > 0 else error + self.slow_deadzone
        self.integral += error_dz
        derivative = error_dz - self.prev_error
        self.prev_error = error_dz

        # 2. 计算输出修正量并做安全裁剪
        delta_L = self.kp * error_dz + self.ki * self.integral + self.kd * derivative
        delta_L = np.clip(delta_L, -self.max_out, self.max_out)

        # 3. 更新左右腿各自的卸载系数（快降慢升）——快通道也复用这里更新的 scale
        self.scale_left = self._update_scale(self.scale_left, left_airborne)
        self.scale_right = self._update_scale(self.scale_right, right_airborne)

        # 4. 差模分配到左右腿，各自乘卸载系数；基础腿长始终生效不受卸载影响
        target_length_left = np.clip(base_length - delta_L * self.scale_left, self.min_leg, self.max_leg)
        target_length_right = np.clip(base_length + delta_L * self.scale_right, self.min_leg, self.max_leg)

        return float(target_length_left), float(target_length_right)

