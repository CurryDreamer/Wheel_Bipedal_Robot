import math

# ==================== 可调参数宏定义 ====================
# 单腿支撑质量对应的重力 (kg * m/s^2)，用于补偿静态重力项
LEG_SUPPORT_WEIGHT = 0.6 * 9.8  # ≈ 5.88 N

# 低通滤波系数 (0~1)：越小越平滑、滞后越大。dt≈1ms 下 0.05 ≈ 20ms 时间常数
FN_LPF_ALPHA = 0.05

# 双阈值迟滞 (N)：
#   着地 → 离地：滤波后支持力 < GROUND_FN_LOW 才判定离地
#   离地 → 着地：滤波后支持力 > GROUND_FN_HIGH 才判定恢复着地
# 两个阈值拉开间隙可抑制临界抖动来回翻转；实测着地稳态 FN 高于离地，故 HIGH>LOW。
# 阈值由飞坡实测数据标定 (log/ground_20260704_184442.csv, 5次飞坡)：
#   稳态着地 lpf: mean+12, p5=+10.7；真实腾空 lpf: mean-0.5, p95=+2.5
#   LOW=4.0 在腾空p95(2.5)之上留余量触发离地；HIGH=8.0 落在腾空与着地之间，恢复稳定
GROUND_FN_LOW = 0.0
GROUND_FN_HIGH = 2.0
# ======================================================


class LegGroundDetector:
    """单腿离地检测器（有状态）：对重构支持力做低通滤波 + 双阈值迟滞判定。

    每条腿各持有一个实例，滤波值与迟滞状态在多次调用间保持。
    """

    def __init__(self, alpha=FN_LPF_ALPHA, fn_low=GROUND_FN_LOW, fn_high=GROUND_FN_HIGH, debug=False, name=""):
        self.alpha = alpha
        self.fn_low = fn_low
        self.fn_high = fn_high
        self.fn_filtered = None   # 滤波后支持力，None 表示未初始化
        self.fn_raw = 0.0         # 最近一帧原始支持力（未滤波，用实际 L_0），供日志记录用
        self.fn_raw_tgt = 0.0     # 用目标腿长计算的原始支持力，供对比/日志用
        self.airborne = False     # 迟滞状态：True=离地, False=着地
        self.debug = debug        # 打开后每帧打印原始/滤波 FN，用于标定阈值
        self.name = name

    def compute_fn(self, five_links, F, T_p, l0=None):
        """由逆 VMC 反算的虚拟力计算垂直地面方向支持力 FN。

        :param l0: 若给定则用该腿长代入 T_p 项（如目标腿长），否则用实测 five_links.L_0。
                   cosθ 项不含 L_0，不受影响。
        """
        L0 = five_links.L_0 if l0 is None else l0
        return (F * math.cos(five_links.theta)
                + T_p * math.sin(five_links.theta) / L0
                + LEG_SUPPORT_WEIGHT)

    def update(self, five_links, F, T_p, target_l0=None) -> int:
        """更新并返回离地状态。

        :param five_links: 对应腿的五连杆动力学参数对象 (需具备 theta, L_0)
        :param F: 逆 VMC 反算出的实际虚拟轴向力 (N)
        :param T_p: 逆 VMC 反算出的实际虚拟劈叉力矩 (N·m)
        :param target_l0: roll 控制器当前给定的目标腿长。给定时判定用它算 FN
                          （腿被地形压伸长时避免 T_p 项被稀释造成误判）；
                          同时仍记录用实测 L_0 的 FN 到 fn_raw 供对比。
        :return: 1 表示离地，0 表示着地
        """
        # 用实测 L_0 的 FN（供对比/日志）
        self.fn_raw = self.compute_fn(five_links, F, T_p)
        # 用目标腿长的 FN（供对比/日志），无目标时退化为实测值
        self.fn_raw_tgt = self.compute_fn(five_links, F, T_p, l0=target_l0)

        # 实际参与迟滞判定的值：优先用目标腿长版本
        fn_raw = self.fn_raw_tgt if target_l0 is not None else self.fn_raw

        # 一阶低通滤波，抑制单帧噪声（实测原始 FN 瞬时 std 很大）
        if self.fn_filtered is None:
            self.fn_filtered = fn_raw
        else:
            self.fn_filtered += self.alpha * (fn_raw - self.fn_filtered)

        # 双阈值迟滞判定
        if self.airborne:
            # 当前判为离地：只有滤波值升过高阈值才恢复着地
            if self.fn_filtered > self.fn_high:
                self.airborne = False
        else:
            # 当前判为着地：只有滤波值降到低阈值以下才判离地
            if self.fn_filtered < self.fn_low:
                self.airborne = True

        if self.debug:
            print(f"[GND {self.name}] raw(L0实测)={self.fn_raw:+7.2f} raw(L0目标)={self.fn_raw_tgt:+7.2f} "
                  f"lpf={self.fn_filtered:+7.2f} air={int(self.airborne)}")

        return 1 if self.airborne else 0


# ---- 兼容旧接口：无状态单帧判定（不推荐，保留以防其他调用点依赖）----
def detect_leg_ground_status(five_links, F, T_p) -> int:
    FN = (F * math.cos(five_links.theta)
          + T_p * math.sin(five_links.theta) / five_links.L_0
          + LEG_SUPPORT_WEIGHT)
    print(FN)
    return 1 if FN < GROUND_FN_LOW else 0
