import math

def detect_leg_ground_status(five_links, F_0, T_p) -> int:
    """
    通用单腿离地检测
    :param five_links: 对应腿的五连杆动力学参数对象 (需具备 theta, L_0 属性)
    :param F_0: 当前计算出的 VMC 轴向虚拟收缩力 (N)
    :param T_p: 当前 LQR/VMC 计算出的虚拟髋关节劈叉力矩 (N·m)
    :return: 1 表示该腿离地，0 表示该腿着地
    """
    # 计算垂直地面方向的支持力 FN
    # 注意：five_links.theta 已经是弧度
    FN = F_0 * math.cos(five_links.theta) + T_p * math.sin(five_links.theta) / five_links.L_0 + 6.0
    
    # 阈值判断：如果支持力小于 5.0 N，判定为离地
    if FN < 1.0:
        return 1  # 离地
    else:
        return 0  # 着地