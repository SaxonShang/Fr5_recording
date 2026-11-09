"""FR5 的解析正逆运动学，只用于离线测试：不连机械臂时代替控制器的 GetInverseKinRef。录制时一律用控制器逆解。

和 10-15 开发录制 6 条的控制器位姿对照，正运动学误差 0.00 mm。
"""

from __future__ import annotations

import math

import numpy as np

D = [0.152, 0, 0, 0.102, 0.102, 0.100]
A = [0, -0.425, -0.395, 0, 0, 0]
AL = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]


def dh(i: int, q: float) -> np.ndarray:
    ct, st, ca, sa = math.cos(q), math.sin(q), math.cos(AL[i]), math.sin(AL[i])
    return np.array([[ct, -st * ca, st * sa, A[i] * ct], [st, ct * ca, -ct * sa, A[i] * st], [0, sa, ca, D[i]], [0, 0, 0, 1]])


def fk(q_deg) -> np.ndarray:
    """关节角（°）→ 基坐标系下的法兰位姿 4×4。"""
    t = np.eye(4)
    for i, q in enumerate(np.deg2rad(q_deg[:6])):
        t = t @ dh(i, q)
    return t


def ik(t06: np.ndarray) -> dict:
    """法兰位姿 → 最多 8 组解：{(s1, s5, s3): 关节角（°，−180～180）}。"""
    d4, d6, a2, a3 = D[3], D[5], A[1], A[2]
    d5 = D[4]
    out = {}
    p05 = t06 @ np.array([0, 0, -d6, 1])
    r = math.hypot(p05[0], p05[1])
    if r < 1e-9 or abs(d4 / r) > 1:
        return out
    psi, phi = math.atan2(p05[1], p05[0]), math.acos(d4 / r)
    t60 = np.linalg.inv(t06)
    for s1 in (1, -1):
        th1 = psi + s1 * phi + math.pi / 2
        s1t, c1t = math.sin(th1), math.cos(th1)
        c5 = (t06[0, 3] * s1t - t06[1, 3] * c1t - d4) / d6
        if abs(c5) > 1:
            continue
        for s5 in (1, -1):
            th5 = s5 * math.acos(c5)
            s5t = math.sin(th5)
            if abs(s5t) < 1e-9:
                continue
            th6 = math.atan2((-t60[1, 0] * s1t + t60[1, 1] * c1t) / s5t, (t60[0, 0] * s1t - t60[0, 1] * c1t) / s5t)
            t14 = np.linalg.inv(dh(0, th1)) @ t06 @ np.linalg.inv(dh(4, th5) @ dh(5, th6))
            p13 = t14 @ np.array([0, -d4, 0, 1])
            px, pz = p13[0], p13[1]
            l2 = px * px + pz * pz
            c3 = (l2 - a2 * a2 - a3 * a3) / (2 * a2 * a3)
            if abs(c3) > 1:
                continue
            for s3 in (1, -1):
                th3 = s3 * math.acos(c3)
                th2 = math.atan2(-pz, -px) - math.asin(-a3 * math.sin(th3) / math.sqrt(l2))
                t34 = np.linalg.inv(dh(1, th2) @ dh(2, th3)) @ t14
                th4 = math.atan2(t34[1, 0], t34[0, 0])
                q = np.rad2deg([th1, th2, th3, th4, th5, th6])
                out[(s1, s5, s3)] = (q + 180) % 360 - 180
    return out


def near(q, ref) -> np.ndarray:
    """把 q 的每个角换成离 ref 最近的 ±360 等价值。"""
    ref = np.asarray(ref, dtype=float)
    return ref + (np.asarray(q, dtype=float) - ref + 180) % 360 - 180


class OfflineRPC:
    """代替控制器的只读接口：离 ref 最近的解，和控制器 GetInverseKinRef 的约定相同。"""

    def __init__(self, soft_limits=None):
        self.soft_limits = soft_limits

    def GetInverseKinRef(self, typ, pose6, ref):  # noqa: N802 - SDK 的名字
        import fr5_motion as fm
        ref = np.asarray(ref, dtype=float)
        best, gap = None, math.inf
        for q in ik(fm.fr5_to_matrix(pose6)).values():
            q = near(q, ref)
            d = float(np.abs(q - ref).max())
            if d < gap:
                best, gap = q, d
        return (0, best.tolist()) if best is not None else (112, None)

    def GetJointSoftLimitDeg(self):  # noqa: N802 - SDK 的名字
        return (0, list(self.soft_limits)) if self.soft_limits else (1, None)
