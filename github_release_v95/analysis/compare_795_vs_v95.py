"""
795 原程序 vs v95 Pipeline 模拟对比
====================================
公平对比条件：
  - 同一 SLM (1024×1272)、同一 beam、同一 target、同一 Wcg
  - 同一 propagate / evaluate_metrics
  - 795 原程序：phase_guess 物理初相 + CG overlap → CG overlap×eff
  - v95 Pipeline：LG vortex → Bowman warm → anchor → multi-stage
  - 都用 v95 的 evaluate_metrics 做最终评判

CPU-only，输出尺寸缩到 2048×2048 以控制运行时间。
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import scipy.optimize

torch.set_default_dtype(torch.float64)

# ===========================================================================
# 物理参数（与 v95 notebook 完全一致，仅输出尺寸缩小）
# ===========================================================================
SLM_H, SLM_W = 1024, 1272
OUT_H, OUT_W = 2048, 2048          # 最小可工作尺寸 (>= SLM_W=1272)
PIXEL_PITCH = 12.5e-6
WAVELENGTH = 795e-9
FOCAL_LENGTH = 0.2
device = torch.device("cpu")

focal_spx = FOCAL_LENGTH * WAVELENGTH / (PIXEL_PITCH * OUT_W)
focal_spy = FOCAL_LENGTH * WAVELENGTH / (PIXEL_PITCH * OUT_H)
print(f"系统参数: SLM={SLM_H}x{SLM_W}, Output={OUT_H}x{OUT_W}")
print(f"焦平面像素尺寸: focal_spx={focal_spx*1e6:.2f} um, focal_spy={focal_spy*1e6:.2f} um")


# ===========================================================================
# 核心函数（与 v95 cell 16 一致）
# ===========================================================================
def make_beam():
    sx = sy = 3.5e-3
    sigmax = math.sqrt(2.0) * (sx / PIXEL_PITCH)
    sigmay = math.sqrt(2.0) * (sy / PIXEL_PITCH)
    y = torch.arange(SLM_H, device=device, dtype=torch.float64) - SLM_H / 2
    x = torch.arange(SLM_W, device=device, dtype=torch.float64) - SLM_W / 2
    X, Y = torch.meshgrid(x, y, indexing="xy")
    beam = torch.exp(-2.0 * ((X / sigmax) ** 2 + (Y / sigmay) ** 2))
    I_L_tot = beam.square().sum()
    beam = beam * (10000.0 / I_L_tot).sqrt()
    return beam


def propagate(field_slm):
    h, w = field_slm.shape
    pad_h = (OUT_H - h) // 2
    pad_w = (OUT_W - w) // 2
    padded = F.pad(field_slm.unsqueeze(0).unsqueeze(0),
                   [pad_w, pad_w, pad_h, pad_h], mode="constant", value=0).squeeze()
    A0 = 1.0 / math.sqrt(OUT_H * OUT_W)
    output = A0 * torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(padded)))
    return output


def compute_overlap(field, target):
    inner = (field.conj() * target).sum().abs()
    norm = field.abs().square().sum().sqrt() * target.abs().square().sum().sqrt()
    return (inner / norm.clamp(min=1e-30)).real.item()


def make_target_and_weight():
    sigma = round(100e-6 / focal_spx)
    sigmax_line = round(20e-6 / focal_spy)
    sigmay_line = round(20e-6 / focal_spx)
    dx = round(-100e-6 / focal_spx)
    dy = round(-100e-6 / focal_spx)
    r0 = [dx + OUT_H / 2, dy + OUT_W / 2]
    print(f"目标参数: d={sigma}px, sigmax={sigmax_line}px, sigmay={sigmay_line}px")
    print(f"偏移: dx={dx}px, dy={dy}px, r0={[f'{v:.1f}' for v in r0]}")

    cols, rows = OUT_H, OUT_W
    y = torch.arange(rows, device=device, dtype=torch.float64)
    x = torch.arange(cols, device=device, dtype=torch.float64)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    fx = 0.5 * (torch.abs(X - sigma / 2.0 - r0[0]) + torch.abs(X + sigma / 2.0 - r0[0]) - sigma)
    target_amp = torch.exp(-(fx ** 2 / max(sigmax_line, 1) ** 2 + (Y - r0[1]) ** 2 / max(sigmay_line, 1) ** 2))
    target_phase = torch.zeros_like(target_amp)

    wcg_radius = round(120e-6 / focal_spx)
    r = torch.sqrt((X - r0[0]) ** 2 + (Y - r0[1]) ** 2)
    flat_mask = torch.zeros((cols, rows), device=device, dtype=torch.float64)
    flat_mask[r < wcg_radius / 2] = 1.0
    Wcg = torch.ones_like(target_amp)
    threshold = 1e-4 * target_amp.max()
    Wcg[torch.abs(target_amp) < threshold] = 0.0
    Wcg = Wcg * flat_mask

    I_L_tot = make_beam().square().sum()
    target_amp = target_amp * Wcg
    target_phase = target_phase * Wcg
    I_Ta_w = target_amp.square().sum()
    target_amp = target_amp * torch.sqrt(I_L_tot / I_Ta_w.clamp(min=1e-30))

    return target_amp, target_phase, Wcg, r0


# ===========================================================================
# 评估函数（与 v95 evaluate_metrics 一致）
# ===========================================================================
def proper_uniformity(output_amp, target_amp, frac=0.9):
    flat_mask = target_amp > frac * target_amp.max()
    n = int(flat_mask.sum().item())
    if n < 10:
        return float('nan'), float('nan'), n
    I = output_amp[flat_mask].square()
    mean = I.mean().clamp(min=1e-30)
    rms = (I.std() / mean).item()
    pv = ((I.max() - I.min()) / mean).item()
    return rms, pv, n


def phase_flatness_metric(output, target_amp, frac=0.5):
    flat_mask = target_amp > frac * target_amp.max()
    if int(flat_mask.sum().item()) < 10:
        return float('nan')
    amp = output.abs()
    phi = torch.angle(output)
    w = amp * flat_mask.to(amp.dtype)
    w_sum = w.sum().clamp(min=1e-30)
    w_n = w / w_sum
    cmean = (w_n * torch.exp(1j * phi)).sum()
    phi_ref = torch.angle(cmean)
    dphi = torch.atan2(torch.sin(phi - phi_ref), torch.cos(phi - phi_ref))
    var = (w_n * dphi.square()).sum()
    return var.sqrt().item()


def spillover_ratio(output_amp, Wcg, beam):
    I = output_amp.square()
    total_input = beam.square().sum().clamp(min=1e-30)
    in_band = I[Wcg > 0].sum() / total_input
    out_of_band = I[Wcg == 0].sum() / total_input
    return out_of_band.item(), in_band.item()


def evaluate_metrics(phase, beam, target_amp, target_phase, Wcg, label=""):
    with torch.no_grad():
        field = beam * torch.exp(1j * phase)
        output = propagate(field)
        output_amp = output.abs()
        target_field = target_amp * torch.exp(1j * target_phase)
        # 使用与 cost 函数一致的 overlap 定义（Wcg 区域内）
        overlap = _differentiable_overlap(output, target_amp, target_phase, Wcg).item()

        Ta_sq = target_amp.square()
        E_out_sq = output_amp.square()
        cost_se = ((Ta_sq - E_out_sq) * Wcg).square().sum().item()

        target_region = target_amp > target_amp.max() * 0.5
        total_power = beam.square().sum().item()
        signal_power = output_amp[target_region].square().sum().item()
        efficiency = signal_power / max(total_power, 1e-12)

        in_wcg = output_amp[Wcg > 0]
        tgt_in_wcg = target_amp[Wcg > 0]
        if in_wcg.numel() > 1:
            wcg_corr = torch.corrcoef(torch.stack([tgt_in_wcg.flatten(), in_wcg.flatten()]))[0, 1].item()
        else:
            wcg_corr = 0.0

        proper_rms, proper_pv, flat_pix = proper_uniformity(output_amp, target_amp, frac=0.9)
        phase_std = phase_flatness_metric(output, target_amp, frac=0.5)
        spill, in_band = spillover_ratio(output_amp, Wcg, beam)

        return {
            "label": label,
            "overlap": overlap,
            "efficiency": efficiency,
            "wcg_corr": wcg_corr,
            "cost_se": cost_se,
            "flat_rms": proper_rms,
            "flat_pv": proper_pv,
            "phase_flatness": phase_std,
            "spillover": spill,
            "in_band": in_band,
        }


def print_metrics_table(results_list):
    keys = ["overlap", "efficiency", "wcg_corr", "flat_rms", "flat_pv",
            "phase_flatness", "spillover", "in_band", "cost_se"]
    header = f"{'指标':<18}" + "".join(f"{r['label']:>18}" for r in results_list)
    print(header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:<18}"
        for r in results_list:
            v = r[k]
            if isinstance(v, float):
                if k == "cost_se":
                    row += f"{v:>18.1f}"
                else:
                    row += f"{v:>18.4f}"
            else:
                row += f"{str(v):>18}"
        print(row)
    print()


# ===========================================================================
# 795 原程序方法：phase_guess 物理初相
# ===========================================================================
def phase_guess_795(dx, dy):
    """复刻 795.py 的 phase_guess()：基于物理几何的初始相位。"""
    mu = np.arctan2(dy, dx)
    ang = torch.tensor(mu, device=device)
    space = 1.8499
    # D 计算（与 795.py 完全一致）
    if abs(np.cos(mu)) > 1e-10:
        D_sign = 1.0 if (dy * dx > 0 and dy > 0) or (dy * dx < 0 and dy < 0) else -1.0
        D = D_sign * 2 * np.pi * PIXEL_PITCH * 1e3 / (WAVELENGTH * 1e6) / (FOCAL_LENGTH * 1e6) * (abs(dx) * space) / abs(np.cos(mu)) * 4
    else:
        D = 0.0

    y = torch.arange(SLM_W, device=device, dtype=torch.float64) - SLM_W / 2
    x = torch.arange(SLM_H, device=device, dtype=torch.float64) - SLM_H / 2
    X, Y = torch.meshgrid(x, y, indexing="xy")
    shr = SLM_H / 256

    asp = 0.9
    R = 0.9 / 1000
    B = 0.0
    KL = D * ((X / shr) * torch.cos(ang) + (Y / shr) * torch.sin(ang))
    KQ = 3 * R * (asp * (X / shr) ** 2 + (1 - asp) * (Y / shr) ** 2)
    KC = B * torch.sqrt((X / shr) ** 2 + (Y / shr) ** 2)
    z = KL + KQ + KC
    return z


# ===========================================================================
# 795 原程序方法：overlap-based cost (cost_SE_gpu)
# ===========================================================================
def _differentiable_overlap(output, target_amp, target_phase, Wcg):
    """可微分的 overlap 计算（返回 tensor，不调用 .item()）。

    复刻 795.py 的 overlap 公式：
      overlap = sum(Ta * E_out_amp * Wcg * cos(phi_out - phi_tgt))
                / (||Ta|| * ||E_out_amp * Wcg||)
    其中 ||·|| 是全范数（sum of squares），不是 Wcg 内范数。
    """
    E_out_amp = output.abs()
    E_out_p = torch.angle(output)
    inner = (target_amp * E_out_amp * Wcg * torch.cos(E_out_p - target_phase)).sum()
    norm_tgt = target_amp.square().sum().sqrt().clamp(min=1e-30)
    norm_out = (E_out_amp * Wcg).square().sum().sqrt().clamp(min=1e-30)
    return (inner / (norm_out * norm_tgt)).abs()


def make_cost_overlap(beam, target_amp, target_phase, Wcg):
    """复刻 795.py 的 cost_SE_gpu：overlap-based cost。"""
    Ta_sq = target_amp.square()

    def cost_fn(phase_flat):
        phase = phase_flat.reshape(SLM_H, SLM_W)
        field = beam * torch.exp(1j * phase)
        output = propagate(field)
        overlap = _differentiable_overlap(output, target_amp, target_phase, Wcg)
        C1 = 9
        cost = (10.0 ** C1) * (1.0 - overlap) ** 2
        return cost

    return cost_fn


def make_cost_overlap_eff(beam, target_amp, target_phase, Wcg):
    """复刻 795.py 的 cost_SE_gpu_e4：overlap × (1/eff)^8。"""
    Ta_sq = target_amp.square()

    def cost_fn(phase_flat):
        phase = phase_flat.reshape(SLM_H, SLM_W)
        field = beam * torch.exp(1j * phase)
        output = propagate(field)
        E_out_2 = output.abs().square()
        overlap = _differentiable_overlap(output, target_amp, target_phase, Wcg)
        I_out_w_tot = (E_out_2 * Wcg).sum()
        I_out_tot = E_out_2.sum()
        efficiency = I_out_w_tot / I_out_tot.clamp(min=1e-30)
        C1 = 9
        cost = (10.0 ** C1) * (1.0 - overlap) ** 2 * (1.0 / efficiency.clamp(min=1e-10)) ** 4
        return cost

    return cost_fn


# ===========================================================================
# 795 原程序：scipy CG 优化器（替代 torchmin）
# ===========================================================================
def run_795_cg_optimize(cost_fn, phase_init, maxiter, label=""):
    """用 scipy.optimize.fmin_cg 优化（复刻 795 原程序的 CG 行为）。"""
    phase_np = phase_init.detach().cpu().numpy().flatten().copy()
    eval_count = [0]
    t0 = time.time()

    def cost_and_grad(phi_flat):
        phase_t = torch.from_numpy(phi_flat).reshape(SLM_H, SLM_W).requires_grad_(True)
        cost = cost_fn(phase_t.flatten())
        cost.backward()
        grad = phase_t.grad.detach().cpu().numpy().flatten()
        eval_count[0] += 1
        if eval_count[0] % 10 == 0:
            print(f"    [{label}] eval {eval_count[0]}: cost={cost.item():.4e}")
        return cost.item(), grad

    cache = [None, None, None]
    def cost_c(p):
        if cache[0] is None or not np.array_equal(p, cache[0]):
            cache[0] = p.copy()
            cache[1], cache[2] = cost_and_grad(p)
        return cache[1]
    def grad_c(p):
        if cache[0] is None or not np.array_equal(p, cache[0]):
            cache[0] = p.copy()
            cache[1], cache[2] = cost_and_grad(p)
        return cache[2]

    res = scipy.optimize.fmin_cg(
        f=cost_c, x0=phase_np, fprime=grad_c,
        maxiter=maxiter, disp=False,
    )
    total_time = time.time() - t0
    print(f"  [{label}] 完成: evals={eval_count[0]}, 耗时={total_time:.1f}s")
    return torch.from_numpy(res).reshape(SLM_H, SLM_W), total_time


# ===========================================================================
# v95 Pipeline：LG vortex 初相
# ===========================================================================
def make_lg_phase(seed=42, curv_base=3.0, charge_spread=0.1):
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    curv = curv_base + torch.randn(1, generator=rng, device=device).item() * 0.5
    charge = (math.pi / 4.0) + torch.randn(1, generator=rng, device=device).item() * charge_spread
    y = torch.arange(SLM_H, device=device, dtype=torch.float64) - SLM_H / 2
    x = torch.arange(SLM_W, device=device, dtype=torch.float64) - SLM_W / 2
    X, Y = torch.meshgrid(x, y, indexing="xy")
    R = torch.sqrt(X ** 2 + Y ** 2)
    theta = torch.atan2(Y, X)
    phase = curv * (R / (SLM_H / 2)) ** 2 + charge * theta
    return phase


# ===========================================================================
# v95 Pipeline：Bowman warm-start (Stage 0)
# ===========================================================================
def run_v95_stage0_bowman(beam, target_amp, Wcg, target_phase, maxiter=80):
    """Stage 0: LG init + fmin_cg on ((Ta²-E²)·Wcg)²，track-by-overlap。"""
    print("\n  [v95 Stage 0] Bowman warm-start...")
    target_field = target_amp * torch.exp(1j * target_phase)
    Ta_sq = target_amp.square()
    init_phase = make_lg_phase(42)
    phase_np = init_phase.detach().cpu().numpy().flatten().copy()
    eval_count = [0]
    best_overlap = [-1.0]
    best_phase_arr = [phase_np.copy()]

    def cost_and_grad(phi_flat):
        phase_t = torch.from_numpy(phi_flat).reshape(SLM_H, SLM_W).requires_grad_(True)
        field = beam * torch.exp(1j * phase_t)
        output = propagate(field)
        E_out_2 = output.abs().square()
        cost = ((Ta_sq - E_out_2) * Wcg).square().sum()
        cost.backward()
        grad = phase_t.grad.detach().cpu().numpy().flatten()
        eval_count[0] += 1
        with torch.no_grad():
            overlap = _differentiable_overlap(output, target_amp, target_phase, Wcg).item()
            if overlap > best_overlap[0]:
                best_overlap[0] = overlap
                best_phase_arr[0] = phi_flat.copy()
        if eval_count[0] % 20 == 0:
            print(f"    [Stage0] eval {eval_count[0]}: overlap={overlap:.6f}")
        return cost.item(), grad

    cache = [None, None, None]
    def cost_c(p):
        if cache[0] is None or not np.array_equal(p, cache[0]):
            cache[0] = p.copy()
            cache[1], cache[2] = cost_and_grad(p)
        return cache[1]
    def grad_c(p):
        if cache[0] is None or not np.array_equal(p, cache[0]):
            cache[0] = p.copy()
            cache[1], cache[2] = cost_and_grad(p)
        return cache[2]

    t0 = time.time()
    scipy.optimize.fmin_cg(f=cost_c, x0=phase_np, fprime=grad_c, maxiter=maxiter, disp=False)
    total_time = time.time() - t0
    print(f"  [Stage 0] 完成: best_overlap={best_overlap[0]:.6f}, 耗时={total_time:.1f}s")
    return torch.from_numpy(best_phase_arr[0]).reshape(SLM_H, SLM_W), total_time


# ===========================================================================
# v95 Pipeline：Stage A — Anchored efficiency lift
# ===========================================================================
def run_v95_stage_a(beam, target_amp, Wcg, target_phase, bowman_phase, maxiter=50):
    """Stage A: 保持相位接近 Bowman 锚点，推高 efficiency/overlap。"""
    print("\n  [v95 Stage A] Anchored efficiency lift...")
    target_field = target_amp * torch.exp(1j * target_phase)
    phase_np = bowman_phase.detach().cpu().numpy().flatten().copy()
    eval_count = [0]
    best = {"cost": float("inf"), "phase": phase_np.copy()}
    lambda_init = 5000.0
    lambda_min = 300.0
    decay_every = 20

    def cost_and_grad(phi_flat):
        phase_t = torch.from_numpy(phi_flat).reshape(SLM_H, SLM_W).requires_grad_(True)
        field = beam * torch.exp(1j * phase_t)
        output = propagate(field)
        overlap_t = _differentiable_overlap(output, target_amp, target_phase, Wcg)
        # efficiency (differentiable)
        E_out_2 = output.abs().square()
        I_out_w_tot = (E_out_2 * Wcg).sum()
        I_out_tot = E_out_2.sum()
        eff_t = I_out_w_tot / I_out_tot.clamp(min=1e-30)
        eff_power = 2.0
        eff_loss = (1.0 / eff_t.clamp(min=1e-10)) ** eff_power
        main_cost = 1e9 * (1.0 - overlap_t) ** 2 * eff_loss
        # anchor penalty
        lam = max(lambda_init * (0.5 ** (eval_count[0] // decay_every)), lambda_min)
        dphi = torch.atan2(torch.sin(phase_t - bowman_phase), torch.cos(phase_t - bowman_phase))
        anchor_loss = lam * (dphi ** 2).mean()
        cost = main_cost + anchor_loss
        cost.backward()
        grad = phase_t.grad.detach().cpu().numpy().flatten()
        eval_count[0] += 1
        cv = cost.item()
        overlap_val = overlap_t.item()
        if not math.isfinite(cv) or cv > 1e10:
            return 1e15, np.zeros_like(grad)
        if cv < best["cost"]:
            best["cost"] = cv
            best["phase"] = phi_flat.copy()
        if eval_count[0] % 20 == 0:
            print(f"    [Stage A] eval {eval_count[0]}: overlap={overlap_val:.6f}, λ={lam:.0f}")
        return cv, grad

    cache = [None, None, None]
    def cost_c(p):
        if cache[0] is None or not np.array_equal(p, cache[0]):
            cache[0] = p.copy()
            cache[1], cache[2] = cost_and_grad(p)
        return cache[1]
    def grad_c(p):
        if cache[0] is None or not np.array_equal(p, cache[0]):
            cache[0] = p.copy()
            cache[1], cache[2] = cost_and_grad(p)
        return cache[2]

    t0 = time.time()
    scipy.optimize.fmin_cg(f=cost_c, x0=phase_np, fprime=grad_c, maxiter=maxiter, disp=False)
    total_time = time.time() - t0
    print(f"  [Stage A] 完成: 耗时={total_time:.1f}s")
    return torch.from_numpy(best["phase"]).reshape(SLM_H, SLM_W), total_time


# ===========================================================================
# v95 Pipeline：Stage B — Multi-snapshot fmin_cg
# ===========================================================================
def run_v95_stage_b(beam, target_amp, Wcg, target_phase, stage_a_phase, maxiter=60):
    """Stage B: 无锚点 overlap/eff 推压，多角度记录候选。"""
    print("\n  [v95 Stage B] Multi-snapshot fmin_cg...")
    target_field = target_amp * torch.exp(1j * target_phase)
    phase_np = stage_a_phase.detach().cpu().numpy().flatten().copy()
    eval_count = [0]
    snapshots = {}

    def cost_and_grad(phi_flat):
        phase_t = torch.from_numpy(phi_flat).reshape(SLM_H, SLM_W).requires_grad_(True)
        field = beam * torch.exp(1j * phase_t)
        output = propagate(field)
        overlap_t = _differentiable_overlap(output, target_amp, target_phase, Wcg)
        # efficiency (differentiable)
        E_out_2 = output.abs().square()
        I_out_w_tot = (E_out_2 * Wcg).sum()
        I_out_tot = E_out_2.sum()
        eff_t = I_out_w_tot / I_out_tot.clamp(min=1e-30)
        eff_power = 2.0
        cost = 1e9 * (1.0 - overlap_t) ** 2 * (1.0 / eff_t.clamp(min=1e-10)) ** eff_power
        cost.backward()
        grad = phase_t.grad.detach().cpu().numpy().flatten()
        eval_count[0] += 1
        overlap_val = overlap_t.item()
        eff_val = eff_t.item()
        with torch.no_grad():
            ph = phase_flatness_metric(output, target_amp, frac=0.5)
        cv = cost.item()
        # snapshots
        if "B_cost_min" not in snapshots or cv < snapshots["B_cost_min"]["cost"]:
            snapshots["B_cost_min"] = {"phase": phi_flat.copy(), "cost": cv,
                                        "overlap": overlap_val, "phase_flat": ph, "eff": eff_val}
        if "B_phase_min" not in snapshots or ph < snapshots["B_phase_min"]["phase_flat"]:
            snapshots["B_phase_min"] = {"phase": phi_flat.copy(), "cost": cv,
                                         "overlap": overlap_val, "phase_flat": ph, "eff": eff_val}
        if ph <= 0.30:
            if "B_constrained" not in snapshots or cv < snapshots["B_constrained"]["cost"]:
                snapshots["B_constrained"] = {"phase": phi_flat.copy(), "cost": cv,
                                               "overlap": overlap_val, "phase_flat": ph, "eff": eff_val}
        if eval_count[0] % 20 == 0:
            print(f"    [Stage B] eval {eval_count[0]}: overlap={overlap_val:.6f}, phase_flat={ph:.4f}, eff={eff_val:.4f}")
        return cv, grad

    cache = [None, None, None]
    def cost_c(p):
        if cache[0] is None or not np.array_equal(p, cache[0]):
            cache[0] = p.copy()
            cache[1], cache[2] = cost_and_grad(p)
        return cache[1]
    def grad_c(p):
        if cache[0] is None or not np.array_equal(p, cache[0]):
            cache[0] = p.copy()
            cache[1], cache[2] = cost_and_grad(p)
        return cache[2]

    t0 = time.time()
    scipy.optimize.fmin_cg(f=cost_c, x0=phase_np, fprime=grad_c, maxiter=maxiter, disp=False)
    total_time = time.time() - t0

    for tag, s in snapshots.items():
        print(f"    [Stage B] {tag}: overlap={s['overlap']:.4f}, phase_flat={s['phase_flat']:.4f}, eff={s['eff']:.4f}")

    # 选 constrained_best 作为出口
    if "B_constrained" in snapshots:
        exit_snap = snapshots["B_constrained"]
        print(f"  [Stage B] 出口: B_constrained")
    else:
        exit_snap = snapshots.get("B_phase_min", snapshots.get("B_cost_min"))
        print(f"  [Stage B] 出口: fallback")

    return torch.from_numpy(exit_snap["phase"]).reshape(SLM_H, SLM_W), total_time, snapshots


# ===========================================================================
# v95 Pipeline：Stage C — Adam trust-region polish
# ===========================================================================
def run_v95_stage_c(beam, target_amp, Wcg, target_phase, stage_b_phase, steps=60, lr=0.0015):
    """Stage C: 小步 Adam 同时下压 phase 和 flat_rms。"""
    print("\n  [v95 Stage C] Adam trust-region polish...")
    target_field = target_amp * torch.exp(1j * target_phase)
    phase_t = stage_b_phase.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([phase_t], lr=lr)
    snapshots = []

    init_metrics = evaluate_metrics(phase_t.detach(), beam, target_amp, target_phase, Wcg)
    print(f"    [Stage C] init: phase={init_metrics['phase_flatness']:.4f}, flat_rms={init_metrics['flat_rms']:.4f}")

    t0 = time.time()
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        field = beam * torch.exp(1j * phase_t)
        output = propagate(field)
        amp = output.abs()
        phi = torch.angle(output)

        # uniformity loss
        flat_mask = target_amp > 0.9 * target_amp.max()
        if flat_mask.sum() > 10:
            I_plateau = amp[flat_mask].square()
            uni_loss = I_plateau.std() / I_plateau.mean().clamp(min=1e-12)
        else:
            uni_loss = torch.tensor(0.0)

        # phase flatness loss
        pmask = target_amp > 0.5 * target_amp.max()
        if pmask.sum() > 10:
            w = (amp * pmask.to(amp.dtype)).detach()
            w = w / w.sum().clamp(min=1e-30)
            cmean = (w * torch.exp(1j * phi)).sum()
            phi_ref = torch.angle(cmean)
            dphi = torch.atan2(torch.sin(phi - phi_ref), torch.cos(phi - phi_ref))
            ph_var = (w * dphi.square()).sum()
        else:
            ph_var = torch.tensor(0.0)

        # soft/hard phase cap
        soft_cap = 0.32
        hard_cap = 0.40
        soft_penalty = 10.0 * torch.relu(ph_var - soft_cap ** 2)
        hard_penalty = 1e4 * torch.relu(ph_var - hard_cap ** 2)

        # overlap retention (differentiable)
        overlap_t = _differentiable_overlap(output, target_amp, target_phase, Wcg)
        overlap_loss = 100.0 * torch.relu(0.93 - overlap_t) ** 2

        loss = uni_loss + soft_penalty + hard_penalty + overlap_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([phase_t], 1.0)
        optimizer.step()

        with torch.no_grad():
            phase_t.data = torch.remainder(phase_t.data, 2 * math.pi)

        if step % 10 == 0 or step == steps:
            m = evaluate_metrics(phase_t.detach(), beam, target_amp, target_phase, Wcg)
            print(f"    [Step {step:3d}] phase={m['phase_flatness']:.4f}, flat_rms={m['flat_rms']:.4f}, overlap={m['overlap']:.4f}")
            # accept gate: flat_rms improve ≥1% OR (phase drop ≥0.005 AND flat ≤ best*1.02)
            if not snapshots or m["flat_rms"] < snapshots[-1]["flat_rms"] * 0.99 or \
               (m["phase_flatness"] < snapshots[-1]["phase_flatness"] - 0.005 and m["flat_rms"] <= snapshots[-1]["flat_rms"] * 1.02):
                snapshots.append(m.copy())
                snapshots[-1]["phase_tensor"] = phase_t.detach().clone()

    total_time = time.time() - t0
    if snapshots:
        best_snap = min(snapshots, key=lambda s: s["phase_flatness"])
        print(f"  [Stage C] 完成: {len(snapshots)} snapshots, best phase={best_snap['phase_flatness']:.4f}")
        return best_snap["phase_tensor"], total_time
    return phase_t.detach(), total_time


# ===========================================================================
# v95 Pipeline：Stage D — Damped WGS + TV-prox
# ===========================================================================
def run_v95_stage_d(beam, target_amp, Wcg, target_phase, stage_c_phase, alphas=None):
    """Stage D: 经典 GS 在焦平面强制振幅匹配 + TV 抑制相位毛刺。"""
    if alphas is None:
        alphas = [0.50, 0.35, 0.25, 0.15, 0.08, 0.04]
    print(f"\n  [v95 Stage D] Damped WGS + TV-prox (alphas={alphas})...")
    phase_cap = 0.42
    init_metrics = evaluate_metrics(stage_c_phase, beam, target_amp, target_phase, Wcg)
    print(f"    [Stage D] init: phase={init_metrics['phase_flatness']:.4f}, flat_rms={init_metrics['flat_rms']:.4f}")

    current_phase = stage_c_phase.clone()
    candidates = []
    t0 = time.time()

    for rd in range(len(alphas)):
        alpha = alphas[rd]
        with torch.no_grad():
            field = beam * torch.exp(1j * current_phase)
            output = propagate(field)
            out_amp = output.abs()
            out_phi = torch.angle(output)
            # damped amplitude replacement
            out_amp_new = alpha * target_amp + (1 - alpha) * out_amp
            out_new = out_amp_new * torch.exp(1j * out_phi)
            # inverse propagate (IFFT then crop to SLM size)
            A0_inv = math.sqrt(OUT_H * OUT_W)
            slm_full = A0_inv * torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(out_new)))
            # crop center SLM_H x SLM_W
            crop_y = (OUT_H - SLM_H) // 2
            crop_x = (OUT_W - SLM_W) // 2
            phase_candidate = torch.angle(slm_full[crop_y:crop_y+SLM_H, crop_x:crop_x+SLM_W])
            # TV-prox step (1 step)
            tv_lr = 5e-4
            dx_tv = phase_candidate[:, 1:] - phase_candidate[:, :-1]
            dy_tv = phase_candidate[1:, :] - phase_candidate[:-1, :]
            tv_grad_x = torch.zeros_like(phase_candidate)
            tv_grad_y = torch.zeros_like(phase_candidate)
            tv_grad_x[:, 1:] += dx_tv.sign()
            tv_grad_x[:, :-1] -= dx_tv.sign()
            tv_grad_y[1:, :] += dy_tv.sign()
            tv_grad_y[:-1, :] -= dy_tv.sign()
            phase_candidate = phase_candidate - tv_lr * (tv_grad_x + tv_grad_y)

        # guard check
        m = evaluate_metrics(phase_candidate, beam, target_amp, target_phase, Wcg)
        phase_ok = m["phase_flatness"] is not None and not math.isnan(m["phase_flatness"]) and m["phase_flatness"] <= phase_cap
        init_flat = init_metrics["flat_rms"] if not math.isnan(init_metrics["flat_rms"]) else float("inf")
        flat_ok = m["flat_rms"] is not None and not math.isnan(m["flat_rms"]) and m["flat_rms"] < init_flat * 0.97
        overlap_ok = m["overlap"] >= init_metrics["overlap"] * 0.97
        eff_ok = m["efficiency"] >= init_metrics["efficiency"] * 0.95

        accepted = phase_ok and flat_ok and overlap_ok and eff_ok
        status = "accept" if accepted else "REJECT"
        print(f"    [rd{rd+1}] α={alpha:.2f}: phase={m['phase_flatness']:.4f}, flat_rms={m['flat_rms']:.4f}, "
              f"overlap={m['overlap']:.4f} [{status}]")

        if accepted:
            current_phase = phase_candidate
            candidates.append({"phase": phase_candidate.clone(), "metrics": m, "alpha": alpha, "rd": rd + 1})
        else:
            break  # alpha backtracking: try next smaller alpha; if all fail, stop

    total_time = time.time() - t0
    print(f"  [Stage D] 完成: {len(candidates)} accepted rounds, 耗时={total_time:.1f}s")
    return current_phase, total_time, candidates


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    print("=" * 70)
    print("795 原程序 vs v95 Pipeline 模拟对比")
    print("=" * 70)

    # 共享物理设置
    beam = make_beam()
    target_amp, target_phase, Wcg, r0 = make_target_and_weight()
    target_field = target_amp * torch.exp(1j * target_phase)
    norm_tgt = target_field.abs().square().sum().sqrt()

    results = []

    # ===================================================================
    # 方法 1：795 原程序（phase_guess + CG overlap + CG overlap×eff）
    # ===================================================================
    print("\n" + "=" * 70)
    print("方法 1：795 原程序")
    print("=" * 70)

    # 物理初相
    dx_px = round(-100e-6 / focal_spx)
    dy_px = round(-100e-6 / focal_spx)
    init_phi_795 = phase_guess_795(dx_px, dy_px)
    print(f"  phase_guess 初相: dx={dx_px}, dy={dy_px}")

    # 第一轮 CG：overlap-based cost
    print("\n--- 795 第一轮 CG: overlap-based cost ---")
    cost_overlap = make_cost_overlap(beam, target_amp, target_phase, Wcg)
    phase_795_r1, t1 = run_795_cg_optimize(cost_overlap, init_phi_795, maxiter=20, label="795-CG-round1")

    # 第二轮 CG：overlap × efficiency cost
    print("\n--- 795 第二轮 CG: overlap × (1/eff)^4 ---")
    cost_eff = make_cost_overlap_eff(beam, target_amp, target_phase, Wcg)
    phase_795_r2, t2 = run_795_cg_optimize(cost_eff, phase_795_r1, maxiter=20, label="795-CG-round2")

    # 评估
    m_795_r1 = evaluate_metrics(phase_795_r1, beam, target_amp, target_phase, Wcg, label="795 CG round1")
    m_795_r2 = evaluate_metrics(phase_795_r2, beam, target_amp, target_phase, Wcg, label="795 CG round2")
    m_795_r1["time"] = t1
    m_795_r2["time"] = t2
    results.append(m_795_r1)
    results.append(m_795_r2)

    # ===================================================================
    # 方法 2：v95 Pipeline
    # ===================================================================
    print("\n" + "=" * 70)
    print("方法 2：v95 Pipeline")
    print("=" * 70)

    # Stage 0: Bowman warm-start
    phase_s0, t_s0 = run_v95_stage0_bowman(beam, target_amp, Wcg, target_phase, maxiter=30)

    # Stage A: Anchored efficiency lift
    phase_sA, t_sA = run_v95_stage_a(beam, target_amp, Wcg, target_phase, phase_s0, maxiter=20)

    # Stage B: Multi-snapshot fmin_cg
    phase_sB, t_sB, snaps_B = run_v95_stage_b(beam, target_amp, Wcg, target_phase, phase_sA, maxiter=20)

    # Stage C: Adam trust-region polish
    phase_sC, t_sC = run_v95_stage_c(beam, target_amp, Wcg, target_phase, phase_sB, steps=20)

    # Stage D: Damped WGS + TV-prox
    phase_sD, t_sD, candidates_D = run_v95_stage_d(beam, target_amp, Wcg, target_phase, phase_sC)

    # 评估所有 v95 候选
    m_s0 = evaluate_metrics(phase_s0, beam, target_amp, target_phase, Wcg, label="v95 Stage0")
    m_sA = evaluate_metrics(phase_sA, beam, target_amp, target_phase, Wcg, label="v95 StageA")
    m_sB = evaluate_metrics(phase_sB, beam, target_amp, target_phase, Wcg, label="v95 StageB")
    m_sC = evaluate_metrics(phase_sC, beam, target_amp, target_phase, Wcg, label="v95 StageC")
    m_sD = evaluate_metrics(phase_sD, beam, target_amp, target_phase, Wcg, label="v95 StageD")

    total_v95_time = t_s0 + t_sA + t_sB + t_sC + t_sD
    m_sD["time"] = total_v95_time
    m_795_r2["time"] = t1 + t2

    results.append(m_s0)
    results.append(m_sA)
    results.append(m_sB)
    results.append(m_sC)
    results.append(m_sD)

    # D 候选池
    for cand in candidates_D:
        mc = cand["metrics"].copy()
        mc["label"] = f"v95 D_rd{cand['rd']}_a{cand['alpha']:.2f}"
        results.append(mc)

    # ===================================================================
    # 输出对比表
    # ===================================================================
    print("\n" + "=" * 70)
    print("完整对比结果")
    print("=" * 70)
    print_metrics_table(results)

    # 核心对比：795 round2 vs v95 best D candidate
    print("=" * 70)
    print("核心对比: 795 CG round2 vs v95 Pareto knee")
    print("=" * 70)
    # 找 v95 最佳候选（phase<0.30 且 flat_rms 最低）
    v95_candidates = [r for r in results if r["label"].startswith("v95") and
                      r.get("phase_flatness") is not None and r["phase_flatness"] < 0.35]
    if v95_candidates:
        v95_best = min(v95_candidates, key=lambda r: r.get("flat_rms", float("inf")))
    else:
        v95_best = m_sD

    key_results = [m_795_r2, v95_best]
    print_metrics_table(key_results)

    # 总结
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print(f"795 原程序 (round2): phase={m_795_r2['phase_flatness']:.4f}, flat_rms={m_795_r2['flat_rms']:.4f}, "
          f"eff={m_795_r2['efficiency']:.4f}, overlap={m_795_r2['overlap']:.4f}")
    print(f"v95 Pipeline (best): phase={v95_best['phase_flatness']:.4f}, flat_rms={v95_best['flat_rms']:.4f}, "
          f"eff={v95_best['efficiency']:.4f}, overlap={v95_best['overlap']:.4f}")
    print()
    phase_winner = "v95" if v95_best["phase_flatness"] < m_795_r2["phase_flatness"] else "795"
    flat_winner = "v95" if v95_best["flat_rms"] < m_795_r2["flat_rms"] else "795"
    eff_winner = "795" if m_795_r2["efficiency"] > v95_best["efficiency"] else "v95"
    overlap_winner = "795" if m_795_r2["overlap"] > v95_best["overlap"] else "v95"
    print(f"等相位 (phase_flatness): {phase_winner} 胜出")
    print(f"均匀性 (flat_rms):      {flat_winner} 胜出")
    print(f"光效率 (efficiency):     {eff_winner} 胜出")
    print(f"重叠度 (overlap):        {overlap_winner} 胜出")


if __name__ == "__main__":
    main()
