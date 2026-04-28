import torch

def create_cost_function(
    # --- 核心物理参数 ---
    L, Ta, P, Wcg, N, NT, A0,
    # --- 外部依赖的函数 ---
    fft_op, get_centre_range_op,
    # --- 三个目标的权衡参数 (λ) ---
    lambda_intensity=0.6, 
    lambda_phase=0.1,
    lambda_efficiency=0.3
):
    """
    一个先进的成本函数工厂，用于平衡三个核心目标：
    1. 强度保真度 (Intensity Fidelity)
    2. 相位保真度 (Phase Fidelity)
    3. 衍射效率 (Efficiency)

    通过三个 lambda 参数，可以直观地控制优化方向。
    """
    n_pixelsx, n_pixelsy = N
    
    # 预处理，避免在循环中重复计算
    if torch.is_complex(L):
        profile_s_r = L.real.double()
        profile_s_i = L.imag.double()
    else:
        profile_s_r = L.double()
        profile_s_i = torch.zeros_like(L, dtype=torch.float64)

    # 预计算目标能量，用于归一化强度误差
    # 我们用 I_target^2 = Ta^4 作为能量项
    target_intensity_energy = torch.sum(torch.pow(Ta, 4) * Wcg) + 1e-9

    # 这个内部函数才是最终传递给优化器的成本函数
    def cost_function(phi_flat):
        
        # --- 核心物理计算 ---
        phi_reshaped = phi_flat.view(n_pixelsx, n_pixelsy)

        E_in_r = A0 * (profile_s_r * torch.cos(phi_reshaped) - profile_s_i * torch.sin(phi_reshaped))
        E_in_i = A0 * (profile_s_i * torch.cos(phi_reshaped) + profile_s_r * torch.sin(phi_reshaped))

        E_in_r_pad = torch.zeros(NT, dtype=torch.float64, device=L.device)
        E_in_i_pad = torch.zeros(NT, dtype=torch.float64, device=L.device)

        idx_0x, idx_1x = get_centre_range_op(n_pixelsx, NT[0])
        idx_0y, idx_1y = get_centre_range_op(n_pixelsy, NT[1])
        
        E_in_r_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_r
        E_in_i_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_i

        # 假设 fft_op 返回一个复数张量
        E_out_complex = fft_op(torch.complex(E_in_r_pad, E_in_i_pad))
        
        I_out = E_out_complex.abs().pow(2)
        E_out_p = E_out_complex.angle()
        
        # --- 1. 计算强度保真度误差 (Normalized MSE) ---
        intensity_error_raw = torch.sum(torch.pow((torch.pow(Ta, 2) - I_out) * Wcg, 2))
        norm_intensity_error = intensity_error_raw / target_intensity_energy
        
        # --- 2. 计算相位保真度误差 ---
        # 使用 (1 - cos(delta_phi)) 的形式，因为它对 2π 周期性友好且平滑
        phase_error = torch.sum(Wcg * (1 - torch.cos(E_out_p - P))) / (torch.sum(Wcg) + 1e-9)
        
        # --- 3. 计算效率误差 ---
        I_out_w_tot = torch.sum(I_out * Wcg)
        I_out_tot = torch.sum(I_out)
        efficiency = I_out_w_tot / (I_out_tot + 1e-9)
        efficiency_error = (1 - efficiency)
        
        # --- 4. 最终的加权成本 ---
        # 确保权重总和为1
        total_lambda = lambda_intensity + lambda_phase + lambda_efficiency
        if total_lambda <= 0: total_lambda = 1 # 避免除以零
        
        cost = (lambda_intensity / total_lambda) * norm_intensity_error + \
               (lambda_phase / total_lambda) * phase_error + \
               (lambda_efficiency / total_lambda) * efficiency_error
               
        return cost

    return cost_function

