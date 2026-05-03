# -*- coding: utf-8 -*-

#%%
import numpy as np                          # 用于数组操作
import matplotlib.pyplot as plt             # 绘图
import torch                                 # 使用 PyTorch 进行张量计算
import SLM_1X as slm                         # 包含 SLM 属性、场计算、目标和绘图属性
# import CG_1 as cg                           # CG 计算和诊断图
import os, shutil                           # 文件夹/文件操作
import time
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
# from sipyco.pc_rpc import Client
from scipy.ndimage import binary_dilation, rotate
import torch.nn as nn
from PIL import Image
import CG_2 as cg2
import CG_new as cg1

def gaussian_peak_only(dim, r0, d, sigmax, sigmay, range, A=1.0, bump_amp=1.0, bump_width_ratio=0.1, save_param=False, device='cuda'):
    """
    只生成一个高斯峰（无平顶分布）
    """
    cols, rows = dim
    x = torch.arange(rows, dtype=torch.float32, device=device)
    y = torch.arange(cols, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    x_centered = X - r0[0]
    y_centered = Y - r0[1]

    bump_width = d * bump_width_ratio
    # 峰的中心位置
    bump_center = 0.5 * (range[0] + range[1])
    fx_bump = ((x_centered - bump_center).pow(2)) / (bump_width ** 2)

    # 只保留凸起部分
    z = bump_amp * A * torch.exp(-fx_bump) * torch.exp(-(y_centered.pow(2)) / (sigmay ** 2))

    if save_param:
        param_str = f"GaussianPeakOnly | dim={dim} | r0={r0} | d={d} | σ={sigmax:.2f} | A={A} | bump_amp={bump_amp} | bump_w={bump_width_ratio}"
        return z, param_str
    else:
        return z                  
def gaussian_line_peak(dim, r0, d, sigmax,sigmay,range,  A=1.0 ,bump_amp=0.02, bump_width_ratio=0.1, save_param=False, device='cuda'):
    """生成带平顶区域局部凸起的高斯线形相位
    Args:
        dim (tuple): 相位图尺寸 (cols, rows)
        r0 (tuple): 中心坐标 (x0, y0)
        d (float): 平顶区域宽度
        sigma (float): 高斯衰减系数
        A (float): 整体幅值
        bump_amp (float): 凸起高度（相对A的比例）
        bump_width_ratio (float): 凸起宽度与d的比例
        save_param (bool): 是否返回参数信息
        device (str): 计算设备
    Returns:
        z (Tensor): 生成的相位图
        param_used (str, optional): 参数信息
    """
    cols, rows = dim
    x = torch.arange(rows, dtype=torch.float32, device=device)
    y = torch.arange(cols, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    x_centered = X - r0[0]
    y_centered = Y - r0[1]
    
    fx = 0.5 * (
        torch.abs(x_centered - d / 2) + 
        torch.abs(x_centered + d / 2) - 
        d
    )

    bump_width = d * bump_width_ratio
    fx_bump =( ((x_centered-0.5*(range[0]+range[1])).pow(2)) / ((bump_width ** 2)))
    bump_mask = (range[0]<=x_centered) & (x_centered<= range[1]) 
    x2d=(bump_mask*torch.exp(-(fx_bump)))
    min_nonzero = x2d[x2d != 0].min()
    # 高斯分布合成
    z = (A * torch.exp( 
        -(fx.pow(2)) / 
        (sigmax ** 2))+bump_mask*bump_amp*A*(torch.exp( 
        -(fx_bump))-min_nonzero))* torch.exp(-(y_centered.pow(2)) / 
        (sigmay ** 2))

    if save_param:
        param_str = f"GausLinePeak | dim={dim} | r0={r0} | d={d} | σ={sigmax:.2f} | A={A} | bump_amp={bump_amp} | bump_w={bump_width_ratio}"
        return z, param_str
    else:
        return z

def get_centre_range(n,N):
    # returns the indices to use given an nxn SLM
    return int(N/2)-int(n/2),int(N/2)+int(n/2)

class InverseFourierOp:
    def __init__(self):
        pass

    def make_node(self, xr, xi):
        xr = torch.as_tensor(xr)
        xi = torch.as_tensor(xi)
        return xr, xi  # Return the input tensors for now
    
    def perform(self, node, inputs, output_storage):
        xr, xi = inputs  # 保持输入方式不变
        x = xr + 1j * xi
        nx, ny = xr.shape
        s = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x))) * (nx * ny)
        output_storage[0][0] = s.real
        output_storage[1][0] = s.imag

class FourierOp:
    __props__ = ()
    
    def make_node(self, xr, xi):
        xr = torch.as_tensor(xr)
        xi = torch.as_tensor(xi)
        return xr, xi  # 返回输入张量

    def perform(self, node, inputs, output_storage):
        x = inputs[0] + 1j * inputs[1]
        s = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))
        output_storage[0] = s.real
        output_storage[1] = s.imag
        z_r = output_storage[0]
        z_i = output_storage[1]
        
    def grad(self, inputs, output_gradients):
        z_r = output_gradients[0]
        z_i = output_gradients[1]
        y = InverseFourierOp()(z_r, z_i)
        return y

    def __call__(self, xr, xi):
        # 创建节点
        inputs = self.make_node(xr, xi)
        output_storage = [torch.empty_like(xr), torch.empty_like(xi)]
        # 执行傅里叶变换
        self.perform(None, inputs, output_storage)
        return output_storage[0], output_storage[1]  # 返回实部和虚部

fft = FourierOp()

device = torch.device("cuda")
torch.set_default_device(device)


# def phase_guess(dim, D, asp, R, ang, B, save_param=False):
#     """
#     Create n x n guess phase: 
#     'D' required radius of shift from origin
#     'asp' aspect ratio of "spreading" for quadratic profile
#     'R' required curvature of quadratic profile
#     'ang' required angle of shift from origin
#     'B' radius of ring in output plane
#     """
#     cols, rows = dim
    
#     # Initialization
#     x = torch.arange(rows) - rows / 2  # Columns
#     y = torch.arange(cols) - cols / 2  # Rows
#     X, Y = torch.meshgrid(x, y, indexing='xy')  # Use meshgrid for 2D arrays
#     z = torch.zeros(size=(rows, cols))

#     # target definition
#     KL = D*((X/shr)*torch.cos(ang)+(Y/shr)*torch.sin(ang));
#     KQ = 3*R*((asp*(torch.pow((X/shr),2))+(1-asp)*(torch.pow((Y/shr),2))));
#     KC = B*torch.pow((torch.pow((X/shr),2)+torch.pow((Y/shr),2)),0.5);
#     z = KC+KQ+KL;
#     z = torch.reshape(z, (rows * cols,))
    
#     if save_param :
#         param_used = "phase_guess | n={0} | D={1} | asp={2} | R={3} | ang={4} | B={5}".format(rows, D, asp, R, ang, B)
#         return z, param_used
#     else :
#         return z
def generate_tilted_grating(size, period = 16,angle_deg=45):
    height, width = size[0], size[1]
    x_coords = (np.arange(width) - width / 2)
    y_coords = (np.arange(height) - height / 2)
    X, Y = np.meshgrid(x_coords, y_coords)


    angle_rad = np.deg2rad(angle_deg)
    X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)


    phase = np.mod(2 * np.pi * X_rot / period, 2 * np.pi)

    return phase
def phase_guess(dim,dx, dy, asp, R,  B, save_param=False):
    """
    Create n x n guess phase: 
    'D' required radius of shift from origin
    'asp' aspect ratio of "spreading" for quadratic profile
    'R' required curvature of quadratic profile
    'ang' required angle of shift from origin
    'B' radius of ring in output plane
    """
    cols, rows = dim
    mu=np.arctan(dy/dx)
    ang=torch.tensor(mu)
    space=1.8499
    if dy/dx>0:
        if dy<0:
            D=-2*np.pi*12.5/(lambda_*1e6)/(focallength*1e6)*(abs(dx)*space)/np.cos(mu)*4
        else:
            D=2*np.pi*12.5/(lambda_*1e6)/(focallength*1e6)*(abs(dx)*space)/np.cos(mu)*4
    if dy/dx<0:
        if dy<0:
            D=2*np.pi*12.5/(lambda_*1e6)/(focallength*1e6)*(abs(dx)*space)/np.cos(mu)*4
        else:
            D=-2*np.pi*12.5/(lambda_*1e6)/(focallength*1e6)*(abs(dx)*space)/np.cos(mu)*4
    # Initialization
    x = torch.arange(rows) - rows / 2  # Columns
    y = torch.arange(cols) - cols / 2  # Rows
    X, Y = torch.meshgrid(x, y, indexing='xy')  # Use meshgrid for 2D arrays
    z = torch.zeros(size=(rows, cols))

    # target definition
    KL = D*((X/shr)*torch.cos(ang)+(Y/shr)*torch.sin(ang));
    KQ = 3*R*((asp*(torch.pow((X/shr),2))+(1-asp)*(torch.pow((Y/shr),2))));
    KC = B*torch.pow((torch.pow((X/shr),2)+torch.pow((Y/shr),2)),0.5);
    z = KC+KQ+KL;
    z = torch.reshape(z, (rows * cols,))
    
    if save_param :
        param_used = "phase_guess | n={0} | D={1} | asp={2} | R={3} | ang={4} | B={5}".format(rows, D, asp, R, ang, B)
        return z, param_used
    else :
        return z
    

def get_init_phi(dim,a1,a2):

    cols, rows = dim
    mu=np.arctan(dy/dx)
    ang=torch.tensor(mu)
    space=1.4
    if dy/dx>0:
        if dy<0:
            D=-2*np.pi*12.5/(lambda_*1e6)/(focallength*1e6)*(abs(dx)*space)/np.cos(mu)*4
        else:
            D=2*np.pi*12.5/(lambda_*1e6)/(focallength*1e6)*(abs(dx)*space)/np.cos(mu)*4
    if dy/dx<0:
        if dy<0:
            D=2*np.pi*12.5/(lambda_*1e6)/(focallength*1e6)*(abs(dx)*space)/np.cos(mu)*4
        else:
            D=-2*np.pi*12.5/(lambda_*1e6)/(focallength*1e6)*(abs(dx)*space)/np.cos(mu)*4
    # Initialization
    x = torch.arange(rows) - rows / 2  # Columns
    y = torch.arange(cols) - cols / 2  # Rows
    X, Y = torch.meshgrid(x, y, indexing='xy')  # Use meshgrid for 2D arrays
    z = torch.zeros(size=(rows, cols))

    z=1/(lambda_*focallength)*4*a1*a2*torch.sqrt( (12.5*X/(0.35*a1)/1000)**2+1 )
    z = torch.reshape(z, (rows * cols,))
    return z

def get_init_phi(dim,a1,a2):
    """
    Create n x n guess phase: 
    'D' required radius of shift from origin
    'asp' aspect ratio of "spreading" for quadratic profile
    'R' required curvature of quadratic profile
    'ang' required angle of shift from origin
    'B' radius of ring in output plane
    """
    cols, rows = dim
    mu=np.arctan(dy/dx)
    ang=torch.tensor(mu)
    space=1.4
    if dy/dx>0:
        if dy<0:
            D=-2*np.pi*12.5/(lambda_*1e6)/(focallength*1e6)*(abs(dx)*space)/np.cos(mu)*4
        else:
            D=2*np.pi*12.5/(lambda_*1e6)/(focallength*1e6)*(abs(dx)*space)/np.cos(mu)*4
    if dy/dx<0:
        if dy<0:
            D=2*np.pi*12.5/(lambda_*1e6)/(focallength*1e6)*(abs(dx)*space)/np.cos(mu)*4
        else:
            D=-2*np.pi*12.5/(lambda_*1e6)/(focallength*1e6)*(abs(dx)*space)/np.cos(mu)*4
    # Initialization
    x = torch.arange(rows) - rows / 2  # Columns
    y = torch.arange(cols) - cols / 2  # Rows
    X, Y = torch.meshgrid(x, y, indexing='xy')  # Use meshgrid for 2D arrays
    term1 = torch.sqrt(torch.tensor(2 * np.pi)) * a1 * a2 * torch.exp(-2 * (X*12.5/ a1/1000000)**2)
    term2 = 2 * np.pi * a2 * X*12.5/1000000 * torch.erf(torch.sqrt(torch.tensor(2)) * X*12.5/1000000 / a1)
    z = (1/(lambda_ * focallength)) * (term1 + term2)+D*((X/shr)*torch.cos(ang)+(Y/shr)*torch.sin(ang))

    z = torch.reshape(z, (rows * cols,))
    return z
def gaussian_linex(dim, r0, d,d2, sigma, A=1.0, save_param=False, device='cuda'):

    cols, rows = dim

    # Initialization
    device = 'cuda'
    x = torch.arange(rows, dtype=torch.float64, device=device)  # Grid points along cols
    y = torch.arange(cols, dtype=torch.float64, device=device)  # Grid points along rows
    X, Y = torch.meshgrid(x, y, indexing='xy')  # Cartesian indexing    
    # Target definition
    fx = 0.5 * (torch.abs(X - d / 2. - r0[1]) + torch.abs(X + d / 2. - r0[1]) - d)
    fy= 0.5*(torch.abs(Y-d2/2.-r0[0])+torch.abs(Y+d2/2.-r0[0])-d2)
    z = A * torch.exp(-(fx**2 + fy**2) / sigma**2)

    if save_param:
        param_used = f"gaussian_line | n={rows} | r0={r0} | d={d} | sigma={sigma} | A={A}"
        return z, param_used
    else:
        return z
    

#%%
N = [1024,1272] # SLM 是 NxN 像素
shr=1024/256
#   ================================================================================================
#   |          SLM 像素
#   ================================================================================================
numb=2
# NT =  [x * numb for x in N]  # 模型输出平面为 NTxNT 像素阵列 - 更高分辨2率
NT =  [2514,2514]
NT =  [4096,4096]
# NT =  [7540,7540]  # 模型输出平面为 NTxNT 像素阵列 - 更高分辨率
NT=[6876,6876]
#   ================================================================================================
#   |          激光束
#   ================================================================================================
spix = 0.0125  # SLM 像素大小，单位为 mm
lambda_ = 795e-9
magnification=1
focallength=0.2
focal_spy=lambda_*focallength/(NT[0]*(spix/1000)*magnification)
focal_spx=lambda_*focallength/(NT[1]*(spix/1000)*magnification)
sx = 3.5  # x 轴强度束大小，单位为 mm (1/e^2)
sy = 3.5#y 轴强度束大小，单位为 mm (1/e^2)
#%%
#   ===   激光幅度   ===============================================
L, Lp = slm.laser_gaussian(dim=N, r0=(0, 0), sigmax=sx/spix, sigmay=sy/spix, A=1.0, save_param=True)
# L=torch.from_numpy(np.sqrt(np.load('L.npy'))).to('cuda')
# 用于将激光强度的总和匹配到目标强度的总和的归一化
# ===  激光归一化 | 请勿删除  ================
I_L_tot = torch.sum(torch.pow(L, 2.))                           #
L = L * torch.pow(10000 / I_L_tot, 0.5)                      #
I_L_tot = torch.sum(torch.pow(L, 2.))                               #
# ===  激光归一化 | 请勿删除  ================
dx=round(-100e-6/focal_spx)
dy=round(-100e-6/focal_spx)
#   ================================================================================================
#   |          目标幅度、目标相位、加权 cg、加权 i
#   ================================================================================================
# param = [1., round(150e-6/focal_spx), round(25e-6/focal_spx), NT/2., NT/2., 9]  # [d2, sigma, l, roi, roj, C1]
# param = [torch.tensor(1.), torch.tensor(round(150e-6/focal_spx)), torch.tensor(round(50e-6/focal_spy)), torch.tensor(dx+NT[0]/2.), torch.tensor(dy+NT[1]/2.), torch.tensor(9)]  # [d2, sigma, l, roi, roj, C1]
param = [torch.tensor(1.), torch.tensor(round(100e-6/focal_spx)), torch.tensor(round(20e-6/focal_spy)), torch.tensor(dx+NT[0]/2.), torch.tensor(dy+NT[1]/2.), torch.tensor(9)]  # [d2, sigma, l, roi, roj, C1]

d2 = param[0]  # 加权区域宽度
sigma = param[1]  # Laguerre 高斯宽度
sigmax = param[2]
sigmay=torch.tensor(round(20e-6/focal_spx))
r0i = param[3]
r0j = param[4]
r0 = torch.tensor([r0i, r0j], dtype=torch.float64)   # 模式位置
C1 = param[5]  # 陡度因子
#   ===   目标幅度   ==============================================
# Ta, Tap = slm.target_lg(n=NT, r0=r0, w=sigma, l=l, A=1.0, save_param=True)
# Ta, Tap =slm.gaussian_top_round(dim=NT, r0=r0,d=round(100e-6/focal_spx),sigma=round(80e-6/focal_spx),A=1.0, save_param=True)
# Ta, Tap =slm.gaussian_top_round(dim=NT, r0=r0,d=round(0e-6/focal_spx),sigma=round(80e-6/focal_spx),A=1.0, save_param=True)
# Ta, Tap =slm.gaussian_top_square(dim=NT, r0=r0,dx=round(80e-6/focal_spx),dy=round(10e-6/focal_spx),sigmax=round(30e-6/focal_spx),sigmay=round(20e-6/focal_spx),A=1.0, save_param=True)
# Ta, Tap = slm.gaussian_line(dim=NT, r0=r0, d=sigma,sigma=sigmax, A=1.0, save_param=True)
Ta, Tap = slm.gaussian_line2(dim=NT, r0=r0, d=sigma,sigmax=sigmax,sigmay=sigmay, A=1.0, save_param=True)

# Ta, Tap = gaussian_linex(dim=NT, r0=r0, d2=2,d=sigma,sigma=l, A=1.0, save_param=True)
# Ta, Tap = gaussian_line_peak(dim=NT, r0=r0, d=sigma,sigmax=sigmax,sigmay=sigmay,bump_amp=0.05,bump_width_ratio=0.15, A=1.0, range=[-sigma/2,-sigma/4],save_param=True)
# Ta, Tap =slm.gaussian_top_round(dim=NT, r0=r0,d=18, sigma=4,A=1.0, save_param=True)
#   ===   目标相位   ==================================================
# P, Pp = slm.phase_spinning_continuous(n=NT, r0=r0, save_param=True)
# P, Pp = slm.phase_inverse_square(n=NT[0], r0=r0, save_param=True)
# P, Pp =slm.gaussian_line_phase(n=NT, r0=r0, d=l,sigma=sigma, save_param=True)
P, Pp = slm.phase_flat(dim=NT,v=0, save_param=True)
# P, Pp = slm.phase_inverse_square(dim=NT, r0=r0, save_param=True)

#   ===   加权 cg   ==================================================
# Weighting, Weightingp = slm.gaussian_top_round(dim=NT, r0=r0, d=2+d2 + sigma, sigma=2, A=1.0, save_param=True)
# Weighting, Weightingp =slm.gaussian_line(dim=NT, r0=r0, d=1.15*sigma,sigma=1.15*l, A=1.0, save_param=True)
# Weighting, Weightingp = slm.gaussian_top_round(dim=NT, r0=r0, d=1.5*sigma, sigma=1, A=1.0, save_param=True)#1.35-1.5
wcg=round(120e-6/focal_spx)
Weighting, Weightingp =slm.flat_top_round(dim=NT, r0=r0,d=wcg,A=1.0, save_param=True)

Wcg, Wcgp = slm.weighting_value(M=Weighting, p=1E-4, v=0, save_param=True)
I_Ta = torch.pow(torch.abs(Ta), 2.)
W99 = slm.weighting_value(M=I_Ta, p=0.01, v=0, save_param=False) # 前99%的加权
W99 = W99 * Wcg
Wcg=W99
# 用于将激光强度的总和匹配到目标强度的总和的归一化
# ===  目标归一化 | 请勿删除  ================
# Ta, Tap = gaussian_peak_only(dim=NT, r0=r0, d=sigma,sigmax=sigmax,sigmay=sigmay,bump_amp=0.05,bump_width_ratio=0.15, A=1.0, range=[-sigma/2,-sigma/4],save_param=True)
Ta = Ta * Wcg                                              #
P = P * Wcg                                                #
I_Ta_w = torch.sum(torch.pow(Ta, 2.))                           #
Ta = Ta * torch.pow(I_L_tot / (I_Ta_w), 0.5)                #
I_Ta = torch.pow(torch.abs(Ta), 2.)                             #
# ===  目标归一化 | 请勿删除  ================

profile_s=L
if not torch.is_complex(profile_s):
    profile_s = profile_s.to(torch.complex128)
n_pixelsx = int(N[0])
n_pixelsy = int(N[1]) 
profile_s_r = profile_s.real.type(torch.float64)
profile_s_i = profile_s.imag.type(torch.float64)
A0 = 1. / np.sqrt(NT[0] * NT[1])  # Linked to the fourier transform. Keeps the same quantity of light between the input and the output

def cost_SE_gpu_amp(phi):
    zero_frame = torch.zeros((NT[0], NT[1]), dtype=torch.float64)
    zero_matrix = torch.as_tensor(zero_frame, dtype=torch.float64)

    # Phi and its momentum for use in gradient descent with momentum:
    phi_reshaped = phi.view(n_pixelsx, n_pixelsy)

    # E_in (n_pixels**2):
    S_r = torch.tensor(profile_s_r, dtype=torch.float64)
    S_i = torch.tensor(profile_s_i, dtype=torch.float64)
    E_in_r = A0 * (S_r * torch.cos(phi_reshaped) - S_i * torch.sin(phi_reshaped))
    E_in_i = A0 * (S_i * torch.cos(phi_reshaped) + S_r * torch.sin(phi_reshaped))

    # 填充输入场
    idx_0x, idx_1x = get_centre_range(n_pixelsx,NT[0])
    idx_0y, idx_1y = get_centre_range(n_pixelsy,NT[1])

    E_in_r_pad = zero_matrix.clone()
    E_in_r_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_r  # 填充实际部分

    E_in_i_pad = zero_matrix.clone()
    E_in_i_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_i  # 填充虚部

    phi_padded = zero_matrix.clone()
    phi_padded[idx_0x:idx_1x, idx_0y:idx_1y] = phi_reshaped

    # 计算输出场（傅里叶变换）
    E_out_r, E_out_i = fft(E_in_r_pad, E_in_i_pad)  # 使用FFT计算输出场的实部和虚部

    # 计算输出强度
    E_out_2 = E_out_r ** 2 + E_out_i ** 2  # 输出强度为实部和虚部的平方和
    # E_out_2=E_out_2* Wcg
    # E_out_2 = E_out_2 * torch.pow(I_L_tot / torch.sum(E_out_2), 0.5)    
    # 输出振幅
    cost=torch.sum(torch.pow((Ta**2 - E_out_2)* Wcg , 2))
    return cost

def cost_SE_gpu(phi):
    zero_frame = torch.zeros((NT[0], NT[1]), dtype=torch.float64)
    zero_matrix = torch.as_tensor(zero_frame, dtype=torch.float64)

    # Phi and its momentum for use in gradient descent with momentum:
    phi_rate = torch.zeros_like(phi, dtype=torch.float64)
    phi_reshaped = phi.view(n_pixelsx, n_pixelsy)

    # E_in (n_pixels**2):
    S_r = torch.tensor(profile_s_r, dtype=torch.float64)
    S_i = torch.tensor(profile_s_i, dtype=torch.float64)
    E_in_r = A0 * (S_r * torch.cos(phi_reshaped) - S_i * torch.sin(phi_reshaped))
    E_in_i = A0 * (S_i * torch.cos(phi_reshaped) + S_r * torch.sin(phi_reshaped))

    # 填充输入场
    idx_0x, idx_1x = get_centre_range(n_pixelsx,NT[0])
    idx_0y, idx_1y = get_centre_range(n_pixelsy,NT[1])

    E_in_r_pad = zero_matrix.clone()
    E_in_r_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_r  # 填充实际部分

    E_in_i_pad = zero_matrix.clone()
    E_in_i_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_i  # 填充虚部

    phi_padded = zero_matrix.clone()
    phi_padded[idx_0x:idx_1x, idx_0y:idx_1y] = phi_reshaped

    # 计算输出场（傅里叶变换）
    E_out_r, E_out_i = fft(E_in_r_pad, E_in_i_pad)  # 使用FFT计算输出场的实部和虚部

    # 计算输出强度
    E_out_2 = E_out_r ** 2 + E_out_i ** 2  # 输出强度为实部和虚部的平方和

    # 计算输出相位
    E_out_p = torch.atan2(E_out_i, E_out_r)  # 输出相位

    # 输出振幅
    E_out_amp = torch.sqrt(E_out_2)  # 输出振幅
    overlap = torch.sum(Ta * E_out_amp * Wcg * torch.cos(E_out_p - P))
    overlap = overlap / (torch.pow(torch.sum(torch.pow(Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_amp * Wcg, 2)), 0.5))
    cost = torch.pow(10, C1) * torch.pow((1 - overlap), 2)

    return cost

def cost_SE_gpu_e(phi):
    zero_frame = torch.zeros((NT[0], NT[1]), dtype=torch.float64)
    zero_matrix = torch.as_tensor(zero_frame, dtype=torch.float64)

    # Phi and its momentum for use in gradient descent with momentum:
    phi_rate = torch.zeros_like(phi, dtype=torch.float64)
    phi_reshaped = phi.view(n_pixelsx, n_pixelsy)

    # E_in (n_pixels**2):
    S_r = torch.tensor(profile_s_r, dtype=torch.float64)
    S_i = torch.tensor(profile_s_i, dtype=torch.float64)
    E_in_r = A0 * (S_r * torch.cos(phi_reshaped) - S_i * torch.sin(phi_reshaped))
    E_in_i = A0 * (S_i * torch.cos(phi_reshaped) + S_r * torch.sin(phi_reshaped))

    # 填充输入场
    idx_0x, idx_1x = get_centre_range(n_pixelsx,NT[0])
    idx_0y, idx_1y = get_centre_range(n_pixelsy,NT[1])

    E_in_r_pad = zero_matrix.clone()
    E_in_r_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_r  # 填充实际部分

    E_in_i_pad = zero_matrix.clone()
    E_in_i_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_i  # 填充虚部

    phi_padded = zero_matrix.clone()
    phi_padded[idx_0x:idx_1x, idx_0y:idx_1y] = phi_reshaped

    # 计算输出场（傅里叶变换）
    E_out_r, E_out_i = fft(E_in_r_pad, E_in_i_pad)  # 使用FFT计算输出场的实部和虚部

    # 计算输出强度
    E_out_2 = E_out_r ** 2 + E_out_i ** 2  # 输出强度为实部和虚部的平方和

    # 计算输出相位
    E_out_p = torch.atan2(E_out_i, E_out_r)  # 输出相位

    # 输出振幅
    E_out_amp = torch.sqrt(E_out_2)  # 输出振幅
    
    # 计算重叠积分
    overlap = torch.sum(Ta * E_out_amp * Wcg * torch.cos(E_out_p - P))
    overlap = overlap / (torch.pow(torch.sum(torch.pow(Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_amp * Wcg, 2)), 0.5))
    
    # 计算光效率（加权区域内的光强占总光强的比例）
    I_out_w_tot = torch.sum(E_out_2 * Wcg)  # 加权区域内的总光强
    I_out_tot = torch.sum(E_out_2)          # 整个输出平面的总光强
    efficiency = I_out_w_tot / I_out_tot
    
    # 计算成本（添加与光效率负相关的项）
    #cost = torch.pow(10 , C1) * torch.pow((1 - overlap), 2) *torch.pow(1.0, 1/efficiency)   
    # cost = torch.pow(10 , C1) * torch.pow((1 - overlap), 2) * torch.pow(2.718281828459045, torch.pow(1/efficiency,1)-1) 
    cost = torch.pow(10 , C1) * torch.pow((1 - overlap), 2) * torch.pow(1/efficiency,2)

    return cost

def cost_SE_gpu_e1(phi):
    zero_frame = torch.zeros((NT[0], NT[1]), dtype=torch.float64)
    zero_matrix = torch.as_tensor(zero_frame, dtype=torch.float64)

    # Phi and its momentum for use in gradient descent with momentum:
    phi_rate = torch.zeros_like(phi, dtype=torch.float64)
    phi_reshaped = phi.view(n_pixelsx, n_pixelsy)

    # E_in (n_pixels**2):
    S_r = torch.tensor(profile_s_r, dtype=torch.float64)
    S_i = torch.tensor(profile_s_i, dtype=torch.float64)
    E_in_r = A0 * (S_r * torch.cos(phi_reshaped) - S_i * torch.sin(phi_reshaped))
    E_in_i = A0 * (S_i * torch.cos(phi_reshaped) + S_r * torch.sin(phi_reshaped))

    # 填充输入场
    idx_0x, idx_1x = get_centre_range(n_pixelsx,NT[0])
    idx_0y, idx_1y = get_centre_range(n_pixelsy,NT[1])

    E_in_r_pad = zero_matrix.clone()
    E_in_r_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_r  # 填充实际部分

    E_in_i_pad = zero_matrix.clone()
    E_in_i_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_i  # 填充虚部

    phi_padded = zero_matrix.clone()
    phi_padded[idx_0x:idx_1x, idx_0y:idx_1y] = phi_reshaped

    # 计算输出场（傅里叶变换）
    E_out_r, E_out_i = fft(E_in_r_pad, E_in_i_pad)  # 使用FFT计算输出场的实部和虚部

    # 计算输出强度
    E_out_2 = E_out_r ** 2 + E_out_i ** 2  # 输出强度为实部和虚部的平方和

    # 计算输出相位
    E_out_p = torch.atan2(E_out_i, E_out_r)  # 输出相位

    # 输出振幅
    E_out_amp = torch.sqrt(E_out_2)  # 输出振幅
    
    # 计算重叠积分
    overlap = torch.sum(Ta * E_out_amp * Wcg * torch.cos(E_out_p - P))
    overlap = overlap / (torch.pow(torch.sum(torch.pow(Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_amp * Wcg, 2)), 0.5))
    
    # 计算光效率（加权区域内的光强占总光强的比例）
    I_out_w_tot = torch.sum(E_out_2 * Wcg)  # 加权区域内的总光强
    I_out_tot = torch.sum(E_out_2)          # 整个输出平面的总光强
    efficiency = I_out_w_tot / I_out_tot
    
    # 计算成本（添加与光效率负相关的项）
    #cost = torch.pow(10 , C1) * torch.pow((1 - overlap), 2) *torch.pow(1.0, 1/efficiency)   
    # cost = torch.pow(10 , C1) * torch.pow((1 - overlap), 2) * torch.pow(2.718281828459045, torch.pow(1/efficiency,1)-1) 
    cost = torch.pow(10 , C1) * torch.pow((1 - overlap), 2) * torch.pow(1/efficiency,4)

    return cost
#   ================================================================================================
def cost_SE_gpu_e4(phi):
    zero_frame = torch.zeros((NT[0], NT[1]), dtype=torch.float64)
    zero_matrix = torch.as_tensor(zero_frame, dtype=torch.float64)

    # Phi and its momentum for use in gradient descent with momentum:
    phi_rate = torch.zeros_like(phi, dtype=torch.float64)
    phi_reshaped = phi.view(n_pixelsx, n_pixelsy)

    # E_in (n_pixels**2):
    S_r = torch.tensor(profile_s_r, dtype=torch.float64)
    S_i = torch.tensor(profile_s_i, dtype=torch.float64)
    E_in_r = A0 * (S_r * torch.cos(phi_reshaped) - S_i * torch.sin(phi_reshaped))
    E_in_i = A0 * (S_i * torch.cos(phi_reshaped) + S_r * torch.sin(phi_reshaped))

    # 填充输入场
    idx_0x, idx_1x = get_centre_range(n_pixelsx,NT[0])
    idx_0y, idx_1y = get_centre_range(n_pixelsy,NT[1])

    E_in_r_pad = zero_matrix.clone()
    E_in_r_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_r  # 填充实际部分

    E_in_i_pad = zero_matrix.clone()
    E_in_i_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_i  # 填充虚部

    phi_padded = zero_matrix.clone()
    phi_padded[idx_0x:idx_1x, idx_0y:idx_1y] = phi_reshaped

    # 计算输出场（傅里叶变换）
    E_out_r, E_out_i = fft(E_in_r_pad, E_in_i_pad)  # 使用FFT计算输出场的实部和虚部

    # 计算输出强度
    E_out_2 = E_out_r ** 2 + E_out_i ** 2  # 输出强度为实部和虚部的平方和

    # 计算输出相位
    E_out_p = torch.atan2(E_out_i, E_out_r)  # 输出相位

    # 输出振幅
    E_out_amp = torch.sqrt(E_out_2)  # 输出振幅
    
    # 计算重叠积分
    overlap = torch.sum(Ta * E_out_amp * Wcg * torch.cos(E_out_p - P))
    overlap = overlap / (torch.pow(torch.sum(torch.pow(Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_amp * Wcg, 2)), 0.5))
    
    # 计算光效率（加权区域内的光强占总光强的比例）
    I_out_w_tot = torch.sum(E_out_2 * Wcg)  # 加权区域内的总光强
    I_out_tot = torch.sum(E_out_2)          # 整个输出平面的总光强
    efficiency = I_out_w_tot / I_out_tot
    
    # 计算成本（添加与光效率负相关的项）
    #cost = torch.pow(10 , C1) * torch.pow((1 - overlap), 2) *torch.pow(1.0, 1/efficiency)   
    # cost = torch.pow(10 , C1) * torch.pow((1 - overlap), 2) * torch.pow(2.718281828459045, torch.pow(1/efficiency,1)-1) 
    cost = torch.pow(10 , C1) * torch.pow((1 - overlap), 2) * torch.pow(1/efficiency,8)

    return cost

N=[1024,1272]
def fresnel_lens_phase_generate(shift_distance, SLMRes=(1024,1272), x0=N[0]/2, y0=N[1]/2, pixelpitch=12.5,wavelength=0.813,focallength=200000,magnification=1):

        Xps, Yps = torch.meshgrid(torch.linspace(0, SLMRes[0], SLMRes[0]), torch.linspace(0, SLMRes[1], SLMRes[1]))


        X = (Xps-x0)*pixelpitch
        Y = (Yps-y0)*pixelpitch

        fresnel_lens_phase = torch.fmod(torch.pi*(X**2+Y**2)*shift_distance/(wavelength*focallength**2)*magnification**2,2*torch.pi)

        return np.array(torch.remainder(fresnel_lens_phase,2*torch.pi))

#   ================================================================================================
nb_iter0 = 5
nb_iter = 100
#%%
# mu = np.arctan(200/200)
# D = -2*np.pi*12.5/0.813/200e3*(40*3.45)/np.cos(mu) *4
# init_phi, ipp = phase_guess(N, dx-30e-6/focal_spy,dy, 0.2, 0.2/1000,  0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位

init_phi, ipp = phase_guess(N, dx,dy, 0.9, 0.9/1000,  0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
# init_phi, ipp = phase_guess(N, -0.18*torch.pi, 0.9, 3.7/1000, torch.tensor(torch.pi/4.4), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
# init_phi=get_init_phi(N,3.5/1000,45/1000000)
# init_phi=torch.from_numpy(np.reshape(np.load('test1.npy'),1024*1272)).to('cuda')
plt.imshow(I_Ta.detach().cpu()[3300:3510,3360:3570])
plt.show()
slm_opt = slm.SLM(NT=NT,N=N,numb=numb, initial_phi=init_phi, profile_s=L)

plt.imshow(slm_opt.E_out_2.detach().cpu()[3300:3510,3360:3570])
plt.show()
#%%
path='phaseold_wcg=99.9_guassianx=29.600000381469727_guassiany=20.350000381469727.npy'
x=np.angle(np.exp(1j*(np.load(path)))+0.0*np.exp(1j*(np.load('test2.npy'))))

init_phi=torch.from_numpy(x).cuda()
slm_opt = slm.SLM(NT=NT,N=N,numb=numb, initial_phi=init_phi.flatten(), profile_s=L)

plt.imshow(slm_opt.E_out_2.detach().cpu()[3460:3570,3460:3570],cmap='jet')
plt.show()
#%%
#double step
fft = FourierOp()
cg = cg1.CG(L_Lp=(L, Lp),
                r0=r0,
                Ta_Tap=(Ta, Tap),
                P_Pp=(P, Pp),
                Wcg_Wcgp=(Wcg, Wcgp),
                init_phi_ipp=(init_phi, ipp),
                nb_iter=20,
                numb=numb,
                slm_opt=slm_opt,
                cost_SE=cost_SE_gpu,
                show=True,
                goal=0.001,
                method='newton-cg',
                lr=0.5)
fft = FourierOp()
cg = cg1.CG(L_Lp=(L, Lp),
                r0=r0,
                Ta_Tap=(Ta, Tap),
                P_Pp=(P, Pp),
                Wcg_Wcgp=(Wcg, Wcgp),
                init_phi_ipp=(cg.slm_phase_end.flatten(), ipp),
                nb_iter=20,
                numb=numb,
                slm_opt=slm_opt,
                cost_SE=cost_SE_gpu_e4,
                show=True,
                goal=0.001,
                method='newton-cg',
                lr=0.5)
#%%
init_phi_new0 = cg_amp.slm_phase_end.to('cuda')
init_phi_new = torch.reshape(init_phi_new0, ([N[0]*N[1]]))

cg1 = cg2.CG(L_Lp=(L, Lp),
            r0=r0,
            Ta_Tap=(Ta, Tap),
            P_Pp=(P, Pp),
            Wcg_Wcgp=(Wcg, Wcgp),
            init_phi_ipp=(init_phi_new, ipp),
            nb_iter=nb_iter,
            numb=numb,
            slm_opt=slm_opt,
            cost_SE=cost_SE_gpu_e,
            show=True,
            goal=0.001,
            lr=0.10)
#%%
fft = FourierOp()
cg0 = cg2.CG(L_Lp=(L, Lp),
                r0=r0,
                Ta_Tap=(Ta, Tap),
                P_Pp=(P, Pp),
                Wcg_Wcgp=(Wcg, Wcgp),
                init_phi_ipp=(init_phi, ipp),
                nb_iter=100,
                numb=numb,
                slm_opt=slm_opt,
                cost_SE=cost_SE_gpu_e,
                show=True,
                goal=0.001,
                lr=0.1)
#%%
# init_phi_new0 = torch.from_numpy(np.load('test1.npy')).to('cuda')
# init_phi= torch.reshape(init_phi_new0, ([N[0]*N[1]]))
fft = FourierOp()
cg = cg1.CG(L_Lp=(L, Lp),
                r0=r0,
                Ta_Tap=(Ta, Tap),
                P_Pp=(P, Pp),
                Wcg_Wcgp=(Wcg, Wcgp),
                init_phi_ipp=(init_phi, ipp),
                nb_iter=400,
                numb=numb,
                slm_opt=slm_opt,
                cost_SE=cost_SE_gpu_e4,
                show=True,
                goal=0.001,
                method='cg',
                lr=0.5)
   #%%

# init_phi_new0 = cg0.slm_phase_end.to('cuda')
# init_phi_new = torch.reshape(init_phi_new0, ([N[0]*N[1]]))

cg1 = cg2.CG(L_Lp=(L, Lp),
            r0=r0,
            Ta_Tap=(Ta, Tap),
            P_Pp=(P, Pp),
            Wcg_Wcgp=(Wcg, Wcgp),
            init_phi_ipp=(init_phi, ipp),
            nb_iter=100,
            numb=numb,
            slm_opt=slm_opt,
            cost_SE=cost_SE_gpu_e4,
            show=True,
            goal=0.1,
            lr=0.10)
#%%
np.save('phaseold_wcg={}_guassianx={}_guassiany={}.npy'.format(wcg*1.85,sigmax*1.85,sigmay*1.85),cg1.slm_phase_end)

#%%
I=(cg0.I_out*Wcg).detach().cpu()
plt.imshow(I[3384-12:3384+13,3384-29:3384+30])
np.save('I_phaseold_wcg={}_guassianx={}_guassiany={}.npy'.format(wcg*1.85,sigmax*1.85,sigmay*1.85),I[3384-12:3384+13,3384-29:3384+30])
#%%

iter=5
for i in range(iter):
    print(i)
    image = IDS_Camera.GetImage()
    image=image[165:194,63:130]
    # image=image
    plt.imshow(image)
    plt.colorbar()
    plt.show()
    print('max:',image.max())
    print('sum:',image.sum())
    np.save(f"img{i}.npy",image.copy())
    # IDS_Camera.StopAcquisition()
    # IDS_Camera.Close()
    center=np.round(d4sigma_centroid(image))
    cutted_image=image[int(center[1])-9:int(center[1])+10,int(center[0])-26:int(center[0])+27]
    plt.imshow(cutted_image)
    plt.colorbar()

    # cutted_image=avg_img(10)
    # plt.imshow(cutted_image)
    # plt.colorbar()

    # cutted_image = np.flip(cutted_image,0)
    # cutted_image = image[1002:1022,1100:1192]

    # rotated_image = rotate(cutted_image,angle=-4.8,reshape=False)
    # rotated_image = rotate(cutted_image,angle=-0.2,reshape=False)
    y_numb=cutted_image.shape[0]-1
    x_numb=cutted_image.shape[1]-1
    path1='TA0.npy'
    path2=f'TA{i}.npy'
    Ta=torch.from_numpy(np.load(path1))
    last_Ta=torch.from_numpy(np.load(path2))

    last_Ta_cut=last_Ta[int(r0[0])-int((y_numb)/2):int(r0[0])+int((y_numb)/2)+1,int(r0[1])-int((x_numb)/2):int(r0[1])+int((x_numb)/2)+1]
    last_I_Ta=torch.pow(torch.abs(last_Ta_cut), 2.)

    Ta_cut=Ta[int(r0[0])-int((y_numb)/2):int(r0[0])+int((y_numb)/2)+1,int(r0[1])-int((x_numb)/2):int(r0[1])+int((x_numb)/2)+1]
    I_Ta=torch.pow(torch.abs(Ta_cut), 2.)
    img_nomlz = cutted_image * (I_Ta.sum().item() / cutted_image.sum())
    # plt.figure(figsize=(20,20))
    # plt.imshow(img_nomlz)
    # plt.colorbar()
    # plt.show()
    # cost=I_Ta/img_nomlz
    # # costx = calculate_cost(I_Ta, img_nomlz)
    # # cost = spatial_smoothing(costx)
    # # cost=I_Ta/img_nomlz*W20.cpu()[int(r0[0])-int((y_numb-1)/2):int(r0[0])+int((y_numb+1)/2),int(r0[1])-int((x_numb-1)/2):int(r0[1])+int((x_numb+1)/2)]
    # I_new=last_I_Ta*torch.pow(cost,0.5)
    # I_new=last_I_Ta*cost
    # I_new_nomlz=I_new * (I_Ta.sum().item() / I_new.sum())
    # new_Tax=torch.pow(I_new_nomlz,0.5)
    # new_Ta=last_Ta
    # new_Ta[int(r0[0])-int((y_numb)/2):int(r0[0])+int((y_numb)/2)+1,int(r0[1])-int((x_numb)/2):int(r0[1])+int((x_numb)/2)+1]=new_Tax
    # print((new_Ta**2).sum().item())


    cost=(I_Ta-img_nomlz)
    # cost=(I_Ta-img_nomlz)*W20.cpu()[int(r0[0])-int((y_numb-1)/2):int(r0[0])+int((y_numb+1)/2),int(r0[1])-int((x_numb-1)/2):int(r0[1])+int((x_numb+1)/2)]
    new_goal=last_I_Ta+0.15*cost
    new_goal = torch.clamp(new_goal, min=0)
    new_goal=new_goal*I_Ta.sum().item()/new_goal.sum().item()
    new_Tax=torch.pow(new_goal,0.5)
    new_Ta=last_Ta
    new_Ta[int(r0[0])-int((y_numb)/2):int(r0[0])+int((y_numb)/2)+1,int(r0[1])-int((x_numb)/2):int(r0[1])+int((x_numb)/2)+1]=new_Tax
    # new_Ta=new_Tay*W20+(1-W20)*torch.from_numpy(np.load(path))
    # new_Ta=new_Ta*torch.pow(I_L_tot.cpu() / (new_Ta**2).sum(), 0.5)
    print((new_Ta**2).sum().item())


    # #%%
    # ta_nomlz=np.pow(img_nomlz,0.5)
    # cost=Ta_cut-ta_nomlz
    # new_goal=Ta_cut+1*cost
    # new_goal = torch.clamp(new_goal, min=0)
    # new_Tax=new_goal*torch.pow(I_Ta.sum()/(torch.pow(torch.abs(new_goal), 2.).sum()),0.5)
    # new_Ta=Ta
    # new_Ta[int(r0[0])-int((y_numb-1)/2):int(r0[0])+int((y_numb+1)/2),int(r0[1])-int((x_numb-1)/2):int(r0[1])+int((x_numb+1)/2)]=new_Tax
    np.save(f'TA{i+1}.npy',new_Ta.cpu())
    new_Ta=torch.from_numpy(np.load(f'TA{i+1}.npy'))
    # cols, rows = N
    # new_init_phi=torch.reshape(torch.from_numpy(np.load('813-a-1.npy')),(rows * cols,))
    # init_phi, ipp = phase_guess(N, D, 0.9, 3.7/1000, torch.tensor(mu), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
    init_phi, ipp = phase_guess(N, dx,dy, 0.9, 3.7/1000,  0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
    fft = FourierOp()
    new_Ta=new_Ta.to('cuda')


    def cost_SE_gpu_x(phi):
        zero_frame = torch.zeros((NT[0], NT[1]), dtype=torch.float64)
        zero_matrix = torch.as_tensor(zero_frame, dtype=torch.float64)

        # Phi and its momentum for use in gradient descent with momentum:
        phi_reshaped = phi.view(n_pixelsx, n_pixelsy)

        # E_in (n_pixels**2):
        S_r = torch.tensor(profile_s_r, dtype=torch.float64)
        S_i = torch.tensor(profile_s_i, dtype=torch.float64)
        E_in_r = A0 * (S_r * torch.cos(phi_reshaped) - S_i * torch.sin(phi_reshaped))
        E_in_i = A0 * (S_i * torch.cos(phi_reshaped) + S_r * torch.sin(phi_reshaped))

        # 填充输入场
        idx_0x, idx_1x = get_centre_range(n_pixelsx,NT[0])
        idx_0y, idx_1y = get_centre_range(n_pixelsy,NT[1])

        E_in_r_pad = zero_matrix.clone()
        E_in_r_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_r  # 填充实际部分

        E_in_i_pad = zero_matrix.clone()
        E_in_i_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_i  # 填充虚部

        phi_padded = zero_matrix.clone()
        phi_padded[idx_0x:idx_1x, idx_0y:idx_1y] = phi_reshaped

        # 计算输出场（傅里叶变换）
        E_out_r, E_out_i = fft(E_in_r_pad, E_in_i_pad)  # 使用FFT计算输出场的实部和虚部

        # 计算输出强度
        E_out_2 = E_out_r ** 2 + E_out_i ** 2  # 输出强度为实部和虚部的平方和

        # 计算输出相位
        E_out_p = torch.atan2(E_out_i, E_out_r)  # 输出相位

        # 输出振幅
        E_out_amp = torch.sqrt(E_out_2)  # 输出振幅
        overlap = torch.sum(new_Ta * E_out_amp * Wcg * torch.cos(E_out_p - P))
        overlap = overlap / (torch.pow(torch.sum(torch.pow(new_Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_amp * Wcg, 2)), 0.5))
        cost = torch.pow(10, C1) * torch.pow((1 - overlap), 2)
        return cost
    cg1 = cg2.CG(L_Lp=(L, Lp),
                r0=r0,
                Ta_Tap=(new_Ta, Tap),
                P_Pp=(P, Pp),
                Wcg_Wcgp=(Wcg, Wcgp),
                init_phi_ipp=(init_phi, ipp),
                nb_iter=100,
                numb=numb,
                slm_opt=slm_opt,
                cost_SE=cost_SE_gpu_x,
                show=True,
                goal=0.01)

    np.save(f'813ad{i+1}.npy',cg1.slm_phase_end)
    path=f'813ad{i+1}.npy'
    tophatscreen=(255*np.load(path)/(2*np.pi))
    tophat_screen_Corrected=SLM_screen_Correct(tophatscreen)
    # tophatscreen=(255*phase/(2*np.pi))
    # tophatscreen=255*np.array(cg1.slm_phase_end)/(2*np.pi)
    # tophat_screen_Corrected=(215*tophatscreen/255).astype('uint8')
    slm_play.update(tophat_screen_Corrected)
#%%
n = 8
d = sigma.item() if isinstance(sigma, torch.Tensor) else sigma
for i in range(n-1):
    # bump_start = -d/2 + i * (d/n)
    # bump_end = bump_start + d/n
    # bump_range = [bump_start, bump_end]

    bump_start = -d/2+d/n/2 + i * (d/n)
    bump_end = bump_start + d/n
    bump_range = [bump_start, bump_end]
    Ta, Tap = gaussian_line_peak(
        dim=NT, r0=r0, d=sigma, sigmax=sigmax, sigmay=sigmay,
        bump_amp=0.4, bump_width_ratio=0.15, A=1.0,
        range=bump_range, save_param=True
    )
    P, Pp = slm.phase_flat(dim=NT,v=0, save_param=True)
    wcg=round(100e-6/focal_spx)
    Weighting, Weightingp =slm.flat_top_round(dim=NT, r0=r0,d=wcg,A=1.0, save_param=True)

    Wcg, Wcgp = slm.weighting_value(M=Weighting, p=1E-4, v=0, save_param=True)
    I_Ta = torch.pow(torch.abs(Ta), 2.)
    W99 = slm.weighting_value(M=I_Ta, p=0.01, v=0, save_param=False) # 前99%的加权
    W99 = W99 * Wcg
    Wcg=W99
    Ta, Tap = gaussian_peak_only(dim=NT, r0=r0, d=sigma,sigmax=sigmax,sigmay=sigmay,bump_amp=0.05,bump_width_ratio=0.15, A=1.0, range=bump_range,save_param=True)
    Ta = Ta * Wcg                                              #
    P = P * Wcg                                                #
    I_Ta_w = torch.sum(torch.pow(Ta, 2.))                           #
    Ta = Ta * torch.pow(I_L_tot / (I_Ta_w), 0.5)                #
    I_Ta = torch.pow(torch.abs(Ta), 2.)                             #


    init_phi, ipp = phase_guess(N, dx+(bump_start+bump_end)/2,dy, 0.2, 0.2/1000,  0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
    plt.imshow(I_Ta.detach().cpu()[3400:3550,3400:3550])
    plt.show()
    slm_opt = slm.SLM(NT=NT,N=N,numb=numb, initial_phi=init_phi, profile_s=L)
    plt.imshow(slm_opt.E_out_2.detach().cpu()[3400:3550,3400:3550])
    plt.show()

    fft = FourierOp()
    cg1 = cg2.CG(L_Lp=(L, Lp),
                    r0=r0,
                    Ta_Tap=(Ta, Tap),
                    P_Pp=(P, Pp),
                    Wcg_Wcgp=(Wcg, Wcgp),
                    init_phi_ipp=(init_phi, ipp),
                    nb_iter=50,
                    numb=numb,
                    slm_opt=slm_opt,
                    cost_SE=cost_SE_gpu_e4,
                    show=True,
                    goal=0.001,
                    lr=0.5)

    # 保存结果
    np.save(f'peak_only_shift_8_{i}.npy', cg1.slm_phase_end)

    # 释放显存，防止爆内存
    del cg1
    torch.cuda.empty_cache()

# %%
