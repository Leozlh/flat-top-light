""" 运行CG最小化

通过Laguerre_Gaussian1.py调用，以执行带有输入参数的最小化：
L_Lp - 激光场幅度
r0 - 图案位置（像素坐标）
Ta_Tap - 目标幅度
P_Pp - 目标相位
Wcg_Wcgp - 加权
init_phi_ipp - SLM初始相位
nb_iter - CG计算中的最大迭代次数
slm_opt - 与SLM相位相关的输入和输出平面场
cost_SE - 成本函数

根据这些输入参数，运行最小化以生成最终相位数组，
计算结果输出场和各种误差指标。

请引用Optics Express 25, 11692 (2017) - https://doi.org/10.1364/OE.25.011692 
14/05/2017
"""

#________________________________________________________________________________________________________________________________
import numpy as np                          # 用于数组操作
import matplotlib.pyplot as plt             # 绘图
import torch                                 # 用于张量操作
# import torch.optim as optim                 # 优化器
import scipy.optimize                        # 调用CG最小化
from scipy.optimize import minimize
import SLM_1X as slm                         # 包含SLM属性、场计算、目标和绘图属性
import time                                  # 时间计算和生成时间戳

def conjugate_gradient(f, grad_f, x0, max_iter=300, tol=1e-5):
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    r = grad_f(x).clone()  # 初始梯度
    p = -r.clone()  # 初始搜索方向
    rs_old = torch.dot(r, r)  # 初始r和r的内积（用于停止准则）

    for i in range(max_iter):
        # 计算梯度在搜索方向上的步长
        Ap = grad_f(x + p) - grad_f(x)  # A*p
        alpha = rs_old / torch.dot(p, Ap)  # 步长计算
        x = x + alpha * p  # 更新参数
        r = grad_f(x).clone()  # 重新计算梯度

        rs_new = torch.dot(r, r)  # 计算新的r和r的内积
        if torch.sqrt(rs_new) < tol:  # 如果梯度足够小，则停止
            break

        p = -r + (rs_new / rs_old) * p  # 计算新的搜索方向
        rs_old = rs_new  # 更新rs_old

    return x.detach().numpy(), i  # 返回最优解和迭代次数
class CG(object):
    
    def __init__(self, numb,goal,L_Lp, r0, Ta_Tap, P_Pp, Wcg_Wcgp, init_phi_ipp, nb_iter, slm_opt, cost_SE,\
                 show=True,lr=0.1):

        r0i = int(round(r0[0].item())) # 图案中心x位置
        r0j = int(round(r0[1].item())) # 图案中心y位置

        if type(L_Lp) == tuple:
            L, Lp = L_Lp
            self.Lp = Lp
        else:
            L = L_Lp

        if type(Ta_Tap) == tuple:
            Ta, Tap = Ta_Tap
            self.Tap = Tap
        else:
            Ta = Ta_Tap

        if type(P_Pp) == tuple:
            P, Pp = P_Pp
            self.Pp = Pp
        else:
            P = P_Pp

        if type(Wcg_Wcgp) == tuple:
            Wcg, Wcgp = Wcg_Wcgp
            self.Wcgp = Wcgp
        else:
            Wcg = Wcg_Wcgp
        
        if type(init_phi_ipp) == tuple:
            init_phi, ipp = init_phi_ipp
            self.ipp = ipp
        else:
            init_phi = init_phi_ipp

        N = L.shape # SLM的像素大小
        NT = Ta.shape # 目标的像素大小

        I_Ta = torch.pow(torch.abs(Ta), 2.)
        W10 = slm.weighting_value(M=I_Ta, p=0.9, v=0, save_param=False) # 前10%的加权
        W99 = slm.weighting_value(M=I_Ta, p=0.01, v=0, save_param=False) # 前99%的加权

        W10 = W10 * Wcg
        W99 = W99 * Wcg
        Etest = (Ta * torch.exp(1j * P)) # 目标的完整复场

        # 获取目标的缩放窗口
        imin, imax, jmin, jmax = slm.give_plot_scale(M=Wcg.clone().cpu().numpy(), p=1E-4, extension=1.1)

        # i和j需要反转，因为matplotlib使用的是笛卡尔坐标而不是矩阵坐标
        zoom = [jmin, jmax, imin, imax]
        
        #  _____________________________________________________________
        # |_____ 初始绘图 _____|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_| 

        # 目标绘图
        p1 = [I_Ta.clone().detach().cpu().numpy(), I_Ta.detach().clone().cpu().numpy(), I_Ta.detach().clone().cpu().numpy(), (Wcg + W99 + W10).clone().cpu().numpy(), P.clone().cpu().numpy(), P.clone().cpu().numpy(), torch.pow(torch.abs(Etest), 2).clone().cpu().numpy(), torch.angle(Etest).clone().cpu().numpy()] # 数据
        d1 = [2, 2, 3, 2, 2, 2, 2, 2] # 绘图维度
        sc1 = [[], zoom, zoom, zoom, [], zoom, zoom, zoom] # 缩放轴
        t1 = ['I_target', 'I_target zoomed', 'I_target 3D zoomed', 'Wcg + W99 + W10 zoomed', 'Target phase', 'Target phase zoomed', 'Etest Intensity', 'Etest phase'] # 标题
        v1 = [[], [], [], [], [-np.pi, np.pi], [-np.pi, np.pi], [], [-np.pi, np.pi]] # 限制
        c1 = [[], [], [], [], [], [], [], []] # 颜色
        plot1 = slm.n_plot(p=p1, d=d1, t=t1, v=v1, sc=sc1, save=True)
        if show: plt.show()

        date = time.strftime('%d-%m-%y__%H-%M-%S', time.localtime())
        time_code = int(time.time())
        self.date = date
        self.time_code = time_code

        print('\n最大迭代次数: {0}'.format(nb_iter))
        print(("计算开始: %s\n" % date))
        
        start_time = time.time()
        phi = init_phi
        phi.requires_grad=True

        optimizer = torch.optim.LBFGS([phi], lr=lr)#%%
        optimizer.zero_grad()
        def closure():
            optimizer.zero_grad()  # 清空梯度
            loss = cost_SE(phi)  # 计算损失，假设目标函数需要乘以 50
            loss.backward()
            return loss

        # 执行优化
        for i in range(nb_iter):
            optimizer.step(closure)  # 使用优化器执行一步更新
            # 每隔 10 步打印损失和当前参数值
            if i % 4 == 0:
                loss_value = closure().item()
                print(f"Step {i}, Loss: {loss_value:.4f}, Phi: {phi[:5]}")  # 打印前五个参数值
                if loss_value<goal:
                    optimizer.zero_grad()
                    break
        # 最终优化结果
        final_phi = phi.clone().detach().cpu()  # 获取优化后的参数并转移到 CPU
        end_time = time.time()
        runtime = end_time - start_time
        print(('运行时间为 %.3fs' % runtime))
        print(('运行时间为 %.0f分钟和%.3fs' % (runtime // 60, runtime % 60)))

        #  _____________________________________________________________________________________________________________________________
        # |                                 |                                                                                           |
        # |   结果和可视化                ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 
        # |_________________________________|___________________________________________________________________________________________| 

        # slm_opt = slm.SLM(NT=NT, initial_phi=res.x, profile_s=L)
        slm_opt = slm.SLM(NT=NT,N=[1024,1272],numb=numb,initial_phi=final_phi, profile_s=L)

        # 定义从对象slm_opt中的变量内容生成函数
        I_out =slm_opt.E_out_2
        E_out_p =slm_opt.E_out_p
        # E_out_r = slm_opt.E_out_r
        # E_out_i = slm_opt.E_out_i
        E_out_amp = slm_opt.E_out_amp

        # 获取这些函数的值
        # I_out = I_out_fn
        # E_out_p = E_out_fn_p()
        # E_out_r = E_out_fn_r()
        # E_out_i = E_out_fn_i()
        # E_out_amp = E_out_fn_amp()

        slm_phase_init = torch.remainder(init_phi.reshape(N[0], N[1]), 2 * np.pi).detach().cpu()
        slm_phase_end = np.mod(final_phi.reshape(N[0], N[1]), 2 * np.pi)
        
        #  ____________________________________________________________
        # |_____ 信息  _____|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_| 

        Efficiency = slm.Efficiency(Wcg, I_out) # 光效率
        F = slm.Fidelity(Wcg, Ta, P, E_out_amp, E_out_p) # 保真度（最大值在1）        
        RMS_error = slm.RMS_error(Wcg, I_Ta, I_out) # RMS error (total weighting region)
        RMS_error_10 = slm.RMS_error(W10, I_Ta, I_out) # RMS error (top 10% weighting region)
        RMS_error_99 = slm.RMS_error(W99, I_Ta, I_out) # RMS error (top 99% weighting region)
        Phase_error = slm.Phase_error(Wcg, P, E_out_p) # Phase error (total weighting region)
        Phase_error_99 = slm.Phase_error(W99, P, E_out_p) # Phase error (top 99% weighting region)

        I_out_Wcg = I_out * Wcg
        E_out_p_Wcg = E_out_p * Wcg
        I_Ta_max = I_Ta.max()
        I_out_max = I_out.max()
        I_out_Wcg_max = I_out_Wcg.max()

        print('\nError Metrics ')
        print('Efficiency : ', Efficiency.item())
        print('Fidelity   : ', F.item())
        print('Fractional rms error : ', RMS_error.item())
        print('Fractional rms error 10% : ', RMS_error_10.item())
        print('Fractional rms error 99% : ', RMS_error_99.item())
        print('Relative phase error : ', Phase_error.item())
        print('Relative phase error 99%: ', Phase_error_99.item())
        print('Efficiency : ', Efficiency.item())

        print('')

        #  ____________________________________________________________
        # |_____ Plots ____|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
            
        p1D2_1 = [(I_Ta[imin:imax, r0i]/I_Ta_max).detach().cpu() , (I_Ta[r0j, jmin:jmax]/I_Ta_max).detach().cpu(), (I_out_Wcg[imin:imax, r0i]/I_out_Wcg_max).detach().cpu(), (I_out_Wcg[r0j, jmin:jmax]/I_out_Wcg_max).detach().cpu()] # Intensity profile
        p1D3_1 = [P[imin:imax, r0i].detach().cpu() , P[r0j, jmin:jmax].detach().cpu(), E_out_p_Wcg[imin:imax, r0i].detach().cpu(), E_out_p_Wcg[r0j, jmin:jmax].detach().cpu()] # Phase profile


        # Intensity result plots
        p2 = [I_out.detach().cpu(), I_out.detach().cpu(), I_out.detach().cpu(), I_Ta.detach().cpu(), I_out_Wcg.detach().cpu(), p1D2_1 , (I_Ta/I_Ta_max-I_out_Wcg/I_out_Wcg_max).detach().cpu() , (Wcg + W99 + W10).cpu()] # data
        d2 = [2, 2, 2, 2, 3, 1, 2, 2] # plot dimensions
        sc2 = [[], zoom, zoom, zoom, zoom, [], zoom, zoom] # axes
        t2 = ['I_out', 'I_out zoomed', 'I_out zoomed on max', 'Target^2 zoomed', 'I_out*Wcg', 'Profiles', 'T^2-I_out norm by max in Wcg', 'Wcg + W99 + W10 zoomed'] # titles
        v2 = [[], [], [0,I_out_Wcg_max], [], [], [], [], []] # limits
        plot2 = slm.n_plot(p=p2, d=d2, sc=sc2, t=t2, v=v2, save=True)
        if show == True: plt.show()
            

        # Phase result plots
        p3 = [E_out_p.detach().cpu(), E_out_p.detach().cpu(), P.detach().cpu(),(Wcg + W99 + W10).detach().cpu(), E_out_p_Wcg.detach().cpu(), p1D3_1, (P-E_out_p_Wcg).detach().cpu()] # data
        d3 = [2, 2, 2, 2, 3, 1, 2] # plot dimensions
        sc3 = [[], zoom, zoom, zoom, zoom, [], zoom] # axes
        t3 = ['E_out_p', 'E_out_p zoomed', 'Phase zoomed', 'Wcg + W99 + W10 zoomed', 'E_out_p*Wcg', 'Profiles', 'phase-E_out in Wcg'] # titles
        v3 = [[], [], [], [], [], [], []] # limits
        plot3 = slm.n_plot(p=p3, d=d3, sc=sc3, t=t3, v=v3, save=True)
        if show == True: plt.show()


        # SLM plane plots
        p4 = [L.cpu(), slm_phase_init.cpu(), slm_phase_end] # data
        t4 = ['S_profile','slm_phase_init', 'slm_phase_end'] # titles
        plot4 = slm.n_plot(p=p4, t=t4, save=True)
        if show == True: plt.show()


        plt.close('all')

            
        #  ____________________________________________________________
        # |_____ Object Assignment ____|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|

        self.nb_iter = nb_iter
        self.time_code = time_code
        self.date = date
        self.runtime = runtime
        self.show = show

        self.L = L
        self.Ta = Ta
        self.I_Ta = I_Ta
        self.I_Ta_max = I_Ta_max
        self.P = P
        self.Wcg = Wcg
        self.W10 = W10
        self.W99 = W99
        self.N = N
        self.NT = NT
            
        self.slm_phase_init = slm_phase_init
        self.slm_phase_end = slm_phase_end

        self.I_out = I_out
        self.E_out_p = E_out_p
        self.E_out_amp = E_out_amp
        self.Efficiency = Efficiency
        self.F = F
        self.RMS_error = RMS_error
        self.RMS_error_10 = RMS_error_10
        self.RMS_error_99 = RMS_error_99
        self.Phase_error = Phase_error
        self.Phase_error_99 = Phase_error_99

        self.zoom = zoom
        self.plot1 = plot1
        self.plot2 = plot2
        self.plot3 = plot3
        self.plot4 = plot4
