import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as tvu
import os
import math

class_num = 951


def get_scalings_for_boundary_condition(sigma, sigma_data=0.5, sigma_min=0.002):
    c_skip = sigma_data ** 2 / (
            (sigma - sigma_min) ** 2 + sigma_data ** 2
    )
    c_out = (
            (sigma - sigma_min)
            * sigma_data
            / (sigma ** 2 + sigma_data ** 2) ** 0.5
    )
    c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out, c_in


def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def get_next_alpha(prev_alpha, gamma):
    return torch.clamp((prev_alpha * (1 + gamma)), 0, 0.9999)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(1)
    return a


def _write_metrics_to_file(file_path, img_ind, psnr_b, psnr_a, lpips_b, lpips_a):
    header = "Image_ID,PSNR_Before,PSNR_After,PSNR_Gain,LPIPS_Before,LPIPS_After,LPIPS_Gain\n"
    psnr_gain = psnr_a - psnr_b
    lpips_gain = lpips_a - lpips_b
    line = f"{img_ind},{psnr_b:.4f},{psnr_a:.4f},{psnr_gain:+.4f},{lpips_b:.4f},{lpips_a:.4f},{lpips_gain:+.4f}\n"
    write_header = not os.path.exists(file_path)
    with open(file_path, 'a') as f:
        if write_header:
            f.write(header)
        f.write(line)


def calculate_metrics_consistent(pred_x0, gt_x0, lpips_fn):
    pred_01 = torch.clamp((pred_x0 + 1.0) / 2.0, 0.0, 1.0)
    gt_01 = torch.clamp((gt_x0 + 1.0) / 2.0, 0.0, 1.0)
    mse = torch.mean((pred_01 - gt_01) ** 2)
    if mse == 0:
        psnr = torch.tensor(100.0)
    else:
        psnr = 10 * torch.log10(1.0 / mse)

    lpips_val = 0.0
    if lpips_fn is not None:
        pred_cpu = pred_x0.to('cpu')
        gt_cpu = gt_x0.to('cpu')
        with torch.no_grad():
            val = lpips_fn(pred_cpu, gt_cpu)
            lpips_val = val.view(-1).item()
    return psnr.item(), lpips_val


def _calculate_toxicity_metrics(xt, y_0, A_funcs):
    """
    计算中间含噪状态的评估指标：
    1. Consistency MSE (||Ax - y||^2) -- 物理一致性
    """
    metrics = {}
    
    # Consistency Error (物理一致性)
    with torch.no_grad():
        y_pred = A_funcs.A(xt)
        consistency_mse = torch.mean((y_pred - y_0) ** 2).item()
    metrics['consistency'] = consistency_mse
        
    return metrics


def _save_fft_spectrum(tensor, save_path):
    """
    对输入的 Tensor (C, H, W) 进行 2D FFT，并保存幅度谱图片。
    """
    # 确保在 CPU 上处理
    x = tensor.detach().cpu()
    
    # 1. 2D FFT (对最后两个维度 H, W 进行变换)
    fft_x = torch.fft.fft2(x, norm='ortho')
    
    # 2. 频移 (将零频移到中心)
    fft_x_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))
    
    # 3. 计算幅度 (Magnitude)
    magnitude = torch.abs(fft_x_shifted)
    
    # 4. 对数变换 (Log Scale) 以便更好地显示动态范围
    # 加 1e-8 防止 log(0)
    log_magnitude = torch.log(magnitude + 1e-8)
    
    # 5. 对通道取平均 (C, H, W) -> (1, H, W) 用于灰度显示
    avg_log_mag = torch.mean(log_magnitude, dim=0, keepdim=True)
    
    # 6. 归一化到 [0, 1]
    min_val = avg_log_mag.min()
    max_val = avg_log_mag.max()
    norm_mag = (avg_log_mag - min_val) / (max_val - min_val + 1e-8)
    
    # 7. 保存图片
    tvu.save_image(norm_mag, save_path)


def _run_visualization_and_analysis(
    xt_guide, noise_original, range_space_of_noise, noise_projected,
    img_ind, iter_ind, config, save_root=None,
    y_0=None, A_funcs=None 
):
    sample_idx = 0 
    
    # --- Part 1: Noise Component Analysis (Spatial & Frequency) ---
    noise_original_sample = noise_original[sample_idx]
    range_space_of_noise_sample = range_space_of_noise[sample_idx]
    noise_projected_sample = noise_projected[sample_idx]

    # 路径处理
    if save_root is None: base_dir = "."
    else: base_dir = save_root

    # A. 保存空间域图片 (Spatial Domain)
    save_path_noise_spatial = os.path.join(base_dir, "noise_vis_spatial", f"image_{img_ind:03d}")
    os.makedirs(save_path_noise_spatial, exist_ok=True)
    
    def normalize_noise_for_saving(noise_tensor):
        tensor = noise_tensor.clone().detach().cpu()
        min_val, max_val = torch.min(tensor), torch.max(tensor)
        return (tensor - min_val) / (max_val - min_val + 1e-8)
    
    tvu.save_image(normalize_noise_for_saving(noise_original_sample), os.path.join(save_path_noise_spatial, f"step_{iter_ind + 1}_1_original_spatial.png"))
    tvu.save_image(normalize_noise_for_saving(range_space_of_noise_sample), os.path.join(save_path_noise_spatial, f"step_{iter_ind + 1}_2_range_spatial.png"))
    tvu.save_image(normalize_noise_for_saving(noise_projected_sample), os.path.join(save_path_noise_spatial, f"step_{iter_ind + 1}_3_null_spatial.png"))
    
    # B. 保存频域图片 (Frequency Domain / FFT)
    save_path_noise_fft = os.path.join(base_dir, "noise_vis_fft", f"image_{img_ind:03d}")
    os.makedirs(save_path_noise_fft, exist_ok=True)
    
    _save_fft_spectrum(noise_original_sample, os.path.join(save_path_noise_fft, f"step_{iter_ind + 1}_1_original_fft.png"))
    _save_fft_spectrum(range_space_of_noise_sample, os.path.join(save_path_noise_fft, f"step_{iter_ind + 1}_2_range_fft.png"))
    _save_fft_spectrum(noise_projected_sample, os.path.join(save_path_noise_fft, f"step_{iter_ind + 1}_3_null_fft.png"))

    print(f"  Saved Noise Spatial & FFT images to '{save_path_noise_spatial}' & '{save_path_noise_fft}'")


    # --- Part 2: Toxicity Metrics Analysis (Consistency Only) ---
    
    # 构造带 batch 维度的 tensor
    xt_guide_sample = xt_guide[sample_idx].unsqueeze(0)
    noise_orig_sample = noise_original[sample_idx].unsqueeze(0)
    noise_range_sample = range_space_of_noise[sample_idx].unsqueeze(0)
    noise_null_sample = noise_projected[sample_idx].unsqueeze(0)
    
    # 构造三种中间状态 (xt)
    xt_next_null = xt_guide_sample + noise_null_sample
    xt_next_range = xt_guide_sample + noise_range_sample
    xt_next_original = xt_guide_sample + noise_orig_sample
    
    print(f"\n>>> Toxicity Metrics Analysis (Step {iter_ind+1})")
    print(f"{'State Type':<15} | {'Consistency MSE':<30}")
    print("-" * 50)

    if y_0 is not None:
        # 准备 batch 数据
        if y_0.ndim == 1: y_0_batch = y_0.unsqueeze(0)
        elif y_0.ndim == 2: y_0_batch = y_0[sample_idx].unsqueeze(0)
        else: y_0_batch = y_0
        
        # [修复点] 这里去掉了多余的 None 参数
        # 1. Null Space (Ours)
        m_null = _calculate_toxicity_metrics(xt_next_null, y_0_batch, A_funcs)
        print(f"{'Null-Space':<15} | {m_null['consistency']:<30.10f}")
        
        # 2. Original
        m_orig = _calculate_toxicity_metrics(xt_next_original, y_0_batch, A_funcs)
        print(f"{'Original':<15} | {m_orig['consistency']:<30.10f}")
        
        # 3. Range Space (Toxic)
        m_range = _calculate_toxicity_metrics(xt_next_range, y_0_batch, A_funcs)
        print(f"{'Range-Space':<15} | {m_range['consistency']:<30.10f}")
        
        # [可选] 将数据写入文件
        if save_root:
            tox_file = os.path.join(save_root, "nsp_consistency_check.txt")
            header = "Image,Step,Type,ConsistencyMSE\n"
            write_header = not os.path.exists(tox_file)
            with open(tox_file, "a") as f:
                if write_header: f.write(header)
                f.write(f"{img_ind},{iter_ind+1},Null,{m_null['consistency']:.10f}\n")
                f.write(f"{img_ind},{iter_ind+1},Original,{m_orig['consistency']:.10f}\n")
                f.write(f"{img_ind},{iter_ind+1},Range,{m_range['consistency']:.10f}\n")

    print("="*50 + "\n")

    # Save Images (xt)
    def normalize_xt_for_saving(xt_tensor):
        tensor = xt_tensor.clone().detach().cpu().squeeze(0)
        min_val, max_val = torch.min(tensor), torch.max(tensor)
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    save_path_xt = os.path.join(base_dir, "xt_toxicity_experiment", f"image_{img_ind:03d}")
    os.makedirs(save_path_xt, exist_ok=True)
    tvu.save_image(normalize_xt_for_saving(xt_next_null), os.path.join(save_path_xt, f"step_{iter_ind + 2}_xt_from_null_space.png"))
    tvu.save_image(normalize_xt_for_saving(xt_next_range), os.path.join(save_path_xt, f"step_{iter_ind + 2}_xt_from_range_space.png"))
    tvu.save_image(normalize_xt_for_saving(xt_next_original), os.path.join(save_path_xt, f"step_{iter_ind + 2}_xt_from_original.png"))
    print(f"  Saved xt toxicity test images to '{save_path_xt}'")

# ====================================================================
# [Main Function]
# ====================================================================

def NGCM_restoration(
        x, model, A_funcs, y_0, sigma_y, betas, eta, deltas, deltas_injection_type,
        gamma, iN, zeta, classes, config, mu, mu_factor=1,
        sigma_t_min=0.002, sigma_t_max=80.0,
        use_denoising_refinement=True, # [重要] 默认值
        
        img_ind=0, save_root=None, x_orig=None, lpips_fn=None
):
    with torch.no_grad():
        iter_ind = -1
        t = (torch.ones(1) * (iN + 1)).to(x.device)
        aN = compute_alpha(betas, t.long())
        alphas = [aN]
        for i in range(config.sampling.T_sampling - 1):
            alphas.append(get_next_alpha(alphas[-1], gamma).reshape(1))

        sigma_t_init = torch.sqrt(1 - alphas[0])
        xt = x + torch.randn_like(x) * sigma_t_init
        
        alphas_next = alphas[1:] + [-1]
        bp_eta_reg = sigma_y ** 2 * zeta

        for at, at_next in tqdm(zip(alphas, alphas_next), desc=f"Processing Image {img_ind}"):
            iter_ind += 1

            sigma_t = torch.sqrt(1 - at)
            sigma_t = torch.clamp(sigma_t, sigma_t_min, sigma_t_max)
            
            if deltas_injection_type == 0:
                scaling_values = get_scalings_for_boundary_condition(sigma_t)
            elif deltas_injection_type == 1:
                scaling_values = get_scalings_for_boundary_condition((1 + deltas[iter_ind]) * sigma_t)
            else:
                raise NotImplementedError(f"Unsupported deltas_injection_type: {deltas_injection_type}.")
            
            c_skip, c_out, c_in = [append_dims(s, xt.ndim) for s in scaling_values]
            rescaled_sigma_t = 1000 * 0.25 * torch.log((1 + deltas[iter_ind]) * sigma_t + 1e-44)

            # --- Initial Prediction ---
            if classes is None:
                et = model(c_in * xt, rescaled_sigma_t)
            else:
                et = model(c_in * xt, rescaled_sigma_t, classes)
            x0_t_initial = c_out * et + c_skip * xt
            x0_t_initial = torch.clamp(x0_t_initial, -1.0, 1.0)
            x0_t = x0_t_initial

            # --- DSR Verification (Step 1 Only) ---
            if use_denoising_refinement and iter_ind == 0:
                print("\n--- [DSR Verification Start] ---")
                
                psnr_before, lpips_before = 0.0, 0.0
                if x_orig is not None:
                    psnr_before, lpips_before = calculate_metrics_consistent(x0_t_initial, x_orig, lpips_fn)
                    print(f"  [Step 1 Initial Prediction] PSNR: {psnr_before:.2f} dB | LPIPS: {lpips_before:.4f}")
                    
                    if save_root and img_ind < 0:
                        save_path_dsr = os.path.join(save_root, "dsr_validation", f"image_{img_ind:03d}")
                        os.makedirs(save_path_dsr, exist_ok=True)
                        tvu.save_image((x0_t_initial + 1)/2, os.path.join(save_path_dsr, "1_x0_initial_before_DSR.png"))
                
                print("Performing Denoising State Refinement (1 extra NFE)...")
                
                BP_guidance_prelim = A_funcs.A_pinv_add_eta(A_funcs.A(x0_t.view(x0_t.size(0), -1)) - y_0.view(y_0.size(0), -1), eta=bp_eta_reg).view(*x.size())
                xt_guide_prelim = x0_t - (mu * mu_factor ** iter_ind) * BP_guidance_prelim

                z_hat_minus_prelim = (x0_t - xt) / sigma_t
                next_sigma_t_temp = torch.sqrt(1 - at_next)
                c1_prelim = next_sigma_t_temp * eta
                c2_prelim = next_sigma_t_temp * ((1 - eta ** 2) ** 0.5)
                noise_original_prelim = c1_prelim * torch.randn_like(xt) + c2_prelim * z_hat_minus_prelim
                
                noise_original_reshaped_prelim = noise_original_prelim.reshape(noise_original_prelim.size(0), -1)
                range_space_of_noise_prelim = A_funcs.A_pinv_add_eta(A_funcs.A(noise_original_reshaped_prelim), eta=bp_eta_reg).reshape(*xt.size())
                noise_projected_prelim = noise_original_prelim - range_space_of_noise_prelim
                
                xt_ideal_prev = xt_guide_prelim + noise_projected_prelim

                # Refinement
                sigma_t_for_refine = torch.sqrt(1 - at_next)
                sigma_t_for_refine = torch.clamp(sigma_t_for_refine, sigma_t_min, sigma_t_max)
                delta_idx = min(iter_ind + 1, len(deltas) - 1)
                effective_sigma_refine = (1 + deltas[delta_idx]) * sigma_t_for_refine if deltas_injection_type == 1 else sigma_t_for_refine
                scaling_values_refine = get_scalings_for_boundary_condition(effective_sigma_refine)
                c_skip_r, c_out_r, c_in_r = [append_dims(s, xt.ndim) for s in scaling_values_refine]
                rescaled_sigma_t_refine = 1000 * 0.25 * torch.log(effective_sigma_refine + 1e-44)

                if classes is None:
                    et_final = model(c_in_r * xt_ideal_prev, rescaled_sigma_t_refine)
                else:
                    et_final = model(c_in_r * xt_ideal_prev, rescaled_sigma_t_refine, classes)
                
                x0_t_refined = c_out_r * et_final + c_skip_r * xt_ideal_prev
                x0_t_refined = torch.clamp(x0_t_refined, -1.0, 1.0)
                x0_t = x0_t_refined

                if x_orig is not None:
                    psnr_after, lpips_after = calculate_metrics_consistent(x0_t_refined, x_orig, lpips_fn)
                    print(f"  [Step 1 Refined Prediction] PSNR: {psnr_after:.2f} dB | LPIPS: {lpips_after:.4f}")
                    print(f"  >>> DSR Gain: PSNR {psnr_after - psnr_before:+.2f} dB | LPIPS {lpips_after - lpips_before:+.4f}")
                    
                    if save_root:
                        metrics_file_path = os.path.join(save_root, "dsr_metrics_gain.txt")
                        _write_metrics_to_file(metrics_file_path, img_ind, psnr_before, psnr_after, lpips_before, lpips_after)

                    if save_root and img_ind < 0:
                         tvu.save_image((x0_t_refined + 1)/2, os.path.join(save_path_dsr, "2_x0_refined_after_DSR.png"))
                         tvu.save_image((x_orig + 1)/2, os.path.join(save_path_dsr, "0_ground_truth.png"))

                print("--- [DSR Verification End] ---\n")

            # --- Standard Cycle ---
            BP_guidance = A_funcs.A_pinv_add_eta(A_funcs.A(x0_t.view(x0_t.size(0), -1)) - y_0.view(y_0.size(0), -1), eta=bp_eta_reg).view(*x.size())
            xt_guide = x0_t - (mu * mu_factor ** iter_ind) * BP_guidance
            
            if at_next == -1:
                if sigma_y != 0: return [x0_t.to('cpu')]
                return [xt_guide.to('cpu')]

            next_sigma_t = torch.sqrt(1 - at_next)
            next_sigma_t = torch.clamp(next_sigma_t, sigma_t_min, sigma_t_max)
            
            z_hat_minus = (x0_t - xt) / sigma_t
            c1 = next_sigma_t * eta
            c2 = next_sigma_t * ((1 - eta ** 2) ** 0.5)
            noise_original = c1 * torch.randn_like(xt) + c2 * z_hat_minus

            noise_original_reshaped = noise_original.reshape(noise_original.size(0), -1)
            range_space_of_noise = A_funcs.A_pinv_add_eta(A_funcs.A(noise_original_reshaped), eta=bp_eta_reg).reshape(*xt.size())
            noise_projected = noise_original - range_space_of_noise
            
            # --- Toxicity Check (NSP) with FFT ---
            if img_ind < 50:
                _run_visualization_and_analysis(
                    xt_guide, noise_original, range_space_of_noise, noise_projected,
                    img_ind, iter_ind, config,
                    save_root=save_root,
                    y_0=y_0,
                    A_funcs=A_funcs
                )
            
            xt = xt_guide + noise_projected
        
        return [xt_guide.to('cpu')]
