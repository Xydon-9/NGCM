import numpy as np
import torch
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

# Added NNP and step helpers here
def _range_space_project(vec, A_funcs, eta_reg=0.0):
    flat = vec.reshape(vec.size(0), -1)
    return A_funcs.A_pinv_add_eta(A_funcs.A(flat), eta=eta_reg).reshape(*vec.size())


def _null_space_project(vec, A_funcs, eta_reg=0.0):
    return vec - _range_space_project(vec, A_funcs, eta_reg=eta_reg)


def _step_sm(iter_ind, mu, mu_factor):
    return float(mu) * (float(mu_factor) ** int(iter_ind))


def _parse_nnp_steps(nnp_steps):
    if isinstance(nnp_steps, str) and nnp_steps.strip():
        return {int(s.strip()) for s in nnp_steps.split(",") if s.strip()}
    return None


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
    metrics = {}
    with torch.no_grad():
        y_pred = A_funcs.A(xt)
        consistency_mse = torch.mean((y_pred - y_0) ** 2).item()
    metrics['consistency'] = consistency_mse
    return metrics

def _save_fft_spectrum(tensor, save_path):
    x = tensor.detach().cpu()
    fft_x = torch.fft.fft2(x, norm='ortho')
    fft_x_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))
    magnitude = torch.abs(fft_x_shifted)
    log_magnitude = torch.log(magnitude + 1e-8)
    avg_log_mag = torch.mean(log_magnitude, dim=0, keepdim=True)
    min_val = avg_log_mag.min()
    max_val = avg_log_mag.max()
    norm_mag = (avg_log_mag - min_val) / (max_val - min_val + 1e-8)
    tvu.save_image(norm_mag, save_path)

def _run_visualization_and_analysis(
    xt_guide, noise_original, range_space_of_noise, noise_projected,
    img_ind, iter_ind, config, save_root=None,
    y_0=None, A_funcs=None 
):
    pass 


def NGCM_restoration(
        x, model, A_funcs, y_0, sigma_y, betas, eta, deltas, deltas_injection_type,
        gamma, iN, zeta, classes, config, mu, mu_factor=1,
        sigma_t_min=0.002, sigma_t_max=80.0,
        use_dsr=False,
        use_nnp=False, nnp_steps="",
        dsr_repeat=1, dsr_repeat_damp=1.0, dsr_repeat_noise_damp=1.0,
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
        
        nnp_step_set = _parse_nnp_steps(nnp_steps)

        def nnp_on_this_step(step_idx):
            if not use_nnp:
                return False
            return (nnp_step_set is None) or (step_idx in nnp_step_set)

        for at, at_next in zip(alphas, alphas_next):
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

            # --- DSR Block ---
            if use_dsr and iter_ind == 0 and at_next != -1:
                dsr_k = max(1, int(dsr_repeat) if dsr_repeat is not None else 1)
                sm_dsr = _step_sm(iter_ind, mu, mu_factor)
                
                next_sigma_t_temp = torch.sqrt(1 - at_next)
                next_sigma_t_temp = torch.clamp(next_sigma_t_temp, sigma_t_min, sigma_t_max)
                c1_prelim = next_sigma_t_temp * float(eta)
                c2_prelim = next_sigma_t_temp * ((1 - float(eta)**2) ** 0.5)
                
                sigma_t_for_refine = next_sigma_t_temp
                delta_idx = min(iter_ind + 1, len(deltas) - 1)
                effective_sigma_refine = (1 + deltas[delta_idx]) * sigma_t_for_refine if deltas_injection_type == 1 else sigma_t_for_refine
                scaling_values_refine = get_scalings_for_boundary_condition(effective_sigma_refine)
                c_skip_r, c_out_r, c_in_r = [append_dims(s, xt.ndim) for s in scaling_values_refine]
                rescaled_sigma_t_refine = 1000 * 0.25 * torch.log(effective_sigma_refine + 1e-44)
                
                x0_work = x0_t
                x0_vel_work = x0_t
                rand_prelim = torch.randn_like(xt)
                y_0_flat = y_0.view(y_0.size(0), -1)

                for rep_i in range(dsr_k):
                    res_pre = A_funcs.A(x0_work.view(x0_work.size(0), -1)) - y_0_flat
                    bp_pre = A_funcs.A_pinv_add_eta(res_pre, eta=bp_eta_reg).view(*x.size())
                    xt_guide_pre = x0_work - sm_dsr * float(dsr_repeat_damp) * bp_pre
                    
                    z_hat_minus_pre = (x0_vel_work - xt) / (sigma_t + 1e-8)
                    
                    if rep_i == 0:
                        c1_use, c2_use = c1_prelim, c2_prelim
                    else:
                        damp = float(dsr_repeat_noise_damp)
                        c1_use, c2_use = c1_prelim * damp, c2_prelim * damp
                        
                    noise_pre = c1_use * rand_prelim + c2_use * z_hat_minus_pre
                    if nnp_on_this_step(iter_ind):
                        noise_pre = _null_space_project(noise_pre, A_funcs, eta_reg=bp_eta_reg)
                        
                    xt_prev = xt_guide_pre + noise_pre
                    
                    if classes is None:
                        et_final = model(c_in_r * xt_prev, rescaled_sigma_t_refine)
                    else:
                        et_final = model(c_in_r * xt_prev, rescaled_sigma_t_refine, classes)
                    
                    x0_new = c_out_r * et_final + c_skip_r * xt_prev
                    x0_new = torch.clamp(x0_new, -1.0, 1.0)
                    
                    x0_work = x0_new
                    x0_vel_work = x0_new
                
                x0_t = x0_work

            # Main processing guidance step
            BP_guidance = A_funcs.A_pinv_add_eta(A_funcs.A(x0_t.view(x0_t.size(0), -1)) - y_0.view(y_0.size(0), -1), eta=bp_eta_reg).view(*x.size())
            xt_guide = x0_t - _step_sm(iter_ind, mu, mu_factor) * BP_guidance
            
            if at_next == -1:
                if sigma_y != 0: return [x0_t.to('cpu')]
                return [xt_guide.to('cpu')]

            next_sigma_t = torch.sqrt(1 - at_next)
            next_sigma_t = torch.clamp(next_sigma_t, sigma_t_min, sigma_t_max)
            
            z_hat_minus = (x0_t - xt) / (sigma_t + 1e-8)
            c1 = next_sigma_t * eta
            c2 = next_sigma_t * ((1 - eta ** 2) ** 0.5)
            noise_original = c1 * torch.randn_like(xt) + c2 * z_hat_minus

            # --- NNP Block ---
            if nnp_on_this_step(iter_ind):
                noise_projected = _null_space_project(noise_original, A_funcs, eta_reg=bp_eta_reg)
            else:
                noise_projected = noise_original

            xt = xt_guide + noise_projected
        
        return [xt_guide.to('cpu')]
