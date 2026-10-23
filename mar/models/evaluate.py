# mar/models/evaluate.py
"""
`main.task=test_refinement`: runs the full pipeline (coarse diffusion -> refinement ->
uniformity optimization) on the test split and reports the paper metrics for each stage.
"""
import json
import logging
import os

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from tqdm.auto import tqdm

from mar.models.refinement import optimize_uniformity, refine, sample_coarse
from mar.utils.visualization import center_com

log = logging.getLogger(__name__)

STAGES = ("coarse", "refined", "optimized")
METRICS = ("chamfer_3d", "chamfer_2d", "mean_nn_dist", "std_nn_dist")


def nn_distances(pc: torch.Tensor) -> torch.Tensor:
    """First-nearest-neighbour distance of every point in a (1, N, D) point cloud."""
    if pc.shape[1] <= 1:
        return torch.zeros(pc.shape[1], device=pc.device)
    dists = torch.cdist(pc, pc)
    dists.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
    return dists.min(dim=2).values.squeeze(0)


def _save_plot(gt, coarse, refined, optimized, existence, image, out_dir, sample_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = existence[0] > 0.5
    datas = [gt, coarse, refined, optimized]
    titles = ["Ground Truth", "Coarse", "Refined", "Refined + Optimized"]
    colors = ["C0", "k", "red", "orange"]
    fig, axs = plt.subplots(3, 4, figsize=(20, 15), sharex="col", sharey="row")
    for col, (data, color) in enumerate(zip(datas, colors)):
        for row, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
            axs[row, col].scatter(data[a, mask], data[b, mask], s=10, alpha=0.7, c=color)
        axs[0, col].set_title(titles[col])
    for ax in axs.flat:
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    fig.savefig(os.path.join(out_dir, f"sample_{sample_idx}.png"), dpi=150)
    plt.close(fig)

    plt.imsave(os.path.join(out_dir, f"sample_{sample_idx}_image.png"), image, cmap="gray")


def evaluate_refinement(accelerator, pc_model, refinement_model, projection_model, test_loader,
                        noise_scheduler, cfg):
    device = accelerator.device
    pc_model = accelerator.unwrap_model(pc_model).eval()
    refinement_model = accelerator.unwrap_model(refinement_model).eval()
    projection_model = accelerator.unwrap_model(projection_model).eval()

    eval_cfg = cfg.evaluation
    opt_cfg = cfg.refinement.optimization
    nm_per_unit = cfg.data.cell_size / 2  # normalized coordinates -> nm
    max_batches = eval_cfg.get("max_batches")
    plot_dir = None
    if eval_cfg.get("save_plots", False) and accelerator.is_main_process:
        plot_dir = "refinement_plots"
        os.makedirs(plot_dir, exist_ok=True)

    results = {stage: {m: [] for m in METRICS} for stage in STAGES}

    progress_bar = tqdm(test_loader, desc="Evaluating", disable=not accelerator.is_main_process)
    for batch_idx, batch in enumerate(progress_bar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch['image'].to(device)
        pixel_sizes = batch['pixel_size'].to(device)
        com_2d = batch['CoM'].to(device)
        gt = batch['point_cloud'].to(device)
        existence = gt[:, 3:, :]

        coarse = sample_coarse(pc_model, projection_model, noise_scheduler, images, pixel_sizes,
                               com_2d, existence, cfg, eval_cfg.num_inference_steps)
        refined = refine(refinement_model, projection_model, coarse, images, pixel_sizes, com_2d, cfg)
        optimized = optimize_uniformity(
            refined,
            steps=opt_cfg.steps,
            lr=opt_cfg.lr,
            momentum=opt_cfg.momentum,
            uniformity_weight=opt_cfg.uniformity_weight,
            fidelity_weight=opt_cfg.fidelity_weight,
            ks=tuple(opt_cfg.knn_ks),
        )

        with torch.no_grad():
            gt_centered = center_com(gt[:, :3, :], mask=existence)
            stage_xyz = {
                "coarse": center_com(coarse[:, :3, :], mask=existence),
                "refined": center_com(refined[:, :3, :], mask=existence),
                "optimized": center_com(optimized[:, :3, :], mask=existence),
            }

            for i in range(gt.shape[0]):
                mask = existence[i, 0, :] > 0.5
                if mask.sum() == 0:
                    continue
                gt_valid = gt_centered[i, :, mask].T.unsqueeze(0) * nm_per_unit
                for stage, xyz in stage_xyz.items():
                    pred_valid = xyz[i, :, mask].T.unsqueeze(0) * nm_per_unit
                    nn = nn_distances(pred_valid)
                    results[stage]["chamfer_3d"].append(chamfer_distance(gt_valid, pred_valid)[0].item())
                    results[stage]["chamfer_2d"].append(
                        chamfer_distance(gt_valid[:, :, :2], pred_valid[:, :, :2])[0].item())
                    results[stage]["mean_nn_dist"].append(nn.mean().item())
                    results[stage]["std_nn_dist"].append(nn.std().item())

                if plot_dir is not None:
                    _save_plot(
                        gt_centered[i].cpu().numpy(),
                        stage_xyz["coarse"][i].cpu().numpy(),
                        stage_xyz["refined"][i].cpu().numpy(),
                        stage_xyz["optimized"][i].cpu().numpy(),
                        existence[i].cpu().numpy(),
                        images[i, 0].cpu().numpy(),
                        plot_dir,
                        batch['dataset_idx'][i].item(),
                    )

        progress_bar.set_postfix({
            "CD3D opt": f"{np.mean(results['optimized']['chamfer_3d']):.6f}",
            "std NN opt": f"{np.mean(results['optimized']['std_nn_dist']):.6f}",
        })

    if not accelerator.is_main_process:
        return

    summary = {
        stage: {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                for m, v in metrics.items()}
        for stage, metrics in results.items()
    }
    with open("refinement_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [f"{'metric':<14}" + "".join(f"{stage:>26}" for stage in STAGES)]
    for m in METRICS:
        row = f"{m:<14}"
        for stage in STAGES:
            s = summary[stage][m]
            row += f"{s['mean']:>14.6f} ± {s['std']:<9.6f}"
        lines.append(row)
    table = "\n".join(lines)
    with open("refinement_metrics.txt", "w") as f:
        f.write(table + "\n")
    log.info(f"Evaluation on {summary['optimized']['chamfer_3d']['n']} samples "
             f"({eval_cfg.num_inference_steps} diffusion steps):\n{table}")
    log.info(f"Saved metrics to {os.path.abspath('refinement_metrics.json')}")
