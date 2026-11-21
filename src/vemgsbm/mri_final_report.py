"""Build the Chinese multi-seed TEST report from a completed frozen experiment matrix."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from .mri_compare import paired_seed_comparison
from .runtime import fingerprint


METRICS = ("all_psnr_db", "mean_slice_ssim", "all_mse", "foreground_psnr_db", "hfen", "edge_mse", "z_difference_mse")
CONTRASTS = (("bridge", "direct"), ("mls", "pca"), ("mls_geometry", "mls_shuffled"))
HIST = "5281446_original"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def case_counts(summary):
    cases = summary["cases"]
    return (len(cases), sum(c["metrics"]["evaluated_slices"] for c in cases),
            sum(c["metrics"]["pixel_count"] for c in cases))


def verify_metric_totals(summary):
    counts = case_counts(summary)
    if counts != (84, 14991, 3929800704) or counts[1] != summary["total_slices"]:
        raise ValueError("Expected the original 84-case, 14,991-slice, 3,929,800,704-voxel MRI cohort")
    rows = [c["metrics"] for c in summary["cases"]]
    mse = sum(row["squared_error_sum"] for row in rows) / counts[2]
    n_ssim = sum(row["ssim_slice_count"] for row in rows)
    ssim = sum(row["mean_slice_ssim"] * row["ssim_slice_count"] for row in rows) / n_ssim
    for metric, value in (("all_mse", mse), ("all_psnr_db", -10 * math.log10(mse)), ("mean_slice_ssim", ssim)):
        if not math.isclose(value, summary[metric], rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError(f"Stored {metric} differs from pooled case evidence")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def delta_row(candidate, reference, training_seed, pairs, draws):
    diff = paired_seed_comparison(pairs, resamples=draws)
    row = dict(candidate=candidate, reference=reference, training_seed=training_seed)
    for key in ("pooled_psnr_db", "pooled_mse", "mean_slice_ssim"):
        stat = diff[key]
        row[key] = stat["difference"]
        row[key + "_low95"], row[key + "_high95"] = stat["ci95"]
    return row


def f(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def with_sd(value, spread, digits):
    return f(value, digits) if spread is None else f"{f(value, digits)} ± {f(spread, digits)}"


def conclusion(row):
    lo, hi = row["pooled_psnr_db_low95"], row["pooled_psnr_db_high95"]
    verdict = "成对区间支持提升" if lo > 0 else ("成对区间支持下降" if hi < 0 else "成对区间跨越或触及零")
    return (f"PSNR 差 {row['pooled_psnr_db']:+.4f} dB，病例配对 95% 区间 [{lo:+.4f}, {hi:+.4f}]，{verdict}。"
            f"SSIM 差 {row['mean_slice_ssim']:+.6f}，95% 区间 "
            f"[{row['mean_slice_ssim_low95']:+.6f}, {row['mean_slice_ssim_high95']:+.6f}]。")


def build(matrix_path, historical_path, output, resamples=10000, independent_root=None):
    historical = read(historical_path)
    verify_metric_totals(historical)
    if (round(historical["all_psnr_db"], 4), round(historical["mean_slice_ssim"], 6)) != (30.6781, .907021):
        raise ValueError("The historical summary must match job 5281446 epoch 38")
    grouped, runs, labels, all_cases, independent = defaultdict(dict), [], set(), [], []
    data_identity = None
    for record in read(matrix_path)["runs"]:
        label, method, seed, group = (record[k] for k in ("label", "method", "training_seed", "group"))
        if group not in {"new", "historical", "external", "bilinear"}:
            raise ValueError(f"Unexpected comparison group {group}")
        if label in labels or method == HIST or seed in grouped[method]:
            raise ValueError(f"Duplicate label or method/seed in frozen matrix: {label}")
        labels.add(label)
        path = Path(record["output"]) / "summary.json"
        s = read(path)
        freeze_dir = Path(record["freeze_dir"])
        freeze = read(freeze_dir / "freeze.json")
        selected = freeze["selected"]
        if (s.get("split") != "test" or s["subset"] != historical["subset"]
                or read(freeze_dir / "manifest.json") != s["subset"]
                or selected["checkpoint_sha256"] != s["checkpoint_sha256"]
                or not math.isclose(selected["pooled_psnr"], record["validation_psnr"], rel_tol=1e-10)):
            raise ValueError(f"Frozen model, validation score, or TEST cohort differs for {label}")
        for key in ("seed", "steps", "samples", "predictor"):
            if selected["inference"][key] != s[key]:
                raise ValueError(f"Frozen {key} differs from evaluation for {label}")
        inference = selected["inference"]
        for key in ("drift_sampler", "readout"):
            if s["predictor"] != "bilinear" and key in inference and s.get(key) != inference[key]:
                raise ValueError(f"Frozen {key} differs from evaluation for {label}")
        if "inference_sigma" in inference and inference["inference_sigma"] != s["sampling_sigma"]:
            raise ValueError(f"Frozen sampling noise differs for {label}")
        if "disable_guidance" in inference and inference["disable_guidance"] == bool(s.get("geometry_guidance")):
            raise ValueError(f"Frozen online guidance differs for {label}")
        verify_metric_totals(s)
        grouped[method][seed] = s
        ext = s.get("external", {})
        nfe = s.get("network_calls_per_tile_sample", s.get("drift_network_calls_per_tile_sample"))
        if nfe is None and "momentum" in s["inference_variant"]:
            nfe = s["steps"]
        rows = s["cases"]
        identity = {case["case_id"]: case["input_fingerprint"] for case in rows}
        if data_identity is not None and identity != data_identity:
            raise ValueError("Frozen TEST runs reference different inputs")
        data_identity = identity
        if group == "new" and s.get("training_seed", seed) != seed:
            raise ValueError(f"Training seed differs from frozen matrix for {label}")
        for case in rows:
            m = case["metrics"]
            all_cases.append(dict(method=method, label=label, training_seed=seed, case_id=case["case_id"],
                                  slices=m["evaluated_slices"], voxels=m["pixel_count"],
                                  **{key: m.get(key) for key in METRICS}))
        if independent_root and record.get("confirm_archive"):
            separate = Path(independent_root) / label / "summary.json"
            held_out = read(separate)
            if (held_out["split"] != "independent_archive" or held_out["checkpoint_sha256"] != s["checkpoint_sha256"]
                    or held_out["freeze_sha256"] != fingerprint(freeze_dir / "freeze.json")):
                raise ValueError(f"Mismatched independent archive confirmation for {label}")
            independent.append(dict(label=label, seed=seed, cohort_sha256=held_out["cohort_sha256"],
                                    cases=len(held_out["cases"]), slices=held_out["total_slices"],
                                    voxels=held_out["pixel_count"], psnr=held_out["all_psnr_db"],
                                    ssim=held_out["mean_slice_ssim"], source=str(separate)))
        arm = record.get("arm", {})
        runs.append(dict(method=method, group=group, training_seed=seed, label=label,
                         validation_psnr=selected["pooled_psnr"], **{k: s.get(k) for k in METRICS},
                         dimensionality=s.get("input_dimensionality", "2D" if ext or group == "bilinear" else "5-slice"),
                         pretraining=s.get("pretraining", ext.get("pretraining")),
                         network_calls_per_tile_sample=nfe,
                         local_dimension=arm.get("dim"), geometry_in_path=arm.get("path"),
                         endpoint_readout=arm.get("readout"),
                         case_evaluation_seconds=sum(c["evaluation_seconds"] for c in rows),
                         cases=len(rows), slices=s["total_slices"], voxels=s["pixel_count"],
                         summary=str(path), summary_sha256=fingerprint(path), freeze_dir=str(freeze_dir),
                         checkpoint_sha256=s["checkpoint_sha256"]))

    methods, intervals = [], []
    for method, summaries in grouped.items():
        group_rows = [r for r in runs if r["method"] == method]
        group = group_rows[0]["group"]
        if any(r["group"] != group for r in group_rows):
            raise ValueError(f"Mixed comparison groups for {method}")
        if group == "new" and set(summaries) != {42, 43, 44}:
            raise ValueError(f"Expected three frozen seeds 42, 43, 44 for {method}")
        if len({r["checkpoint_sha256"] for r in group_rows}) < len(group_rows):
            raise ValueError(f"Repeated checkpoint among purported training seeds of {method}")
        row = dict(method=method, group=group, training_seeds=len(summaries),
                   validation_psnr=mean(r["validation_psnr"] for r in group_rows))
        for metric in METRICS:
            vals = [r[metric] for r in group_rows]
            row[metric] = mean(vals) if all(v is not None for v in vals) else None
        row["psnr_seed_sd"] = stdev(r["all_psnr_db"] for r in group_rows) if len(summaries) > 1 else None
        row["ssim_seed_sd"] = stdev(r["mean_slice_ssim"] for r in group_rows) if len(summaries) > 1 else None
        paired = delta_row(method, HIST, "all", [(historical, s) for s in summaries.values()], resamples)
        intervals.append(paired)
        row.update({"vs_5281446_" + k: v for k, v in paired.items()
                    if k not in ("reference", "candidate", "training_seed")})
        methods.append(row)
    by_method = {row["method"]: row for row in methods}
    methods.sort(key=lambda r: -r["all_psnr_db"])
    for i, row in enumerate(methods, 1):
        row["rank"] = i
    eligible = [r for r in methods if r["group"] == "new" and r["method"] != "mls_shuffled"]
    # Keep manifest order for equal validation scores. No TEST-dependent tie breaking.
    best = max(eligible, key=lambda r: (r["validation_psnr"], -list(grouped).index(r["method"])))
    contrasts = []
    for cand, ref in CONTRASTS:
        if cand not in grouped or ref not in grouped or set(grouped[cand]) != set(grouped[ref]):
            raise ValueError(f"Incomplete same-seed contrast {cand} versus {ref}")
        pairs = [(grouped[ref][s], grouped[cand][s]) for s in sorted(grouped[cand])]
        for s, pair in zip(sorted(grouped[cand]), pairs, strict=True):
            intervals.append(delta_row(cand, ref, s, [pair], resamples))
        row = delta_row(cand, ref, "all", pairs, resamples)
        intervals.append(row)
        contrasts.append(row)
    external = [r for r in methods if r["group"] == "external"]
    full = by_method["mls_geometry"]
    ext_rank = 1 + sum(r["all_psnr_db"] > full["all_psnr_db"] for r in external)
    best_diff = next(r for r in intervals if r["candidate"] == best["method"] and r["reference"] == HIST)
    lines = ["# 配对 MRI XY×4、Z×1 正式测试", "",
             f"1. 按平均整病例 VAL PSNR 选定 **{best['method']}**。相对第 38 轮历史模型，{conclusion(best_diff)}",
             f"2. 无图册桥相对同骨干直接回归，{conclusion(contrasts[0])}",
             f"3. 验证选定的 MLS 配置相对局部 PCA 配置，{conclusion(contrasts[1])}",
             f"4. 真实协方差方向相对同谱随机方向，{conclusion(contrasts[2])}",
             (f"5. 完整模型 mls_geometry 与 {len(external)} 个已评估外部方法按 TEST PSNR 排序，"
              f"位列 **第 {ext_rank}/{1 + len(external)}**。"
              if external else "5. 当前矩阵缺少同协议外部方法，外部排名待确认。"),
             "", "## 冻结配置的整病例成绩", "",
             "历史参考为作业 5281446 的第 38 轮。固定清单含 84 例、14,991 张切片、3,929,800,704 个体素。"
             "候选方法按 VAL 冻结后再评分。每个训练种子先计算全体素 MSE 与 PSNR、逐切片平均 SSIM，"
             "表中按方法对种子取等权算术平均。更高的 PSNR 和 SSIM、更低的误差表示更好的结果。",
             "", "| TEST 排名 | 方法 | 种子数 | VAL PSNR | TEST PSNR ± 种子 SD | SSIM ± 种子 SD | MSE | 前景 PSNR | HFEN | 连续 Z 差分 MSE |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in methods:
        lines.append(f"| {r['rank']} | {r['method']} | {r['training_seeds']} | {f(r['validation_psnr'],4)} | "
                     f"{with_sd(r['all_psnr_db'],r['psnr_seed_sd'],4)} | "
                     f"{with_sd(r['mean_slice_ssim'],r['ssim_seed_sd'],6)} | {f(r['all_mse'],9)} | "
                     f"{f(r['foreground_psnr_db'],4)} | {f(r['hfen'])} | {f(r['z_difference_mse'],9)} |")
    lines.extend(["", f"95% 区间采用 {resamples:,} 次病例配对自助重采样，随机种子为 42。"
                  "同一抽样在全部训练种子和两种方法间复用，区间描述固定训练种子下的病例抽样变异。"
                  "种子间标准差单列，单种子方法留空。各方法配置分别由 VAL 选择，"
                  "比较 MLS 与 PCA 的机制时还需结合相同局部维数的验证消融。", "",
                  "[各方法汇总及相对历史模型的区间](methods.csv) · [逐病例指标](cases.csv) · [逐种子得分、输入、预训练、网络调用、评估耗时及运行记录](runs.csv) · "
                  "[逐种子和跨种子配对差值](paired.csv)", "",
                  f"历史 summary 的 SHA-256 为 `{fingerprint(historical_path)}`。冻结矩阵 SHA-256 为 `{fingerprint(matrix_path)}`。"])
    lines += ["", "## 同种子比较", "", "| 候选 | 对照 | PSNR 差为正的种子 | PSNR 差范围（dB） |", "|---|---|---:|---:|"]
    for cand, ref in CONTRASTS:
        differences = [r["pooled_psnr_db"] for r in intervals
                       if r["candidate"] == cand and r["reference"] == ref and r["training_seed"] != "all"]
        lines.append(f"| {cand} | {ref} | {sum(x > 0 for x in differences)}/{len(differences)} | "
                     f"{min(differences):+.4f} 至 {max(differences):+.4f} |")
    if independent:
        if len({row["cohort_sha256"] for row in independent}) != 1:
            raise ValueError("Independent confirmations used different case cohorts")
        lines += ["", "## 单独留出的档案病例", "",
                  f"同一数据档案的预留病例共 {independent[0]['cases']} 例，跨中心来源未经确认。"
                  f"VAL 选定模型在这些病例上的跨种子平均 PSNR 为 {mean(r['psnr'] for r in independent):.4f} dB，"
                  f"SSIM 为 {mean(r['ssim'] for r in independent):.6f}。[独立确认记录](independent.csv)"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "methods.csv", methods)
    write_csv(output / "runs.csv", runs)
    write_csv(output / "cases.csv", all_cases)
    write_csv(output / "paired.csv", intervals)
    if independent:
        write_csv(output / "independent.csv", independent)
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("matrix", "historical", "output"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--independent-root")
    args = parser.parse_args()
    build(args.matrix, args.historical, args.output, independent_root=args.independent_root)


if __name__ == "__main__":
    main()
