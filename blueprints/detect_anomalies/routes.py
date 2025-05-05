from . import detect_anomalies_bp
from flask import request, render_template
from utils.ml import norm_a_scoring
# from utility import *

# np.random.seed(42)


@detect_anomalies_bp.route('norm_a', methods=['POST'])
def norm_a():
    json_data = request.get_json()
    data = json_data.get('data')
    flags = json_data.get('flags')
    areas = json_data.get('areas')

    return norm_a_scoring(data, flags, areas)


# @detect_anomalies_bp.route('show_distribution', methods=['GET'])
# def show_distribution():
#     score_files = sorted(list_all_filenames('results'))
#     if not score_files:
#         return "No score files found", 404
#
#     requested = request.args.get('score_file')
#     if requested not in score_files:
#         requested = score_files[0]
#
#     base = os.path.splitext(requested)[0]
#     matched_label = next(
#         lf for lf in list_all_filenames('data')
#         if os.path.splitext(lf)[0] == base
#     )
#
#     scores, labels = load_data(requested, matched_label)
#
#     num_bins = 50
#     lo, hi = float(scores.min()), float(scores.max())
#     bins = np.linspace(lo, hi, num_bins + 1)
#     bin_ranges = [f"{bins[i]:.3f}-{bins[i + 1]:.3f}" for i in range(num_bins)]
#
#     experiments = []
#     for _ in range(10):
#         start, end = select_training_range(scores)
#         sel = simulate_user_labels(labels, start, end)
#
#         train_scores = scores[start:end]
#         label_scores = scores[sel]
#
#         train_counts, _ = np.histogram(train_scores, bins=bins)
#         label_counts, _ = np.histogram(label_scores, bins=bins)
#
#         # 计算训练集中的真实异常数量和总数量
#         train_labels = labels[start:end]
#         num_anomalies = int((train_labels == 1).sum())
#         num_total = end - start
#         anomaly_ratio_str = f"{num_anomalies}/{num_total} ({(num_anomalies / num_total * 100):.2f}%)"
#
#         experiments.append({
#             "trainCounts": train_counts.tolist(),
#             "labelCounts": label_counts.tolist(),
#             "trainScores": train_scores.tolist(),
#             "labelScores": label_scores.tolist(),
#             "anomalyRatio": anomaly_ratio_str  # ← 添加这个值
#         })
#
#     result = {
#         "score_file": requested,
#         "all_files": score_files,
#         "binRanges": bin_ranges,
#         "experiments": experiments
#     }
#
#     return render_template('distribution.html', result=result)
