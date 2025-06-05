import numpy as np
import pandas as pd
import os


def simulate_user_labels(label, start_index, end_index, min_labels=100, max_labels=500):
    label = np.asarray(label)
    indices = np.arange(start_index, end_index)
    num_indices = len(indices)
    if num_indices == 0:
        return np.array([], dtype=int)

    # 随机决定要标记多少点
    n_labels = np.random.randint(min_labels, max_labels + 1)
    n_labels = min(n_labels, num_indices)

    # 划分 anomaly / normal 索引
    local = label[start_index:end_index]
    anomaly_idx = indices[local == 1]
    normal_idx  = indices[local == 0]

    # 1) 至少一半是真异常
    # 2) 确保剩下的 normal 抽样量不超过 normal_idx 的大小
    min_correct = n_labels // 2
    min_correct = max(min_correct, n_labels - len(normal_idx))
    max_correct = min(len(anomaly_idx), n_labels)
    if max_correct < min_correct:
        min_correct = max_correct

    # 从 [min_correct, max_correct] 中抽一个 correct 数量
    n_correct   = np.random.randint(min_correct, max_correct + 1) if max_correct > 0 else 0
    n_incorrect = n_labels - n_correct

    # 分别不放回抽样
    selected_anom = (
        np.random.choice(anomaly_idx, size=n_correct, replace=False)
        if n_correct > 0 else np.array([], dtype=int)
    )
    selected_norm = (
        np.random.choice(normal_idx, size=n_incorrect, replace=False)
        if n_incorrect > 0 else np.array([], dtype=int)
    )

    # 合并并打乱
    selected = np.concatenate([selected_anom, selected_norm])
    np.random.shuffle(selected)
    return selected


def select_training_range(scores, min_fraction=0.1, max_fraction=0.3):
    total_length = len(scores)
    random_fraction = np.random.uniform(min_fraction, max_fraction)
    train_length = int(total_length * random_fraction)

    max_start = total_length - train_length
    start_idx = np.random.randint(0, max_start + 1)
    end_idx = start_idx + train_length

    return start_idx, end_idx


def load_data(score_filename, label_filename):
    for dirpath, dirnames, filenames in os.walk('results'):
        if score_filename in filenames:
            score_path = os.path.join(dirpath, score_filename)
            break
    else:
        raise FileNotFoundError(f"Score file '{score_filename}' not found under 'results/'")

    for dirpath, dirnames, filenames in os.walk('data'):
        if label_filename in filenames:
            label_path = os.path.join(dirpath, label_filename)
            break
    else:
        raise FileNotFoundError(f"Label file '{label_filename}' not found under 'data/'")

    scores = np.load(score_path)
    df = pd.read_csv(label_path, header=None).to_numpy()
    labels = df[:, 1]

    return scores, labels


def list_all_filenames(root_dir):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            all_files.append(filename)
    return all_files


def plot_score_histogram(scores, selected_indices, start_index, end_index, score_file, num_exp):
    # 1. 提取两组分数
    train_scores = scores[start_index:end_index]
    label_scores = scores[selected_indices]

    # 2. 各自按十分位数分箱
    train_bins = np.percentile(train_scores, np.linspace(0, 100, 11))
    label_bins = np.percentile(label_scores, np.linspace(0, 100, 11))

    # 3. 统计每个 bin 内的数量
    train_counts, _ = np.histogram(train_scores, bins=train_bins)
    label_counts, _ = np.histogram(label_scores, bins=label_bins)

    # 4. 计算 bin 中心和宽度
    train_centers = (train_bins[:-1] + train_bins[1:]) / 2
    train_widths = np.diff(train_bins)
    label_centers = (label_bins[:-1] + label_bins[1:]) / 2
    label_widths = np.diff(label_bins)

    # 5. 作图 - 现在有四个子图了
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2行2列
    ax1, ax2, ax3, ax4 = axes.flatten()

    # 左上：训练集quartile分箱柱状图
    ax1.bar(
        train_centers,
        train_counts,
        width=train_widths * 0.8,
        color='blue',
        alpha=0.6
    )
    ax1.set_ylabel('Train count')
    ax1.set_xlabel('Anomaly score (train deciles)')
    ax1.set_title('Train Scores (Decile Binned)')
    ax1.set_xticks(train_centers)
    train_labels = [f"{train_bins[i]:.2f}–{train_bins[i + 1]:.2f}" for i in range(10)]
    ax1.set_xticklabels(train_labels, rotation=45, ha='right')

    # 右上：训练集原始分数直方图
    bins_0_1 = np.arange(0, 1.05, 0.05)  # 0,0.05,...,1.0
    ax2.hist(
        train_scores,
        bins=bins_0_1,
        color='skyblue',
        alpha=0.7,
        edgecolor='black'
    )
    ax2.set_ylabel('Train count')
    ax2.set_xlabel('Anomaly score (0–1)')
    ax2.set_title('Train Scores Distribution (0–1)')

    # 左下：标签集quartile分箱柱状图
    ax3.bar(
        label_centers,
        label_counts,
        width=label_widths * 0.8,
        color='orange',
        alpha=0.6
    )
    ax3.set_ylabel('Label count')
    ax3.set_xlabel('Anomaly score (label deciles)')
    ax3.set_title('Label Scores (Decile Binned)')
    ax3.set_xticks(label_centers)
    label_labels = [f"{label_bins[i]:.2f}–{label_bins[i + 1]:.2f}" for i in range(10)]
    ax3.set_xticklabels(label_labels, rotation=45, ha='right')

    # 右下：标签集原始分数直方图
    ax4.hist(
        label_scores,
        bins=bins_0_1,
        color='orange',
        alpha=0.7,
        edgecolor='black'
    )
    ax4.set_ylabel('Label count')
    ax4.set_xlabel('Anomaly score (0–1)')
    ax4.set_title('Label Scores Distribution (0–1)')

    # 整体调整
    fig.suptitle(f"Scores Histogram Exp"+str(num_exp)+ f": ({os.path.basename(score_file)})", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 6. 保存
    imgs_dir = os.path.join(os.path.dirname(__file__), 'imgs')
    os.makedirs(imgs_dir, exist_ok=True)
    save_name = os.path.splitext(os.path.basename(score_file))[0] + "_exp" + str(num_exp) + '.png'
    save_path = os.path.join(imgs_dir, save_name)
    fig.savefig(save_path)
    plt.close(fig)

    return save_path

