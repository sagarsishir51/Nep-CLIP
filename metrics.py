import numpy as np

def calculate_ap(found_ranks, total_relevant):
    """
    Calculates Average Precision for a single query.
    found_ranks: 0-indexed positions of all correct items in the sorted list.
    total_relevant: Total number of ground-truth items (e.g., 5 for Flickr8k I2T).
    """
    if len(found_ranks) == 0:
        return 0.0

    # Sort ranks to ensure we calculate P@k in order (1st found, 2nd found...)
    sort_ranks = np.sort(found_ranks) + 1
    precision_sum = 0.0
    for i, r in enumerate(sort_ranks):
        # (i + 1) is the count of relevant items found so far
        precision_sum += (i + 1) / r
    return precision_sum / total_relevant

def calculate_mar(found_ranks, total_relevant, k=10):
    """
    Calculates Average Recall @ K for a single query.
    found_ranks: 0-indexed positions of all correct items.
    total_relevant: Total possible relevant items (5 for Flickr8k I2T).
    k: The cutoff for recall.
    """
    # Count how many of the found_ranks are within the top K
    hits_at_k = np.sum(found_ranks < k)
    return hits_at_k / total_relevant

def calculate_maf1(found_ranks, total_relevant, k=10):
    """
    Calculates F1 @ K for a single query.
    found_ranks: 0-indexed positions of all correct items.
    total_relevant: Total possible relevant items (5 for Flickr8k I2T).
    k: The cutoff for precision and recall.
    """
    hits_at_k = np.sum(found_ranks < k)

    precision_at_k = hits_at_k / k
    recall_at_k = hits_at_k / total_relevant

    if (precision_at_k + recall_at_k) == 0:
        return 0.0

    f1 = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    return f1


def compute_metrics(similarity, query_type='i2t', captions_per_image=5):
    """
    I2T: Row-wise search (Is the correct caption in the top K for this image?)
    T2I: Column-wise search (Is the correct image in the top K for this caption?)
    """
    aps, ars, af1s = [], [], []
    if query_type == 'i2t':
        # [N_images, N_texts] -> Axis 0 is images
        num_queries = similarity.shape[0]
        ranks = []
        for i in range(num_queries):
            scores = similarity[i]
            inds = np.argsort(scores)[::-1]
            gt_indices = np.arange(i * captions_per_image, (i + 1) * captions_per_image)

            # Find all positions where the ground truth captions appear
            all_found_ranks = np.where(np.isin(inds, gt_indices))[0]

            # Use the best (first) rank for standard R@K metrics
            best_rank = all_found_ranks[0]
            ranks.append(best_rank)

            # Calculate AP using the external function
            aps.append(calculate_ap(all_found_ranks, captions_per_image))
            ars.append(calculate_mar(all_found_ranks, captions_per_image, k=10))
            af1s.append(calculate_maf1(all_found_ranks, captions_per_image, k=10))
    else:
        # [N_images, N_texts] -> Axis 1 is texts
        num_queries = similarity.shape[1]
        ranks = []
        for j in range(num_queries):
            scores = similarity[:, j]
            inds = np.argsort(scores)[::-1]
            gt_image_index = j // captions_per_image

            rank = np.where(inds == gt_image_index)[0][0]
            ranks.append(rank)

            # For T2I (1 relevant item), AP is just 1/rank
            aps.append(calculate_ap(np.array([rank]), 1))
            ars.append(calculate_mar(np.array([rank]), 1, k=10))
            af1s.append(calculate_maf1(np.array([rank]), 1, k=10))
    ranks = np.array(ranks)
    metrics = {
        "R@1":  float(np.mean(ranks < 1) * 100),
        "R@5":  float(np.mean(ranks < 5) * 100),
        "R@10": float(np.mean(ranks < 10) * 100),
        "MedR": float(np.median(ranks) + 1),
        "MAP":  float(np.mean(aps) * 100),
        "MAR":  float(np.mean(ars) * 100),
        "MAF1": float(np.mean(af1s) * 100)
    }

    r_indexed = ranks + 1
    for k in range(1, 11):
        mrr_k = np.where(r_indexed <= k, 1.0 / r_indexed, 0.0).mean()
        metrics[f"MRR@{k}"] = float(mrr_k)

    return metrics

import re
def get_numeric_step(name):
    """Extracts numeric step from string 'step_1234' (expects a string)."""
    match = re.search(r'step_(\d+)', name)
    return int(match.group(1)) if match else 0


def log_report(reports):
  # Header updated with MAP, MAR, MAF1, and MRR
  header = f"{'Model Checkpoint':<30} | {'I2T (R@1 / R@5 / R@10  |  M  |  MAP  |  MAR  |  MF1  |  MRR)':<80} | {'T2I (R@1 / R@5 / R@10  |  M  |  MAP  |  MAR  |  MF1  |  MRR)':<80}"
  print(header)
  print("-" * len(header))

  for model, metrics in sorted(reports.items(), key=lambda x: get_numeric_step(x[0])):
      i2t = metrics['I2T']
      t2i = metrics['T2I']

      # Formatting strings to include MAF1 (MF1)
      i2t_str = (f"{i2t['R@1']:>4.2f}/{i2t['R@5']:>4.2f}/{i2t['R@10']:>4.2f} | "
                f"M:{i2t['MedR']:<2.0f} | {i2t['MAP']:>5.2f} | {i2t['MAR']:>5.2f} | "
                f"{i2t['MAF1']:>5.2f} | {i2t['MRR@10']*100:>5.2f}")

      t2i_str = (f"{t2i['R@1']:>4.2f}/{t2i['R@5']:>4.2f}/{t2i['R@10']:>4.2f} | "
                f"M:{t2i['MedR']:<2.0f} | {t2i['MAP']:>5.2f} | {t2i['MAR']:>5.2f} | "
                f"{t2i['MAF1']:>5.2f} | {t2i['MRR@10']*100:>5.2f}")

      print(f"{model:<30} | {i2t_str:<60} | {t2i_str:<60}")