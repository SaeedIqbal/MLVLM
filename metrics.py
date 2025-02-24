import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score
from scipy.spatial.distance import cosine

# Step 1: Factuality Score (Ground Truth vs. Generated Reports)
def compute_factuality_score(generated_reports, ground_truth_reports):
    """
    Compute the factuality score by comparing generated reports with ground truth.
    
    Args:
        generated_reports (list): List of generated medical reports.
        ground_truth_reports (list): List of ground truth medical reports.
    
    Returns:
        float: Factuality score.
    """
    factuality_scores = []
    for gen_report, gt_report in zip(generated_reports, ground_truth_reports):
        # Example metric: Jaccard similarity between sets of tokens
        gen_tokens = set(gen_report.split())
        gt_tokens = set(gt_report.split())
        intersection = len(gen_tokens.intersection(gt_tokens))
        union = len(gen_tokens.union(gt_tokens))
        factuality_scores.append(intersection / union if union > 0 else 0.0)
    return np.mean(factuality_scores)


# Step 2: Precision and Recall of Retrieved Knowledge
def compute_precision_recall(true_positives, false_positives, false_negatives):
    """
    Compute precision and recall for retrieved knowledge.
    
    Args:
        true_positives (int): Number of true positives.
        false_positives (int): Number of false positives.
        false_negatives (int): Number of false negatives.
    
    Returns:
        tuple: Precision and recall.
    """
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    return precision, recall


# Step 3: Hallucination Rate
def compute_hallucination_rate(hallucinated_content_count, total_content_count):
    """
    Compute the hallucination rate.
    
    Args:
        hallucinated_content_count (int): Count of hallucinated content.
        total_content_count (int): Total content count.
    
    Returns:
        float: Hallucination rate.
    """
    return hallucinated_content_count / total_content_count if total_content_count > 0 else 0.0


# Step 4: Cross-Modal Alignment Index (CMAI)
def compute_cmai(image_embeddings, text_embeddings):
    """
    Compute the Cross-Modal Alignment Index (CMAI) using cosine similarity.
    
    Args:
        image_embeddings (torch.Tensor): Image embeddings of shape (N, D).
        text_embeddings (torch.Tensor): Text embeddings of shape (N, D).
    
    Returns:
        float: CMAI score.
    """
    cmais = []
    for img_emb, txt_emb in zip(image_embeddings, text_embeddings):
        similarity = 1 - cosine(img_emb.detach().numpy(), txt_emb.detach().numpy())
        cmais.append(similarity)
    return np.mean(cmais)


# Step 5: Pathological Feature Extraction Accuracy (Dice Coefficient)
def compute_dice_coefficient(pred_mask, gt_mask):
    """
    Compute the Dice coefficient for pathological feature extraction.
    
    Args:
        pred_mask (torch.Tensor): Predicted mask of shape (H, W).
        gt_mask (torch.Tensor): Ground truth mask of shape (H, W).
    
    Returns:
        float: Dice coefficient.
    """
    intersection = torch.sum(pred_mask * gt_mask)
    dice = (2.0 * intersection) / (torch.sum(pred_mask) + torch.sum(gt_mask))
    return dice.item()


# Step 6: Epistemic Uncertainty Quantification (Predictive Entropy and Mutual Information)
def compute_predictive_entropy(probabilities):
    """
    Compute predictive entropy for uncertainty quantification.
    
    Args:
        probabilities (torch.Tensor): Predicted probabilities of shape (N, C).
    
    Returns:
        float: Predictive entropy.
    """
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1).mean()
    return entropy.item()


def compute_mutual_information(predictive_entropy, expected_entropy):
    """
    Compute mutual information for uncertainty quantification.
    
    Args:
        predictive_entropy (float): Predictive entropy.
        expected_entropy (float): Expected entropy.
    
    Returns:
        float: Mutual information.
    """
    return predictive_entropy - expected_entropy


# Step 7: Hallucination Reduction Rate (HRR)
def compute_hrr(hallucination_rate_baseline, hallucination_rate_improved):
    """
    Compute the Hallucination Reduction Rate (HRR).
    
    Args:
        hallucination_rate_baseline (float): Baseline hallucination rate.
        hallucination_rate_improved (float): Improved hallucination rate.
    
    Returns:
        float: HRR percentage.
    """
    return ((hallucination_rate_baseline - hallucination_rate_improved) / hallucination_rate_baseline) * 100


# Step 8: Clinical Consistency Score (CCS)
def compute_diagnostic_accuracy(correct_diagnoses, total_diagnoses):
    """
    Compute diagnostic accuracy.
    
    Args:
        correct_diagnoses (int): Number of correct diagnoses.
        total_diagnoses (int): Total number of diagnoses.
    
    Returns:
        float: Diagnostic accuracy.
    """
    return correct_diagnoses / total_diagnoses if total_diagnoses > 0 else 0.0


def compute_treatment_recommendation_accuracy(weights, correct_treatments, total_treatments):
    """
    Compute treatment recommendation accuracy.
    
    Args:
        weights (list): Weights for each diagnosis-treatment pair.
        correct_treatments (list): Correct treatment counts.
        total_treatments (list): Total treatment counts.
    
    Returns:
        float: Treatment recommendation accuracy.
    """
    numerator = sum(w * ct for w, ct in zip(weights, correct_treatments))
    denominator = sum(w * tt for w, tt in zip(weights, total_treatments))
    return numerator / denominator if denominator > 0 else 0.0


def compute_medical_information_coherence(model_info, expert_info, weights):
    """
    Compute coherence of medical information.
    
    Args:
        model_info (torch.Tensor): Model-generated medical information.
        expert_info (torch.Tensor): Expert-validated medical information.
        weights (torch.Tensor): Weights for different levels of medical information.
    
    Returns:
        float: Medical information coherence.
    """
    numerator = torch.sum(weights * (model_info * expert_info))
    denominator = torch.sqrt(torch.sum(weights * (model_info**2))) * torch.sqrt(torch.sum(weights * (expert_info**2)))
    return numerator / denominator if denominator > 0 else 0.0


def compute_clinical_consistency_score(diagnostic_accuracy, treatment_accuracy, coherence, beta_weights):
    """
    Compute the Clinical Consistency Score (CCS).
    
    Args:
        diagnostic_accuracy (float): Diagnostic accuracy.
        treatment_accuracy (float): Treatment recommendation accuracy.
        coherence (float): Medical information coherence.
        beta_weights (list): Weights for each component [beta1, beta2, beta3].
    
    Returns:
        float: CCS.
    """
    beta1, beta2, beta3 = beta_weights
    return beta1 * diagnostic_accuracy + beta2 * treatment_accuracy + beta3 * coherence


# Main Function
if __name__ == "__main__":
    # Example usage
    # Dummy data
    generated_reports = ["Patient has pneumonia", "Patient has diabetes"]
    ground_truth_reports = ["Patient has pneumonia", "Patient has hypertension"]
    true_positives = 5
    false_positives = 2
    false_negatives = 3
    hallucinated_content_count = 10
    total_content_count = 100
    image_embeddings = torch.randn(5, 128)
    text_embeddings = torch.randn(5, 128)
    pred_mask = torch.randint(0, 2, (256, 256)).float()
    gt_mask = torch.randint(0, 2, (256, 256)).float()
    probabilities = torch.softmax(torch.randn(10, 5), dim=-1)
    predictive_entropy = compute_predictive_entropy(probabilities)
    expected_entropy = 0.5  # Placeholder
    hallucination_rate_baseline = 0.2
    hallucination_rate_improved = 0.1
    correct_diagnoses = 8
    total_diagnoses = 10
    weights = [1.0, 0.5]
    correct_treatments = [4, 3]
    total_treatments = [5, 6]
    model_info = torch.randn(5)
    expert_info = torch.randn(5)
    coherence_weights = torch.tensor([0.5, 0.3, 0.2])
    beta_weights = [0.4, 0.3, 0.3]

    # Step 1: Factuality Score
    factuality_score = compute_factuality_score(generated_reports, ground_truth_reports)

    # Step 2: Precision and Recall
    precision, recall = compute_precision_recall(true_positives, false_positives, false_negatives)

    # Step 3: Hallucination Rate
    hallucination_rate = compute_hallucination_rate(hallucinated_content_count, total_content_count)

    # Step 4: Cross-Modal Alignment Index (CMAI)
    cmai = compute_cmai(image_embeddings, text_embeddings)

    # Step 5: Dice Coefficient
    dice = compute_dice_coefficient(pred_mask, gt_mask)

    # Step 6: Predictive Entropy and Mutual Information
    mutual_information = compute_mutual_information(predictive_entropy, expected_entropy)

    # Step 7: Hallucination Reduction Rate (HRR)
    hrr = compute_hrr(hallucination_rate_baseline, hallucination_rate_improved)

    # Step 8: Clinical Consistency Score (CCS)
    diagnostic_accuracy = compute_diagnostic_accuracy(correct_diagnoses, total_diagnoses)
    treatment_accuracy = compute_treatment_recommendation_accuracy(weights, correct_treatments, total_treatments)
    coherence = compute_medical_information_coherence(model_info, expert_info, coherence_weights)
    ccs = compute_clinical_consistency_score(diagnostic_accuracy, treatment_accuracy, coherence, beta_weights)

    # Print results
    print("Factuality Score:", factuality_score)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Hallucination Rate:", hallucination_rate)
    print("Cross-Modal Alignment Index (CMAI):", cmai)
    print("Dice Coefficient:", dice)
    print("Predictive Entropy:", predictive_entropy)
    print("Mutual Information:", mutual_information)
    print("Hallucination Reduction Rate (HRR):", hrr)
    print("Clinical Consistency Score (CCS):", ccs)