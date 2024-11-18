
from typing import List
from data.graph_builder.code_graph import Sample
from util.setting import log
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict


def evaluation(data_bcb, data_gcj, data_gpt):
    bcb_evaluation(data_bcb)
    gcj_evaluation(data_gcj)
    gptclonebench_evaluation(data_gpt)

def general_p_r_f(data):
    log.info("Evaluating General Precision, Recall and F1")
    samples, predicted_labels = zip(*data)
    true_labels = [sample.clone_label for sample in samples]
    precision = precision_score(true_labels, predicted_labels, average='binary')
    recall = recall_score(true_labels, predicted_labels, average='binary')
    f1 = f1_score(true_labels, predicted_labels, average='binary')
    log.info(f"Precision: {precision}, Recall: {recall}, F1: {f1}")


def clone_type_p_r_f(data):

    grouped_data = defaultdict(list)
    for sample, label in data:
        grouped_data[sample.clone_type].append((sample.clone_label, label))

    results = {}
    for clone_type, values in grouped_data.items():
        if clone_type == 'Non_Clone':
            continue
        # Separate predicted and true labels for this clone_type
        true_labels, predicted_labels  = zip(*values)
    
        # Compute precision, recall, and F1-score
        recall = recall_score(true_labels, predicted_labels, average='binary')    
        # Store results
        results[clone_type] = {
            "recall": recall
        }

    # Output the results
    for clone_type, metrics in results.items():
        log.info(f"Clone Type: {clone_type}, Recall: {metrics['recall']}")



def bcb_evaluation(data_bcb):
    log.info("Evaluating BCB data-----------------------------------------------------")
    if len(data_bcb) == 0:
        log.info("No BCB data to evaluate")
        return
    general_p_r_f(data_bcb)

    log.info("Strat evaluation BCB per type-------------------------------------------------")
    clone_type_p_r_f(data_bcb)


def gcj_evaluation(data_gcj):
    log.info("Evaluating GCJ data-----------------------------------------------------")
    if len(data_gcj) == 0:
        log.info("No GCJ data to evaluate")
        return
    general_p_r_f(data_gcj)

def gptclonebench_evaluation(data_gpt):
    log.info("Evaluating GPTCloneBench data-----------------------------------------------------")
    if len(data_gpt) == 0:
        log.info("No GPTCloneBench data to evaluate")
        return
    general_p_r_f(data_gpt)

    log.info("Strat evaluation GPTCloneBench per type-------------------------------------------------")
    clone_type_p_r_f(data_gpt)

