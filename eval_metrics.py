import numpy as np




def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def accuracy_at_k(actual, predicted, topk):
    test_num = 0.0
    hit_num = 0.0
    num_users = len(predicted)

    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])

        if len(act_set) != 0:
            if len(act_set & pred_set) != 0:
                hit_num = hit_num + 1
            test_num += 1
    return hit_num / test_num

def ndcg_at_k(actual, predicted, topk):
    sum_ndcg_score = 0.0
    num_users = len(predicted)
    true_users = 0
    # idcg score
    idcg_score = 0.0
    for index in range(topk):
        idcg_score += np.reciprocal(np.log2(index+2))

    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            true_users += 1
            # dcg score
            dcg_score = 0.0
            for (index,cand_poi) in enumerate(predicted[i][:topk]):
                if cand_poi in act_set:
                    dcg_score += np.reciprocal(np.log2(index+2))
            ndcg_score = dcg_score/idcg_score
            sum_ndcg_score += ndcg_score

    return sum_ndcg_score/true_users

def hit_ratio_at_k(actual, predicted, topk):
    num_users = len(predicted)
    true_users = 0
    sum_num_of_hit = 0
    sum_GT = 0

    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            true_users += 1
            sum_num_of_hit += len(act_set & pred_set)
            sum_GT += len(act_set)

    return sum_num_of_hit/sum_GT

