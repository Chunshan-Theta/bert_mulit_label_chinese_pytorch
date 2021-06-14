
def __get_f1__(TP, FN, FP):

    recall = (TP / (TP + FN)) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return f1, recall, precision




def __convert_to_validation__(anss,pred):
    TP, FN, FP = 0, 0, 0

    #
    # TP 答案是有 ， 預測也有
    # FP 答案沒有 ， 預測是有
    # FN 答案是有 ， 預測沒有
    ## 正規
    for p_gs, a_gs in zip(pred, anss):
        for p, a in zip(p_gs, a_gs):
            if p == 1 and a == 1:
                TP += 1
            elif p == 1 and a == 0:
                FP += 1
            elif p == 0 and a == 1:
                FN += 1

    return TP, FN, FP