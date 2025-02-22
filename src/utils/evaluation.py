def calculate_accuracy(y_true, y_pred):
    correct_predictions = sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def generate_confusion_matrix(y_true, y_pred):
    true_positive = sum((y_true == 1) & (y_pred == 1))
    true_negative = sum((y_true == 0) & (y_pred == 0))
    false_positive = sum((y_true == 0) & (y_pred == 1))
    false_negative = sum((y_true == 1) & (y_pred == 0))
    
    return {
        'TP': true_positive,
        'TN': true_negative,
        'FP': false_positive,
        'FN': false_negative
    }