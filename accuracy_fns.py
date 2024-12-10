from postproc_utils import gqa_postproc, general_postprocessing

def GQA_accuracy(prediction, ground_truth, *args):
        """
        Args:
            prediction (list): List of predicted answers.
            ground_truth (list): List of ground truth answers.
        Returns:
            score (float): Score of the prediction.
        """
        if len(prediction) == 0:  # if no prediction, return 0
            return 0
        assert len(prediction) == len(ground_truth)
        score = 0
        for p, g in zip(prediction, ground_truth):
            if gqa_postproc(p) == g:
                score += 1
        return score / len(prediction)

def general_accuracy(prediction, ground_truth, *args):
        """
        Args:
            prediction (list): List of predicted answers.
            ground_truth (list): List of ground truth answers.
        Returns:
            score (float): Score of the prediction.
        """
        if len(prediction) == 0:  # if no prediction, return 0
            return 0
        assert len(prediction) == len(ground_truth)
        score = 0
        for p, g in zip(prediction, ground_truth):
            if general_postprocessing(p) == g:
                score += 1
        return score / len(prediction)
    

def get_accuracy_fn(dataset):
    if dataset == "GQA":
        return GQA_accuracy
    else:
        return general_accuracy