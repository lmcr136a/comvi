def run(data, network, cfg_run):
    """
    한방에 train/val을 진행, validation accuracy가 가장 높은 파라미터를 가지고
    test 진행. test accuracy 를 포함한 학습과정 전체를 가지고 있는 정보를 반환.

    Args: dataloader, network, configuration with run

    Output: history of the run & classification test accuracy
    """
    best_network = _trainNval(data, network, cfg_run)
    test_accuracy = _test(data['test'], best_network, cfg_run)
    return test_accuracy

def _trainNval(data, network, cfg_run):
    """
    cfg_run에 담긴대로 training/validation 진행
    가장 높은 validation accuracy를 가진 네트워크를 출력

    여기서 data는 dictionary type이다.
    """
    pass

def _test(data, network, cfg_run):
    """
    test accuracy 반환, 여기서 data는 dataloader type이다.
    """
    pass