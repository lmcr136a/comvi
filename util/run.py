import torch
import torch.nn as nn
import torch.optim as optim

def run(dataset, dataloader, network, cfg_run):
    """
    한방에 train/val을 진행, validation accuracy가 가장 높은 파라미터를 가지고
    test 진행. test accuracy 를 포함한 학습과정 전체를 가지고 있는 정보를 반환.

    Args: dataset, dataloader, network, configuration with run

    Output: history of the run & classification test accuracy
    """
    device = is_cuda()

    # network to device
    network = network.to(device)

    # set criterion/optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=cfg_run["lr"])

    # Train/Val step, return the best accuracy network
    best_network = _trainNval(dataset, dataloader, network, cfg_run, criterion, optimizer, device)

    # Get the train accuracy
    test_accuracy = _test(dataset, dataloader, best_network, criterion, device)
    return test_accuracy

def _trainNval(dataset, dataloader, network, cfg_run, criterion, optimizer, device):
    """
    cfg_run에 담긴대로 training/validation 진행
    가장 높은 validation accuracy를 가진 네트워크를 출력

    여기서 data는 dictionary type이다.
    """
    best_acc = 0.0
    best_network = None

    for epoch in range(cfg_run["epoch"]):
        print('Epoch {}/{}'.format(epoch, cfg_run["epoch"] - 1))
        print('-' * 10)

        # Train step for every epoch
        network.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = network(inputs)
                _, preds = torch.max(outputs,1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / dataset['train']
        epoch_acc = running_corrects / dataset['train']

        print('[TRAIN] Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Validation step for every cfg_run["val_interval"] times
        if (epoch+1) % cfg_run["val_interval"] == 0:
            network.eval()

            # Do validation
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = network(inputs)
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            val_loss = running_loss / dataset['val']
            val_acc = running_corrects / dataset['val']

            print('[ VAL ] Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))

            # if validation accuracy is greater than best_acc, save that model
            if val_acc >= best_acc:
                best_network = network
                best_acc = val_acc
        print()

    return best_network

def _test(dataset, dataloader, network, criterion, device):
    """
    test accuracy 반환, 여기서도 dataset, dataloader는 dictionary type이다.
    """
    network.eval()

    # Do validation with test dataset
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = network(inputs)
            _, preds = torch.max(outputs,1)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    test_loss = running_loss / dataset['test']
    test_acc = running_corrects / dataset['test']

    print('[TEST ] Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))

    return test_acc

def is_cuda():
    if torch.cuda.is_available():
        print("CUDA available")
        return "cuda"
    else:
        print("No CUDA. Working on CPU.")
        return "cpu"