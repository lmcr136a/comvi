git clone https://github.com/lmcr136a/comvi.git 실행 고고

# Update Note
## ver.211103
- 일단 전반적인 코드 구조는 만들어놓음, 이제 추가적으로 `preprocess` 부분에서 cv technique 추가하면 될듯
- 워낙 코드가 간단해서 설명이 필요해보이지는 않지만 그래도 굳이 설명하면
  - comvi.py에서 전체 과정 진행, .yml 파일 읽어와서 configuration 생성하고 이에 따른 dataset, dataloader, network 생성해서 train/val/test까지 한번에 진행
  - result에 이것저것 넣으려고 구상은 했는데 현재는 간단하게 test accuracy만 포함함
- 지금 configuration을 담고 있는 .yml 파일 체계가 지저분한 상태임. 마음에 안들면 바꾸면 됨

## ver.211106
- 각각 주요 단계 별로 콘솔에 로그 출력할 수 있도록 print해놓음 (아래는 test.yml의 출력 결과이다.)
```
[ DATADIR ]  ./data_for_test/stl10_small
[ DATASET ] [ train ] N_CLASS: 2 , SIZE: 100
[ DATASET ] [ val ] N_CLASS: 2 , SIZE: 20
[ DATASET ] [ test ] N_CLASS: 2 , SIZE: 20
[ DEVICE  ] No CUDA. Working on CPU.
[OPTIMIZER] ADAM FIXED [LearningRate]  5e-05

TRAINING START!
Epoch 0/5
----------------------------------------
[  TRAIN  ] Loss: 0.6881 Acc: 0.4900

Epoch 1/5
----------------------------------------
[  TRAIN  ] Loss: 0.6299 Acc: 0.6400
[   VAL   ] Loss: 0.7590 Acc: 0.5000

Epoch 2/5
----------------------------------------
[  TRAIN  ] Loss: 0.5654 Acc: 0.7500

Epoch 3/5
----------------------------------------
[  TRAIN  ] Loss: 0.5033 Acc: 0.7200
[   VAL   ] Loss: 1.0544 Acc: 0.5000

Epoch 4/5
----------------------------------------
[  TRAIN  ] Loss: 0.4864 Acc: 0.7700

Epoch 5/5
----------------------------------------
[  TRAIN  ] Loss: 0.5973 Acc: 0.7500
[   VAL   ] Loss: 0.4150 Acc: 0.8000

[  TEST   ] Loss: 0.6093 Acc: 0.7000
Test accuracy : 0.699999988079071
```


# TODO LIST
- Custom Transform으로 Computer Vision Technique 추가해주는 부분 만들기
- ResNet50 말고 사용하고 싶은 network 있으면 구현 or 아이디어 제시 (그러면 이원석이 만들어낼수도?)
  - VGG16, GoogLeNet, MobileNet 같은거, 굉장히 유명하거나 baseline이 되는거 쓰면 될듯
- **[중요]** 실험에 사용하기로 한 데이터셋을 실제로 구축해야됨
- First Layer에 사용되는 weight를 강제로 주입하는 방법 알아보기
  - 아마 `load_state`같은거 사용하면 되지 않을까 싶다.
- 현재 학습이 진행되는 `run`함수에서 best network를 train에 사용하고 버리는 방식을 사용하고 있는데, 이 네트워크 파라미터를 저장하는 부분을 구현하지 않음. 필요시 구현할 필요 있어보임
  - network parameter 저장 경로 지정해줄 필요 있음
  - 저장해놓은 경로를 불러오는 과정도 필요하지 않을까 싶음
  - 만약 네트워크 파라미터를 저장한다고 하면 `run`에서 training을 거치지 않고 바로 test만 하는 상황이 발생할 수도 있는데 이는 나중에 생각하기로 하자