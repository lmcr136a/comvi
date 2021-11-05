git clone https://github.com/lmcr136a/comvi.git 실행 고고

ver.211103
- 일단 전반적인 코드 구조는 만들어놓음, 이제 추가적으로 `preprocess` 부분에서 cv technique 추가하면 될듯
- 워낙 코드가 간단해서 설명이 필요해보이지는 않지만 그래도 굳이 설명하면
  - comvi.py에서 전체 과정 진행, .yml 파일 읽어와서 configuration 생성하고 이에 따른 dataset, dataloader, network 생성해서 train/val/test까지 한번에 진행
  - result에 이것저것 넣으려고 구상은 했는데 현재는 간단하게 test accuracy만 포함함
- 지금 configuration을 담고 있는 .yml 파일 체계가 지저분한 상태임. 마음에 안들면 바꾸면 됨
