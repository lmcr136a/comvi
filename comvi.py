################################################################################
# 개발자 메모
# - 일단 전반적인 흐름은 만들어놓고
# - logger를 추가해야할 것 같긴 한데 동시에 구상하기 빡세보여서 일단 후발주자한테
#   맡김 (그냥 콘솔에 출력만 하게 만들어도 괜찮을듯?)
################################################################################

# 실험을 진행할 메인 코드
import argparse

from util import configuration, getDataSet, getDataLoader, run, getResult
from backbone import getNetwork


def main(args):
    # configuration 파일 읽어오는 모듈
    cfg = configuration(args.config)

    # Data 불러오는 모듈
    dataset = getDataSet(cfg["data"])
    dataloader = getDataLoader(dataset, cfg["data"])

    # Network 불러오는 모듈
    network = getNetwork(cfg["network"])

    # RUN(train/val/test): validation 이후 가장 좋은 성능 보이는 걸로 test 진행
    result = run(dataset, dataloader, network, cfg["run"])

    # result 적당한 방식으로 출력해주는 모듈
    getResult(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        "-c",
        nargs="?",
        type=str,
        default="config/test.yml",
        help="Configuration file to use, .yml format",
    )
    args = parser.parse_args()
    main(args)
