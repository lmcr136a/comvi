import argparse

from util import configuration, getDataSet, getDataLoader, run, getResult
from backbone import getNetwork


def main(args):
    # configuration 파일 읽어오는 모듈
    cfg = configuration(args.config)

    # Data 불러오는 모듈
    dataset, n_class = getDataSet(cfg["data"])
    dataloader = getDataLoader(dataset, cfg["data"])

    # Network 불러오는 모듈
    network = getNetwork(cfg["network"], n_class)

    # RUN(train/val/test): validation 이후 가장 좋은 성능 보이는 걸로 test 진행
    result = run(dataset, dataloader, network, cfg["run"])

    # result 적당한 방식으로 출력해주는 모듈
    getResult(result)


if __name__ == '__main__':
    # 우리 코드 실행시킬때 argument로 config 파일 경로만 넣어주면 되게 처리함
    # 만약에 정말 동시에 여러개의 코드를 실행시킬 일이 있다면 그때는 .sh파일을 사용하자
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
