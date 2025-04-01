from logicmate.bot.controller import start_bot
from logicmate.utils.logger.logger import setup_logging


def main(path_to_file: str) -> None:
    setup_logging()
    start_bot(path_to_file=path_to_file)


if __name__ == "__main__":
    path_to_file = "media/videos/6c26a5d1-5f67-4752-b1a4-7b5911cdd157.mp4"
    main(path_to_file=path_to_file)
