from logicmate.bot.controller import start_bot
from logicmate.utils.logger.logger import setup_logging


def main(path_to_file: str) -> None:
    setup_logging()
    start_bot(path_to_file=path_to_file)


if __name__ == "__main__":
    path_to_file = "media/videos/21a10a54-4ac2-4eb8-8807-dcfbaab749ca.mp4"
    main(path_to_file=path_to_file)
