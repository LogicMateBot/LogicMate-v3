from logicmate.bot.controller import start_bot
from logicmate.utils.env.env import get_required_env_variables, load_env_variables
from logicmate.utils.logger.logger import setup_logging


def main(video_path: str) -> None:
    setup_logging()
    load_env_variables(env_path="/teamspace/studios/this_studio/LogicMate-v3/.env")
    config: dict[str, str] = get_required_env_variables()
    start_bot(video_path=video_path, config=config)


if __name__ == "__main__":
    video_path = "/teamspace/studios/this_studio/LogicMate-v3/media/videos/21a10a54-4ac2-4eb8-8807-dcfbaab749ca.mp4"
    main(video_path=video_path)
