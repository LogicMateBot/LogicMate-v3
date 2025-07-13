# poetry run celery -A logicmate.main worker --loglevel=info

import logging
from celery import Celery

from logicmate.bot.controller import start_bot
from logicmate.utils.env.env import get_required_env_variables, load_env_variables
from logicmate.utils.logger.logger import setup_logging


setup_logging()
load_env_variables(env_path=".env")
config: dict[str, str] = get_required_env_variables()

celery = Celery(
    main="logicmate_bot",
    broker=config.get("CELERY_BROKER_URL"),
    backend=config.get("CELERY_RESULT_BACKEND"),
)

celery.conf.imports = ["logicmate.main"]

logging.info(
    msg="Celery broker and backend configured successfully.",
)

logging.info(
    msg="Bot is waiting for tasks to process videos.",
)


@celery.task(name="logicmate_bot.process_video")
def process_video_task(
    video_bytes: bytes, users_emails: list[str], current_user_email: str
) -> None:
    with open(file="/tmp/video.mp4", mode="wb") as f:
        f.write(video_bytes)

    logging.info(
        msg=f"Processing video for user {current_user_email} and users {users_emails}. to notify them.",
    )

    # start_bot(video_path="/tmp/video.mp4", config=config, users_emails=users_emails, current_user_email=current_user_email)


if __name__ == "__main__":
    start_bot(
        video_path="media/videos/21a10a54-4ac2-4eb8-8807-dcfbaab749ca.mp4",
        config=config,
    )
