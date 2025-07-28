# poetry run celery -A logicmate.main worker --loglevel=info
# curl -fsSL https://ollama.com/install.sh | sh
# OLLAMA_HOST="127.0.0.1:11435" ollama serve
# ollama pull gemma3:12b

import logging
from celery import Celery
from logicmate.utils.env.env import get_required_env_variables, load_env_variables
from logicmate.utils.logger.logger import setup_logging
from logicmate.bot.controller import start_bot

setup_logging()
load_env_variables(env_path=".env")
config = get_required_env_variables()

celery = Celery(
    main="logicmate_bot",
    broker=config.get("CELERY_BROKER_URL"),
    backend=config.get("CELERY_RESULT_BACKEND"),
)

celery.conf.imports = ["logicmate.main"]

logging.info("Celery broker and backend configured successfully.")
logging.info("Bot is waiting for tasks to process videos.")


@celery.task(name="logicmate_bot.process_video")
def process_video_task(
    video_bytes: bytes, users_emails: list[str], current_user_email: str
) -> None:
    with open("/tmp/video.mp4", "wb") as f:
        f.write(video_bytes)

    logging.info(
        f"Processing video for user {current_user_email} and users {users_emails}.",
    )

    video = start_bot(video_path="/tmp/video.mp4", config=config)


# Solo para pruebas manuales
if __name__ == "__main__":
    from datetime import datetime

    logging.info("Starting bot manually...")
    moment = datetime.now()

    video = start_bot(
        video_path="21a10a54-4ac2-4eb8-8807-dcfbaab749ca.mp4",
        config=config,
    )

    if video is None:
        logging.warning("No video processed. Exiting.")
        exit(0)

    final_moment = datetime.now()
    duration = final_moment - moment

    logging.info(f"Bot finished in {duration.total_seconds()} seconds.")
    print(video.model_dump_json(indent=2))
