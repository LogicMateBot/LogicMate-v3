# poetry run celery -A logicmate.main worker --loglevel=info
# curl -fsSL https://ollama.com/install.sh | sh
# OLLAMA_HOST="127.0.0.1:11435" ollama serve
# ollama pull gemma3:12b

from datetime import datetime, timedelta
import requests
import logging
from celery import Celery
from logicmate.models.video.video import Video
from logicmate.utils.env.env import get_required_env_variables, load_env_variables
from logicmate.utils.logger.logger import setup_logging
from logicmate.bot.controller import start_bot

setup_logging()
load_env_variables(env_path=".env")
config: dict[str, str] = get_required_env_variables()

celery = Celery(
    main="logicmate_bot",
    broker=config.get("CELERY_BROKER_URL"),
    backend=config.get("CELERY_RESULT_BACKEND"),
)

celery.conf.imports = ["logicmate.main"]

logging.info(msg="Celery broker and backend configured successfully.")
logging.info(msg="Bot is waiting for tasks to process videos.")


@celery.task(name="logicmate_bot.process_video")
def process_video_task(
    video_bytes: bytes, users_emails: list[str], current_user_email: str
) -> None:
    with open(file="/tmp/video.mp4", mode="wb") as f:
        f.write(video_bytes)

    logging.info(
        msg=f"Processing video for user {current_user_email} and users {users_emails}.",
    )

    video: Video | None = start_bot(video_path="/tmp/video.mp4", config=config)

    if video is None:
        logging.error(msg="Video processing failed. No result from bot.")
        return

    all_users: list[str] = users_emails + [current_user_email]

    payload = {
        "id": video.id,
        "duration": video.duration,
        "categories": video.categories,
        "scenes": [scene.model_dump() for scene in video.scenes],
        "title": video.title,
        "explanation": video.explanation,
        "code": video.code,
        "diagram": video.diagram,
        "approaches": [a.model_dump() for a in (video.approaches or [])],
        "exercises": [e.model_dump() for e in (video.exercises or [])],
        "users": all_users,
    }

    try:
        response: requests.Response = requests.post(
            url="http://localhost:8000/", json=payload
        )
        response.raise_for_status()
        logging.info(msg="Video successfully sent to backend.")
    except requests.RequestException as e:
        logging.exception(msg=f"Failed to send video to backend: {e}")


if __name__ == "__main__":
    from datetime import datetime

    logging.info(msg="Starting bot manually...")
    moment: datetime = datetime.now()

    video: Video | None = start_bot(
        video_path="6c26a5d1-5f67-4752-b1a4-7b5911cdd157.mp4",
        config=config,
    )

    if video is None:
        logging.warning(msg="No video processed. Exiting.")
        exit(code=0)

    final_moment: datetime = datetime.now()
    duration: timedelta = final_moment - moment

    print(video.model_dump_json(indent=2))
    logging.info(msg=f"Bot finished in {duration.total_seconds()} seconds.")
