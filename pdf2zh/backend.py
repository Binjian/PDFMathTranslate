from flask import Flask, request, send_file
from celery import Celery, Task
from celery.result import AsyncResult
from pdf2zh import translate_stream
import tqdm
import json
import io
from string import Template
from pdf2zh.doclayout import ModelInstance
from pdf2zh.config import ConfigManager
from pdf2zh.mongo_store import JobArtifactStore

# MongoDB is the source of truth for result PDFs: the task stores the mono/dual
# blobs in GridFS keyed by the Celery task id and the result route serves them
# from MongoDB instead of round-tripping large payloads through the result backend.
_artifact_store = JobArtifactStore()

flask_app = Flask("pdf2zh")
flask_app.config.from_mapping(
    CELERY=dict(
        broker_url=ConfigManager.get("CELERY_BROKER", "redis://127.0.0.1:6379/0"),
        result_backend=ConfigManager.get("CELERY_RESULT", "redis://127.0.0.1:6379/0"),
    )
)


def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.Task = FlaskTask
    celery_app.set_default()
    celery_app.autodiscover_tasks()
    app.extensions["celery"] = celery_app
    return celery_app


celery_app = celery_init_app(flask_app)


@celery_app.task(bind=True)
def translate_task(
    self: Task,
    stream: bytes,
    args: dict,
):
    def progress_bar(t: tqdm.tqdm):
        self.update_state(state="PROGRESS", meta={"n": t.n, "total": t.total})  # noqa
        print(f"Translating {t.n} / {t.total} pages")

    if "prompt" in args:
        args["prompt"] = Template(args["prompt"])

    doc_mono, doc_dual = translate_stream(
        stream,
        callback=progress_bar,
        model=ModelInstance.value,
        **args,
    )
    # Store the result PDFs in MongoDB (the source of truth) keyed by task id;
    # return only the stored file names so the result backend stays lightweight.
    task_id = self.request.id
    _artifact_store.put_file(
        doc_mono, f"{task_id}-mono.pdf", job_id=task_id, variant="mono"
    )
    _artifact_store.put_file(
        doc_dual, f"{task_id}-dual.pdf", job_id=task_id, variant="dual"
    )
    return {"mono": f"{task_id}-mono.pdf", "dual": f"{task_id}-dual.pdf"}


@flask_app.route("/v1/translate", methods=["POST"])
def create_translate_tasks():
    file = request.files["file"]
    stream = file.stream.read()
    print(request.form.get("data"))
    args = json.loads(request.form.get("data"))
    task = translate_task.delay(stream, args)
    return {"id": task.id}


@flask_app.route("/v1/translate/<id>", methods=["GET"])
def get_translate_task(id: str):
    result: AsyncResult = celery_app.AsyncResult(id)
    if str(result.state) == "PROGRESS":
        return {"state": str(result.state), "info": result.info}
    else:
        return {"state": str(result.state)}


@flask_app.route("/v1/translate/<id>", methods=["DELETE"])
def delete_translate_task(id: str):
    result: AsyncResult = celery_app.AsyncResult(id)
    result.revoke(terminate=True)
    _artifact_store.delete_files({"job_id": id})
    return {"state": str(result.state)}


@flask_app.route("/v1/translate/<id>/<format>")
def get_translate_result(id: str, format: str):
    result = celery_app.AsyncResult(id)
    if not result.ready():
        return {"error": "task not finished"}, 400
    if not result.successful():
        return {"error": "task failed"}, 400
    variant = "mono" if format == "mono" else "dual"
    if not _artifact_store.available():
        return {"error": "artifact storage (MongoDB) is unavailable"}, 503
    blob = _artifact_store.get_file({"job_id": id, "variant": variant})
    if blob is None:
        return {"error": "result not found"}, 404
    data, _ = blob
    return send_file(io.BytesIO(data), "application/pdf")


if __name__ == "__main__":
    flask_app.run()
