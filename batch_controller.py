from clearml import Task, Logger
import random
import config
from dataset_utils import get_dataset_size


def batch_controller(
    hf_dataset_name,
    hf_config_name,
    batch_size,
    input_artifacts_task_id,
    output_artifacts_task_id,
    queue_name="gpu_worker",
):

    dataset_len = int(get_dataset_size(hf_dataset_name, hf_config_name, split="train"))

    total_batches = (dataset_len + batch_size - 1) // batch_size

    transcription_base_task = Task.get_task(
        project_name=config.task_templates_project_name,
        task_name=config.transcribe_base_task_name,
        allow_archived=False,
    )
    transcription_tasks_project = Task.get_project_id(
        project_name=config.transcription_tasks_project_name
    )

    enqueued_task_ids = []

    for i in range(total_batches):
        task = Task.clone(
            transcription_base_task,
            parent=Task.current_task(),
            project=transcription_tasks_project,
            name=f"transcribe_batch_{i}",
        )
        task.set_parameters(
            {
                "General/batch_index": i,
                "General/batch_size": batch_size,
                "General/input_task_id": input_artifacts_task_id,
                "General/output_task_id": output_artifacts_task_id,
            }
        )
        Task.enqueue(
            task=task,
            queue_name=queue_name,
        )

        enqueued_task_ids.append(task.id)

    Logger.current_logger().report_scalar(
        title="Batch Info", series="Total Batches", value=total_batches, iteration=0
    )

    current_task = Task.current_task()
    current_task.upload_artifact("enqueued_task_ids", enqueued_task_ids)


if __name__ == "__main__":

    params = Task.current_task().get_parameters()

    batch_controller(
        params["General/hf_dataset_name"],
        params["General/hf_config_name"],
        int(params["General/batch_size"]),
        params["General/input_task_id"],
        params["General/output_task_id"],
        params["General/queue_name"]
    )
