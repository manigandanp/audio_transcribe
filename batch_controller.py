from clearml import Task, Logger
import config
from dataset_utils import get_dataset_size


def batch_controller(
    hf_dataset_name,
    hf_config_name,
    batch_size,
    input_task_id,
    output_task_id,
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
    current_task = Task.current_task()
    
    enqueued_task_ids = []
    input_task = Task.get_task(task_id=input_task_id)
    for batch_index in range(total_batches):
        artifacts = input_task.artifacts
        all_keys = list(artifacts.keys())
        start_idx = batch_index * batch_size
        end_idx = min((batch_index + 1) * batch_size, len(all_keys))
        batch = all_keys[start_idx:end_idx]
        
        current_task.upload_artifact(str(batch_index), batch)
        
        task: Task = Task.clone(
            transcription_base_task,
            parent=Task.current_task(),
            project=transcription_tasks_project,
            name=f"transcribe_batch_{batch_index}",
        )
        task.set_parameters(
            {
                "General/batch_index": batch_index, 
                "General/controller_task_id": current_task.id,
                "General/input_task_id": input_task_id,
                "General/output_task_id": output_task_id,
            }
        )
        # task.set_packages("./requirements.txt")
        Task.enqueue(
            task=task,
            queue_name=queue_name,
        )
        print(f"Enqueued {task.name} in {task.project} tasks - {task.id}")
        enqueued_task_ids.append(task.id)

    Logger.current_logger().report_scalar(
        title="Batch Info", series="Total Batches", value=total_batches, iteration=0
    )

    
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
