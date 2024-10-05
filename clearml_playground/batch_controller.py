from clearml import Task, Logger
import random
import config

def batch_controller(
    batch_size, input_artifacts_task_id, output_artifacts_task_id
):
    # input_artifacts_task = Task.get_task(task_id=input_artifacts_task_id)
    # artifacts = [a for a in input_artifacts_task.artifacts if ".wav" in a]
    # dataset_len = int(get_dataset_size(hf_dataset_name, hf_config_name, split="train"))
    
    dataset_len = random.randint(1, 30)
    total_batches = (dataset_len + batch_size - 1) // batch_size
    
    transcription_base_task = Task.get_task(
        project_name=f"{config.base_project_name}/template",
        task_name=config.transcribe_base_task_name,
        allow_archived=False,
    )

    for i in range(total_batches):
        task = Task.clone(
            transcription_base_task,
            parent=Task.current_task(),
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
            queue_name="gpu_worker",
        )

    Logger.current_logger().report_scalar(
        title="Batch Info", series="Total Batches", value=total_batches, iteration=0
    )

if __name__ == "__main__":

    params = Task.current_task().get_parameters()
    
    batch_controller(
        int(params['General/batch_size']),
        params['General/input_artifacts_task_id'],
        params['General/output_artifacts_task_id']
    )

