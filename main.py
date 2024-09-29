from clearml import PipelineController, Task
import config


def main():
    base_project_name = config.base_project_name  # "Test/Audio Transcription"
    dataset_download_queue_name = "cpu_worker"

    input_artifacts_task = Task.init(
        project_name=base_project_name, task_name=config.input_artifacts_task_name
    )
    output_artifacts_task = Task.init(
        project_name=base_project_name, task_name=config.output_artifacts_task_name
    )

    dataset_download_task: Task = Task.init(
        project_name=f"{base_project_name}/template",
        task_name=config.download_dataset_base_task_name,
    )
    dataset_download_task.set_parameters(
        {
            "hf_dataset_name": "mastermani305/ps-raw",
            "hf_config_name": "ps-2-2-sample",
            "input_task_id": input_artifacts_task.id,
        }
    )
    Task.enqueue(dataset_download_task, queue_name=dataset_download_queue_name)
    dataset_download_task.wait_for_status(Task.TaskStatusEnum.completed)
    print("completed downloading from huggingface and uploading to clearml")
    print(input_artifacts_task.artifacts)

    pipe = PipelineController(
        name="audio_transcription_pipeline",
        project=base_project_name,
        version="1.0",
        add_pipeline_tags=False,
        repo="https://github.com/manigandanp/audio_transcribe.git",
        repo_branch="main",
    )

    # pipe.add_parameter(
    #     name="hf_dataset_name", default="mastermani305/ps-raw", param_type="string"
    # )
    pipe.add_parameter(
        name="hf_config_name", default="ps-2-2-sample", param_type="string"
    )
    pipe.add_parameter(name="batch_size", default="10", param_type="int")
    pipe.add_parameter(
        name="hf_output_dataset_name",
        default="mastermani305/ps-transcribed",
        param_type="string",
    )
    pipe.add_parameter(name="is_private_dataset", default="True", param_type="bool")
    # pipe.add_parameter(name="base_project_name", default=base_project_name, param_type="string")

    # hf_dataset_name = pipe.get_parameters().get("hf_dataset_name")
    hf_config_name = pipe.get_parameters().get("hf_config_name")
    batch_size = pipe.get_parameters().get("batch_size")
    hf_output_dataset_name = pipe.get_parameters().get("hf_output_dataset_name")
    is_private_dataset = pipe.get_parameters().get("is_private_dataset")
    # base_project_name = pipe.get_parameters().get("base_project_name")

    # Dataset Loader Step
    # pipe.add_step(
    #     name="download_dataset",
    #     base_task_project=f"{base_project_name}/template",
    #     base_task_name="dataset_loader_base",
    #     parameter_override={
    #         "hf_dataset_name": hf_dataset_name,
    #         "hf_config_name": hf_config_name,
    #         "input_task_id": input_artifacts_task.id
    #     }
    # )

    # Dynamically add batch processing steps
    batch_step_names = []
    artifacts = [a for a in input_artifacts_task.artifacts if ".wav" in a]
    total_batches = (len(artifacts) + batch_size - 1) // batch_size
    for i in range(total_batches):
        batch_step_name = f"transcribe_batch_{i}"
        pipe.add_step(
            name=batch_step_name,
            base_task_project=f"{base_project_name}/template",
            base_task_name=config.transcribe_base_task_name,
            # parents=["download_dataset"],
            parameter_override={
                "batch_index": i,
                "batch_size": batch_size,
                "input_task_id": input_artifacts_task.id,
                "output_task_id": output_artifacts_task.id,
            },
        )
        batch_step_names.append(batch_step_name)

    # Final processing step
    pipe.add_step(
        name="upload_dataset",
        base_task_project=f"{base_project_name}/template",
        base_task_name=config.upload_dataset_base_task_name,
        parents=batch_step_names,
        parameter_override={
            "hf_output_dataset_name": hf_output_dataset_name,
            "hf_config_name": hf_config_name,
            "is_private_dataset": is_private_dataset,
        },
    )

    pipe.start(queue=None)


if __name__ == "__main__":
    main()
