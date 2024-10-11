from clearml import PipelineController, Task
import config

def main():
    base_project_name = config.base_project_name  # "Test/Audio Transcription"
    cpu_queue = "cpu_worker"
    gpu_queue = "gpu_worker"
    project_template = config.task_templates_project_name
    input_artifacts_base_task = Task.get_task(
        project_name=project_template,
        task_name=config.input_artifacts_task_name,
        allow_archived=False,
    )
    
    input_artifacts_task = Task.clone(
        input_artifacts_base_task,
        project=config.transcription_tasks_project_name,
        parent=Task.current_task(),
        name=config.input_artifacts_task_name,
    )
    
    output_artifacts_base_task = Task.get_task(
        project_name=project_template,
        task_name=config.output_artifacts_task_name,
        allow_archived=False,
    )
    
    output_artifacts_task = Task.clone(
        output_artifacts_base_task,
        project=config.transcription_tasks_project_name,
        parent=Task.current_task(),
        name=config.output_artifacts_task_name,
    )

    pipe = PipelineController(
        name="audio_transcription_pipeline",
        project=base_project_name,
        version="1.0",
        add_pipeline_tags=False,
        repo="https://github.com/manigandanp/audio_transcribe.git",
        repo_branch="main",
        packages=["clearml"],
    )

    pipe.add_parameter(
        name="hf_dataset_name", default="mastermani305/ps-raw", param_type="string"
    )
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

    # Dataset Loader Step
    pipe.add_step(
        name="download_dataset",
        base_task_project=project_template,
        base_task_name=config.download_dataset_base_task_name,
        execution_queue=cpu_queue,
        parameter_override={
            "General/hf_dataset_name": "${pipeline.hf_dataset_name}",
            "General/hf_config_name": "${pipeline.hf_config_name}",
            "General/input_task_id": input_artifacts_task.id,
        },
        task_overrides={
            "script.requirements.pip": ["clearml", "datasets", "soundfile", "librosa"]
        },
    )

    pipe.add_step(
        name="transcription_batch_controller",
        base_task_project=project_template,
        base_task_name=config.batch_controller_base_task_name,
        parents=["download_dataset"],
        execution_queue=cpu_queue,
        parameter_override={
            "General/batch_size": "${pipeline.batch_size}",
            "General/input_task_id": input_artifacts_task.id,
            "General/output_task_id": output_artifacts_task.id,
            "General/hf_dataset_name": "${pipeline.hf_dataset_name}",
            "General/hf_config_name": "${pipeline.hf_config_name}",
            "General/queue_name": gpu_queue,
        },
        task_overrides={"script.requirements": {"pip": ["clearml", "datasets"]}},
    )

    pipe.add_step(
        name="wait_for_batches",
        base_task_project=project_template,
        base_task_name=config.wait_for_batches_base_task_name,
        parents=["transcription_batch_controller"],
        execution_queue=cpu_queue,
        parameter_override={
            "General/controller_task_id": "${transcription_batch_controller.id}",
        },
        task_overrides={"script.requirements": {"pip": ["clearml"]}},
    )

    pipe.add_step(
        name="upload_dataset",
        base_task_project=project_template,
        base_task_name=config.upload_dataset_base_task_name,
        parents=["wait_for_batches"],
        execution_queue=cpu_queue,
        parameter_override={
            "General/output_task_id": output_artifacts_task.id,
            "General/hf_output_dataset_name": "${pipeline.hf_output_dataset_name}",
            "General/hf_config_name": "${pipeline.hf_config_name}",
            "General/is_private_dataset": "${pipeline.is_private_dataset}",
        },
        task_overrides={"script.requirements.pip": ["clearml", "datasets", "soundfile", "librosa"]},
    )

    pipe.start(queue=None)


if __name__ == "__main__":
    main()
