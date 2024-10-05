from clearml import PipelineController, Task, Logger
import config
import random



def main():
    base_project_name = config.base_project_name  
    cpu_queue = "cpu_worker"
    gpu_queue = "gpu_worker"
    project_template = f"{base_project_name}/template"
    input_artifacts_task = Task.get_task(
        project_name=project_template,
        task_name=config.input_artifacts_task_name,
        allow_archived=False,
    )
    output_artifacts_task = Task.get_task(
        project_name=project_template,
        task_name=config.output_artifacts_task_name,
        allow_archived=False,
    )

    pipe = PipelineController(
        name="audio_transcription_pipeline_dummy",
        project=base_project_name,
        version="1.0",
        add_pipeline_tags=False,
        repo="https://github.com/manigandanp/audio_transcribe.git",
        repo_branch="main",
        packages=["clearml"],
    )

    pipe.add_parameter(
        name="hf_dataset_name",
        default="default-mastermani305/ps-raw",
        param_type="string",
    )
    pipe.add_parameter(
        name="hf_config_name", default="default-ps-2-2-sample", param_type="string"
    )
    pipe.add_parameter(name="batch_size", default="10", param_type="int")
    pipe.add_parameter(
        name="hf_output_dataset_name",
        default="default-mastermani305/ps-transcribed",
        param_type="string",
    )
    pipe.add_parameter(name="is_private_dataset", default="True", param_type="bool")
    # pipe.add_parameter(name="base_project_name", default=base_project_name, param_type="string")

    hf_dataset_name = pipe.get_parameters().get("hf_dataset_name")
    hf_config_name = pipe.get_parameters().get("hf_config_name")
    batch_size = int(pipe.get_parameters().get("batch_size"))
    hf_output_dataset_name = pipe.get_parameters().get("hf_output_dataset_name")
    is_private_dataset = pipe.get_parameters().get("is_private_dataset")
    # base_project_name = pipe.get_parameters().get("base_project_name")
    print("pipe.get_paramenters")
    print(pipe.get_parameters())

    # print(pipeline)
    # print(f"${pipeline}")
    # Dataset Loader Step
    pipe.add_step(
        name="download_dataset",
        base_task_project=f"{base_project_name}/template",
        base_task_name=config.download_dataset_base_task_name,
        execution_queue=cpu_queue,
        parameter_override={
            "General/hf_dataset_name": "${pipeline.hf_dataset_name}",
            "General/hf_config_name": "${pipeline.hf_config_name}",
            "General/input_task_id": input_artifacts_task.id,
        },
        task_overrides={"script.requirements.pip": ["clearml"]},
    )

    pipe.add_step(
        name="batch_controller",
        base_task_project=f"{base_project_name}/template",
        base_task_name=config.batch_controller_base_task_name,  
        parents=["download_dataset"],
        execution_queue=cpu_queue,
        parameter_override={
            "General/batch_size": "${pipeline.batch_size}",
            "General/input_artifacts_task_id": input_artifacts_task.id,
            "General/output_artifacts_task_id": output_artifacts_task.id,
        },
        task_overrides={'script.requirements': {'pip': ['clearml']}}
    )
    
    # Final processing step
    pipe.add_step(
        name="upload_dataset",
        base_task_project=f"{base_project_name}/template",
        base_task_name=config.upload_dataset_base_task_name,
        parents=["batch_controller"],
        execution_queue=cpu_queue,
        parameter_override={
            "General/output_task_id": output_artifacts_task.id,
            "General/hf_output_dataset_name": "${pipeline.hf_output_dataset_name}",
            "General/hf_config_name": "${pipeline.hf_config_name}",
            "General/is_private_dataset": "${pipeline.is_private_dataset}",
        },
        task_overrides={"script.requirements.pip": "clearml"},
    )

    pipe.start(queue=None)


if __name__ == "__main__":
    main()
