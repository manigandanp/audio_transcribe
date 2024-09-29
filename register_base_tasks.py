from clearml import Task, TaskTypes


def register_base_task(project_name, task_name, script_path, task_type, params_dict={}):

    existing_tasks = Task.get_tasks(
        project_name=project_name,
        task_name=task_name,
        task_filter=dict(
            system_tags=["-archived"],
        ),
    )
    # Archiving exising task
    for existing_task in existing_tasks:
        if (
            not existing_task.get_system_tags()
            or "archived" not in existing_task.get_system_tags()
        ):
            existing_task.set_archived(True)
            print(
                f"Existing task {existing_task.name} id {existing_task.id} has been archived"
            )
    # Registering
    task = Task.create(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type,
        repo="https://github.com/manigandanp/audio_transcribe.git",
        branch="main",
        script=script_path,
    )
    task.set_parameters(params_dict)
    task.close()
    print(f"Registered base task: {task_name}")


if __name__ == "__main__":
    import config

    base_project_template = f"{config.base_project_name}/template"

    download_dataset_params = {
        "hf_dataset_name": "mastermani305/ps-raw",
        "hf_config_name": "ps-2-2-sample",
        "input_task_id": "",
    }

    transcription_task_params = {
        "batch_index": 0,
        "batch_size": 10,
        "input_task_id": "",
        "output_task_id": "",
    }

    upload_task_params = {
        "output_task_id": "",
        "hf_output_dataset_name": "ps-transcribed",
        "hf_config_name": "ps-2-2-sample",
        "is_private_dataset": True,
    }

    register_base_task(
        project_name=base_project_template,
        task_name=config.download_dataset_base_task_name,
        script_path="dataset_loader.py",
        task_type=TaskTypes.data_processing,
        params_dict=download_dataset_params,
    )
    register_base_task(
        project_name=base_project_template,
        task_name=config.transcribe_base_task_name,
        script_path="transcription_pipeline.py",
        task_type=TaskTypes.data_processing,
        params_dict=transcription_task_params,
    )
    register_base_task(
        project_name=base_project_template,
        task_name=config.upload_dataset_base_task_name,
        script_path="dataset_uploader.py",
        task_type=TaskTypes.data_processing,
        params_dict=upload_task_params,
    )

    register_base_task(
        project_name=base_project_template,
        task_name=config.input_artifacts_task_name,
        script_path="",
        task_type=TaskTypes.custom,
    )

    register_base_task(
        project_name=base_project_template,
        task_name=config.output_artifacts_task_name,
        script_path="",
        task_type=TaskTypes.custom,
    )
