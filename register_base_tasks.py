from clearml import Task, TaskTypes


def register_base_task(project_name, task_name, script_path, task_type):
    task = Task.create(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type,
        repo="https://github.com/manigandanp/audio_transcribe.git",
        branch="main",
    )
    task.set_script(script_path)
    task.close()
    print(f"Registered base task: {task_name}")


if __name__ == "__main__":
    import config

    base_project_name = f"{config.base_project_name}/template"

    register_base_task(
        project_name=base_project_name,
        task_name=config.download_dataset_base_task_name,
        script_path="dataset_loader.py",
        task_type=TaskTypes.data_processing,
    )
    register_base_task(
        project_name=base_project_name,
        task_name=config.transcribe_base_task_name,
        script_path="transcription_pipeline.py",
        task_type=TaskTypes.data_processing,
    )
    register_base_task(
        project_name=base_project_name,
        task_name=config.upload_dataset_base_task_name,
        script_path="dataset_upload.py",
        task_type=TaskTypes.data_processing,
    )
