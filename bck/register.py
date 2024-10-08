from clearml import Task, TaskTypes


project_name = "Test/AudioProcessing/template"
task_name = "process_audio"

existing_tasks = Task.get_tasks(
    project_name=project_name,
    task_name=task_name,
    task_filter=dict(
        system_tags=["-archived"],  # Exclude archived tasks
    ),
)

for existing_task in existing_tasks:
    if (
        not existing_task.get_system_tags()
        or "archived" not in existing_task.get_system_tags()
    ):
        # new_name = f"{existing_task.name}_old_{existing_task.id}"
        # existing_task.rename(new_name)
        existing_task.set_archived(True)
        print(
            f"Existing task {existing_task.name} id {existing_task.id} has been archived"
        )


audio_process_task = Task.create(
    project_name=project_name,
    task_name=task_name,
    task_type=TaskTypes.data_processing,
    repo="https://github.com/manigandanp/audio_transcribe.git",
    script="process_audio.py",
    argparse_args=[
        ("hf_dataset_name", "mastermani305/ps-raw"),
        ("audio_index", 0),
        ("audio_name", "1.1"),
        ("hf_config_name", "ps-2-2-sample"),
        ("project_name", "Test/AudioProcessing/test_run"),
    ],
    branch="main",
)


final_task = Task.create(
    project_name=project_name,
    task_name="final_step",
    task_type=TaskTypes.data_processing,
    repo="https://github.com/manigandanp/audio_transcribe.git",
    script="final.py",
    branch="main",
    argparse_args=[
        ("parents", "parents_from_base"),
    ],
)

print(audio_process_task, final_task)
