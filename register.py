from clearml import Task, TaskTypes, TaskQueryParameters


project_name = "Test/AudioProcessing/template"
task_name = "process_audio"

existing_tasks = Task.get_tasks(
    project_name=project_name,
    task_name=task_name,
    task_filter=dict(
        system_tags=['-archived'],  # Exclude archived tasks
    )
)

for existing_task in existing_tasks:
    if not existing_task.get_archived():
        print(f"Archiving existing task: {existing_task.id}")
        existing_task.archive()


audio_process_task = Task.create(
    project_name=project_name,
    task_name=task_name,
    task_type=TaskTypes.data_processing,
    repo="https://github.com/manigandanp/audio_transcribe.git",
    script="process_audio.py",
    argparse_args=[("--hf_dataset_name", "mastermani305/ps-raw")],
    branch="main",
)

print(audio_process_task)
