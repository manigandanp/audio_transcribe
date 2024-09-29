from clearml import Task, PipelineController

# all_task = Task.get_tasks(project_name="Test/AudioProcessing")

all_task = PipelineController.get(pipeline_name="audio_transcribe_pipeline", pipeline_project="Test/AudioProcessing")

task = Task.get_task(task_id=all_task.id)

# print(all_task.artifacts)

print(all_task._task.id, all_task._task.name)

print("task", task)

# task = Task.get_task(task_id='8e6716b19b0547de8fd2f917befd0a66')
# Task.artifacts()

# print(task.id, task.name, task.artifacts)