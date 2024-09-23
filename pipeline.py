import os
from clearml import PipelineController
from dataset_loader import get_dataset

project_name = "Test/AudioProcessing"


pipe = PipelineController(
    project=project_name,
    name="audio_transcribe_pipeline",
    version="0.1",
    repo="https://github.com/manigandanp/audio_transcribe.git",
    repo_branch="main",
)

pipe.set_default_execution_queue("default")

pipe.add_parameter(
    name="hf_dataset_name", default="mastermani305/ps-raw", param_type="string"
)

pipe.add_parameter(name="hf_config_name", default="ps-2-2-sample", param_type="string")


token = os.environ.get("HF_TOKEN")
# Load the specific audio file from Hugging Face dataset
hf_dataset_name = pipe.get_parameters().get("hf_dataset_name")
hf_config_name = pipe.get_parameters().get("hf_config_name")

dataset = get_dataset(hf_dataset_name, hf_config_name, token)

for audio_index in range(len(dataset)):
    pipe.add_step(
        name=f"process_audio-{audio_index}",
        base_task_name="process_audio",
        base_task_project=f"{project_name}/template",
        execution_queue="audio_process",
        parameter_override={
            "Args/project_name": project_name,
            "Args/hf_dataset_name": hf_dataset_name,
            "Args/hf_config_name": hf_config_name,
            "Args/audio_index": str(audio_index),
            "Args/audio_name": "random_audio_name",
        },
        clone_base_task=True,
    )

pipe.add_step(
    name="final_step",
    base_task_name="final_step",
    base_task_project=f"{project_name}/template",
    execution_queue="audio_process",
    parameter_override={
        "Args/parents": project_name,
    },
)


pipe.start()
