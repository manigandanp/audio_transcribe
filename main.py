# main_task.py

import os
import re
import shutil
import pandas as pd
from clearml import Task
from datasets import Dataset, load_dataset
from dataset_loader import get_dataset


def main(hf_dataset_name, project_name="Test", hf_config_name=None):
    # Initialize ClearML Task
    task = Task.init(project_name=project_name, task_name="Main Task")
    token = os.environ.get("HF_TOKEN")

    dataset = get_dataset(hf_dataset_name, hf_config_name, token)

    # List to keep track of processing tasks
    processing_tasks = []

    # Create and enqueue processing tasks for each audio file
    for idx, data in enumerate(dataset):
        audio_name = (
            data.get("chp_id", None) or re.search(r"\d+\.\d+", data["audio"]["path"]).group()
        )
        audio_index = idx

        # Create a processing task for each audio file
        processing_task = Task.create(
            project_name=project_name,
            task_name=f"Process Audio - {audio_name}-{audio_index}",
            script="process_audio.py",
            # args=[
            #     "--hf_dataset_name",
            #     hf_dataset_name,
            #     "--hf_config_name",
            #     str(hf_config_name),
            #     "--audio_index",
            #     str(audio_index),
            #     "--audio_name",
            #     str(audio_name),
            # ],
        )
        
        processing_task.set_script(
            # script='process_audio.py',
            working_dir=os.getcwd(),
            arguments=[
                "--hf_dataset_name",
                hf_dataset_name,
                "--hf_config_name",
                str(hf_config_name),
                "--audio_index",
                str(audio_index),
                "--audio_name",
                str(audio_name),
            ],
        )

        # Enqueue the task
        processing_task.enqueue("process_audio")

        processing_tasks.append(processing_task)

    # Wait for all processing tasks to complete
    for p_task in processing_tasks:
        p_task.wait_for_status(status=["completed", "failed"], timeout=None)

    # Collect artifacts from all processing tasks
    metadata_list = []
    wavs_dir = "combined_wavs"
    os.makedirs(wavs_dir, exist_ok=True)

    for p_task in processing_tasks:
        if p_task.get_status() != "completed":
            print(f"Task {p_task.name} did not complete successfully.")
            continue

        artifacts = p_task.artifacts

        # Download metadata.csv
        metadata_csv_path = artifacts["metadata.csv"].get_local_copy()
        metadata_df = pd.read_csv(metadata_csv_path)
        metadata_list.append(metadata_df)

        # Download wavs folder
        wavs_artifact = artifacts["wavs"]
        wavs_path = wavs_artifact.get_local_copy()
        for wav_file in os.listdir(wavs_path):
            src = os.path.join(wavs_path, wav_file)
            dst = os.path.join(wavs_dir, wav_file)
            shutil.move(src, dst)

    # Combine all metadata into a single DataFrame
    combined_metadata_df = pd.concat(metadata_list, ignore_index=True)
    combined_metadata_csv_path = "combined_metadata.csv"
    combined_metadata_df.to_csv(combined_metadata_csv_path, index=False)

    # Upload the combined dataset to Hugging Face
    # Ensure you have the necessary permissions and API tokens set up
    combined_dataset = load_dataset("audiofolder", data_dir="./")
    combined_dataset.push_to_hub(
        "mastermani305/ps-transcribed",
        config_name=hf_config_name,
        private=True,
        token=token,
    )  # Replace with your target dataset name

    # Finalize the main task
    task.close()


if __name__ == "__main__":
    project_name = "Test/Audio Processing"
    hf_dataset_name = "mastermani305/ps-raw"
    hf_config_name = "ps-2-2-sample"
    main(
        project_name=project_name,
        hf_dataset_name=hf_dataset_name,
        hf_config_name=hf_config_name,
    )
