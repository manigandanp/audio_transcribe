from clearml import Task
import pandas as pd
from datasets import Dataset, Audio, load_dataset
import os
import shutil

class DatasetUploader:
    def __init__(self, input_task_id, hf_output_dataset_name, hf_config_name, is_private_dataset):
        self.task = Task.current_task()
        self.input_task = Task.get_task(task_id=input_task_id)
        self.hf_output_dataset_name = hf_output_dataset_name
        self.output_dir = "final_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.wavs_dir = os.path.join(self.output_dir, "wavs")
        os.makedirs(self.wavs_dir, exist_ok=True)
        self.token = os.getenv("HF_TOKEN")
        self.hf_config_name = hf_config_name
        self.is_private_dataset = is_private_dataset

    def upload(self):
        all_metadata = []
        
        for artifact_name, artifact in self.input_task.artifacts.items():
            if artifact_name.endswith("_metadata"):
                df = pd.read_csv(artifact.get_local_copy())
                all_metadata.append(df)
            elif artifact_name.endswith("-wavs"):
                wav_folder = artifact.get_local_copy()
                self.move_wav_files(wav_folder)

        # Merge all metadata CSVs
        merged_metadata = pd.concat(all_metadata, ignore_index=True, sort=False)
        merged_metadata_path = os.path.join(self.output_dir, "merged_metadata.csv")
        merged_metadata.to_csv(merged_metadata_path, index=False)

        # Update paths in merged metadata
        merged_metadata['audio'] = merged_metadata['chunk_name'].apply(lambda x: os.path.join('wavs', x))
        
        # Create and upload Hugging Face dataset
        self.create_and_upload_hf_dataset(merged_metadata)

        # # Upload artifacts to ClearML
        # self.task.upload_artifact("merged_metadata", merged_metadata_path)
        # self.task.upload_artifact("audio_chunks", self.wavs_dir)

    def move_wav_files(self, source_folder):
        for root, _, files in os.walk(source_folder):
            for file in files:
                if file.endswith('.wav'):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(self.wavs_dir, file)
                    shutil.move(source_path, dest_path)

    def create_and_upload_hf_dataset(self, metadata):
        # Save the updated metadata CSV
        updated_metadata_path = os.path.join(self.output_dir, "dataset_metadata.csv")
        metadata.to_csv(updated_metadata_path, index=False)
        dataset = load_dataset("audiofolder", data_dir=self.output_dir, split="train")

        dataset.push_to_hub(self.hf_output_dataset_name, self.hf_config_name, token=self.token, private=self.is_private_dataset)

if __name__ == "__main__":
    task = Task.current_task()
    input_task_id = task.get_parameter("input_task_id")
    hf_output_dataset_name = task.get_parameter("hf_output_dataset_name")
    hf_config_name = task.get_parameter("hf_config_name")
    is_private_dataset = task.get_parameter("is_private_dataset")
    
    uploader = DatasetUploader(input_task_id, hf_output_dataset_name, hf_config_name, is_private_dataset)
    uploader.upload()
    
    task.close()