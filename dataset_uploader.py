from clearml import Task
import pandas as pd
from datasets import load_dataset
import os
import shutil
from dataset_utils import delete_clearml_artifcats


class DatasetUploader:
    def __init__(
        self,
        output_artifacts_task_id,
        hf_output_dataset_name,
        hf_config_name,
        is_private_dataset,
    ):
        self.output_artifacts_task:Task = Task.get_task(task_id=output_artifacts_task_id)
        self.hf_output_dataset_name = hf_output_dataset_name
        self.output_dir = "final_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.wavs_dir = os.path.join(self.output_dir, "wavs")
        os.makedirs(self.wavs_dir, exist_ok=True)
        self.token = os.getenv("HF_TOKEN")
        self.hf_config_name = hf_config_name
        self.is_private_dataset = is_private_dataset
        print("Dataset Uploader initialized")

    def upload(self):
        all_metadata = []
        all_artifacts = self.output_artifacts_task.artifacts
        print(f"Uploading artifacts... \n {len(all_artifacts.keys())}")
        print("All artifcat names - ", all_artifacts.keys())
        for artifact_name, artifact in all_artifacts.items():
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
        merged_metadata["file_name"] = merged_metadata["chunk_name"].apply(
            lambda x: os.path.join("wavs", x)
        )

        # Create and upload Hugging Face dataset
        self.create_and_upload_hf_dataset(merged_metadata)
        delete_clearml_artifcats(self.output_artifacts_task, all_artifacts.keys())
        

    def move_wav_files(self, source_folder):
        for root, _, files in os.walk(source_folder):
            for file in files:
                if file.endswith(".wav"):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(self.wavs_dir, file)
                    shutil.move(source_path, dest_path)

    def create_and_upload_hf_dataset(self, metadata):
        # Save the updated metadata CSV
        updated_metadata_path = os.path.join(self.output_dir, "metadata.csv")
        metadata.to_csv(updated_metadata_path, index=False)
        dataset = load_dataset("audiofolder", data_dir=self.output_dir, split="train")

        dataset.push_to_hub(
            self.hf_output_dataset_name,
            self.hf_config_name,
            token=self.token,
            private=self.is_private_dataset,
        )


if __name__ == "__main__":
    task = Task.current_task()
    task_parameters = task.get_parameters_as_dict(cast=True)["General"]
    output_task_id = task_parameters.get("output_task_id")
    hf_output_dataset_name = task_parameters.get("hf_output_dataset_name")
    hf_config_name = task_parameters.get("hf_config_name")
    is_private_dataset = task_parameters.get("is_private_dataset")
  
    uploader = DatasetUploader(
        output_task_id, hf_output_dataset_name, hf_config_name, is_private_dataset
    )
    uploader.upload()
    task.close()