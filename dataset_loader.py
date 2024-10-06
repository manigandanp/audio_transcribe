from datasets import load_dataset
from clearml import Task
import os


class DatasetLoader:
    def __init__(
        self,
        input_task_id: str,
        hf_dataset_name: str,
        hf_config_name: str = None,
    ):
        self.token = os.getenv("HF_TOKEN")
        self.hf_dataset_name = hf_dataset_name
        self.hf_config_name = hf_config_name
        self.task = Task.get_task(task_id=input_task_id)

    def upload_to_clearml(self):

        dataset = load_dataset(
            self.hf_dataset_name,
            self.hf_config_name,
            split="train",
            token=self.token,
        )
        for item in dataset:
            filename = item["filename"]
            audio_file = item["audio"]
            print(f"Uploading {filename} to ClearML")
            self.task.upload_artifact(filename, audio_file)
        return self.task.id


if __name__ == "__main__":
    task: Task = Task.current_task()
    task_parameters = task.get_parameters_as_dict()["General"]
    input_task_id = task_parameters.get("input_task_id")
    hf_dataset_name = task_parameters.get("hf_dataset_name")
    hf_config_name = task_parameters.get("hf_config_name")
    print(f"Input Task ID: {input_task_id}", hf_dataset_name, hf_config_name)
    loader = DatasetLoader(input_task_id, hf_dataset_name, hf_config_name)
    loader.upload_to_clearml()
    print("waiting for upload to complete")
    task.flush(wait_for_uploads=True)
    print("upload complete...")
    task.close()
