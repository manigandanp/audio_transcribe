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
        ds_with_config = load_dataset(
            self.hf_dataset_name, self.hf_config_name, split="train", token=self.token
        )
        ds_without_config = load_dataset(
            self.hf_dataset_name, split="train", token=self.token
        )
        dataset = ds_with_config if self.hf_config_name else ds_without_config
        for item in dataset:
            filename = item["filename"]
            audio_file = item["audio"]
            print(f"Uploading {filename} to ClearML", audio_file)
            self.task.upload_artifact(filename, audio_file)
        return self.task.id


if __name__ == "__main__":
    task: Task = Task.current_task()
    print(task.id)
    print(task.get_parameters())
    print(task.get_parameters_as_dict())
    input_task_id = task.get_parameter("input_task_id")
    hf_output_dataset_name = task.get_parameter("hf_output_dataset_name")
    hf_config_name = task.get_parameter("hf_config_name")
    print(f"Input Task ID: {input_task_id}", hf_output_dataset_name, hf_config_name)
    loader = DatasetLoader(input_task_id, hf_output_dataset_name, hf_config_name)
    loader.upload_to_clearml()
    
    task.close()
  
    # dataset_loader = DatasetLoader(
    #     hf_dataset_name="mastermani305/ps-raw",
    #     clearml_project="Test/dev/Audio Transcribe",
    #     clearml_task_name="Dataset Task",
    #     hf_config_name="ps-2-2-sample",
    # )
    # task_id = dataset_loader.upload_to_clearml()
    # print(f"Task ID: {task_id}")
