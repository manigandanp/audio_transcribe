from clearml import Task
import time 

class DatasetUploader:
    def __init__(self, output_task_id, hf_output_dataset_name, hf_config_name, is_private_dataset):
        self.output_task_id = output_task_id
        self.hf_output_dataset_name = hf_output_dataset_name
        self.hf_config_name = hf_config_name
        self.is_private_dataset = is_private_dataset

    def upload(self):
        print(f"Uploading dataset {self.hf_output_dataset_name} to HF Hub")
        print(f"Output task ID: {self.output_task_id}")
        print(f"HF config name: {self.hf_config_name}")
        print(f"Is private dataset: {self.is_private_dataset}")
        time.sleep(5)

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