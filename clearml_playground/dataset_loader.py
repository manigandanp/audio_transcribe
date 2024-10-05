import time
from clearml import Task


def download_dataset(hf_dataset_name, hf_config_name, input_task_id):
    print(f"Downloaded {hf_dataset_name} dataset")
    print(f"Dataset config: {hf_config_name}")
    print(f"Input task: {input_task_id}")
    time.sleep(5)
    print("Download dataset function called")


if __name__ == "__main__":
    task = Task.current_task()
    task_parameters = task.get_parameters_as_dict()["General"]
    print(f"download_dataset current task parameters: {task_parameters}")
    input_task_id = task_parameters.get("input_task_id")
    hf_dataset_name = task_parameters.get("hf_dataset_name")
    hf_config_name = task_parameters.get("hf_config_name")
    # print(f"Input Task ID: {input_task_id}", hf_dataset_name, hf_config_name)
    download_dataset(
        input_task_id=input_task_id,
        hf_dataset_name=hf_dataset_name,
        hf_config_name=hf_config_name,
    )
