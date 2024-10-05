from clearml import Task
import time

class BatchProcessor:
    def __init__(self, input_task_id, output_task_id, batch_index, batch_size):
        self.input_task_id = input_task_id
        self.output_task_id = output_task_id
        self.batch_index = batch_index
        self.batch_size = batch_size

    def process_batch(self):
        print(f"Processing batch {self.batch_index} of size {self.batch_size}")
        print("Processing logic goes here...")
        print(f"Output task ID: {output_task_id}")
        print(f"Input task ID: {input_task_id}")
        print("Processing completed.")
        time.sleep(5)


if __name__ == "__main__":
    task = Task.current_task()
    task_parameters = task.get_parameters_as_dict()["General"]
    input_task_id = task_parameters.get("input_task_id")
    output_task_id = task_parameters.get("output_task_id")
    batch_index = int(task_parameters.get("batch_index"))
    batch_size = int(task_parameters.get("batch_size"))
    batch_processor = BatchProcessor(
        input_task_id, output_task_id, batch_index, batch_size
    )
    batch_processor.process_batch()
