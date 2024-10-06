from clearml import Task

def wait_for_batches(controller_task_id):
    controller_task = Task.get_task(task_id=controller_task_id)
    enqueued_task_ids = controller_task.artifacts['enqueued_task_ids'].get()
    
    for task_id in enqueued_task_ids:
        Task.get_task(task_id=task_id).wait_for_status()

if __name__ == "__main__":    
    params = Task.current_task().get_parameters()
    wait_for_batches(params['General/controller_task_id'])