from datasets import load_dataset_builder

def get_dataset_size(dataset_name, config_name=None, split='train'):
    dataset_builder = load_dataset_builder(dataset_name, config_name)
    split_info = dataset_builder.info.splits.get(split)

    if split_info is None:
        raise ValueError(f"Split '{split}' not found in dataset '{dataset_name}'")
    return split_info.num_examples


if __name__ == '__main__':
  try:
      dataset_name = "mastermani305/ps-raw"
      config_name = "ps-1"  
      split = "train"
      
      num_rows = get_dataset_size(dataset_name, config_name, split)
      print(f"Number of rows in {dataset_name} ({config_name}) {split} split: {num_rows}")
  except Exception as e:
      print(f"An error occurred: {e}")