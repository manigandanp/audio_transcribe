from datasets import load_dataset


def get_dataset(hf_dataset_name, hf_config_name, token):
    print("Loading dataset...")
    print(hf_dataset_name, hf_config_name)
    dataset = (
        load_dataset(hf_dataset_name, hf_config_name, split="train", token=token)
        if hf_config_name
        else load_dataset(hf_dataset_name, split="train", token=token)
    )

    return dataset
