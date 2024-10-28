import yaml
import json
import pandas as pd


def read_text(file_path):
    with open(file_path, "r") as f:
        return f.read()
    

def read_config(config_path ="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        return config
    
def save_to_json(data, file_path):
    with open(file_path, "w", encoding="utf-8")as fp:
        json.dump(data, fp)


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8")as fp:
        return json.load(fp)
    
def singleton(cls):
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

def split_train_test_by_title(df, title_col="Category"):
    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    
    title_counts = df[title_col].value_counts()
    
    for title, count in title_counts.items():
        title_group = df[df[title_col] == title]
        
        if count > 1:
            # Sample at least 3 or min-1 data point for the test set
            test_sample = title_group.sample(min(3,count-1), random_state = 43)
            # Remaining data points for the train set
            train_sample = title_group.drop(test_sample.index)
            
            test_df = pd.concat([test_df, test_sample], ignore_index=True)
            train_df = pd.concat([train_df, train_sample], ignore_index=True)
        else:
            # If only one data point, include it in the train set
            train_df = pd.concat([train_df, title_group], ignore_index=True)
    
    return train_df, test_df


def encode_data(df, column_to_encode= "Role"):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    roles = le.fit_transform(df[column_to_encode])
    role_to_id = {role: int(role_id) for role, role_id in zip(le.classes_, le.transform(le.classes_))}
    return role_to_id

TEMPLATE = read_text("data/template.txt")
def fill_template(input_dict, template = TEMPLATE):
    """
    Fills the given template with values from the input_dict.
    
    Parameters:
    template (str): The template to fill in.
    input_dict (dict): A dictionary containing lists of values or None for each placeholder in the template.
    
    Returns:
    str: The template with placeholders filled in.
    """
    if not isinstance(input_dict, dict):
        input_dict = json.loads(input_dict)
    for key, value in input_dict.items():
        if not value or len(value) == 0:
            replacement = f"[{key}]"
        if isinstance(value, str):
            replacement = value
        else:  # if value is a list and not empty
            replacement = ', '.join(value)
        
        # Replace the placeholder in the template
        template = template.replace(f"[{key}]", replacement)
    
    return template


    
def concat_output(json_output):
    template = ""
    for k,v in json_output.items():
        if isinstance(v, str):
            template += v
        elif len(v):
            template += ", ".join(v)
            template += ", "

    return template


if __name__ == "__main__":
    print(read_text("data/template.txt"))