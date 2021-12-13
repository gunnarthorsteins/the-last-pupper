import os
import glob

def get_latest_model():
    """Gets the latest model path for training classifier"""

    list_of_models = glob.glob('models/*')
    list_of_models_modified = []
    for model in list_of_models:
        if 'cae' in model:
            pass
        else:
            list_of_models_modified.append(model)

    latest_model_path = max(list_of_models_modified,
                            key=os.path.getctime)

    return latest_model_path
