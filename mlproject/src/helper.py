
import os
import pickle
import json
from datetime import datetime


def create_directory(path):
    os.makedirs(path, exist_ok=True)


def save_object(file_path, obj):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_object(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_json(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def get_current_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
