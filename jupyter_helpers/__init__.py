from .github_paperspace import create_github_sshkey_on_machine
from .gpu_memory import clean_memory
from .gpu_training import ToDeviceCallback, device, safely_train_with_gpu
from .kaggle import create_submission, download_data
