# run_preprocess.py
from preprocessing import load_and_preprocess_adni  # replace 'your_module' with the actual file name without .py

# Path to your ADNIMERGE CSV
csv_path = "data/ADNIMERGE.csv"

# Call the function
(basic_train, basic_test, yb_train, yb_test), (adv_train, adv_test, ya_train, ya_test) = \
    load_and_preprocess_adni(csv_path, save_dir="models")
