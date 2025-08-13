from setuptools import setup, find_packages
if __name__ == '__main__':
    setup(
        name="ppo_mask_recurrent",
        version="0.0.1",
        py_modules=["ppo_mask_recurrent", "policies"],  # for single file .py
        packages=find_packages(include=["common*"]),   # for folder package
    )