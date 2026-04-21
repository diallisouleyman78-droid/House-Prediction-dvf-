from setuptools import setup, find_packages
from typing import List

def get_requirements()->List[str]:
    """
    This function will return list of requirements
    """
    requirements_list:List[str] = []
    
    with open("requirements.txt", "r") as req_file:
        for line in req_file:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements_list.append(line)
    
    return requirements_list

setup(
    name="house_prediction",
    version="0.0.1",
    author="Diallo Souleyman",
    author_email="diallisouleyman78@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
