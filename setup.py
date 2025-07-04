from setuptools import find_packages,setup
from typing import List


HYPEN_DOT_E='-e .'
def get_requirement(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        
    if HYPEN_DOT_E in requirements:
        requirements.remove(HYPEN_DOT_E)
        
    return requirements


setup(
    name="ml project",
    version="0.0.1",
    author="rishad",
    author_email="muhammed.risshad@gmail.com",
    packages=find_packages(),
    install_requires=get_requirement('requirement.txt')
    )