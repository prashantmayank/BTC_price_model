from setuptools import find_packages, setup
from typing import List


hyphen_check = '-e .'


def fetch_requirements(file_path: str) -> List[str]:
    '''
    This function will return list of requirement
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if hyphen_check in requirements:
            requirements.remove(hyphen_check)
    return requirements


setup(
    name='BTC_PRICE_MODEL',
    version='0.0.1',
    author='prashantmayank',
    author_email='mayankprashant67@gmail.com',
    packages=find_packages(),
    install_requires=fetch_requirements('requirements.txt')

)
