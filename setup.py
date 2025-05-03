from setuptools import find_packages,setup

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path):
    with open(file_path) as file:
        modules = file.readlines()
        packages = [i.replace('\n','') for i in modules]
        if HYPHEN_E_DOT in packages:
            packages.remove(HYPHEN_E_DOT)
    return packages

setup(
    name='mlproject',
    version='0.0.1',
    author='Azhar',
    author_email='azhar.sid@icloud.com',
    packages= find_packages(),
    # install_requires = ['pandas','numpy','seaborn']
    install_requires = get_requirements('requirements.txt')
)