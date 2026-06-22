from setuptools import find_packages,setup
from typing import List

hy='-e .'

def get_requriments(file_path:str)->List[str]:
    
  with open(file_path)as file_obj:
   requirements=file_obj.readlines()
   
   requirements=[req.replace('\n','') for req in requirements]

   
   if hy in requirements:
     requirements.remove(hy)
   return requirements

setup(

     name='Diseases prediction project',
     version='0.0.1',
     author='Babar Ali khan',
     author_email='babaralikhanaiexpert098@gmail',
     packages=find_packages(),
     install_requires=get_requriments('requirements.txt')
)
