from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'gait_intent'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
        ('share/' + package_name + '/scripts',
            glob('scripts/*.py')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib',
    ],
    zip_safe=True,
    maintainer='Gokul Soman',
    maintainer_email='gokul@example.com',
    description='Gait intent recognition for exoskeleton assistive control',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gait_ros2_node = gait_intent.gait_ros2_node:main',
        ],
    },
)
