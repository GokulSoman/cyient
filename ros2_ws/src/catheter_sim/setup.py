from setuptools import setup

package_name = 'catheter_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gokul Soman',
    maintainer_email='gokul@example.com',
    description='Robotic arm simulation for femoral artery catheterization guidance',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vessel_detector = catheter_sim.vessel_detector:main',
            'image_publisher  = catheter_sim.image_publisher:main',
            'target_planner   = catheter_sim.target_planner:main',
            'arm_controller   = catheter_sim.arm_controller:main',
        ],
    },
)
