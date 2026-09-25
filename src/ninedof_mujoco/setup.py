from setuptools import find_packages, setup

package_name = 'ninedof_mujoco'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Grover Aruquipa',
    maintainer_email='grover.adelante9@gmail.com',
    description='MuJoCo helpers for the 9-DoF parallel robot.',
    license='Apache-2.0',
)
