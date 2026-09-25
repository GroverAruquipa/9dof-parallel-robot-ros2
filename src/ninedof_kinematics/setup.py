from setuptools import find_packages, setup

package_name = 'ninedof_kinematics'

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
    description='Kinematics of the 9-DoF 5PSS-S-4PSS parallel robot.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'pose_to_joint_states = ninedof_kinematics.pose_to_joint_states:main',
            'fk_joint_state_publisher = ninedof_kinematics.fk_joint_state_publisher:main',
            'demo_pose_publisher = ninedof_kinematics.demo_pose_publisher:main',
        ],
    },
)
