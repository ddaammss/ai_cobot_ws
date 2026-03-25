from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'bartender'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    package_data={
        package_name: [
            'shake/*.pt',
            'recipe/*.pt',
            'recipe/*.json',
            'recipe/*.npy',
            'recipe/*.yaml',
        ],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dorong',
    maintainer_email='ehdud2312@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'shake = bartender.shake.shake_node:main',
        'recipe = bartender.recipe.recipe_node:main',
        'tracking = bartender.ob_tracking.tracking_node:main',
        'db = bartender.db.mariadb_node:main',
        'query = bartender.db.query_node:main',
        'recovery = bartender.recovery.recovery_node:main',
        'stt = bartender.stt.stt_node:main',
        'srv_test = bartender.topping.srv_test:main',
        'topping_node = bartender.topping.topping_node:main',
        'supervisor = bartender.supervisor.supervisor_node:main',
        'cup_pick = bartender.recipe.cup_pick_node:main',
        'bottle = bartender.recipe.bottle_test_node:main'
        ],
}   ,
)
