from setuptools import setup, find_packages


setup(
    name='mhr',  # 包的名字
    version='0.1.0',  # 包的版本
    description='mhr is a alignment package for LVLM',  # 包的简短描述
    author='anonymous',  # 你的名字
    author_email='somone@somemail.com',  # 你的邮箱
    packages=find_packages(),  # 自动找到包含在你的包中的所有子包
    install_requires=[  # 定义你的包需要的依赖项
        'numpy',
        'matplotlib',
    ],
    classifiers=[  # 定义你的包的分类信息
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)