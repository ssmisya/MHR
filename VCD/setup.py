
from setuptools import setup, find_packages

setup(
    name='vcd',  # 项目名称
    version='0.1.0',  # 项目版本
    packages=find_packages(),  # 自动发现项目中的包
    include_package_data=True,  # 包含非 .py 文件
    install_requires=[  # 项目依赖
        'torchvision==0.15.2',
        'transformers==4.31.0',
        'torch==2.0.1',
        'tokenizers>=0.12.1,<0.14',
        'sentencepiece==0.1.99',
        'shortuuid',
        'accelerate==0.21.0',
        'peft==0.4.0',
        'bitsandbytes==0.41.0',
        'numpy',
        'scikit-learn==1.2.2',
        'gradio==3.35.2',
        'gradio_client==0.2.9',
        'requests', 
        'httpx==0.24.0',
        'uvicorn',
        'fastapi',
        'einops==0.6.1',
        'einops-exts==0.0.4',
        'timm==0.6.13',
    ],
    # entry_points={  # 可选，定义命令行工具
    #     'console_scripts': [
    #         'your-command=your_package.your_module:main_function',
    #     ],
    # },
    # 其他元数据...
    author='Your Name',
    author_email='your.email@example.com',
    description='A short description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your-project-name',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        # 其他分类器...
    ],
)