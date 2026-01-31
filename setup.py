from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="audio-enhancer-gpu",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="GPU-accelerated audio enhancement tool for amplifying quiet recordings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audio-enhancer-gpu",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Conversion",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gui": ["tkinter"],
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "audio-enhance=cli:main",
            "audio-enhance-gui=gui:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
)
