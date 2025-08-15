"""
Setup script for HWiNFO Analyzer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hwinfo-analyzer",
    version="2.0.0",
    author="HWiNFO Analyzer Team",
    description="Analizador científicamente preciso para logs CSV de HWiNFO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=[
        "improved_analyzer",
        "thermal_thresholds", 
        "data_processor",
        "anomaly_detector",
        "thermal_analyzer"
    ],
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "hwinfo-analyzer=improved_analyzer:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: System Administrators",
        "Intended Audience :: End Users/Desktop",
        "Topic :: System :: Hardware",
        "Topic :: System :: Monitoring",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="hwinfo, hardware, monitoring, temperature, analysis, diagnostics",
    project_urls={
        "Bug Reports": "https://github.com/your-username/hwinfo-analyzer/issues",
        "Source": "https://github.com/your-username/hwinfo-analyzer",
    },
)