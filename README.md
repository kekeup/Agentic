 Agentic VSC System

 [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
 [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

  A video semantic communication system based on multi-agent and continual learning, supporting edge-cloud collaborative processing, EWC continual learning, and adaptive model switching.

1.Environment Configuration
#Dataset (Baidu Netdisk)
 Link:https://pan.baidu.com/s/1c0vdr5uqBaCBWNsgY068iA?pwd=tdgw 
 Extraction code:tdgw 
Note: Extract the dataset to the data/ directory

#Create Virtual Environment
 conda create -n agentic_vsc python=3.8
 conda activate agentic_vsc

#Install Dependencies
 pip install -r requirements.txt


2.API
In the file config/system_config.yaml
   modify the OpenAI API key to your_apikey
   modify the GLM API key to your_apikey

3.Launch Project
python start.py
