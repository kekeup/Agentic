 Agentic VSC System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

一个基于多智能体和持续学习的视频语义通信系统，支持边缘-云端协同处理、EWC持续学习和自适应模型切换。

1.环境配置
#克隆项目与数据集（百度网盘）
链接：https://pan.baidu.com/s/1Ow0toxFfTa7MGAMIq7IBYA?pwd=1pil 
提取码：1pil 
#注意：将数据集解压至data/目录

#创建虚拟环境
conda create -n agentic_vsc python=3.8
conda activate agentic_vsc

#安装依赖
pip install -r requirements.txt


2.Openapi
在文件config/system_config.yaml与src/agents/cloud_agent.py中将api修改成your_apikey

3.启动项目
python start.py
