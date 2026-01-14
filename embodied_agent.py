import asyncio
import json
import random
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import cv2
import numpy as np

from .base_agent import BaseAgent
from src.communication.protocols import MessageType
from src.datasets.kitti_loader import KITTIDataset

class EmbodiedAgent(BaseAgent):
    def __init__(self, agent_id: str, message_bus, data_source="kitti"):
        super().__init__(agent_id, message_bus)
        self.data_source = data_source
        self.device_type = "roadside_camera"
        self.location = "intersection_001"
        self.current_index = 0
        self.dataset = None
        
        # 加载真实KITTI数据集（道路车辆专用）
        self._initialize_kitti_dataset()
        
        print(f"📷 Embodied Agent {agent_id} 初始化完成")
        print(f"   数据源: KITTI本地真实数据集")
        print(f"   设备类型: {self.device_type}")
        print(f"   位置: {self.location}")
        print(f"   数据集大小: {len(self.dataset)} 个样本" if self.dataset else "数据集加载失败")
        if self.dataset:
            print(f"   车辆类别: {self.dataset.get_classes()[:5]}...")
    
    def _initialize_kitti_dataset(self):
        """初始化本地KITTI数据集"""
        try:
            print("📂 加载本地KITTI数据集...")
            self.dataset = KITTIDataset(
                data_dir="data/kitti",
                split="val",  # 使用验证集进行评估
                img_size=640,
                augment=False
            )
            self.current_index = 0
            
            print(f"✅ KITTI数据集加载成功")
            print(f"   总样本数: {len(self.dataset)}")
            print(f"   图像尺寸: 640x640")
            
        except Exception as e:
            print(f"❌ KITTI数据集加载失败: {e}")
            # 创建空的dataset占位符
            self.dataset = type('SimpleDataset', (), {
                '__len__': lambda self: 0,
                'get_class_name': lambda self, cls_id: f'class_{cls_id}'
            })()
    
    async def process_message(self, message):
        """处理接收到的消息"""
        print(f"📥 {self.agent_id} 收到消息类型: {message.msg_type}")
        
        if message.msg_type == MessageType.JSON_CMD:
            await self._handle_json_command(message)
        elif message.msg_type == MessageType.TASK:
            await self._handle_task(message)
        else:
            print(f"⚠️  忽略未知消息类型: {message.msg_type}")
    
    async def _handle_json_command(self, message):
        """处理JSON命令 - 从本地KITTI数据集中检索真实数据"""
        command = message.payload
        
        # 兼容多种命令格式
        if "action" in command:
            action = command["action"]
        elif "command" in command:
            action = command["command"]
        else:
            action = ""
        
        print(f"📥 {self.agent_id} 收到JSON命令: {action}")
        
        if action in ["data_retrieval", "retrieve_video", "get_data"]:
            # 执行真实数据检索
            params = command.get("parameters", {})
            num_frames = params.get("frame_count", 5)
            
            # 从本地KITTI数据集检索真实数据
            data = await self._retrieve_real_kitti_data(num_frames)
            
            # 发送数据给Edge Agent
            await self.send_message(
                receiver="edge_agent",
                msg_type=MessageType.DATA,
                payload={
                    "data_type": "video",
                    "frames": data.get("frames", []),
                    "source": self.agent_id,
                    "command_id": command.get("command_id", ""),
                    "timestamp": datetime.now().isoformat(),
                    "data_source": "kitti_local",
                    "dataset_info": data.get("dataset_info", {})
                }
            )
    
    async def _retrieve_real_kitti_data(self, num_frames: int = 5) -> Dict[str, Any]:
        """从本地KITTI数据集检索真实道路数据"""
        print(f"🎥 {self.agent_id} 从本地KITTI数据集检索真实道路数据: {num_frames} 帧")
        
        frames = []
        
        if self.dataset is None or len(self.dataset) == 0:
            print(f"❌ KITTI数据集未加载或为空")
            return self._create_fallback_data(num_frames)
        
        try:
            total_samples = len(self.dataset)
            actual_frames = min(num_frames, total_samples)
            
            for i in range(actual_frames):
                if self.current_index >= total_samples:
                    self.current_index = 0
                
                idx = self.current_index
                
                # 获取真实KITTI数据样本
                try:
                    sample = self.dataset[idx]
                except Exception as e:
                    print(f"⚠️  获取样本{idx}失败: {e}")
                    self.current_index += 1
                    continue
                
                # 提取真实标注信息
                annotations = []
                vehicle_count = 0
                
                if sample['boxes'].numel() > 0:
                    boxes = sample['boxes'].cpu().numpy()
                    for box in boxes:
                        if len(box) >= 5:
                            cls_id = int(box[0])
                            cx, cy, w, h = box[1:5]
                            
                            # 转换类别ID到类别名称
                            try:
                                class_name = self.dataset.get_class_name(cls_id)
                            except:
                                # 默认KITTI类别映射
                                class_names = {
                                    0: 'car', 1: 'van', 2: 'truck', 3: 'pedestrian',
                                    4: 'Person_sitting', 5: 'cyclist', 6: 'tram', 7: 'misc'
                                }
                                class_name = class_names.get(cls_id, f'class_{cls_id}')
                            
                            # 转换为像素坐标
                            if 'orig_size' in sample:
                                orig_h, orig_w = sample['orig_size'].cpu().numpy()
                            else:
                                orig_h, orig_w = 375, 1242  # KITTI默认尺寸
                            
                            x_center = cx * orig_w
                            y_center = cy * orig_h
                            width = w * orig_w
                            height = h * orig_h
                            
                            x1 = float(x_center - width / 2)
                            y1 = float(y_center - height / 2)
                            x2 = float(x_center + width / 2)
                            y2 = float(y_center + height / 2)
                            
                            # 判断是否为车辆类别
                            is_vehicle = cls_id in [0, 1, 2]  # car, van, truck
                            
                            annotation = {
                                "class_id": cls_id,
                                "class_name": class_name,
                                "bbox": [x1, y1, x2, y2],
                                "bbox_normalized": [float(cx), float(cy), float(w), float(h)],
                                "confidence": 1.0,
                                "is_vehicle": is_vehicle
                            }
                            
                            annotations.append(annotation)
                            
                            if is_vehicle:
                                vehicle_count += 1
                
                # 构建帧数据
                image_path = sample.get('img_path', f'kitti_sample_{idx}')
                
                frame_data = {
                    "frame_id": idx,
                    "image_path": str(image_path),
                    "vehicle_count": vehicle_count,
                    "annotations": annotations,
                    "timestamp": datetime.now().isoformat(),
                    "data_source": "kitti_local_real",
                    "dataset_name": "KITTI Real Dataset",
                    "description": f"真实KITTI道路场景 - 样本{idx}",
                    "sample_info": {
                        "total_annotations": len(annotations),
                        "vehicle_annotations": vehicle_count,
                        "has_ground_truth": len(annotations) > 0
                    }
                }
                
                frames.append(frame_data)
                self.current_index += 1
            
            print(f"✅ 成功检索 {len(frames)} 帧真实KITTI数据")
            
            # 构建数据集信息
            dataset_info = {
                "name": "KITTI",
                "purpose": "自动驾驶道路车辆检测",
                "description": "真实道路场景车辆检测数据集",
                "resolution": "1242x375 (原始)",
                "vehicle_classes": ["car", "van", "truck"],
                "total_classes": 8,
                "source": "KITTI Benchmark Suite",
                "local_path": "data/kitti",
                "data_type": "real"
            }
            
            return {
                "status": "success",
                "data_source": "kitti_local_real",
                "frames": frames,
                "total_frames": len(frames),
                "dataset_info": dataset_info,
                "retrieval_info": {
                    "requested_frames": num_frames,
                    "delivered_frames": len(frames),
                    "current_index": self.current_index,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"❌ KITTI真实数据检索失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回备用数据
            return self._create_fallback_data(num_frames)
    
    def _create_fallback_data(self, num_frames: int) -> Dict[str, Any]:
        """创建备用数据（当真实数据不可用时）"""
        print(f"⚠️  创建备用数据（真实数据不可用）")
        
        frames = []
        for i in range(num_frames):
            # 创建简单的模拟数据
            vehicle_count = random.randint(1, 3)
            annotations = []
            for j in range(vehicle_count):
                annotations.append({
                    "class_id": random.randint(0, 2),
                    "class_name": ["car", "van", "truck"][random.randint(0, 2)],
                    "bbox": [random.randint(0, 1000), random.randint(0, 300), 
                            random.randint(100, 1100), random.randint(50, 350)],
                    "bbox_normalized": [random.random(), random.random(), 
                                       random.random()*0.2, random.random()*0.15],
                    "confidence": random.uniform(0.7, 0.95),
                    "is_vehicle": True
                })
            
            frames.append({
                "frame_id": i,
                "image_path": f"data/kitti/backup_frame_{i}.jpg",
                "vehicle_count": vehicle_count,
                "annotations": annotations,
                "timestamp": datetime.now().isoformat(),
                "data_source": "kitti_backup",
                "dataset_name": "KITTI Backup"
            })
        
        return {
            "status": "backup",
            "data_source": "kitti_backup",
            "frames": frames,
            "total_frames": len(frames),
            "dataset_info": {
                "name": "KITTI (Backup)",
                "purpose": "道路车辆检测演示",
                "vehicle_classes": ["car", "van", "truck"],
                "note": "真实数据不可用，使用模拟数据"
            }
        }
    
    async def _handle_task(self, message):
        """处理任务指令"""
        task_config = message.payload
        print(f"📋 {self.agent_id} 收到任务配置")
        print(f"   任务类型: {task_config.get('task', 'unknown')}")
        print(f"   数据集: {task_config.get('dataset', 'kitti')}")
        print(f"   模型: {task_config.get('model', 'unknown')}")
        print(f"   流程: {task_config.get('training_pipeline', {}).get('method', 'unknown')}")
    
    def get_device_status(self) -> Dict[str, Any]:
        """获取设备状态"""
        return {
            "agent_id": self.agent_id,
            "device_type": self.device_type,
            "location": self.location,
            "status": "active",
            "data_source": self.data_source,
            "dataset_loaded": self.dataset is not None,
            "dataset_size": len(self.dataset) if self.dataset else 0,
            "current_index": self.current_index,
            "operation_mode": "real_data_retrieval",
            "timestamp": datetime.now().isoformat(),
            "capabilities": [
                "real_kitti_data_retrieval",
                "road_vehicle_annotation",
                "local_dataset_access"
            ]
        }