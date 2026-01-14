import asyncio
import json
import yaml
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
from .base_agent import BaseAgent
from src.communication.protocols import MessageType

class CloudAgent(BaseAgent):
    def __init__(self, agent_id: str, message_bus):
        super().__init__(agent_id, message_bus)
        self.config = self._load_system_config()
        self.model_state = "Model_A_COCO"
        self.model_coco_path = "models/yolov8n.pt"
        self.model_kitti_path = "models/model_b_kitti_ewc.pt"
        self.task_history = []
        self.model_history = []
        self.openai_client = None
        self._init_openai()
    
    def _load_system_config(self) -> Dict[str, Any]:
        config_path = Path("config/system_config.yaml")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            default_config = {
                'system': {'name': 'Agentic_VSC_System', 'version': '5.0'},
                'openai': {'api_key': '', 'enabled': False},
                'ewc': {'importance': 1000.0, 'training_epochs': 5},
                'thresholds': {'mAP': 0.35, 'snr_fluctuation': 3.0}
            }
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            return default_config
    
    def _init_openai(self):
        try:
            openai_config = self.config.get('openai', {})
            if not openai_config.get('enabled', False):
                return
            
            api_key = openai_config.get('api_key', '')
            if not api_key or api_key == 'your-openai-key':
                return
            
            import openai
            self.openai_client = openai.OpenAI(api_key=api_key)
        except Exception:
            pass
    
    async def process_message(self, message):
        if message.msg_type == MessageType.REPORT:
            await self._handle_report(message)
        elif message.msg_type == MessageType.ALERT:
            await self._handle_alert(message)
    
    async def deploy_task(self, task_input):
        if isinstance(task_input, str):
            if self.openai_client:
                task_config = await self._parse_task_with_openai(task_input)
            else:
                task_config = self._parse_task_locally(task_input)
        elif isinstance(task_input, dict):
            task_config = task_input
        else:
            return None
        
        if 'description' not in task_config:
            task_config['description'] = task_input if isinstance(task_input, str) else '自定义任务'
        
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_config['task_id'] = task_id
        task_config['deployed_at'] = datetime.now().isoformat()
        
        task_record = {
            "task_id": task_id,
            "description": task_config['description'],
            "config": task_config,
            "timestamp": datetime.now().isoformat(),
            "status": "deployed",
            "model_state_at_deploy": self.model_state
        }
        self.task_history.append(task_record)
        
        await self.send_message(
            receiver="edge_agent",
            msg_type=MessageType.TASK,
            payload=task_config
        )
        
        return task_record
    
    async def _parse_task_with_openai(self, description: str) -> Dict[str, Any]:
        try:
            prompt = f"""解析用户任务描述："{description}"，返回JSON配置。"""
            
            response = self.openai_client.chat.completions.create(
                model=self.config.get('openai', {}).get('model', 'gpt-4'),
                messages=[{"role": "system", "content": "视频语义通信系统任务解析"}, {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    return self._parse_task_locally(description)
            
            return self._parse_task_locally(description)
        except Exception:
            return self._parse_task_locally(description)
    
    def _parse_task_locally(self, description: str) -> Dict[str, Any]:
        desc_lower = description.lower()
        config = {
            "task_type": "vehicle_extraction",
            "model": "Model_A_COCO",
            "dataset": "kitti",
            "description": description,
            "evaluation_needed": True,
            "finetuning_needed": False,
            "report_needed": True,
            "thresholds": {"map_threshold": 0.35, "snr_fluctuation": 3.0},
            "specific_requirements": []
        }
        
        if any(word in desc_lower for word in ["检测", "detect"]):
            config["task_type"] = "vehicle_detection"
        elif any(word in desc_lower for word in ["分析", "analyze"]):
            config["task_type"] = "performance_evaluation"
        elif any(word in desc_lower for word in ["监控", "monitor"]):
            config["task_type"] = "traffic_monitoring"
        elif any(word in desc_lower for word in ["行人", "pedestrian"]):
            config["task_type"] = "pedestrian_detection"
            
        if any(word in desc_lower for word in ["微调", "finetune"]):
            config["finetuning_needed"] = True
            
        return config
    
    async def _handle_report(self, message):
        report_data = message.payload
        summary = report_data.get('summary', {})
        avg_map = summary.get('average_mAP', 1.0)
        if avg_map == 1.0:
            avg_map = report_data.get('metrics', {}).get('mAP', {}).get('current', 1.0)
        
        map_threshold = report_data.get('thresholds', {}).get('map_threshold', 0.35)
        needs_finetuning = avg_map < map_threshold
        
        if needs_finetuning:
            success = await self._trigger_real_ewc_finetuning(report_data)
            if success:
                await self._send_model_switch_instruction()
    
    async def _trigger_real_ewc_finetuning(self, report_data: Dict[str, Any]) -> bool:
        try:
            from src.core.fixed_ewc_trainer import RealEWCTrainer
            
            importance = self.config.get('ewc', {}).get('importance', 1000.0)
            trainer = RealEWCTrainer(
                base_model_path=self.model_coco_path,
                importance=importance
            )
            
            trainer.compute_fisher_matrix(
                coco_data_dir="data/coco",
                num_samples=30
            )
            
            training_results = trainer.finetune_on_kitti(
                epochs=self.config.get('ewc', {}).get('training_epochs', 3),
                batch_size=4
            )
            
            if training_results.get('status') in ['success', 'success_fallback']:
                self.model_state = "Model_B_KITTI_EWC"
                self.model_kitti_path = training_results.get('model_path', 'models/model_b_kitti_ewc.pt')
                
                model_record = {
                    "from_model": "Model_A_COCO",
                    "to_model": "Model_B_KITTI_EWC",
                    "method": training_results.get('method', 'Real_EWC'),
                    "timestamp": datetime.now().isoformat(),
                    "trigger_reason": f"mAP低于阈值: {report_data.get('summary', {}).get('average_mAP', 0):.3f} < 0.35"
                }
                self.model_history.append(model_record)
                
                return True
            return False
        except Exception:
            self.model_history.append({
                "attempt": "Real_EWC_Training",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            })
            return False
    
    async def _send_model_switch_instruction(self):
        switch_prompt = {
            "action": "switch_model",
            "old_model": "Model_A_COCO",
            "new_model": "Model_B_KITTI_EWC",
            "model_path": self.model_kitti_path,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_message(
            receiver="edge_agent",
            msg_type=MessageType.PROMPT,
            payload=switch_prompt
        )
    
    async def _handle_alert(self, message):
        pass
    
    def get_system_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "model_state": self.model_state,
            "task_count": len(self.task_history),
            "model_switches": len(self.model_history),
            "openai_enabled": self.openai_client is not None
        }