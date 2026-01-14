from abc import ABC, abstractmethod
from typing import Dict, Any
import asyncio
import json
from datetime import datetime
from src.communication.message_bus import MessageBus
from src.communication.protocols import BaseMessage, MessageType

class BaseAgent(ABC):
    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.is_running = False
        
    async def start(self):
        self.is_running = True
        self.message_bus.register_agent(self.agent_id)
        asyncio.create_task(self._message_loop())
        
    async def stop(self):
        self.is_running = False
        
    async def _message_loop(self):
        while self.is_running:
            message = await self.message_bus.receive(self.agent_id)
            if message:
                await self.process_message(message)
            await asyncio.sleep(0.1)
    
    @abstractmethod
    async def process_message(self, message: BaseMessage):
        pass
    
    async def send_message(self, receiver: str, msg_type: MessageType, payload: Dict[str, Any]):
        message = BaseMessage(
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            payload=payload
        )
        await self.message_bus.publish(message)
    
    async def collect_data(self, source: str = "roadside_camera") -> Dict[str, Any]:
        json_command = {
            "command": "data_retrieval",
            "agent_type": self.agent_id,
            "data_source": source,
            "timestamp": self._get_timestamp(),
            "parameters": {
                "retrieval_type": "video_frames",
                "duration_seconds": 10,
                "resolution": "1920x1080",
                "format": "jpg",
                "frame_rate": 30
            },
            "status": "pending_execution"
        }
        return json_command
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")