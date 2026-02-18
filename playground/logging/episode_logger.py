"""Episode日志：分解reward、动作序列、终止原因"""
from typing import Dict, Any, List
import json
from datetime import datetime
from env.world_state import WorldState
from env.action_space import Action


class EpisodeLogger:
    """Episode轨迹日志记录器"""
    
    def __init__(self, log_dir: str = "./logs"):
        """初始化日志目录"""
        self.log_dir = log_dir
        self.current_episode = None
    
    def start_episode(self, episode_id: str, initial_state: WorldState):
        """开始记录新episode"""
        self.current_episode = {
            'episode_id': episode_id,
            'timestamp': datetime.now().isoformat(),
            'initial_state': initial_state.to_dict(),
            'steps': [],
            'total_reward': 0.0,
            'termination_reason': None
        }
    
    def log_step(self, state: WorldState, action: Action, next_state: WorldState,
                 reward: float, reward_terms: Dict[str, float], info: Dict[str, Any]):
        """记录一步"""
        step_log = {
            'step': state.step_idx,
            'action': {
                'type': action.type.name,
                'target_id': action.target_id,
                'params': action.params
            },
            'state_before': state.to_dict(),
            'state_after': next_state.to_dict(),
            'reward': reward,
            'reward_terms': reward_terms,
            'info': info
        }
        self.current_episode['steps'].append(step_log)
        self.current_episode['total_reward'] += reward
    
    def end_episode(self, termination_reason: str, final_state: WorldState):
        """结束episode并保存"""
        self.current_episode['termination_reason'] = termination_reason
        self.current_episode['final_state'] = final_state.to_dict()
        
        # 保存为JSON
        filename = f"{self.log_dir}/episode_{self.current_episode['episode_id']}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.current_episode, f, indent=2, ensure_ascii=False)
        
        self.current_episode = None
    
    def get_summary(self) -> Dict[str, Any]:
        """获取当前episode摘要"""
        if self.current_episode is None:
            return {}
        return {
            'episode_id': self.current_episode['episode_id'],
            'total_steps': len(self.current_episode['steps']),
            'total_reward': self.current_episode['total_reward'],
            'termination_reason': self.current_episode.get('termination_reason')
        }
