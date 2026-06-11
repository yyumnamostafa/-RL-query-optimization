# train_rq2.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Union
import numpy as np
import json
import os
import gc
import time
from datetime import datetime
from dotenv import load_dotenv
import psutil
from models.qwen_lora import QwenLoRAQueryEnhancer
from models.qwen_full import QwenFullQueryEnhancer
from llm.deepseek import DeepseekAPI
from utils.reward_util import RewardCalculator
from torch.cuda.amp import GradScaler, autocast
from utils.code_metric import CodeMetricsCalculator

# Load environment variables
load_dotenv()

# Get training mode from environment, default to "lora"
TRAINING_MODE = os.getenv("TRAINING_MODE", "lora").lower()

class RLTrainer:
    def __init__(self, 
                 query_enhancer: Union[QwenFullQueryEnhancer, QwenLoRAQueryEnhancer],
                 deepseek_api: DeepseekAPI,
                 reward_calculator: RewardCalculator,
                 learning_rate: float = 1e-4,
                 checkpoint_dir: str = os.getenv("CHECKPOINT_DIR"),
                 log_dir: str = os.getenv("LOG_DIR"),
                 gradient_accumulation_steps: int = 1,
                 is_lora: bool = True,
                 use_amp: bool = True,
                 validation_interval: int = 50,
                 max_grad_norm: float = 1.0):
        self.query_enhancer = query_enhancer
        self.deepseek_api = deepseek_api
        self.reward_calculator = reward_calculator
        self.is_lora = is_lora
        self.validation_interval = validation_interval
        self.max_grad_norm = max_grad_norm
        
        # AMP initialization with explicit float16
        self.use_amp = use_amp and torch.cuda.is_available() and not is_lora
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda', enabled=True)  
            self.amp_dtype = torch.float16
        else:
            self.scaler = None

        # Optimizer setup based on model type
        if self.is_lora:
            if hasattr(self.query_enhancer, 'model') and hasattr(self.query_enhancer.model, 'parameters'):
                trainable_params = [p for p in self.query_enhancer.model.parameters() if p.requires_grad]
                self.optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        else:
            self.optimizer = optim.AdamW(query_enhancer.parameters(), lr=learning_rate, weight_decay=0.01)
            
        # Directory setup
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize logging files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.train_log_file = os.path.join(self.log_dir, f"train_log_{timestamp}.jsonl")
        self.val_log_file = os.path.join(self.log_dir, f"val_log_{timestamp}.jsonl")
        self.metrics_file = os.path.join(self.log_dir, f"metrics_{timestamp}.json")
        
        # Initialize metrics tracking
        self.metrics = {
            "train_rewards": [],
            "val_rewards": [],
            "test_rewards": [],
            "best_reward": -float('inf'),
            "best_epoch": 0,
            "training_time": 0,
            "start_time": time.time(),
            "memory_usage": [],
            "gpu_memory_usage": [],
            "epoch_times": [],
            "steps_completed": 0,
            "total_samples_processed": 0,
            "oom_events": 0,
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_css": [],
            "test_precision": 0.0,
            "test_recall": 0.0,
            "test_f1": 0.0,
            "test_css": 0.0,
            "val_losses": [],
            "test_losses": [],
            "best_val_reward": -float('inf'),
            "best_val_epoch": 0,
            "validation_times": [],
            "test_times": []
        }
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.code_metrics_calculator = CodeMetricsCalculator()

    def get_memory_stats(self):
        """Get memory usage statistics"""
        stats = {
            "ram_used": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
            "ram_percent": psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_used": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_cached": torch.cuda.memory_reserved() / 1024 / 1024
            })
        
        return stats

    def train_step(self, original_query: str, ground_truth: str, step: int) -> Tuple[float, str, str]:
        """Execute a single training step"""
        try:
            memory_stats = self.get_memory_stats()
            self.metrics["memory_usage"].append(memory_stats)
            
            self.query_enhancer.eval()
            
            with torch.no_grad():
                _, enhanced_queries = self.query_enhancer.forward_with_loss([original_query])
                enhanced_query = enhanced_queries[0]
                response = self.deepseek_api.get_response(enhanced_query)
                generated_code = self._parse_response(response)
                reward = self.reward_calculator.calculate(generated_code, ground_truth)
            

            self.query_enhancer.train()
            
            with autocast(enabled=self.use_amp, dtype=torch.float16):
                model_loss, _ = self.query_enhancer.forward_with_loss([original_query])
                loss = -torch.mean(torch.tensor(reward, device=model_loss.device) * model_loss) / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                if self.is_lora:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.query_enhancer.model.parameters() if p.requires_grad],
                        self.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(self.query_enhancer.parameters(), self.max_grad_norm)
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self._cleanup_memory()

            self.metrics["train_rewards"].append(float(reward))
            self.metrics["steps_completed"] += 1
            self.metrics["total_samples_processed"] += 1
            
            self._log_training_step(original_query, enhanced_query, ground_truth, 
                                  generated_code, reward, step, loss)
            
            return reward, enhanced_query, generated_code
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.metrics["oom_events"] += 1
                self._cleanup_memory()
                print(f"OOM Error (#{self.metrics['oom_events']})")
                raise e
            elif "_amp_foreach_non_finite_check_and_unscale_cuda" in str(e):
                print("AMP scaling error encountered, disabling AMP and retrying...")
                self.use_amp = False
                self.scaler = None
                return self.train_step(original_query, ground_truth, step)
            raise e

    def validate(self, validation_data: List[Dict]) -> float:
        """Evaluate model performance on validation set"""
        self.query_enhancer.eval()
        total_reward = 0
        total_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "css": 0.0}
        total_samples = len(validation_data)
        validation_start_time = time.time()

        with torch.no_grad():
            for idx, data in enumerate(validation_data):
                try:
                    _, enhanced_queries = self.query_enhancer.forward_with_loss([data["prompt"]])
                    enhanced_query = enhanced_queries[0]
                    response = self.deepseek_api.get_response(enhanced_query)
                    generated_code = self._parse_response(response)
                    reward = self.reward_calculator.calculate(generated_code, data["reference_code"])
                    total_reward += reward

                    # Calculate code metrics
                    metrics = self.code_metrics_calculator.calculate_metrics(generated_code, data["reference_code"])
                    for key, value in metrics.items():
                        total_metrics[key] += value

                    print(f"Validation sample {idx+1}/{total_samples}, Reward: {reward:.4f}")

                except Exception as e:
                    print(f"Validation sample {idx+1} failed: {str(e)}")
                    continue

        avg_reward = total_reward / total_samples
        avg_metrics = {k: v / total_samples for k, v in total_metrics.items()}
        validation_time = time.time() - validation_start_time

        # Update metrics
        self.metrics["val_rewards"].append(avg_reward)
        self.metrics["validation_times"].append(validation_time)
        self.metrics["val_precision"].append(avg_metrics["precision"])
        self.metrics["val_recall"].append(avg_metrics["recall"])
        self.metrics["val_f1"].append(avg_metrics["f1"])
        self.metrics["val_css"].append(avg_metrics["css"])

        print(f"\nValidation completed:")
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Metrics: {avg_metrics}")
        print(f"Time taken: {validation_time:.2f}s")

        return avg_reward

    def test(self, test_data: List[Dict]) -> float:
        """Evaluate model performance on test set"""
        self.query_enhancer.eval()
        total_reward = 0
        total_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "css": 0.0}
        total_samples = len(test_data)
        test_start_time = time.time()

        with torch.no_grad():
            for idx, data in enumerate(test_data):
                try:
                    _, enhanced_queries = self.query_enhancer.forward_with_loss([data["prompt"]])
                    enhanced_query = enhanced_queries[0]
                    response = self.deepseek_api.get_response(enhanced_query)
                    generated_code = self._parse_response(response)
                    reward = self.reward_calculator.calculate(generated_code, data["reference_code"])
                    total_reward += reward

                    # Calculate code metrics
                    metrics = self.code_metrics_calculator.calculate_metrics(generated_code, data["reference_code"])
                    for key, value in metrics.items():
                        total_metrics[key] += value

                    print(f"Test sample {idx+1}/{total_samples}, Reward: {reward:.4f}")

                except Exception as e:
                    print(f"Test sample {idx+1} failed: {str(e)}")
                    continue

        avg_reward = total_reward / total_samples
        avg_metrics = {k: v / total_samples for k, v in total_metrics.items()}
        test_time = time.time() - test_start_time

        # Update metrics
        self.metrics["test_rewards"] = avg_reward
        self.metrics["test_times"].append(test_time)
        self.metrics["test_precision"] = avg_metrics["precision"]
        self.metrics["test_recall"] = avg_metrics["recall"]
        self.metrics["test_f1"] = avg_metrics["f1"]
        self.metrics["test_css"] = avg_metrics["css"]

        print(f"\nTest evaluation completed:")
        print(f"Average reward: {avg_reward:.4f}")
        print(f"Metrics: {avg_metrics}")
        print(f"Time taken: {test_time:.2f}s")

        return avg_reward

    def save_checkpoint(self, epoch: int, avg_reward: float, is_best: bool = False, checkpoint_path: str = None):
        """Save model checkpoint"""
        if self.is_lora:
            if checkpoint_path is None:
                checkpoint_dir = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}")
            else:
                checkpoint_dir = checkpoint_path
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            self.query_enhancer.save_lora_weights(checkpoint_dir)
            
            meta_info = {
                'epoch': epoch,
                'reward': avg_reward,
                'model_type': 'QwenLoRAQueryEnhancer',
                'optimizer_state': self.optimizer.state_dict(),
                'metrics': self.metrics,
                'amp_state': self.scaler.state_dict() if self.use_amp else None
            }
            torch.save(meta_info, os.path.join(checkpoint_dir, "meta_info.pt"))
            
            if is_best:
                best_model_dir = os.path.join(self.checkpoint_dir, 'best_model')
                os.makedirs(best_model_dir, exist_ok=True)
                self.query_enhancer.save_lora_weights(best_model_dir)
                torch.save(meta_info, os.path.join(best_model_dir, "meta_info.pt"))
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.query_enhancer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'reward': avg_reward,
                'model_type': 'QwenFullQueryEnhancer',
                'metrics': self.metrics,
                'amp_state': self.scaler.state_dict() if self.use_amp else None
            }
            
            if checkpoint_path is None:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            
            if is_best:
                best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                torch.save(checkpoint, best_model_path)

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0, -float('inf')
            
        if self.is_lora and os.path.isdir(checkpoint_path):
            self.query_enhancer.load_lora_weights(checkpoint_path)
            
            meta_path = os.path.join(checkpoint_path, "meta_info.pt")
            if os.path.exists(meta_path):
                meta_info = torch.load(meta_path, map_location='cpu')
                self.optimizer.load_state_dict(meta_info['optimizer_state'])
                if meta_info.get('amp_state') and self.use_amp:
                    self.scaler.load_state_dict(meta_info['amp_state'])
                epoch = meta_info['epoch']
                reward = meta_info['reward']
                if 'metrics' in meta_info:
                    self.metrics.update(meta_info['metrics'])
                return epoch, reward
            else:
                return 0, -float('inf')
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.query_enhancer.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('amp_state') and self.use_amp:
                self.scaler.load_state_dict(checkpoint['amp_state'])
            if 'metrics' in checkpoint:
                self.metrics.update(checkpoint['metrics'])
            return checkpoint['epoch'], checkpoint['reward']

    def _parse_response(self, response: str) -> str:
        """Parse API response"""
        try:
            if "<answer>" in response and "</answer>" in response:
                code = response.split("<answer>")[1].split("</answer>")[0].strip()
                if code.startswith("```"):
                    first_newline = code.find("\n")
                    if first_newline != -1:
                        last_marker = code.rfind("```")
                        if last_marker != -1:
                            code = code[first_newline:last_marker].strip()
                        else:
                            code = code[first_newline:].strip()
                return code
            print(f"Warning: Could not find <answer> tags in response: {response}")
            return ""
        except Exception as e:
            print(f"Error parsing response: {e}")
            return ""

    def _cleanup_memory(self):
        """Clean up memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _log_training_step(self, original_query: str, enhanced_query: str, 
                          ground_truth: str, generated_code: str, 
                          reward: float, step: int, loss: torch.Tensor):
        """Log training step data"""
        log_entry = {
            "original_query": original_query,
            "enhanced_query": enhanced_query,
            "ground_truth": ground_truth,
            "generated_code": generated_code,
            "reward": float(reward),
            "step": step,
            "loss": float(loss.item()),
            "memory_stats": self.get_memory_stats(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.train_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def save_metrics(self):
        """Save training metrics"""
        self.metrics["training_time"] = time.time() - self.metrics["start_time"]
        
        if self.metrics["train_rewards"]:
            self.metrics.update({
                "avg_train_reward": sum(self.metrics["train_rewards"]) / len(self.metrics["train_rewards"]),
                "max_train_reward": max(self.metrics["train_rewards"]),
                "min_train_reward": min(self.metrics["train_rewards"]),
                "final_memory_stats": self.get_memory_stats()
            })
        
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)


def load_and_process_data(file_path: str, sample_ratio: float = 1.0) -> List[Dict]:
    """
    
    Args:
        file_path (str): JSONL file path
        sample_ratio (float): Sampling ratio, range (0,1)
        
    Returns:
        List[Dict]: Processed data list
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file does not exist: {file_path}")
    
    if not 0 < sample_ratio <= 1:
        raise ValueError("The sampling ratio must be within the range of (0,1].")
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            
                if "prompt" in item and "reference_code" in item:
                    data.append(item)
                else:
                    print(f"Warning: Skip incomplete data items: {line[:50]}...")
            except json.JSONDecodeError:
                print(f"Warning: Skip invalid JSON lines: {line[:50]}...")
    
    total_samples = len(data)
    if total_samples == 0:
        raise ValueError(f"No valid training data was found.: {file_path}")
        
    print(f"Original dataset size: {total_samples}")
    
    if sample_ratio < 1.0:
        import random
        sample_size = int(total_samples * sample_ratio)
        random.seed(42) 
        data = random.sample(data, sample_size)
        print(f"Dataset size after sampling: {len(data)} (Sampling ratio: {sample_ratio:.2%})")
    
    return data

def split_data(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
   The dataset is divided into training set, validation set, and test set.
    
    Args:
        data (List[Dict]): complete dataset
        train_ratio (float): Training set ratio, default 0.8
        val_ratio (float): Validation set ratio, default 0.1
        (Test set proportion = 1 - train_ratio - val_ratio)
        
    Returns:
        Tuple[List[Dict], List[Dict], List[Dict]]: (training set, validation set, testing set)
    """
    if not 0 < train_ratio < 1:
        raise ValueError("The training set ratio must be in the range (0,1).")
    if not 0 < val_ratio < 1:
        raise ValueError("The validation set ratio must be in the range (0,1).")
    if train_ratio + val_ratio >= 1:
        raise ValueError("The sum of the ratios of the training set and the validation set must be less than 1.")
    
    total_samples = len(data)
    if total_samples == 0:
        raise ValueError("数据集为空")
    
    import random
    random.seed(42)
    
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    print(f"\n dataset splitting complete:")
    print(f"training set: {len(train_data)} sample ({len(train_data)/total_samples:.1%})")
    print(f"validation set: {len(val_data)} sample ({len(val_data)/total_samples:.1%})")
    print(f"testing set: {len(test_data)} sample ({len(test_data)/total_samples:.1%})")
    
    return train_data, val_data, test_data

def run_training_epoch(trainer: RLTrainer, 
                      train_data: List[Dict], 
                      val_data: List[Dict], 
                      epoch: int, 
                      num_epochs: int) -> None:
    """
    run training
    
    Args:
        trainer: RLTrainer instance
        train_data: Training data
        val_data: Validation data
        epoch: Current epoch
        num_epochs: Total no of epochs
    """
    print(f"\n====== Epoch {epoch + 1}/{num_epochs} ======")
    epoch_start_time = time.time()
    total_reward = 0
    
    trainer.optimizer.zero_grad()
    
    for idx, data in enumerate(train_data):
        try:
            reward, enhanced_query, generated_code = trainer.train_step(
                data["prompt"],
                data["reference_code"],
                idx
            )
            print(f"train: Epoch {epoch + 1}, Sample {idx+1}/{len(train_data)}, Reward: {reward:.4f}")
            
            total_reward += reward
            
            if (idx + 1) % 5 == 0:
                trainer._cleanup_memory()
        
            if (idx + 1) % trainer.validation_interval == 0:
                print("\n perform val...")
                val_reward = trainer.validate(val_data)
                print(f"Val rewards: {val_reward:.4f}")
                
                if val_reward > trainer.metrics["best_val_reward"]:
                    trainer.metrics["best_val_reward"] = val_reward
                    trainer.metrics["best_val_epoch"] = epoch
                    trainer.save_checkpoint(epoch, val_reward, is_best=True)
                    print(f"Discover a new best model Reward: {val_reward:.4f}")
                
                # 恢复训练模式
                trainer.query_enhancer.train()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Warning: CUDA memory is low. Free up cache and skip the current sample")
                trainer._cleanup_memory()
                continue
            else:
                raise e
 
    avg_train_reward = total_reward / len(train_data)
    print(f"\n[*] Epoch {epoch + 1}")
    print(f"{avg_train_reward:.4f}")
  
    trainer.metrics["train_rewards"].append(avg_train_reward)
    trainer.metrics["epoch_times"].append(time.time() - epoch_start_time)
  
    trainer.save_checkpoint(epoch, avg_train_reward)

    latest_checkpoint_path = os.path.join(trainer.checkpoint_dir, 
                                        "latest_checkpoint" if trainer.is_lora else "latest_checkpoint.pt")
    if trainer.is_lora:
        os.makedirs(latest_checkpoint_path, exist_ok=True)
        trainer.query_enhancer.save_lora_weights(latest_checkpoint_path)
    else:
        trainer.save_checkpoint(epoch, avg_train_reward, checkpoint_path=latest_checkpoint_path)

def print_training_summary(trainer: RLTrainer):
    """Print a summary of the training metrics and results.
    
    Args:
        trainer: The RLTrainer instance containing training metrics
    """
    print("\n====== Training Summary ======")
    
    # Training statistics
    print("\nTraining Statistics:")
    if trainer.metrics["train_rewards"]:
        print(f"Average Training Reward: {sum(trainer.metrics['train_rewards']) / len(trainer.metrics['train_rewards']):.4f}")
        print(f"Best Training Reward: {max(trainer.metrics['train_rewards']):.4f}")
        print(f"Total Steps Completed: {trainer.metrics['steps_completed']}")
        print(f"Total Samples Processed: {trainer.metrics['total_samples_processed']}")
    
    # Validation statistics
    print("\nValidation Statistics:")
    if trainer.metrics["val_rewards"]:
        print(f"Best Validation Reward: {trainer.metrics['best_val_reward']:.4f}")
        print(f"Best Validation Epoch: {trainer.metrics['best_val_epoch']}")
        if trainer.metrics["val_precision"]:
            latest_val_idx = -1
            print(f"Final Validation Metrics:")
            print(f"- Precision: {trainer.metrics['val_precision'][latest_val_idx]:.4f}")
            print(f"- Recall: {trainer.metrics['val_recall'][latest_val_idx]:.4f}")
            print(f"- F1 Score: {trainer.metrics['val_f1'][latest_val_idx]:.4f}")
            print(f"- CSS Score: {trainer.metrics['val_css'][latest_val_idx]:.4f}")
    
    # Test statistics
    print("\nTest Statistics:")
    print(f"Final Test Reward: {trainer.metrics['test_rewards']:.4f}")
    print(f"Test Metrics:")
    print(f"- Precision: {trainer.metrics['test_precision']:.4f}")
    print(f"- Recall: {trainer.metrics['test_recall']:.4f}")
    print(f"- F1 Score: {trainer.metrics['test_f1']:.4f}")
    print(f"- CSS Score: {trainer.metrics['test_css']:.4f}")
    
    # Training time and resource statistics
    print("\nResource Usage:")
    training_time_hours = trainer.metrics["training_time"] / 3600
    print(f"Total Training Time: {training_time_hours:.2f} hours")
    if trainer.metrics["oom_events"] > 0:
        print(f"Out of Memory Events: {trainer.metrics['oom_events']}")
    
    if trainer.metrics.get("final_memory_stats"):
        mem_stats = trainer.metrics["final_memory_stats"]
        print("\nFinal Memory Usage:")
        print(f"RAM Used: {mem_stats['ram_used']:.2f} MB ({mem_stats['ram_percent']}%)")
        if 'gpu_used' in mem_stats:
            print(f"GPU Memory Used: {mem_stats['gpu_used']:.2f} MB")
            print(f"GPU Memory Cached: {mem_stats['gpu_cached']:.2f} MB")
    
    print("\n====== End of Training Summary ======")

    
def main():
    # PyTorch memory management setup
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    
    # Get training mode
    training_mode = TRAINING_MODE
    print(f"Current training mode: {training_mode}")
    
    # Create directories
    checkpoint_dir = f"{os.getenv('CHECKPOINT_DIR')}/checkpoints_{training_mode}"
    log_dir = f"{os.getenv('LOG_DIR')}/logs_{training_mode}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize model
    is_lora = training_mode == "lora"
    if is_lora:
        print("Initializing Qwen LoRA model...")
        query_enhancer = QwenLoRAQueryEnhancer()
    else:
        print("Initializing Qwen full parameter model...")
        query_enhancer = QwenFullQueryEnhancer()
    
    # Initialize components
    deepseek_api = DeepseekAPI()
    reward_calculator = RewardCalculator(method=os.getenv("REWARD_METHOD"))
    
    # Set gradient accumulation steps
    gradient_accumulation_steps = 32 if not is_lora else 8
    
    # Convert SAMPLE_RATIO to float with default value of 1.0
    sample_ratio = float(os.getenv("SAMPLE_RATIO", "1.0"))
    
    full_data = load_and_process_data("dataset/train.jsonl", sample_ratio=sample_ratio)
    train_data, val_data, test_data = split_data(full_data, train_ratio=0.8, val_ratio=0.1)
    

    dataset_info = {
        "total_samples": len(full_data),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "sample_ratio": sample_ratio,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    dataset_info_path = os.path.join(log_dir, "dataset_info.json")
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    # Initialize trainer
    trainer = RLTrainer(
        query_enhancer=query_enhancer,
        deepseek_api=deepseek_api,
        reward_calculator=reward_calculator,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        is_lora=is_lora,
        use_amp=not is_lora,
        validation_interval=50,
        max_grad_norm=1.0
    )
    
    # Load checkpoint if exists
    start_epoch = 0
    resume_checkpoint = os.path.join(checkpoint_dir, 
                                   "latest_checkpoint" if is_lora else "latest_checkpoint.pt")
    
    if os.path.exists(resume_checkpoint):
        start_epoch, _ = trainer.load_checkpoint(resume_checkpoint)
        start_epoch += 1
    
    # Training loop
    num_epochs = int(os.getenv("NUM_EPOCHS"))
    try:
        for epoch in range(start_epoch, num_epochs):
            run_training_epoch(trainer, train_data, val_data, epoch, num_epochs)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {str(e)}")
        raise e
    finally:
        trainer.save_metrics()
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    print("Evaluating on test set...")
    test_reward = trainer.test(test_data)
    trainer.save_metrics()
    
    print_training_summary(trainer)

if __name__ == "__main__":
    main()
