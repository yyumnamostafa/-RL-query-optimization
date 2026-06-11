import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv

load_dotenv()
class QwenLoRAQueryEnhancer(nn.Module):
    def __init__(self, lora_r=int(os.getenv("LORA_R")), lora_alpha=int(os.getenv("LORA_ALPHA")), lora_dropout=float(os.getenv("LORA_DROPOUT"))):
        super().__init__()
        model_name = os.getenv("MODEL_NAME")
        if model_name is None:
            raise ValueError("MODEL_NAME environment variable must be set. Please set it before initializing QwenLoRAQueryEnhancer.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir="huggingface_cache"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
      
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,  # Changed from model_name=model_name
            trust_remote_code=True,
            cache_dir="huggingface_cache",
            device_map="auto",
            # load_in_8bit=True,  
            # load_in_4bit=True,
            # torch_dtype=torch.float16 
        )

        # Add this before creating the LoRA config
        print("Available modules in the model:")
        for name, _ in self.base_model.named_modules():
            if any(keyword in name for keyword in ['attn', 'attention', 'proj', 'mlp']):
                print(name)
        
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        print("LoRA...")
        # In models/qwen_lora.py, around line 46-50

        if model_name in ["Qwen/Qwen1.5-7B-Chat", "Qwen/Qwen1.5-MoE-A2.7B-Chat"]:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif model_name in ["Qwen/Qwen-1_8B-Chat"]:
            target_modules = ["c_attn", "c_proj"]
        else:
            target_modules = ["c_attn", "c_proj", "w1", "w2"]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,                     
            lora_alpha=lora_alpha,        
            lora_dropout=lora_dropout,    
            bias="none",                  
            target_modules=target_modules, 
        )

        # lora_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     r=lora_r,                    
        #     lora_alpha=lora_alpha,        
        #     lora_dropout=lora_dropout,    
        #     bias="none",                 
        #     # Update to match Qwen1.5 architecture
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # )
        
       
        self.model = get_peft_model(self.base_model, lora_config)
        print(f"{self.model.print_trainable_parameters()}")
        
    def forward(self, queries: List[str]) -> List[str]:
        
        prompt_template = """You are a query enhancement assistant. Your task is to improve the given query to make it more specific and detailed for code generation.
Original query: {query}
Enhanced query:"""
        
        enhanced_queries = []
        for query in queries:
            prompt = prompt_template.format(query=query)
            
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                enhanced_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                enhanced_query = enhanced_query.split("Enhanced query:")[-1].strip()
                enhanced_queries.append(enhanced_query)
                
                del outputs, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        return enhanced_queries
        
    def forward_with_loss(self, queries: List[str]):
        
        prompt_template = """You are a query enhancement assistant. Your task is to improve the given query to make it more specific and detailed for code generation.
Original query: {query}
Enhanced query:"""
        
        batch_loss = None
        enhanced_queries = []
        
        for query in queries:
            prompt = prompt_template.format(query=query)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            if batch_loss is None:
                batch_loss = outputs.loss
            else:
                batch_loss = batch_loss + outputs.loss
            
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                enhanced_query = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                enhanced_query = enhanced_query.split("Enhanced query:")[-1].strip()
                enhanced_queries.append(enhanced_query)
            
            del generated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        batch_loss = batch_loss / len(queries)
        return batch_loss, enhanced_queries

    def save_lora_weights(self, path):
        self.model.save_pretrained(path)
        
    def load_lora_weights(self, path):
        self.model = PeftModel.from_pretrained(self.base_model, path)
