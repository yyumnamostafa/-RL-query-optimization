from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from typing import List
import torch
import os
from dotenv import load_dotenv

load_dotenv()

class QwenFullQueryEnhancer(nn.Module):
    def __init__(self, model_name=os.getenv("MODEL_NAME")):
        super().__init__()
        # Load the tokenizer with proper settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir="huggingface_cache",
        )
        
        # For Qwen models, use the eos token as the pad token
        # These models don't support adding new special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Make sure model is updated with the new token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="huggingface_cache",
            device_map="auto"
        )
        
        # Ensure model config has pad token ID set
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, queries: List[str]) -> List[str]:
        # 添加系统提示和任务描述
        prompt_template = """You are a query enhancement assistant. Your task is to improve the given query to make it more specific and detailed for code generation.
Original query: {query}
Enhanced query:"""
        
        enhanced_queries = []
        for query in queries:
            prompt = prompt_template.format(query=query)
            
            # Process a single query at a time (no padding needed)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            enhanced_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 移除原始提示，只保留生成的部分
            enhanced_query = enhanced_query.split("Enhanced query:")[-1].strip()
            enhanced_queries.append(enhanced_query)
            
        return enhanced_queries
        
    def forward_with_loss(self, queries: List[str]):
        # Record operations for gradient calculation
        prompt_template = """You are a query enhancement assistant. Your task is to improve the given query to make it more specific and detailed for code generation.
    Original query: {query}
    Enhanced query:"""
        
        batch_loss = torch.tensor(0.0, requires_grad=True)
        enhanced_queries = []
        
        for query in queries:
            prompt = prompt_template.format(query=query)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Get model outputs with loss computation
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            # Get the loss from the model's output
            if batch_loss.item() == 0.0:
                batch_loss = outputs.loss
            else:
                batch_loss = batch_loss + outputs.loss
            
            # Generate enhanced query as before
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
        
        # Average the loss over the batch
        batch_loss = batch_loss / len(queries)
        return batch_loss, enhanced_queries
