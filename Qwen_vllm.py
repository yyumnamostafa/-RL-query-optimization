from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from typing import List
import torch

class QwenQueryEnhancer(nn.Module):
    def __init__(self, model_name="Qwen/Qwen-7B"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir="huggingface_cache"
        )
        # For Qwen, we need to set pad_token to an existing token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_vllm = False
        
        # First try vLLM (with memory optimization)
        try:
            from vllm import LLM, SamplingParams
            
            # Clear CUDA cache before initializing vLLM
            torch.cuda.empty_cache()
            
            # Initialize vLLM with memory optimization settings
            self.vllm_engine = LLM(
                model=model_name,
                trust_remote_code=True,
                gpu_memory_utilization=0.8,  # Lower value to prevent OOM
                max_model_len=1024,  # Limit context length
                tensor_parallel_size=1  # Use just one GPU
            )
            
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=128
            )
            
            self.use_vllm = True
            print("Successfully initialized vLLM for Qwen")
            
        except (ImportError, ValueError) as e:
            print(f"Could not initialize vLLM: {e}")
            print("Falling back to standard transformers implementation")
            
            # Try with Flash Attention if available
            try:
                # Clear CUDA cache before loading model
                torch.cuda.empty_cache()
                
                # Set lower memory requirements
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir="huggingface_cache",
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16  # Use lower precision
                )
                print("Successfully enabled Flash Attention 2")
                
            except Exception as e:
                print(f"Could not enable Flash Attention: {e}")
                print("Falling back to standard attention implementation")
                
                # Final fallback: standard implementation with memory optimization
                try:
                    torch.cuda.empty_cache()
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        cache_dir="huggingface_cache",
                        device_map="auto",
                        torch_dtype=torch.bfloat16,  # Lower precision
                        low_cpu_mem_usage=True
                    )
                except Exception as e:
                    print(f"Warning: Standard model loading failed: {e}")
                    print("Trying with 8-bit quantization as last resort")
                    
                    # Last resort: 8-bit quantization
                    try:
                        import bitsandbytes as bnb
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            cache_dir="huggingface_cache",
                            device_map="auto",
                            load_in_8bit=True
                        )
                    except:
                        print("All model loading attempts failed. Try using a smaller model.")
                        raise
        
    def forward(self, queries: List[str]) -> List[str]:
        # System prompt template
        prompt_template = """You are a query enhancement assistant. Your task is to improve the given query to make it more specific and detailed for code generation.
Original query: {query}
Enhanced query:"""
        
        enhanced_queries = []
        
        # vLLM implementation if enabled
        if self.use_vllm:
            try:
                prompts = [prompt_template.format(query=query) for query in queries]
                outputs = self.vllm_engine.generate(prompts, self.sampling_params)
                
                for output in outputs:
                    text = output.outputs[0].text
                    # Extract the enhanced query part
                    enhanced_query = text.split("Enhanced query:")[-1].strip()
                    enhanced_queries.append(enhanced_query)
                return enhanced_queries
            except Exception as e:
                print(f"vLLM inference failed: {e}. Falling back to standard implementation.")
                self.use_vllm = False
                # Make sure we have a standard model as backup
                if not hasattr(self, 'model'):
                    self.model = AutoModelForCausalLM.from_pretrained(
                        "Qwen/Qwen-7B",
                        trust_remote_code=True,
                        cache_dir="huggingface_cache",
                        device_map="auto",
                        torch_dtype=torch.bfloat16
                    )
        
        # Standard implementation (used as fallback)
        if not self.use_vllm:
            for query in queries:
                prompt = prompt_template.format(query=query)
                # Process one at a time to save memory
                with torch.no_grad():  # Disable gradient calculation to save memory
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                enhanced_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                enhanced_query = enhanced_query.split("Enhanced query:")[-1].strip()
                enhanced_queries.append(enhanced_query)
                
                # Clear memory after each generation
                del outputs, inputs
                torch.cuda.empty_cache()
            
        return enhanced_queries
        
    def forward_with_loss(self, queries: List[str]):
        enhanced_queries = self.forward(queries)
        loss = torch.tensor(0.0, requires_grad=True)
        return loss, enhanced_queries
