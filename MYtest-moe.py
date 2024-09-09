import os
from llama_cpp import Llama
import rich
import warnings
warnings.filterwarnings(action='ignore')
import datetime
import random
import string
from time import sleep
import tiktoken

# for counting the tokens in the prompt and in the result
#context_count = len(encoding.encode(yourtext))
encoding = tiktoken.get_encoding("r50k_base") 

def writehistory(filename,text):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

def genRANstring(n):
    """
    n = int number of char to randomize
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return res

verbosity = True

MODEL_CONFIG = {
    "director": {
        "name": "gemma-2-2b-it-Q5_K_M.gguf",
        "task": "text-generation",
        "stops": ['<eos>'],
        "nctx": 8192,
        "verbosity": verbosity
    },
    "writing": {
        "name": "gemma-2-2b-it-Q5_K_M.gguf",
        "task": "text-generation",
        "stops": ['<eos>'],
        "nctx": 8192,
        "verbosity": verbosity    
    },
    "summarization": {
        "name": "qwen2-0_5b-instruct-q8_0.gguf",
        "task": "text-generation",
        "stops": ['<|im_end|>'],
        "nctx": 8192,
        "verbosity": verbosity     
    },
    "indexing": {
        "name": "Phi-3.5-mini-instruct_Uncensored-Q6_K_L.gguf",
        "task": "text-generation",
        "stops": ['<|endoftext|>'],
        "nctx": 8192,
        "verbosity": verbosity     
    },    
    "extraction": {
        "name": "NuExtract-tiny.gguf",
        "task": "text-generation",
        "stops": ['<|end-output|>'],
        "nctx": 8192,
        "verbosity": verbosity        
    }, 
}


KEYWORDS = {
    "indexing": [
            "table of contents",          
            "structured",
            "hierarchy",
            "structure",
            "sections",
            "index"
    ],
    "writing": [
            "headings",
            "topic",
            "create",            
            "outline",        
            "subheadings",
            "sections",
            "topics",
            "keywords",
            "phrases",
            "sentences",
            "paragraphs",          
    ],
    "extraction": [
            "output",
            "format",  
            "data extraction",
            "web scraping",
            "information extraction",
            "extraction",
            "data cleansing",
            "data transformation",
            "data analysis",
            "data visualization",
            "JSON format",
            "keywords extraction", 
            "content analysis",
            "text analysis",                
    ],    
    "summarization": [
            "summarization",
            "text summarization",
            "automatic summarization",
            "extractive summarization",
            "abstractive summarization",
            "abstract",
            "sexecutive summary",
            "summary"
    ]
}

class MOELLM:
    def __init__(self):
        self.current_expert = None
        self.current_model = None
        #self.current_tokenizer = None
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #print(f"Using device: {self.device}")
        self.load_director_model()

    def load_director_model(self):
        """Loads the director model."""
        print("Loading director model...")
        model_name = MODEL_CONFIG["director"]["name"]
        #self.director_tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self.director_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.director_pipeline = Llama(
                model_path=f'models/{MODEL_CONFIG["director"]["name"]}',
                #n_gpu_layers=-1,  #enable GPU
                temperature=0.24,
                n_ctx=MODEL_CONFIG["director"]["nctx"],
                max_tokens=2000,
                repeat_penalty=1.176,
                stop=MODEL_CONFIG["director"]["stops"],
                verbose=MODEL_CONFIG["director"]["verbosity"],
        )
        print("Director model loaded.")

    def load_expert_model(self, expert):
        """Dynamically loads an expert model, releasing memory from the previous model."""
        if expert not in MODEL_CONFIG:
            raise ValueError(f"Unknown expert: {expert}")

        if self.current_expert != expert:
            print(f"Loading expert model: {expert}...")
            
            # Free memory from the current model if it exists
            if self.current_model:
                del self.current_model
            
            model_config = MODEL_CONFIG[expert]
            #self.current_tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
            #self.current_model = AutoModelForCausalLM.from_pretrained(model_config["name"], torch_dtype=torch.float16).to(self.device)
            self.current_expert = expert
            
            print(f"{expert.capitalize()} model loaded.")
        
        return Llama(model_path=f'models/{MODEL_CONFIG[expert]["name"]}',
                #n_gpu_layers=-1,  #enable GPU
                temperature=0.24,
                n_ctx=MODEL_CONFIG[expert]["nctx"],
                max_tokens=2000,
                repeat_penalty=1.176,
                stop=MODEL_CONFIG[expert]["stops"],
                verbose=MODEL_CONFIG[expert]["verbosity"],
        )

    def determine_expert_by_keywords(self, question):
        """Determines the expert based on keywords in the question."""
        question_lower = question.lower()
        for expert, keywords in KEYWORDS.items():
            if any(keyword in question_lower for keyword in keywords):
                return expert
        return None

    def determine_expert(self, question):
        """Determines which expert should answer the question."""
        expert = self.determine_expert_by_keywords(question)
        if expert:
            print(f"Expert determined by keyword: {expert}")
            return expert

        prompt = f"""Classify the following user question into one of these categories: writing, summarization, indexing, extraction. 
User question: {question}

Reply only with the assigned category.
Category: """
        message = [{"role": "user", "content": prompt}]
        response = self.director_pipeline.create_chat_completion(
                messages=message,
                temperature=0.1,
                repeat_penalty= 1.176,
                stop=MODEL_CONFIG["director"]["stops"],
                max_tokens=100,)['choices'][0]['message']['content']
        print(response)
        expert = response.split(":")[-1].strip().lower()
        print(expert)
        if expert not in MODEL_CONFIG:
            expert = "director"
        print(f"Redirecting question to: {expert}")
        return expert

    def generate_response(self, question, expert):
        """Generates a response using the appropriate model."""
        try:
            model = self.load_expert_model(expert)
            prompt = f"Answer the following question as an expert in {expert}: {question}\n\nAnswer:"
            message = [{"role": "user", "content": prompt}]
            response = model.create_chat_completion(
                messages=message,
                temperature=0.1,
                repeat_penalty= 1.176,
                stop=MODEL_CONFIG[expert]["stops"],
                max_tokens=200,)['choices'][0]['message']['content']
            print(response)
            return response.split("Answer:")[-1].strip()
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "Sorry, there was an error processing your request. Please try again."

    def chat_interface(self):
        """Simple chat interface."""
        print("Welcome to the MOE-LLM chat. Type 'exit' to quit.")
        while True:
            question = input("\nYou: ")
            if question.lower() in ['exit', 'quit']:
                break
            
            try:
                expert = self.determine_expert(question)
                response = self.generate_response(question, expert)
                print(f"\n{expert.capitalize()}: {response}")
            except Exception as e:
                print(f"Error in chat: {str(e)}")
                print("Please try asking another question.")

if __name__ == "__main__":
    moe_llm = MOELLM()
    moe_llm.chat_interface()