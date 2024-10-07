

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set pad_token_id to eos_token_id to avoid warnings
model.config.pad_token_id = model.config.eos_token_id

def generate_topic(text, max_length=50):
    prompt = f"The main topic of this document is: {text} The topic is"
    
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate the attention mask
    attention_mask = torch.ones(inputs.shape, device=inputs.device)
    
    # Generate text using GPT-2
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=max_length, 
            do_sample=True, 
            top_k=50,
            pad_token_id=model.config.eos_token_id,
            attention_mask=attention_mask
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Post-process: extract the topic from the generated text
    topic_start = generated_text.find("The topic is") + len("The topic is ")
    topic_end = generated_text.find(".", topic_start)
    
    # Return the inferred topic
    return generated_text[topic_start:topic_end].strip()

# Example: Using GPT-2 for zero-shot topic generation
text_document = "Artificial intelligence is transforming industries with innovative solutions like machine learning and deep learning models. These technologies are being applied in healthcare, finance, and manufacturing."
topic = generate_topic(text_document)
print(f"Generated Topic: {topic}")
