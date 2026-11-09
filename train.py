from unsloth import FastLanguageModel
import random
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

import torch
print("Visible GPU:", torch.cuda.current_device(), torch.cuda.get_device_name())

import os
import datetime


os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["WANDB_MODE"] = "offline" # We will manage this in run.sh or TrainingArguments
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# local_model_path = "/home/fluffymelon/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-bnb-4bit/snapshots/70d37f62f4f475a5a42dbbd5b7aad38f420e0960"
local_model_path = "/home/fluffymelon/.cache/huggingface/hub/models--unsloth--Llama-3.1-8B-unsloth-bnb-4bit/snapshots/0a43e2563fa80ebc5eb093baa843c9cb65ab50d5"
# local_model_path = "/home/fluffymelon/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct-bnb-4bit/snapshots/f15c379fb32bb402fa06a7ae9aecb1febf4b79ec"

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name="unsloth/llama-3-8b-bnb-4bit",
    model_name=local_model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

custom_prompt = """Ты ассистент, который учавствует в диалоге между Лешей и Соней. Ниже приведено сообщение, на которое вы должны ответить максимально похоже на то, как бы это сделал Леша

### Сообщение:
{}

### Ответ:
{}"""

custom_context_prompt = """Ты ассистент, который учавствует в диалоге между Лешей и Соней. Прими во внимание контекст диалога. Твоя задача ответить на приведенное ниже сообщение максимально похоже на то, как бы это сделал Леша

### Контекст:
{}

### Сообщение:
{}

### Ответ:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def alpaca_formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # if len(instruction) == 0:
        #     augmented_instruction = ""
        # else:
        #     augmented_instruction = "\n".join(instruction[random.randint(-len(instruction), -1):])
        # # Must add EOS_TOKEN, otherwise your generation will go on forever!
        # text = alpaca_prompt.format(augmented_instruction, input, output) + EOS_TOKEN
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def custom_formatting_prompts_func(examples):
    contexts = examples["context"]
    inputs   = examples["input"]
    outputs  = examples["output"]
    texts = []
    for context, input, output in zip(contexts,inputs, outputs):
        if context == []:
            context = ""
        else:
            context = "\n".join(context)
        output = "\n".join(output)
        text = custom_context_prompt.format(context, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


# dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = load_dataset('json', data_files="dataset_context_min_max_long_output.json", split="train")
dataset = dataset.map(custom_formatting_prompts_func, batched = True,)

# Split the dataset (e.g., 90% train, 10% validation)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42) # seed for reproducibility

# Access the splits
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"] # The 'test' split is commonly used for validation during training

# Generate a unique run ID (e.g., timestamp)
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Use timestamp
# run_id = str(uuid.uuid4())[:8] # Or use UUID for uniqueness
base_output_dir = "outputs"
output_dir = os.path.join(base_output_dir, f"run_{run_id}")
wandb_project_name = "digitaltwin-llama3-finetune" # Define your project name here

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        do_eval = True,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 200,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = output_dir, # Use the unique output directory
        eval_strategy = "steps",      # Added evaluation
        eval_steps = 100,                   # Evaluate every 100 steps
        save_strategy = "steps",
        save_steps = 100,
        load_best_model_at_end = True,
        save_total_limit = 2,
        report_to="wandb", # Ensure wandb reporting is enabled
        # run_name=f"run_{run_id}", # Set wandb run name (Removed to let WandB auto-name)
        # Add wandb project name if not set globally via env var
        # You might need to login to wandb (`wandb login`) if not using offline mode
    ),
    # Add wandb specific arguments if needed, like project name
    # This is often handled by environment variables or TrainingArguments
)

trainer_stats = trainer.train()
