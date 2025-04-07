import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import matplotlib.pyplot as plt

data = []
with open('data/data.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

dataset = Dataset.from_dict({
    'docstring': [entry['docstring'] for entry in data],
    'code': [entry['code'] for entry in data]
})

dataset = dataset.shuffle(seed=42).select(range(100))

model_name = "Salesforce/codegen-350M-mono"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    texts = [doc + tokenizer.eos_token + code for doc, code in zip(examples["docstring"], examples["code"])]
    result = tokenizer(texts, padding="max_length", truncation=True, max_length=128, )
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.2).values()

model = AutoModelForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="checkpoints_codegen350M",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=5,
    max_steps=50,
    fp16=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    logging_dir="logs",
    dataloader_num_workers=0,
    disable_tqdm=False,
    seed=42,
    no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

train_loss_values = [x["loss"] for x in trainer.state.log_history if "loss" in x]
plt.plot(train_loss_values, label="Train Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()
