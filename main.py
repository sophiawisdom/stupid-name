import transformers

print("Loading GPTNeoXForCausalLM!")
model = transformers.GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
print("Completed loading GPTNeoXForCausalLM!")
tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
print("Loaded GPTNeoXForCausalLM Tokenizer")

from torchmetrics.collections import MetricCollection
from composer.models.huggingface import HuggingFaceModel
from composer.metrics import CrossEntropy
from torchmetrics import Accuracy

metrics = [CrossEntropy(), Accuracy()]

composer_model = HuggingFaceModel(model, metrics=metrics, use_logits=True)
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
import json

class FinetuneDataset(IterableDataset):
    def __init__(self, files):
        self.files = files
    def __iter__(self):
        for file in self.files:
            with open(file) as f:
                for sample in f:
                    sample = json.loads(sample)
                    string = sample["context"] + "\n\n\n" + sample["completion"]
                    tokenized = tokenizer(string, return_tensors="pt")
                    # meet equal length requirements... will drop eventually
                    tokenized['input_ids'] = tokenized['input_ids'][:,:100].squeeze()
                    tokenized['attention_mask'] = tokenized['attention_mask'][:,:100].squeeze()
                    # print("Tokenized is", tokenized)
                    yield tokenized

finetune_dataset = FinetuneDataset(["train_sample.jsonl"])
finetune_dataset_loader = DataLoader(finetune_dataset, batch_size=8)



from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR

'''
optimizer = AdamW(
    params=composer_model.parameters(),
    lr=3e-5, betas=(0.9, 0.98),
    eps=1e-6, weight_decay=3e-6
)
'''
optimizer = SGD(params=composer_model.parameters(), lr=3e-5)
linear_lr_decay = LinearLR(
    optimizer, start_factor=1.0,
    end_factor=0, total_iters=150
)

import torch
from composer import Trainer

composer_model.model_inputs = tokenizer.model_input_names

fsdp_config = {
    'sharding_strategy': 'FULL_SHARD',
    'min_params': 1e8,
    'cpu_offload': False, # Not supported yet
    'mixed_precision': 'DEFAULT',
    'backward_prefetch': 'BACKWARD_POST',
    'activation_checkpointing': False,
    'activation_cpu_offload': False,
    'verbose': True
}

# Create Trainer Object
trainer = Trainer(
    model=composer_model,
    train_dataloader=finetune_dataset_loader,
    max_duration="1ep",
    optimizers=optimizer,
    schedulers=[linear_lr_decay],
    device='gpu' if torch.cuda.is_available() else 'cpu',
    train_subset_num_batches=6,
    fsdp_config=fsdp_config,
    seed=17
)
# Start training
trainer.fit()
