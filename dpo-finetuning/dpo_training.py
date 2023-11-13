import torch
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, GPTQConfig, BitsAndBytesConfig
from trl import DPOTrainer
from sklearn.model_selection import train_test_split


def get_dataset(dataset_name, split, val_size=500):
    # Getting all the data
    dataset = load_dataset(
        dataset_name,
        split=split
    )

    # Getting only chosen and rejected values from each column
    df = dataset.to_pandas()
    df["chosen"] = df["chosen"].apply(lambda x: x[1]["content"])
    df["rejected"] = df["rejected"].apply(lambda x: x[1]["content"])

    # Splitting data based on test size
    train, val = train_test_split(df, test_size=val_size)

    train_data = Dataset.from_pandas(train)
    val_data = Dataset.from_pandas(val)

    return train_data, val_data


def prepare_model_and_tokenizer(model_name, model_ref_name, peft_model_link):
    device_arg = {'device_map': 'auto'}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, **device_arg)

    model_ref = AutoModelForCausalLM.from_pretrained(model_ref_name, quantization_config=bnb_config, **device_arg)

    model.load_adapter(peft_model_link, "qlora", **device_arg)

    # Setting some config for training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model_ref.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.config.pretraining_tp = 1

    # Set tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, model_ref, tokenizer


if __name__ == "__main__":
    train_data, val_data = get_dataset(dataset_name="HuggingFaceH4/ultrafeedback_binarized", split="train_prefs",
                                       val_size=500)
    # Peft model link is the repo that contains adapter config and adapter model bin
    model, model_ref, tokenizer = prepare_model_and_tokenizer(model_name="teknium/OpenHermes-2.5-Mistral-7B",
                                                              model_ref_name="HuggingFaceH4/zephyr-7b-beta",
                                                              peft_model_link="")

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        max_steps=-1,
        remove_unused_columns=True,
        gradient_accumulation_steps=2,
        learning_rate=5e-7,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10,
        num_train_epochs=5,
        output_dir="mistral-dpo",
        optim="rmsprop",
        warmup_steps=5,
        eval_steps=15,
        save_steps=5,
        bf16=True,
        report_to="wandb",
        push_to_hub=True
    )

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_length=2048,
        max_target_length=2048,
        max_prompt_length=2048
    )

    dpo_trainer.train()
