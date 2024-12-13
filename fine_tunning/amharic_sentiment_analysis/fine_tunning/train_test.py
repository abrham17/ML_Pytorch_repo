from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from ML_Pytorch_repo.fine_tunning.amharic_sentiment_analysis.fine_tunning.sentiment_analysis_on_amharic_data import model , preparing_data
def get_trainer(model, train_dataset, test_dataset, output_dir, epochs):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    return trainer

train_dataset , test_dataset = preparing_data()    
trainer = get_trainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        output_dir=".\\ML_Pytorch_repo\\fine_tunning\\amharic_sentiment_analysis",
        epochs=3
    )
    
train_result = trainer.train()
eval_results = trainer.evaluate()

print(eval_results)