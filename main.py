import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from evaluate import load
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the pre-trained model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-ti"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Load your data
try:
    translations = pd.read_csv('/Users/joyboy/english2tygrinya/translations.csv')
except FileNotFoundError:
    print("Error: The file 'translations.csv' was not found. Please check the file path.")
    exit()

# Improved Data Quality Check
def check_data_quality(df):
    issues = []
    for index, row in df.iterrows():
        eng_words = row['text_eng'].split()
        tir_words = row['text_tir'].split()
        if len(eng_words) == 0 or len(tir_words) == 0:
            issues.append(f"Empty translation at index {index}")
        elif abs(len(eng_words) - len(tir_words)) > max(len(eng_words), len(tir_words)) * 0.5:
            issues.append(f"Large word count difference at index {index}")
    return issues

print("Performing improved data quality check...")
issues = check_data_quality(translations)
for issue in issues:
    print(issue)

print("Proceeding with the current data...")

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def batch_translate(texts, batch_size=32):
    results = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        results.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    return results

# Fine-tuning
print("Preparing data for fine-tuning...")
dataset = Dataset.from_pandas(translations[['text_eng', 'text_tir']])

def tokenize_function(examples):
    inputs = tokenizer(examples['text_eng'], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(examples['text_tir'], padding="max_length", truncation=True, max_length=128)
    inputs['labels'] = targets['input_ids']
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset
train_dataset = tokenized_datasets.shuffle(seed=42).select(range(int(len(tokenized_datasets) * 0.8)))
eval_dataset = tokenized_datasets.shuffle(seed=42).select(range(int(len(tokenized_datasets) * 0.8), len(tokenized_datasets)))

print("Fine-tuning the model...")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_dir="./logs",
    learning_rate=5e-5,
    evaluation_strategy="steps",  # Change this to "steps" or "epoch"
    save_strategy="steps",        # Use the same strategy for saving
    eval_steps=100,               # Evaluate every 100 steps
    save_steps=100,               # Save every 100 steps
    load_best_model_at_end=True,  # Continue to load the best model at the end
)

# Load BLEU metric
bleu = load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_labels = [[ref] for ref in decoded_labels]
    
    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

print("Fine-tuning complete. Saving the model...")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Load the fine-tuned model
model = MarianMTModel.from_pretrained("./fine_tuned_model")
tokenizer = MarianTokenizer.from_pretrained("./fine_tuned_model")

print("Translating sentences with the fine-tuned model...")
translations['predicted_tigrinya'] = batch_translate(translations['text_eng'].tolist())

# Evaluate accuracy
translations['correct'] = translations['predicted_tigrinya'] == translations['text_tir']
accuracy = translations['correct'].mean()

print(f"Accuracy: {accuracy:.2f}")

# Test with a specific sentence
test_sentence = "i like cats"
predicted_translation = translate(test_sentence)
print(f"Input: {test_sentence}")
print(f"Predicted Translation: {predicted_translation}")

# Error analysis
print("\nError Analysis:")
for index, row in translations.iterrows():
    if row['predicted_tigrinya'] != row['text_tir']:
        print(f"Mismatch at index {index}:")
        print(f"English: {row['text_eng']}")
        print(f"Expected Tigrinya: {row['text_tir']}")
        print(f"Predicted Tigrinya: {row['predicted_tigrinya']}")
        print()

# Save results
translations.to_csv('translation_results.csv', index=False)

print("Results saved to 'translation_results.csv'")