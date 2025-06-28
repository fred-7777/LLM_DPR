from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sequences = ["Hello, my dog is cute", "My dog is cute as well"]
inputs = tokenizer(sequences, padding=False, return_tensors="pt")
print(inputs["input_ids"].shape)
# Output: torch.Size([2, 8])  # 2 sequences, each padded to 8 tokens