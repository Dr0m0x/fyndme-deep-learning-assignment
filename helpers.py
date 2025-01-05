from transformers import AutoTokenizer
import random

def batched_tokenizer(texts):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return encoded

def convert_digits_to_random_text(labels):
    # Feel free to modify - up to you! (some of them are redundant)
    digit_to_text = {
        '0': ['0', 'Zero', 'zero', 'Null', '00'],
        '1': ['1', 'One', 'one', 'Won', 'Onew'],
        '2': ['2', 'Two', 'two', 'Too', 'ToO'],
        '3': ['3', 'Three', 'tree', 'Thre', 'tHREE'],
        '4': ['4', 'Four', 'four', 'For', 'fOUR'],
        '5': ['5', 'Five', 'quinque', 'cinco', 'Paanch'],
        '6': ['6', 'Six', 'six', 'SIX', 'siX'],
        '7': ['7', 'Seven', 'seven', 'SeVeN', 'SEVEN'],
        '8': ['8', 'Eight', 'eight', 'EIGHT', 'eiGht'],
        '9': ['9', 'Nine', 'nine', 'NINE', 'NiNe']
    }

    def _label_to_random_text(label: int) -> str:
        return random.choice(digit_to_text[str(label)])
    random_texts = [_label_to_random_text(label) for label in labels.tolist()]

    return random_texts

if __name__ == '__main__':
    batch_texts = ["The quick brown fox", "jumps over the lazy dog", "hello world"]
    encoded_batch = batched_tokenizer(batch_texts)
