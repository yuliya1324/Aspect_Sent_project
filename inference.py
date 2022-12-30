import re
import torch
from nltk.tokenize import word_tokenize
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch import nn

MODEL_NAME = 'sberbank-ai/ruRoberta-large'

int_to_cat = [
    "O", 
    "Food", 
    "Food", 
    "Interior",
    "Interior", 
    "Price",
    "Price", 
    "Whole", 
    "Whole", 
    "Service", 
    "Service",
]
int_to_sent = ["positive", "negative", "neutral", "both"]
labels = ["absence", "positive", "negative", "both", "neutral"]

class CatClassifier(nn.Module):

    def __init__(self, n_classes):
        super(CatClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.5)
        self.Food = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.Interior = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.Price = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.Whole = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.Service = nn.Linear(self.bert.config.hidden_size, n_classes)
  
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(self.mean_pooling(outputs, attention_mask))
        Food = self.Food(output)
        Interior = self.Interior(output)
        Price = self.Price(output)
        Whole = self.Whole(output)
        Service = self.Service(output)
        return {
            "Food": Food, 
            "Interior": Interior, 
            "Price": Price, 
            "Whole": Whole, 
            "Service": Service, 
            }
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

def parse_args():
    parser = argparse.ArgumentParser(description="Inference on Aspect-Sentiment tasks")
    parser.add_argument("--first_task", action="store_true")
    parser.add_argument("--no-first_task", dest="first_task", action="store_false")
    parser.set_defaults(first_task=False)
    parser.add_argument("--second_task", action="store_true")
    parser.add_argument("--no-second_task", dest="second_task", action="store_false")
    parser.set_defaults(second_task=False)
    parser.add_argument("--aspect_filename", type=Path, required=False, help="Path to file where to write aspects")
    parser.add_argument("--sentiment_filename", type=Path, required=False, help="Path to file where to write sentiments")
    parser.add_argument("--reviews_filename", type=Path, required=True, help="Path to file with reviews")
    parser.add_argument("--device1", type=str, required=False, help="device for model_cat", default="cpu")
    parser.add_argument("--device2", type=str, required=False, help="device for model_sent", default="cpu")
    parser.add_argument("--device3", type=str, required=False, help="device for model", default="cpu")
    parser.add_argument("--model_cat", type=Path, required=False, help="Path to the model for sequence labeling by category")
    parser.add_argument("--model_sent", type=Path, required=False, help="Path to the model for sequence labeling by sentiment")
    parser.add_argument("--model", type=Path, required=False, help="Path to the model for sentiment classification")
    args = parser.parse_args()
    return args


def join_punctuation(seq, right_punct = '!$%\'),-.:;?', left_punct = '('):
    right_punct = set(right_punct)
    left_punct = set(left_punct)
    seq = iter(seq)
    current = next(seq)

    for nxt in seq:
        if nxt in right_punct:
            current += nxt
        elif current in left_punct:
            current += nxt
        else:
            yield current
            current = nxt

    yield current

def inference_aspects(review, model_cat, model_sent, tokenizer, idx, device_1, device_2):
    tokens = word_tokenize(review)
    text = review
    
    model_cat.eval()
    model_sent.eval()
    tokenized = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    word_ids = tokenized.word_ids()
    
    res = []
    pred = model_cat(tokenized["input_ids"].to(device_1), attention_mask=tokenized["attention_mask"].to(device_1)).logits.argmax(dim=2)[0]
    prev = None
    for k, j in enumerate(word_ids):
        if j != None and prev != j:
            res.append(int_to_cat[pred[k].item()])
        prev = j
        
    res_2 = []
    pred = model_sent(tokenized["input_ids"].to(device_2), attention_mask=tokenized["attention_mask"].to(device_2)).logits.argmax(dim=2)[0]
    prev = None
    for k, j in enumerate(word_ids):
        if j != None and prev != j:
            res_2.append(int_to_sent[pred[k].item()])
        prev = j
        
    result = []
    prev = None
    for token, tag, sent in zip(tokens, res, res_2):
        if tag != "O":
            if prev and prev == tag:
                result[-1][0].append(token) 
            else:
                result.append([[token], tag, sent])
        prev = tag
        
    output = []
    length = 0
    for span in result:
        word = ' '.join(join_punctuation(span[0]))
        m = re.search(re.escape(word), text)
        if m:
            s, e = m.span()
            output.append([idx, span[1], word, str(s + length), str(e + length), span[2]])
            text = text[e:]
            length += e
    return output

def inference_sentiment(tokenizer, model, text, device):
    encoding = tokenizer(text, return_tensors='pt')
    model = model.eval()
    with torch.no_grad():
        pred = model(input_ids=encoding["input_ids"].to(device), attention_mask=encoding["attention_mask"].to(device))
    _, Food = torch.max(pred["Food"], dim=1)
    _, Interior = torch.max(pred["Interior"], dim=1)
    _, Price = torch.max(pred["Price"], dim=1)
    _, Whole = torch.max(pred["Whole"], dim=1)
    _, Service = torch.max(pred["Service"], dim=1)
    return {
            "Food": labels[Food.item()], 
            "Interior": labels[Interior.item()], 
            "Price": labels[Price.item()], 
            "Whole": labels[Whole.item()], 
            "Service": labels[Service.item()],
    }

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

    if args.first_task and args.model_cat and args.model_sent:
        model_cat = torch.load(args.model_cat)
        model_sent = torch.load(args.model_sent)
        _ = model_cat.to(args.device1)
        _ = model_sent.to(args.device2)

        if args.aspect_filename and args.reviews_filename:
            with open(args.aspect_filename, "w", encoding="utf-8") as file_write:
                with open(args.reviews_filename, encoding="utf-8") as file_read:
                    for line in file_read:
                        line = line[:-1]
                        idx, review = line.split("\t")
                        res = inference_aspects(review, model_cat, model_sent, tokenizer, idx, args.device1, args.device2)
                        for r in res:
                            file_write.write("\t".join(r) + "\n")
        else:
            raise Exception("You should provide paths to files with reviews and where to write answers")

    if args.second_task and args.model:
        model = torch.load(args.model)
        _ = model.to(args.device3)

        if args.sentiment_filename and args.reviews_filename:
            with open(args.sentiment_filename, "w", encoding="utf-8") as file_write:
                with open(args.reviews_filename, encoding="utf-8") as file_read:
                    for line in file_read:
                        line = line[:-1]
                        idx, text = line.split("\t")
                        pred = inference_sentiment(tokenizer, model, text, args.device3)
                        for cat in pred:
                            file_write.write(f"{idx}\t{cat}\t{pred[cat]}\n")
        else:
            raise Exception("You should provide paths to files with reviews and where to write answers")    


if __name__ == '__main__':
    args = parse_args()
    main(args)
