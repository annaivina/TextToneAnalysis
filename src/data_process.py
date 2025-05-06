import pandas as pd
import torch 
from src.utils import train_val_test_split, pair_comments, pair_comments_deberta
from src.dataset.vectoriser import TextVectoriser
from src.dataset.SarcasmDataset import SarcasmDataset
from datasets import Dataset
from transformers import AutoTokenizer
from functools import partial

def load_data(file_name, is_transformer = False, is_parent = False, is_deberta = False, model_checkpoint= None, max_len = 35, min_freq = 2, batch_size=32, frac_train=0.8, frac_val=0.1, frac_test=0.1):

    df = pd.read_csv(file_name)

    train_ds, val_ds, test_ds = train_val_test_split(df,frac_train=frac_train, frac_val=frac_val, frac_test=frac_test)

    train_sent = train_ds['comment'].tolist()
    train_labels = train_ds['label'].tolist()
    train_parent = train_ds['parent_comment'].tolist()
    
    valid_sent = val_ds['comment'].tolist()
    valid_labels = val_ds['label'].tolist()
    valid_parent = val_ds['parent_comment'].tolist()
    
    test_sent = test_ds['comment'].tolist()
    test_labels = test_ds['label'].tolist()
    test_parent = test_ds['parent_comment'].tolist()

    len_train = len(train_labels)

    if is_deberta:
        train_deberta_sent = pair_comments_deberta(train_parent, train_sent)
        valid_deberta_sent = pair_comments_deberta(valid_parent, valid_sent)
        test_deberta_sent = pair_comments_deberta(test_parent, test_sent)

        train = [{"text": t, "label": l} for t, l in zip(train_deberta_sent, train_labels)]
        valid = [{"text": t, "label": l} for t, l in zip(valid_deberta_sent, valid_labels)]
        test  = [{"text": t, "label": l} for t, l in zip(test_deberta_sent, test_labels)]
        
        train_deberta= Dataset.from_list(train)
        valid_deberta = Dataset.from_list(valid)
        test_deberta = Dataset.from_list(test)

        train_deberta = train_deberta.rename_column("label", "labels")
        valid_deberta = valid_deberta.rename_column("label", "labels")
        test_deberta = test_deberta.rename_column("label", "labels")


        #Load the tokinezer which has been used for the deberta specifically
        #Dont know how to write in gracefully...

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
        def tokenize_inputs(input):
             return tokenizer(input["text"], truncation=True, padding="max_length", max_length=143)

        
        data_train_main_add = train_deberta.map(tokenize_inputs, batched=True)
        data_valid_main_add  = valid_deberta.map(tokenize_inputs, batched=True)
        data_test_main_add = test_deberta.map(tokenize_inputs, batched=True)

        print(data_train_main_add[1])

      
        data_train_main_add.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        data_valid_main_add.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        data_test_main_add.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        return  data_train_main_add,  data_valid_main_add,  data_test_main_add, tokenizer 


    else: 
        vectoriser = TextVectoriser(max_len, min_freq, keep_emojis = True, remove_punctuation=True, is_Transformer=is_transformer)
        #Keep emojis and Remove punctuation are hardcoded for now. 

        if is_parent:
            all_comments = train_parent + train_sent
            vectoriser.build_vocab(all_comments)

            vectorized_data_train = SarcasmDataset(text = train_sent, main_text=train_parent, label=train_labels, vectorizer=vectoriser, is_parent=is_parent)
            vectorized_data_valid = SarcasmDataset(text = valid_sent, main_text=valid_parent, label=valid_labels, vectorizer=vectoriser, is_parent=is_parent)
            vectorized_data_test = SarcasmDataset(text = test_sent, main_text=test_parent, label=test_labels, vectorizer=vectoriser, is_parent=is_parent)
    
        elif is_transformer:
            new_train_sent = pair_comments(train_parent, train_sent)
            new_valid_sent = pair_comments(valid_parent, valid_sent)
            new_test_sent =  pair_comments(test_parent, test_sent)

            vectoriser.build_vocab(new_train_sent)
            vectorized_data_train = SarcasmDataset(text = new_train_sent, label=train_labels, vectorizer=vectoriser, is_parent=is_parent)
            vectorized_data_valid = SarcasmDataset(text = new_valid_sent, label=valid_labels, vectorizer=vectoriser, is_parent=is_parent)
            vectorized_data_test = SarcasmDataset(text = new_test_sent,  label=test_labels, vectorizer=vectoriser, is_parent=is_parent)

        else:
            vectoriser.build_vocab(train_sent)
            vectorized_data_train = SarcasmDataset(text = train_sent, label=train_labels, vectorizer=vectoriser, is_parent=is_parent)
            vectorized_data_valid = SarcasmDataset(text = valid_sent, label=valid_labels, vectorizer=vectoriser, is_parent=is_parent)
            vectorized_data_test = SarcasmDataset(text = test_sent,  label=test_labels, vectorizer=vectoriser, is_parent=is_parent)

        voc_size = len(vectoriser.vocab)

        #Lets pass it thopught the DataLoader
        data_train_main_add = torch.utils.data.DataLoader(vectorized_data_train, batch_size=batch_size, shuffle=True)
        data_valid_main_add = torch.utils.data.DataLoader(vectorized_data_valid, batch_size=32, shuffle=False)
        data_test_main_add = torch.utils.data.DataLoader(vectorized_data_test, batch_size=32, shuffle=False)
        
        return data_train_main_add, data_valid_main_add, data_test_main_add, voc_size, len_train







