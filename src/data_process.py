import pandas as pd
import torch 
from src.utils import train_val_test_split, pair_comments
from src.dataset.vectoriser import TextVectoriser
from src.dataset.SarcasmDataset import SarcasmDataset

def load_data(file_name, is_transformer = False, is_parent = False, max_len = 35, min_freq = 2, batch_size=32, frac_train=0.8, frac_val=0.1, frac_test=0.1):

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





        


