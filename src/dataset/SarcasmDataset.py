import torch 
from typing import List


class SarcasmDataset(torch.utils.data.Dataset):
  def __init__(self, text: List[str], label: List[int], vectorizer, main_text: List[str] = None, is_parent = False) -> None:

    """
        The SarcasmDataset processes the inputs intot vectorised inputs.
        The output is the torch tensor 

        Parameters:
            text: this is the main text (comment) input
            main_text: this isa parent comment
            label: this is the label of the main comment
            vectoriser: the TextVectoriser to use for the vectorisationb
            is_parent: in case you provide main_text you ise True, so you will get the output as vect(main), vect(parent), labels
    """
    self.text = text
    self.main_text = main_text
    self.label = label
    self.vectorizer = vectorizer
    self.is_parent = is_parent

  def __len__(self)->int:
    return len(self.main_text) if self.main_text is not None else len(self.text)

  def __getitem__(self, idx: int):
    #Vectorise main comment
    text = str(self.text[idx])
    vectorised = self.vectorizer.vectorise(text)
    #Extract labels 
    label = self.label[idx]
    
    #Process parent comment if available
    vectorised_main = None
    if self.is_parent:
      main_text = str(self.main_text[idx])
      vectorised_main = self.vectorizer.vectorise(main_text)

    
    return (
            torch.tensor(vectorised, dtype=torch.long),
            torch.tensor(vectorised_main, dtype=torch.long) if vectorised_main is not None else torch.zeros_like(torch.tensor(vectorised, dtype=torch.long)),
            torch.tensor(label, dtype=torch.long)
        )