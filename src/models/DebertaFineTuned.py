from transformers import AutoModelForSequenceClassification
import torch 


class DebertaClassifier(torch.nn.Module):
    def __init__(self, model_checkpoint: str, num_labels: int = 2, freeze_layers: bool = False, freeze_layers_num: int = 3):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

        #Default setting for such models is let all layers to be unfrozen
        if freeze_layers:
            #Free all layers 
            for param in self.model.deberta.parameters():
                param.requires_grad = False

            #Unfreeze some layers if you set freeze_layers_num 
            if freeze_layers_num > 0:
                for layer in self.model.deberta.encoder.layer[freeze_layers_num:]:
                    for param in layer.parameters():
                        param.requires_grad = True

                #If you unfrose some layers you need also to set the embedding, LayerNorm and rel_embed to True 
                for param in self.model.deberta.embeddings.parameters():
                    param.requires_grad = True
                    
                for param in self.model.deberta.encoder.LayerNorm.parameters():
                    param.requires_grad = True
                    
                for param in self.model.deberta.encoder.rel_embeddings.parameters():
                    param.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)







