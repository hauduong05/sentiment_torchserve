from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import transformers
import os
import json
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info('Transformers version %s', transformers.__version__)

class ModelHandler(BaseHandler):

    def initialize(self, context):
      

        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        #logger.info(f'Using device {self.device}')

        model_file = self.manifest['model']['modelFile']
        model_path = os.path.join(model_dir, model_file)

        if os.path.isfile(model_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()
            #logger.info(f'Successfully loaded model from {model_file}')
        else:
            raise RuntimeError('Missing the model file')

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer is not None:
            logger.info('Successfully loaded tokenizer')
        else:
            raise RuntimeError('Missing tokenizer')

        mapping_file_path = os.path.join(model_dir, 'index_to_name.json')
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
            #logger.info('Successfully loaded mapping file')
        else:
            logger.warning('Mapping file is not detected')

        self.initialized = True

    def preprocess(self, requests):
     
        input_ids_batch = None
        attention_mask_batch = None 
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            
            input_text = input_text.get("input")

            inputs = self.tokenizer.encode_plus(
                    input_text,
                    max_length=50,
                    padding='max_length',
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )

            #logger.info('Tokenization process completed')

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat(
                        (attention_mask_batch, attention_mask), 0
                    )
        return (input_ids_batch, attention_mask_batch)

 
    def inference(self, inputs):
    
        input_ids_batch, attention_mask_batch = inputs
        inferences = []
        predictions = self.model(input_ids_batch, attention_mask_batch)[0]

        num_rows, _ = predictions.shape
        for i in range(num_rows):
            out = predictions[i].unsqueeze(0)
            y_hat = out.argmax(1).item()
            predicted_idx = str(y_hat)
            inferences.append(self.mapping[predicted_idx])
        #logger.info('Predictions successfully created.')

        return inferences

    def postprocess(self, outputs):
 
        #logger.info(f'PREDICTED LABELS: {outputs}')

        return outputs