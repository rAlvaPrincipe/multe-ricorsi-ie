from annotations import get_annotations, get_annotations_clf
import os
import hashlib
import json


class Dataset:
    prompt_set = ["deed15e959c0735198276d6998d4e6f9"]
    validation_set = ["4232f3cf40e138828d7ecb31911caa99", "c7afff007fe233cd56440a6c366613c9", "1bc317be515cf46f1120a0becd606124", "e1c18da0691e992c26f48f53fd051181", "c4067aae1e67a7005c2c6445e39e6898", "083fd7f4dd147537cff0d3f7936e37c2", "dfd39b0603d0b1cd64de69d32aa8fcdc"]
    
    def __init__(self):
        annotations, annotations_clean = get_annotations('data/doccano_annotated/admin.jsonl')
        self.annotations = annotations_clean
        self.annotations_articoli = get_annotations_clf('data/doccano_annotated/articoli_violati.jsonl', "articoli_violati")
        self.annotations_motivazioni = get_annotations_clf('data/doccano_annotated/motivazioni.jsonl', "motivazioni")

        self.annotations = sorted( self.annotations, key=lambda x: x['md5_text'])
        self.annotations_articoli = sorted( self.annotations_articoli, key=lambda x: x['md5_text'])
        self.annotations_motivazioni = sorted( self.annotations_motivazioni, key=lambda x: x['md5_text'])


    def get_dataset_articoli_mixed(self, dataset, split):
        X, Y, ids = [], [], []

        for el, el_art in zip(self.annotations, self.annotations_articoli):
            if el["md5_text"] != el_art["md5_text"]:
                raise Exception("annotations e annotations_articoli non hanno gli stessi documenti.")

        if split == "validation":
            for el, el_art in zip(self.annotations, self.annotations_articoli):
                if el["md5_text"] in self.prompt_set:
                    if "articolo_violato" not in el["entities"].values():
                        for articolo in el_art["label"]:
                            el["entities"][list(articolo.values())[0]] = "cod_articolo_violato"

                    Y.append({"entities": el["entities"], "relations": el["relations"]})
                    X.append(el["text"])
                    ids.append(el["md5_text"])

            for el, el_art in zip(self.annotations, self.annotations_articoli):
                if el["md5_text"] in self.validation_set:
                    if "articolo_violato" not in el["entities"].values():
                        for articolo in el_art["label"]:
                            el["entities"][list(articolo.values())[0]] = "cod_articolo_violato"
                    Y.append({"entities": el["entities"], "relations": el["relations"]})
                    X.append(el["text"])
                    ids.append(el["md5_text"])

        elif split == "test":
            for el, el_art in zip(self.annotations, self.annotations_articoli):
                if not(el["md5_text"] in self.validation_set or el["md5_text"] in  self.prompt_set):
                    if "articolo_violato" not in el["entities"].values():
                        for articolo in el_art["label"]:
                            el["entities"][list(articolo.values())[0]] = "cod_articolo_violato"
                    X.append(el["text"])
                    ids.append(el["md5_text"])
                    Y.append({"entities": el["entities"], "relations": el["relations"]})

        return X, Y, ids



    def get_dataset_clf(self, dataset, split):
        if dataset == "articoli_violati":
            annotations = self.annotations_articoli
        elif dataset == "motivazioni":
            annotations = self.annotations_motivazioni
         
        X, Y, ids = [], [], []   
        if split == "validation":
            for el in annotations:
                if el["md5_text"] in  self.prompt_set:
                    X.append(el["text"])
                    Y.append(el["label"])
                    ids.append(el["md5_text"])
            for el in annotations:
                if el["md5_text"] in  self.validation_set:
                    X.append(el["text"])
                    Y.append(el["label"])
                    ids.append(el["md5_text"])
        elif split == "test":
            for el in annotations:  
                if not(el["md5_text"] in self.validation_set or el["md5_text"] in  self.prompt_set):
                    X.append(el["text"])
                    Y.append(el["label"])
                    ids.append(el["md5_text"])
        return X, Y, ids


    def get_dataset_ie(self, split):
        X, Y, ids = [], [], []
        
        if split == "validation":
            for el in self.annotations:
                if el["md5_text"] in  self.prompt_set:
                    X.append(el["text"])
                    Y.append({"entities": el["entities"], "relations": el["relations"]})
                    ids.append(el["md5_text"])
            
            for el in self.annotations:
                if el["md5_text"] in  self.validation_set:
                    X.append(el["text"])
                    Y.append({"entities": el["entities"], "relations": el["relations"]})
                    ids.append(el["md5_text"])
        
        elif split == "test":
            for el in self.annotations:
                if not(el["md5_text"] in self.validation_set or el["md5_text"] in  self.prompt_set):
                    X.append(el["text"])
                    Y.append({"entities": el["entities"], "relations": el["relations"]})               
                    ids.append(el["md5_text"])

        elif split == "chatgpt" or split == "claude-3-opus" or split == "claude-3.5-sonnet" or split == "claude-3-sonnet":
            print(os.getcwd())
            text_folder = 'data/synthetic_data/text/' + split
            for file in os.listdir(text_folder):
                with open(text_folder + "/" + file, "r", encoding='utf-8') as f:
                    doc = f.read()
                    X.append(doc)
                    ids.append(hashlib.md5(doc.encode()).hexdigest())

            label_folder = 'data/synthetic_data/labels/' + split
            for file in os.listdir(label_folder):
                with open(label_folder + "/" + file, "r", encoding='utf-8') as f:
                    text = f.read()
                    entities =  json.loads(text[text.index("entities") + 11 :text.index("}")+1])
                    relations =  json.loads(text[text.index("relations") + 12 :])
                    Y.append({"entities": entities, "relations": relations})
            
        return X, Y, ids


    def get_prompt_set(self):
        out = list()
        for el in self.annotations:
            if el["md5_text"] in  self.prompt_set:
                out.append(el)
        return out
    
    
    def get_validation_set(self):
        out = list()
        for el in self.annotations:
            if el["md5_text"] in  self.validation_set:
                out.append(el)
        return out
        
    
    def get_test_set(self):
        out = list()
        for el in self.annotations:
            if not(el["md5_text"] in self.validation_set or el["md5_text"] in  self.prompt_set):
                out.append(el)    
        return out

    
    def search(self, s):
        for annotation in self.annotations:
            if s in annotation["text"]:
                print(annotation["md5_text"])




    def get_ft_dataset(self, include_synthetic =True, include_validation=True):
        from datasets import Dataset as ds
        data = {"input": [], "output": [], "instruction": []}

        text_folder = "data/synthetic_data/text"
        labels_folder = "data/synthetic_data/labels"
        entries = os.listdir(text_folder)
        models_list = [entry for entry in entries if os.path.isdir(os.path.join(text_folder, entry))]
        with open("data/synthetic_data/instructions.txt", 'r', encoding='utf-8') as file:
            instruction = file.read()

        if include_synthetic:
            for model in models_list:
                for f in os.listdir(text_folder + "/" + model):
                    data["instruction"].append(instruction)
                    with open(text_folder + "/" + model + "/" + f, 'r', encoding='utf-8') as file:
                        text = file.read()
                        data["input"].append(text)

                for f in os.listdir(labels_folder + "/" + model):
                    with open(labels_folder + "/" + model + "/" + f, 'r', encoding='utf-8') as file:
                        label = file.read()

                        entities = json.loads(label[label.index("\"entities\":") + 12 :label.index("}")+1])
                        relations = json.loads(label[label.index("\"relations\":") + 13 : label.rfind("]")+1])
                        output = "\"entities\": " + json.dumps(entities, indent=3) + "\n\n" + "\"relations\": " + json.dumps(relations, indent=3)
                        data["output"].append(output)

        if include_validation:
            validation_set = self.get_validation_set() + self.get_prompt_set()
            for instance in validation_set:
                data["input"].append(instance["text"])
                data["instruction"].append(instruction)
                output = "\"entities\": " + json.dumps(instance["entities"], indent=3) + "\n\n" + "\"relations\": " + json.dumps(instance["relations"], indent=3)
                data["output"].append(output)

        return ds.from_dict(data)