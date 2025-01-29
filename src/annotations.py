import json
from normalizer import Normalizer
import hashlib


normalizer = Normalizer()

def get_annotations_clf(file_path, dataset):
    annotations = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data_new = {}
            data = json.loads(line)
            data_new["text"] = data["text"]
            md5 = hashlib.md5(data["text"].encode()).hexdigest()
            data_new["md5_text"] = md5
            # if len(data["label"]) != 0:
            if dataset == "articoli_violati":
                labels = []
                for el in data["label"]:
                    labels.append({"cod_articolo_violato": el})
                data_new["label"] = labels
            elif dataset == "motivazioni":
                labels = []
                for el in data["label"]:
                    labels.append({"cod_motivazione": el})
                data_new["label"] = labels
            annotations.append(data_new)
                
    return annotations
            
            


def get_annotations(file_path):
    annotations = make_explicit_and_create_id(file_path)
    annotations_clean = normalize(annotations)
    annotations_clean = clean_fields(annotations_clean)
    annotations_clean = deduplicate(annotations_clean)
    annotations_clean = change_format(annotations_clean)
    return annotations, annotations_clean


def make_explicit_and_create_id(file_path):
    annotations = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            for entity in data["entities"]:
                entity["entity"] = data["text"][entity["start_offset"]:entity["end_offset"]]
            
            for relation in data["relations"]:
                source = ""
                target = ""
                for entity in data["entities"]:
                    if entity["id"] == relation["from_id"]:
                        source = entity["entity"]
                        source_label = entity["label"]
                    if entity["id"] == relation["to_id"]:
                        target = entity["entity"]
                        source_target = entity["label"]
                relation["relation"] = {"source": source, "source_label": source_label, "relation": relation["type"], "target": target, "target_label": source_target}
            
            md5 = hashlib.md5(data["text"].encode()).hexdigest()
            data["md5_text"] = md5
            
            annotations.append(data)
            
    return annotations


def clean_fields(annotations):
    annotations_clean = []
    for annotation in annotations:
        entities = list()
        for entity in annotation["entities"]:
            entities.append({entity["entity"]: entity["label"]})
    
        relations = list()
        for relation in annotation["relations"]:
            relations.append({"source": relation["relation"]["source"], "relation": relation["relation"]["relation"], "target": relation["relation"]["target"]})
        
        annotations_clean.append({"md5_text": annotation["md5_text"], "text": annotation["text"], "entities": entities, "relations": relations})
        
    return annotations_clean


def normalize(annotations):
    for annotation in annotations:
        for entity in annotation["entities"]:
            ent = entity["entity"]
            label = entity["label"]
            entity["entity"] = normalizer.normalize(ent, label)
            
        for relation in annotation["relations"]:
            source = relation["relation"]["source"]
            source_label = relation["relation"]["source_label"]
            target = relation["relation"]["target"]
            target_label = relation["relation"]["target_label"]
            relation["relation"]["source"] = normalizer.normalize(source, source_label)
            relation["relation"]["target"] = normalizer.normalize(target, target_label)
    return annotations        
    


def deduplicate(annotations):
    for annotation in annotations:
        unique_entities = []
        for entity in annotation["entities"]:
            if entity not in unique_entities:
                unique_entities.append(entity)

        annotation["entities"] = unique_entities
        
        unique_relations = []
        for relation in annotation["relations"]:
            if relation not in unique_relations:
                unique_relations.append(relation)
        annotation["relations"] = unique_relations
  
    return annotations  
    

def change_format(annotations):
    for annotation in annotations:
        entities = {}   
        for entity in annotation["entities"]:
            key, value = next(iter(entity.items()))
            entities[key] = value
        annotation["entities"] = entities
    return annotations



# annotations, annotations_clean = get_annotations('data/doccano_annotated/admin.jsonl')
# for e1, e2 in zip(annotations, annotations_clean):
#     pprint.pprint(e2["md5_text"])
#     input()
    
    
    

