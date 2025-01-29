
import json
from pathlib import Path
import os

tipi = ["num_verbale", "targa", "mail", "data", "cf_trasgressore", "cf_avvocato", "destinatario"]
tipi_articoli_comma = [ "articolo_violato", "comma"]
relazioni = ["data_infrazione", "data_notifica"]
relazioni_articoli_comma = ["specifica"]
articoli = ["7", "142", "146", "148", "157", "159", "171"]
motivazioni = ["segnaletica", "omologazione", "taratura", "prescrizione", "carenza_dati_verbale", "lettura_errata_targa", "possesso_autorizzazione", "altro"]

stats_template = {"match": 0,
         "tot_y": 0,
         "tot_pred": 0,
         "new": 0,
         "malformed_json_error": 0,
         "empty_json": 0,
         "generic_error": 0,
         "too_many_tokens_error": 0,
         "no_response_error": 0,
         }

stats_short_template = {"match": 0,
         "tot_y": 0,
         "tot_pred": 0,
         "new": 0,
         "empty_json": 0
         }

def clean_entities(enties_dict, is_only_articoli_comma):
    if is_only_articoli_comma:
        filtro = tipi_articoli_comma
    else:
        filtro = tipi

    entities_clean = {}
    for key, value in enties_dict.items():
        if value.lower() in filtro:
            entities_clean[key] = value
                            
    return entities_clean


def clean_relations(relations, is_only_articoli_comma):
    if is_only_articoli_comma:
        filtro = relazioni_articoli_comma
    else:
        filtro = relazioni

    clean_relations = []
    for rel in relations:
        if rel["relation"].lower() in filtro:
            clean_relations.append(rel)
            
    return clean_relations


def final_stats(stats, output_file):
    try:
        stats["precision"] = stats["match"] / (stats["match"] + stats["new"])
    except ZeroDivisionError:
        stats["precision"] = "ZeroDivisionError"
    try:
        stats["recall"] = stats["match"] / stats["tot_y"]
    except ZeroDivisionError:
        stats["recall"] = "ZeroDivisionError"

    if stats["precision"] == "ZeroDivisionError" or stats["recall"] == "ZeroDivisionError":
        stats["f1"] = "non_calcolabile"
    else:
        stats["f1"] = 2 * (( stats["precision"] * stats["recall"]) / (stats["precision"] + stats["recall"]))

    save_stats(stats, output_file)
    return stats


def global_entities_scoring(labels, predictions, output_file):
    stats = stats_template.copy()
    for entities_y, entities_pred in zip(labels, predictions): # for each document
        entities_y = {key.lower(): value.lower() for key, value in entities_y.items()}
        stats["tot_y"] += len(entities_y)

        if isinstance(entities_pred, str) and "error" in entities_pred:
            if entities_pred == "malformed_json_error":
                stats["malformed_json_error"] +=1
            elif entities_pred == "too_many_tokens_error":
                stats["too_many_tokens_error"] +=  1
            elif entities_pred == "generic_error":
                stats["generic_error"] += 1
            elif entities_pred == "no_response_error":
                stats["no_response_error"] += 1 
        else:
            entities_pred = {key.lower(): value.lower() for key, value in entities_pred.items()}
            # entità da matchare, entità trovate ed entità matchate
            for key, value in entities_y.items():
                if key in entities_pred and entities_pred[key] == value:
                    stats["match"] += 1
            
            stats["tot_pred"] += len(entities_pred)

            # entitià non richieste
            if len(entities_pred) == 0:
                stats["empty_json"] += 1
            for key, value in entities_pred.items():
                if not ( key in entities_y and entities_y[key] == value):
                    stats["new"] += 1
    return final_stats(stats, output_file)


def global_relations_scoring(labels, predictions, output_file):
    stats = stats_template.copy()
    for relations_y, relations_pred in zip(labels, predictions):
        stats["tot_y"] += len(relations_y)

        if isinstance(relations_pred, str) and "error" in relations_pred:
            if relations_pred == "malformed_json_error":
                stats["malformed_json_error"] +=1
            elif relations_pred == "too_many_tokens_error":
                stats["too_many_tokens_error"] +=  1
            elif relations_pred == "generic_error":
                stats["generic_error"] += 1
            elif relations_pred == "no_response_error":
                stats["no_response_error"] += 1 
        else: 
            for relation_y in relations_y:
                if relation_y in  relations_pred:
                    stats["match"] += 1
            stats["tot_pred"] += len(relations_pred)
            if len(relations_pred) == 0:
                stats["empty_json"] += 1
            for relation in relations_pred:
                if not relation in relations_y:
                    stats["new"] += 1

    return final_stats(stats, output_file)

    
def global_clf_scoring(labels, predictions, output_dir, field_name):
    stats = stats_template.copy()
    for labs, preds in zip(labels, predictions):
        stats["tot_y"] += len(labs)
        if isinstance(preds, str) and "error" in preds:
            if preds == "malformed_json_error":
                stats["malformed_json_error"] +=1
            elif preds == "too_many_tokens_error":
                stats["too_many_tokens_error"] +=  1
            elif preds == "generic_error":
                stats["generic_error"] += 1
            elif preds == "no_response_error":
                stats["no_response_error"] += 1 
        else: 
            for lab in labs:
                for pred in preds:
                    if lab[field_name] == pred[field_name]:
                        stats["match"] += 1
                        break
            stats["tot_pred"] += len(preds)
            if len(preds) == 0:
                stats["empty_json"] += 1
            for pred in preds:
                found = False
                for lab in labs:
                    if lab[field_name] == pred[field_name]:
                        found = True
                if not found:
                    stats["new"] += 1

    if field_name == "cod_articolo_violato":
        return final_stats(stats, output_dir + "/metrics/global/global_stats_clf_a.json")
    elif field_name == "cod_motivazione":
        return final_stats(stats, output_dir + "/metrics/global/global_stats_clf_m.json")

    
def global_entities_scoring_noerror(labels, predictions, output_file):
    stats = stats_short_template.copy()
    for entities_y, entities_pred in zip(labels, predictions):        # for each document
        entities_y = {key.lower(): value.lower() for key, value in entities_y.items()}
        stats["tot_y"] += len(entities_y)
        entities_pred = {key.lower(): value.lower() for key, value in entities_pred.items()}
        # entità da matchare, entità trovate ed entità matchate
        for key, value in entities_y.items():
            if key in entities_pred and entities_pred[key] == value:
                stats["match"] += 1
        
        stats["tot_pred"] += len(entities_pred)

        # entitià non richieste
        if len(entities_pred) == 0:
            stats["empty_json"] += 1
        for key, value in entities_pred.items():
            if not ( key in entities_y and entities_y[key] == value):
                stats["new"] += 1

    return final_stats(stats, output_file)

       
def global_relations_scoring_noerror(labels, predictions, output_file):
    stats = stats_short_template.copy()
    for relations_y, relations_pred in zip(labels, predictions):
        stats["tot_y"] += len(relations_y)
        for relation in relations_y:
            if relation in relations_pred:
                stats["match"] += 1
        stats["tot_pred"] += len(relations_pred)

        if len(relations_pred) == 0:
            stats["empty_json"] += 1
        for relation in relations_pred:
            if not relation in relations_y:
                stats["new"] += 1

    return final_stats(stats, output_file)


def detailed_entities_scoring(labels, predictions, output_file, is_only_articoli_comma=False):
    stats = dict()
    if is_only_articoli_comma:
        filter = tipi_articoli_comma
    else:
        filter = tipi

    for tipo in filter:
        stats[tipo] = {"tot_y":0, "tot_pred": 0,  "match": 0, "new": 0}
        
    for tipo in filter:
        for entities_y, entities_pred in zip(labels, predictions):        # for each document
            entities_y = {key.lower(): value.lower() for key, value in entities_y.items()}

            for key, value in entities_y.items():
                if value == tipo:
                    stats[tipo]["tot_y"] += 1
            if not(isinstance(entities_pred, str) and "error" in entities_pred):  
                entities_pred = {key.lower(): value.lower() for key, value in entities_pred.items()}
                
                for key, value in entities_pred.items():
                    if value == tipo:
                        stats[tipo]["tot_pred"] += 1
                        
                        if not (key in entities_y and entities_y[key] == value):
                            stats[tipo]["new"] += 1   
                                      
                for key, value in entities_y.items():
                    if value == tipo:
                        if key in entities_pred and entities_pred[key] == value:
                            stats[tipo]["match"] += 1
     
    save_stats(stats, output_file)
    return stats


def detailed_relations_scoring(labels, predictions, output_file, is_only_articoli_comma=False):
    stats = dict()
    if is_only_articoli_comma:
        filter = relazioni_articoli_comma
    else:
        filter = relazioni

    for relazione in filter:
        stats[relazione] = {"tot_y": 0, "tot_pred": 0,  "match": 0, "new": 0}

    for relazione in filter:
        for relations_y, relations_pred in zip(labels, predictions):

            for relation in relations_y:
                if relation["relation"] == relazione:
                    stats[relazione]["tot_y"] += 1
            if not(isinstance(relations_pred, str) and "error" in relations_pred):

                for relation in relations_pred:
                    if relation["relation"] == relazione:
                        stats[relazione]["tot_pred"] += 1

                        if not relation in relations_y:
                            stats[relazione]["new"] += 1

                for relation in relations_y:
                    if relation["relation"] == relazione:
                        if relation in relations_pred:
                            stats[relazione]["match"] += 1
    save_stats(stats, output_file)
    return stats
        
       
def detailed_clf_a_scoring(labels, predictions, output_dir):
    stats = dict()
    for articolo in articoli:
        stats[articolo] = {"tot_y":0, "tot_pred": 0,  "match": 0, "new": 0}

    for articolo in articoli:
        for labs, preds in zip(labels, predictions):

            for lab in labs:
                if lab["cod_articolo_violato"] == articolo:
                    stats[articolo]["tot_y"] += 1 
            if not(isinstance(preds, str) and "error" in preds):

                for pred in preds:
                    if pred["cod_articolo_violato"] == articolo:
                        stats[articolo]["tot_pred"] += 1

                        found = False
                        for lab in labs:
                            if lab["cod_articolo_violato"] == pred["cod_articolo_violato"]:
                                found = True
                        if not found:
                            stats[articolo]["new"] += 1

                for lab in labs:
                    if lab["cod_articolo_violato"] == articolo:
                        for pred in preds:
                            if pred["cod_articolo_violato"] == lab["cod_articolo_violato"]:
                                stats[articolo]["match"] += 1
                                break
    save_stats(stats, output_dir + "/metrics/detailed/detailed_stats_clf_a.json")
    return stats
    
      
def detailed_clf_m_scoring(labels, predictions, output_dir):
    stats = dict()
    for motivazione in motivazioni:
        stats[motivazione] = {"tot_y":0, "tot_pred": 0,  "match": 0, "new": 0}

    for motivazione in motivazioni:
        for labs, preds in zip(labels, predictions):

            for lab in labs:
                if lab["cod_motivazione"] == motivazione:
                    stats[motivazione]["tot_y"] += 1 
            if not(isinstance(preds, str) and "error" in preds):

                for pred in preds:
                    if pred["cod_motivazione"] == motivazione:
                        stats[motivazione]["tot_pred"] += 1

                        found = False
                        for lab in labs:
                            if lab["cod_motivazione"] == pred["cod_motivazione"]:
                                found = True
                        if not found:
                            stats[motivazione]["new"] += 1

                for lab in labs:
                    if lab["cod_motivazione"] == motivazione:
                        for pred in preds:
                            if pred["cod_motivazione"] == lab["cod_motivazione"]:
                                stats[motivazione]["match"] += 1
                                break
    save_stats(stats, output_dir + "/metrics/detailed/detailed_stats_clf_m.json")
    return stats
  

def save_stats(stats, output_f):
    output_dir = Path(output_f).parent.absolute()
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
        
    with open(output_f, "w", encoding='utf-8') as json_file:
        json.dump(stats, json_file, indent=4)
        
       
        
def save(content, output_dir, f_name):
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
    
    if "json" in f_name:
        with open(output_dir + "/" + f_name, "w", encoding='utf-8') as json_file:
            json.dump(content, json_file, indent=4)
    else:
        with open(output_dir + "/" + f_name, "w", encoding='utf-8') as text_file:
            text_file.write(content)