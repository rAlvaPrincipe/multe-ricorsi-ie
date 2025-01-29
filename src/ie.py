from dataset import Dataset
from scoring import global_entities_scoring, detailed_entities_scoring, save, global_entities_scoring_noerror, global_relations_scoring, detailed_relations_scoring, global_relations_scoring_noerror, global_clf_scoring, detailed_clf_a_scoring, detailed_clf_m_scoring, clean_entities, clean_relations
import json
from confs import parse, build_conf, defaults_prompts
from normalizer import Normalizer
import os
from llms import Llms
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

llms = Llms()

def build_prompt(template_f, text):
    if os.path.isdir("prompt_templates"):
        template_f = "prompt_templates/" + template_f
    else:
        template_f = "../prompt_templates/" + template_f

    if ".txt" not in template_f:
        template_f += ".txt"
    with open(template_f, "r", encoding='utf-8') as f:
        template = f.read()

    if "llama" in template_f:
        return template + "\n" + text + "\n\n### Response:"
    else:
        return template + "\n" + text + "\n\nRisposta:"


def separate_NER_from_RE(pred):
    pred = pred.replace("\_", "_")
    if isinstance(pred, str) and " error " in pred:
        return pred, pred
    else:

        try:
            #entities =  json.loads(pred[0:pred.index("}")+1])
            if "entities:" in pred:
                entities = json.loads(pred[pred.index("entities:") + 10 :pred.index("}")+1])
            elif "\"entities\":" in pred:
                entities = json.loads(pred[pred.index("\"entities\":") + 12 :pred.index("}")+1])
            else:
                entities = "malformed_json_error"
        except:
            entities =  "malformed_json_error"

        try:
            if "relations:" in pred:
                relations = json.loads(pred[pred.index("relations") + 11 : pred.rfind("]")+1])
            elif "\"relations\":" in pred:
                relations = json.loads(pred[pred.index("\"relations\":") + 13 : pred.rfind("]")+1])
            else:
                relations = "malformed_json_error"
        except:
            relations = "malformed_json_error"
        return entities, relations


def separate_NER_from_RE_for_mixed_inference(pred):
    pred = pred.replace("\_", "_")
    if isinstance(pred, str) and "error" in pred:
        return pred, pred
    else:
        try:
            entities = json.loads(pred[pred.index('"entities":') + 16 : pred.index("]")+1])
            entities_out = []
            new_entities = []
            for el in entities:
                if "cod_articolo_violato" in el.keys():
                    if el["cod_articolo_violato"] not in entities_out:
                        new_entities.append(el)
                        entities_out.append(el["cod_articolo_violato"])
                if "comma" in el.keys():
                    if el["comma"] not in entities_out:
                        new_entities.append(el)
                        entities_out.append(el["comma"])

            entities = new_entities
        except:
            entities = "malformed_json_error"
        try:
            relations = json.loads(pred[pred.index("relations") + 12 :])
        except:
            relations = "malformed_json_error"
        return entities, relations

def normalize_llm_response(res):
    tipi = ["num_verbale", "targa", "mail", "data", "cf_trasgressore", "cf_avvocato", "destinatario", "articolo_violato", "comma"]
    if isinstance(res, dict):
        out = {}
        normalizer = Normalizer()
        for key, value in res.items():
            if value in tipi:
                out[normalizer.normalize(key, value)] = value
        return out
    else:
        return res


def normalize_re_llm_response(res):
    try:
        if isinstance(res, list):
            out = list()
            normalizer = Normalizer()
            for rel in res:
                rel_norm = {}
                rel_norm["source"] = normalizer.normalize(rel["source"], "num_verbale")
                rel_norm["relation"] = rel["relation"]
                rel_norm["target"] = normalizer.normalize(rel["target"], "data")
                out.append(rel_norm)
            return out
        else:
            return res
    except:
        return "malformed_json_error"


def evaluate_mixed(llm, X, Y, ids, output_dir, template_f):
    pred_entities, pred_relations, y_entities, y_relations, pred_entities_noerror, y_entities_noerror, pred_relations_noerror, y_relations_noerror = [], [], [], [], [], [], [], []
    for id, x, y in zip(ids, X, Y):
        res, prompt, pred_ent, pred_rel = inference_mixed(llm, x, template_f)

        y_ent = y["entities"]
        y_rel = y["relations"]
        y_ent = clean_entities(y_ent, True)
        y_rel = clean_relations(y_rel, True)

        pred_entities.append(pred_ent)
        pred_relations.append(pred_rel)
        y_entities.append(y_ent)
        y_relations.append(y_rel)

        if "error" not in pred_ent:
            y_entities_noerror.append(y_ent)
            pred_entities_noerror.append(pred_ent)

        if "error" not in pred_rel:
            y_relations_noerror.append(y_rel)
            pred_relations_noerror.append(pred_rel)

        new_ent = diff(y_ent, pred_ent, "entities", True)
        missed_ent = diff(y_ent, pred_ent,"entities", False)
        new_rel = diff(y_rel, pred_rel, "relations", True)
        missed_rel = diff(y_rel, pred_rel,"relations", False)

        save(prompt, output_dir + "/logs/" + id + "/mixed_art/", "prompt_mixed.txt")
        save(y_ent, output_dir + "/logs/" + id + "/mixed_art/", "gt_ner.json")
        save(pred_ent, output_dir + "/logs/" + id + "/mixed_art/", "pred_ner.json")
        save(y_rel, output_dir + "/logs/" + id + "/mixed_art/", "gt_re.json")
        save(pred_rel, output_dir + "/logs/" + id + "/mixed_art/" , "pred_re.json")
        save(new_ent, output_dir + "/logs/" + id + "/mixed_art/", "new_ner.json")
        save(missed_ent, output_dir + "/logs/" + id + "/mixed_art/", "missed_ner.json")
        save(new_rel, output_dir + "/logs/" + id + "/mixed_art/", "new_rel.json")
        save(missed_rel, output_dir + "/logs/" + id + "/mixed_art/", "missed_rel.json")
        save(res, output_dir + "/raw/" + id + "/", "mixed_art.txt")

    return pred_entities, pred_relations, y_entities, y_relations, pred_entities_noerror, y_entities_noerror, y_relations_noerror, pred_relations_noerror


def evaluate_ie(llm, X, Y, ids, output_dir, template_f):
    pred_entities, pred_relations, y_entities, y_relations, pred_entities_noerror, y_entities_noerror, pred_relations_noerror, y_relations_noerror = [], [], [], [], [], [], [], []
    for id, x, y in zip(ids, X, Y):
        res, prompt, pred_ent, pred_rel =  inference_ie(llm, x, template_f)

        y_ent = y["entities"]
        y_rel = y["relations"]
        y_ent = clean_entities(y_ent, False)
        y_rel = clean_relations(y_rel, False)

        pred_entities.append(pred_ent)
        pred_relations.append(pred_rel)
        y_entities.append(y_ent)
        y_relations.append(y_rel)

        if "error" not in pred_ent:
            y_entities_noerror.append(y_ent)
            pred_entities_noerror.append(pred_ent)

        if "error" not in pred_rel:
            y_relations_noerror.append(y_rel)
            pred_relations_noerror.append(pred_rel)

        new_ent = diff(y_ent, pred_ent, "entities", True)
        missed_ent = diff(y_ent, pred_ent,"entities", False)
        new_rel = diff(y_rel, pred_rel, "relations", True)
        missed_rel = diff(y_rel, pred_rel,"relations", False)

        save(prompt, output_dir + "/logs/" + id + "/ie/", "prompt_ie.txt")
        save(y_ent, output_dir + "/logs/" + id + "/ie/", "gt_ner.json")
        save(pred_ent, output_dir + "/logs/" + id + "/ie/", "pred_ner.json")
        save(y_rel, output_dir + "/logs/" + id + "/ie/", "gt_re.json")
        save(pred_rel, output_dir + "/logs/" + id + "/ie/" , "pred_re.json")
        save(new_ent, output_dir + "/logs/" + id + "/ie/", "new_ner.json")
        save(missed_ent, output_dir + "/logs/" + id + "/ie/", "missed_ner.json")
        save(new_rel, output_dir + "/logs/" + id + "/ie/" , "new_rel.json")
        save(missed_rel, output_dir + "/logs/" + id + "/ie/", "missed_rel.json")
        save(res, output_dir + "/raw/" + id + "/", "ie.txt")

    return pred_entities, pred_relations, y_entities, y_relations, pred_entities_noerror, y_entities_noerror, y_relations_noerror, pred_relations_noerror


def inference_mixed(llm, doc, template_f):
    prompt = build_prompt(template_f, doc)
    res = llms.ask(prompt, llm)
    pred_ent, pred_rel = separate_NER_from_RE_for_mixed_inference(res)
    pred_ent = normalize_llm_response(pred_ent)
    pred_rel = normalize_re_llm_response(pred_rel)

    # change format (da array di diz a un unico diz, togliendo anche la motivazione)
    tmp = {}
    for el in pred_ent:
        if "cod_articolo_violato" in el.keys():
            tmp[el["cod_articolo_violato"]] = "articolo_violato"
        if "comma" in el.keys():
            tmp[el["comma"]] = "comma"

    pred_ent = tmp

    # mantieni solo i tipi di interesse
    if "error" not in pred_ent:
        pred_ent = clean_entities(pred_ent, True)
    if "error" not in pred_rel:
        pred_rel = clean_relations(pred_rel, True)
    return res, prompt, pred_ent, pred_rel



def inference_ie(llm, doc, template_f):
    prompt = build_prompt(template_f, doc)
    res = llms.ask(prompt, llm)
    pred_ent, pred_rel = separate_NER_from_RE(res)
    pred_ent = normalize_llm_response(pred_ent)
    pred_rel = normalize_re_llm_response(pred_rel)

    # mantieni solo i tipi di interesse
    if "error" not in pred_ent:
        pred_ent = clean_entities(pred_ent, False)
    if "error" not in pred_rel:
        pred_rel = clean_relations(pred_rel, False)
    return res, prompt, pred_ent, pred_rel


def inference_clf_motivazione(llm, doc, template_f):
    prompt = build_prompt(template_f, doc)
    res = llms.ask(prompt, llm)

    try:
        pred = json.loads(res[res.index('"motivazioni":') + 14 : res.index("]")+1])

        motivazioni = []
        new_pred = []
        for el in pred:
            if el["cod_motivazione"] not in motivazioni:
                new_pred.append(el)
                motivazioni.append(el["cod_motivazione"])
        pred = new_pred
        return res, prompt, pred
    except:
        pred =  "malformed_json_error"
        return res, prompt, pred


def inference_clf_articolo(llm, doc, template_f):
    prompt = build_prompt(template_f, doc)
    res = llms.ask(prompt, llm)

    try:
        pred = json.loads(res[res.index('"articoli_violati":') + 19 : res.index("]")+1])

        articoli_violati = []
        new_pred = []
        for el in pred:
            if el["cod_articolo_violato"] not in articoli_violati:
                new_pred.append(el)
                articoli_violati.append(el["cod_articolo_violato"])
        pred = new_pred
        return res, prompt, pred


    except:
        pred = "malformed_json_error"
        return res, prompt, pred


def evaluate_clf_articolo(llm, X, Y, ids, output_dir, template_f):
    preds, preds_noerror, Y_noerror = [], [], []
    for id, x, y in zip(ids, X, Y):
        res, prompt, pred = inference_clf_articolo(llm, x, template_f)

        if "error" not in pred:
            Y_noerror.append(pred)
            preds_noerror.append(pred)

        new_pred = diff(y, pred, "clf_articoli", True)
        missed_pred = diff(y, pred,"clf_articoli", False)

        preds.append(pred)
        save(y, output_dir + "/logs/" + id + "/clf_a/", "gt_clf_a.json")
        save(pred, output_dir + "/logs/" + id + "/clf_a/", "pred_clf_a.json")
        save(prompt, output_dir + "/logs/" + id + "/clf_a/", "prompt_clf_a.txt")
        save(new_pred, output_dir + "/logs/" + id + "/clf_a/", "new_clf_a.json")
        save(missed_pred, output_dir + "/logs/" + id + "/clf_a/", "missed_clf_a.json")
        save(res, output_dir + "/raw/" + id + "/", "clf_a.txt")

    return preds, Y, preds_noerror, Y_noerror


def evaluate_clf_motivazione(llm, X, Y, ids, output_dir, template_f):
    preds, preds_noerror, Y_noerror = [], [], []
    for id, x, y in zip(ids, X, Y):
        res, prompt, pred = inference_clf_motivazione(llm, x, template_f)
        if "error" not in pred:
            Y_noerror.append(pred)
            preds_noerror.append(pred)

        new_pred = diff(y, pred, "clf_motivazioni", True)
        missed_pred = diff(y, pred,"clf_motivazioni", False)

        preds.append(pred)
        save(y, output_dir + "/logs/" + id + "/clf_m/", "gt_clf_m.json")
        save(pred, output_dir + "/logs/" + id  + "/clf_m/", "pred_clf_m.json")
        save(prompt, output_dir + "/logs/" + id  + "/clf_m/", "prompt_clf_m.txt")
        save(new_pred, output_dir + "/logs/" + id  + "/clf_m/", "new_clf_m.json")
        save(missed_pred, output_dir + "/logs/" + id  + "/clf_m/", "missed_clf_m.json")
        save(res, output_dir + "/raw/" + id + "/", "clf_m.txt")

    return preds, Y, preds_noerror, Y_noerror



# questo metodo calcola le differenze tra le predizioni e la ground truth. il flag "is_new" serve per indicare se ritornare i documenti in più o quelli mancati.
def diff(y, pred, type, is_new):
    if type == "clf_motivazioni":
        tmp = list()
        for el in pred:
            if not (isinstance(el, str) and "error" in el):
                tmp.append({"cod_motivazione": el["cod_motivazione"]})
        pred = tmp
    if type == "clf_articoli":
        tmp = list()
        for el in pred:
            if not (isinstance(el, str) and "error" in el):
                tmp.append({"cod_articolo_violato": el["cod_articolo_violato"]})
        pred = tmp


    difference = {}

    if type == "entities":
        if is_new:
            if "error" not in pred:
                for key in pred:
                    if key not in y:
                        difference[key] = pred[key]
                    else:
                        if y[key] != pred[key]:
                            difference[key] = pred[key]
        else:
            if "error" not in pred:
                for key in y:
                    if key not in pred:
                        difference[key] = y[key]
                    else:
                        if y[key] != pred[key]:
                            difference[key] = y[key]
    elif type == "relations" or type == "clf_articoli" or type == "clf_motivazioni":
        difference = list()
        if is_new:
            for el_pred in pred:
                exist = False
                for el_y in y:
                    if el_pred == el_y:
                        exist = True
                if not exist:
                    difference.append(el_pred)
        if not is_new:
            for el_y in y:
                exist = False
                for el_pred in pred:
                    if el_pred == el_y:
                        exist = True
                if not exist:
                    difference.append(el_y)
    return difference



def extract(llm, doc):
    out = {}
    templates = defaults_prompts()

    if templates["ie"]:
        _, _, pred_ent, pred_rel = inference_ie(llm, doc, templates["ie"])
        out["ie"] = {"entities": pred_ent, "relations": pred_rel}
    if templates["mixed_articolo"]:
        _, _, pred_ent, pred_rel = inference_mixed(llm, doc, templates["mixed_articolo"])
        out["ie_clf_mix_articolo"] = {"entities": pred_ent, "relations": pred_rel}
    if templates["clf_motivazione"]:
        _, _, pred = inference_clf_motivazione(llm, doc, templates["clf_motivazione"])
        out["clf_motivazioni"] = pred
    if templates["clf_articolo"]:
        _, _, pred = inference_clf_articolo(llm, doc, templates["clf_articolo"])
        out["clf_articoli"] = pred

    return out


if __name__ == "__main__":
    args = parse()
    conf = build_conf(args)
    dataset = Dataset()

    if "unsloth" in conf["llm"]:
        ft_conf = None
        if "lora_model" in conf["llm"]:
            with open(conf["llm"] + "/../conf.json", 'r') as file:
                ft_conf = json.load(file)
        llms.load_local_llm(conf["llm"], conf["max_seq_length"], conf["constraints"], ft_conf)

    if conf["templates"]["ie"]:
        X, Y, ids = dataset.get_dataset_ie(conf["dataset"])
        pred_entities, pred_relations, y_entities, y_relations, pred_entities_noerror, y_entities_noerror, y_relations_noerror, pred_relations_noerror = evaluate_ie(conf["llm"], X, Y, ids, conf["output_dir"], conf["templates"]["ie"])

        global_entities_scoring(y_entities, pred_entities, conf["output_dir"] + "/metrics/global/global_stats.json" )
        detailed_entities_scoring(y_entities, pred_entities, conf["output_dir"] + "/metrics/detailed/detailed_stats.json")
        global_entities_scoring_noerror(y_entities_noerror, pred_entities_noerror, conf["output_dir"] + "/metrics/global/global_stats_noerror.json")

        global_relations_scoring(y_relations, pred_relations, conf["output_dir"] + "/metrics/global/global_stats_re.json")
        detailed_relations_scoring(y_relations, pred_relations, conf["output_dir"] + "/metrics/detailed/detailed_stats_re.json")
        global_relations_scoring_noerror(y_relations_noerror, pred_relations_noerror, conf["output_dir"] + "/metrics/global/global_stats_noerror_re.json")

    if conf["templates"]["clf_articolo"]:
        X, Y, ids = dataset.get_dataset_clf("articoli_violati", conf["dataset"])
        preds, Y, preds_noerror, Y_noerror = evaluate_clf_articolo(conf["llm"], X, Y, ids, conf["output_dir"], conf["templates"]["clf_articolo"])

        global_clf_scoring(Y, preds, conf["output_dir"], "cod_articolo_violato")
        detailed_clf_a_scoring(Y, preds, conf["output_dir"])

    if conf["templates"]["clf_motivazione"]:
        X, Y, ids = dataset.get_dataset_clf("motivazioni", conf["dataset"])
        preds, Y, preds_noerror, Y_noerror = evaluate_clf_motivazione(conf["llm"], X, Y, ids, conf["output_dir"], conf["templates"]["clf_motivazione"])

        global_clf_scoring(Y, preds, conf["output_dir"], "cod_motivazione")
        detailed_clf_m_scoring(Y, preds, conf["output_dir"])

    if conf["templates"]["mixed_articolo"]:
        X, Y, ids = dataset.get_dataset_articoli_mixed("motivazioni", conf["dataset"])
        pred_entities, pred_relations, y_entities, y_relations, pred_entities_noerror, y_entities_noerror, y_relations_noerror, pred_relations_noerror = evaluate_mixed(conf["llm"], X, Y, ids, conf["output_dir"], conf["templates"]["mixed_articolo"])

        global_entities_scoring(y_entities, pred_entities, conf["output_dir"] + "/metrics/global/global_stats_art.json" )
        detailed_entities_scoring(y_entities, pred_entities, conf["output_dir"] + "/metrics/detailed/detailed_stats_art.json", True)
        global_entities_scoring_noerror(y_entities_noerror, pred_entities_noerror, conf["output_dir"] + "/metrics/global/global_stats_noerror_art.json")

        global_relations_scoring(y_relations, pred_relations, conf["output_dir"] + "/metrics/global/global_stats_art_re.json")
        detailed_relations_scoring(y_relations, pred_relations, conf["output_dir"] + "/metrics/detailed/detailed_stats_art_re.json", True)
        global_relations_scoring_noerror(y_relations_noerror, pred_relations_noerror, conf["output_dir"] + "/metrics/global/global_stats_noerror_art_re.json")