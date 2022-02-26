import json
import os
import _jsonnet
import torch
from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderItem
from ratsql.utils import registry
from roberta import *

exp_config = json.loads(_jsonnet.evaluate_file("experiments/spider-bert-run.jsonnet"))
model_config_path = os.path.join("/content/drive/MyDrive/RatSql-Colab/rat-sql", exp_config["model_config"])
model_config_args = exp_config.get("model_config_args")
infer_config = json.loads(_jsonnet.evaluate_file("configs/spider/nl2code-bert.jsonnet", tla_codes={'args': json.dumps(model_config_args)}))


inferer = Inferer(infer_config)
inferer.device = torch.device("cuda")
model = inferer.load_model("/content/drive/MyDrive/RatSql-Colab/rat-sql/logdir/bert_run/bs=2,lr=7.4e-04,bert_lr=1.0e-05,end_lr=0e0,att=1", 81000)
dataset = registry.construct('dataset', inferer.config['data']['val'])

for _, schema in dataset.schemas.items():
    model.preproc.enc_preproc._preprocess_schema(schema)

def question(q, db_id):
    spider_schema = dataset.schemas[db_id]
    data_item = SpiderItem(
        text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
        code=None,
        schema=spider_schema,
        orig_schema=spider_schema.orig,
        orig={"question": q}
    )
    model.preproc.clear_items()
    enc_input = model.preproc.enc_preproc.preprocess_item(data_item, None)
    preproc_data = enc_input, None
    with torch.no_grad():
        x = inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)
        return inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)


qt = input("Input your question: ")
db = input("Input your DB: ")
terminalQuery = question(qt, db)[0]["inferred_code"]
finalQuery = robertaQnA(terminalQuery, qt)
print(finalQuery)