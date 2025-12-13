import json
import pdb
import random

with open('/home/a1/sdb/lb/Mytest/src/data_synthesis/diverse_entities_wikidata_en.jsonl') as f:
    dataset = [json.loads(line) for line in f]

results = []
for data in dataset:
    ent = data['name']
    desc = data['description']
    s = f'{ent}: {desc}'
    results.append(s)

random.shuffle(results)
results = results[:5000]
with open('//home/a1/sdb/lb/Mytest/src/data_synthesis/example_seed_entities_0-5k.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

