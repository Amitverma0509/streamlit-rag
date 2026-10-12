"""Compare services/text_splitter.py with the original Node.js textSplitter.js.
Usage: python verify_splitter.py path/to/RAG/backend/src/services/textSplitter.js   (needs Node.js)"""
import json, random, subprocess, sys, tempfile, os
from services.text_splitter import split_text

random.seed(7)
W = ["alpha", "beta", "gamma", "delta", "model", "training", "data", "regularization", "overfitting", "neural", "network", "é", "ü", "—"]
sent = lambda: " ".join(random.choice(W) for _ in range(random.randint(5, 25))) + "."
paras = [" ".join(sent() for _ in range(random.choice([1, 3, 8, 30]))) for _ in range(60)]
text = "\n\n".join(paras[:40]) + "\n" + "\n".join(paras[40:50]) + "\r\n\r\n" + "x" * 2500 + "\n\n" + "word " * 700 + "\n\n" + " ".join(paras[50:])
settings = [(1000, 200), (500, 100), (300, 50), (1000, 0)]

js_path = os.path.abspath(sys.argv[1])
tmp = tempfile.mkdtemp()
open(f"{tmp}/t.json", "w").write(json.dumps(text))
open(f"{tmp}/run.mjs", "w").write(f"""
import fs from 'fs'; import {{splitText}} from {json.dumps('file:///' + js_path.replace(os.sep, '/').lstrip('/'))};
const t = JSON.parse(fs.readFileSync({json.dumps(tmp.replace(os.sep, '/') + '/t.json')}, 'utf8')); const o = {{}};
for (const [s, v] of {json.dumps(settings)}) o[s + '-' + v] = splitText(t, {{chunkSize: s, chunkOverlap: v}});
fs.writeFileSync({json.dumps(tmp.replace(os.sep, '/') + '/o.json')}, JSON.stringify(o));""")
subprocess.run(["node", f"{tmp}/run.mjs"], check=True)
node = json.load(open(f"{tmp}/o.json"))

print(f"{'chunk_size-overlap':<21}{'Node.js chunks':<17}{'Python chunks':<16}Result")
for s, v in settings:
    py, js = split_text(text, s, v), node[f"{s}-{v}"]
    print(f"{f'{s}-{v}':<21}{len(js):<17}{len(py):<16}{'IDENTICAL' if py == js else 'DIFFERENT'}")
