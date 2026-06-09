import os, json, time, re, requests, tempfile, shutil, statistics
from huggingface_hub import HfApi
from groq import Groq
from benchmark_data import TEST_CASES, REGEX_PATTERNS, LLAMA_SYSTEM

print("UEG Benchmark starting...")

HF_TOKEN     = os.environ["HF_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
UEG_ENDPOINT = "https://ueg-api.onrender.com"
HF_REPO      = "rufatronics/ueg-benchmark-results"
MODEL        = "llama-3.1-8b-instant"

api         = HfApi(token=HF_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)

try:
    api.repo_info(repo_id=HF_REPO, repo_type="dataset", token=HF_TOKEN)
except Exception:
    api.create_repo(repo_id=HF_REPO, repo_type="dataset", private=False, token=HF_TOKEN)

def regex_classify(text):
    t0 = time.perf_counter()
    for pattern, label, tier in REGEX_PATTERNS:
        if pattern.search(text):
            return {"label": label, "tier": tier,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 4), "matched": True}
    return {"label": "unknown", "tier": "?",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 4), "matched": False}

def llama_classify(text):
    t0 = time.perf_counter()
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": LLAMA_SYSTEM},
                {"role": "user", "content": text},
            ],
            temperature=0, max_tokens=30,
        )
        raw = re.sub(r'```json|```', '', resp.choices[0].message.content.strip()).strip()
        result = json.loads(raw)
        return {"label": result.get("label", "unknown"), "tier": result.get("tier", "?"),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "error": None}
    except Exception as e:
        return {"label": "error", "tier": "?",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "error": str(e)}

def wake_ueg():
    print("[UEG] Waking Render instance...")
    for attempt in range(15):
        try:
            r = requests.get(f"{UEG_ENDPOINT}/health", timeout=15)
            if r.status_code == 200 and r.json().get("ready"):
                print(f"[UEG] Ready after {attempt+1} pings")
                break
        except Exception:
            pass
        time.sleep(10)
    print("[UEG] Warm-up classify call...")
    for _ in range(5):
        try:
            r = requests.post(f"{UEG_ENDPOINT}/classify", json={"text": "hello"}, timeout=60)
            if r.status_code == 200:
                print("[UEG] Warm-up done")
                return True
        except Exception:
            pass
        time.sleep(10)
    return False

def ueg_classify(text):
    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            r = requests.post(f"{UEG_ENDPOINT}/classify", json={"text": text}, timeout=30)
            if r.status_code == 200:
                data = r.json()
                return {
                    "label": data["intent_class_label"], "tier": data["tier"],
                    "confidence": data["confidence_intent"],
                    "resource_class": data["resource_class"],
                    "language_iso": data["language_iso"],
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "api_latency_ms": data["latency_ms"], "error": None,
                }
            time.sleep(3)
        except Exception as e:
            print(f"[UEG] Attempt {attempt+1} failed: {e}")
            time.sleep(8)
    return {"label": "error", "tier": "?", "confidence": 0.0,
            "resource_class": "?", "language_iso": "?",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "api_latency_ms": 0, "error": "All attempts failed"}

wake_ueg()
time.sleep(3)

categories  = ["easy", "ambiguous", "noisy", "non_english", "adversarial", "tier5"]
cat_stats   = {c: {"ueg": [0,0], "regex": [0,0], "llama": [0,0]} for c in categories}
results     = []
ueg_correct = regex_correct = llama_correct = 0
ueg_total   = regex_total   = llama_total   = 0

print(f"\nRunning {len(TEST_CASES)} test cases...\n")

for i, (text, expected_label, expected_tier, category) in enumerate(TEST_CASES):
    print(f"[{i+1:2d}/{len(TEST_CASES)}] [{category:<12}] {text[:55]}")
    ueg   = ueg_classify(text)
    rx    = regex_classify(text)
    time.sleep(1.5)
    llama = llama_classify(text)

    ueg_ok   = ueg["label"]   == expected_label
    rx_ok    = rx["label"]    == expected_label
    llama_ok = llama["label"] == expected_label

    if ueg["error"] is None:
        ueg_total += 1; ueg_correct += ueg_ok
        cat_stats[category]["ueg"][0] += ueg_ok
        cat_stats[category]["ueg"][1] += 1
    if rx["matched"]:
        regex_total += 1; regex_correct += rx_ok
        cat_stats[category]["regex"][0] += rx_ok
        cat_stats[category]["regex"][1] += 1
    llama_total += 1; llama_correct += llama_ok
    cat_stats[category]["llama"][0] += llama_ok
    cat_stats[category]["llama"][1] += 1

    results.append({
        "id": i+1, "text": text, "category": category,
        "expected_label": expected_label, "expected_tier": expected_tier,
        "ueg":   {"label": ueg["label"], "correct": ueg_ok,
                   "confidence": ueg.get("confidence", 0),
                   "resource_class": ueg.get("resource_class", "?"),
                   "language_iso": ueg.get("language_iso", "?"),
                   "api_latency_ms": ueg.get("api_latency_ms", 0),
                   "error": ueg.get("error")},
        "regex": {"label": rx["label"], "correct": rx_ok,
                   "matched": rx["matched"], "latency_ms": rx["latency_ms"]},
        "llama": {"label": llama["label"], "correct": llama_ok,
                   "latency_ms": llama["latency_ms"], "error": llama.get("error")},
    })
    tu = "v" if ueg_ok   else "x"
    tr = "v" if rx_ok    else ("x" if rx["matched"] else "-")
    tl = "v" if llama_ok else "x"
    print(f"  UEG={tu}({ueg['label']}) | Regex={tr}({rx['label']}) | Llama={tl}({llama['label']})")

def avg(lst): return round(statistics.mean(lst), 2) if lst else 0
def pct(c, t): return round(c / max(t, 1) * 100, 1)

ueg_acc   = pct(ueg_correct, ueg_total)
regex_acc = pct(regex_correct, regex_total)
llama_acc = pct(llama_correct, llama_total)
ueg_lats   = [r["ueg"]["api_latency_ms"] for r in results if not r["ueg"]["error"] and r["ueg"]["api_latency_ms"] > 0]
llama_lats = [r["llama"]["latency_ms"]   for r in results if not r["llama"]["error"]]
regex_lats = [r["regex"]["latency_ms"]   for r in results]

print(f"\n{'='*62}")
print(f"  BENCHMARK — {len(TEST_CASES)} test cases")
print(f"{'='*62}")
print(f"  UEG   : {ueg_acc}% ({ueg_correct}/{ueg_total}) | {avg(ueg_lats):.1f}ms avg")
print(f"  Regex : {regex_acc}% on matched ({regex_correct}/{regex_total} of {len(TEST_CASES)}) | {avg(regex_lats):.4f}ms")
print(f"  Llama : {llama_acc}% ({llama_correct}/{llama_total}) | {avg(llama_lats):.0f}ms avg")
print(f"{'='*62}")
for cat in categories:
    s  = cat_stats[cat]
    up = pct(s["ueg"][0], s["ueg"][1])
    rp = f"{pct(s['regex'][0],s['regex'][1])}%({s['regex'][1]}/{s['ueg'][1]})" if s["regex"][1]>0 else f"N/A(0/{s['ueg'][1]})"
    lp = pct(s["llama"][0], s["llama"][1])
    print(f"  {cat:<14} UEG={up}% | Regex={rp} | Llama={lp}%")

summary = {
    "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "test_cases": len(TEST_CASES), "ueg_endpoint": UEG_ENDPOINT, "llama_model": MODEL,
    "overall": {
        "ueg":   {"correct": ueg_correct, "total": ueg_total, "accuracy": ueg_acc, "avg_latency_ms": avg(ueg_lats)},
        "regex": {"correct": regex_correct, "matched": regex_total, "total": len(TEST_CASES),
                   "coverage_pct": pct(regex_total, len(TEST_CASES)),
                   "accuracy_on_matched": regex_acc, "avg_latency_ms": avg(regex_lats)},
        "llama": {"correct": llama_correct, "total": llama_total, "accuracy": llama_acc, "avg_latency_ms": avg(llama_lats)},
    },
    "by_category": {
        cat: {
            "ueg_accuracy":   pct(cat_stats[cat]["ueg"][0],   cat_stats[cat]["ueg"][1]),
            "regex_accuracy": pct(cat_stats[cat]["regex"][0], cat_stats[cat]["regex"][1]) if cat_stats[cat]["regex"][1]>0 else None,
            "regex_coverage": f"{cat_stats[cat]['regex'][1]}/{cat_stats[cat]['ueg'][1]}",
            "llama_accuracy": pct(cat_stats[cat]["llama"][0], cat_stats[cat]["llama"][1]),
        } for cat in categories
    },
    "detailed_results": results,
}

tmp = tempfile.mkdtemp()
try:
    with open(os.path.join(tmp, "benchmark_latest.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(tmp, f"benchmark_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    lines = [
        "# UEG Benchmark Results\n\n",
        "Auto-generated: UEG vs Regex vs Llama-3.1-8B\n\n",
        f"## Overall ({summary['run_at'][:10]})\n\n",
        "| Classifier | Accuracy | Coverage | Avg Latency |\n",
        "|------------|----------|----------|-------------|\n",
        f"| **UEG** | **{ueg_acc}%** | **{ueg_total}/{len(TEST_CASES)}** | **{avg(ueg_lats):.1f}ms** |\n",
        f"| Regex | {regex_acc}% on matched | {regex_total}/{len(TEST_CASES)} ({pct(regex_total,len(TEST_CASES))}%) | {avg(regex_lats):.4f}ms |\n",
        f"| Llama-3.1-8B | {llama_acc}% | {llama_total}/{len(TEST_CASES)} | {avg(llama_lats):.0f}ms |\n\n",
        "## By Category\n\n",
        "| Category | UEG | Regex (coverage) | Llama |\n",
        "|----------|-----|-----------------|-------|\n",
    ]
    for cat in categories:
        s  = cat_stats[cat]
        up = pct(s["ueg"][0], s["ueg"][1])
        rp = f"{pct(s['regex'][0],s['regex'][1])}% ({s['regex'][1]}/{s['ueg'][1]})" if s["regex"][1]>0 else f"N/A (0/{s['ueg'][1]})"
        lp = pct(s["llama"][0], s["llama"][1])
        lines.append(f"| {cat} | {up}% | {rp} | {lp}% |\n")
    lines += [
        f"\nTest cases: {len(TEST_CASES)} | Endpoint: {UEG_ENDPOINT}\n\n",
        "## Why UEG\n\n",
        "- **Coverage**: Regex fails on non-English, noisy text, and Tier 5 inputs entirely\n",
        "- **Accuracy**: Regex gets ambiguous cases wrong — same words, different intent\n",
        f"- **Speed**: UEG is {round(avg(llama_lats)/max(avg(ueg_lats),0.1))}x faster than Llama for classification\n",
        "- **Output**: UEG adds resource density + language ISO — Llama gives none of that\n",
        "- **Cost**: UEG runs at near-zero cost vs burning frontier tokens on every Hi\n",
    ]
    with open(os.path.join(tmp, "README.md"), "w") as f:
        f.writelines(lines)
    api.upload_folder(
        folder_path=tmp, repo_id=HF_REPO, repo_type="dataset",
        commit_message=f"Benchmark {summary['run_at'][:10]} UEG={ueg_acc}% Regex={regex_acc}%matched Llama={llama_acc}%",
    )
    print(f"\n[HF] https://huggingface.co/datasets/{HF_REPO}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
