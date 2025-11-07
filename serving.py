from flask import Flask, request, jsonify, render_template_string
import numpy as np
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
import logging
import time
from sampling import autoregressive_sampling, speculative_sampling_bass_pad
import argparse
import subprocess, threading, json, re
from queue import Queue

# ===========================================================
# 全局状态
# ===========================================================
app = Flask(__name__)
log_queue = Queue()
simulation_running = False
process_thread = None
GLOBAL_SERVER = None

# ===========================================================
# 页面 HTML 模板
# ===========================================================
@app.route("/")
def home():
    return render_template_string("""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>BASS++ 推测推理演示系统</title>
<style>
  body{font-family:'Segoe UI',Arial,sans-serif;background:linear-gradient(135deg,#e8f5e9,#f9fff5);margin:0;padding:0;color:#1b5e20;}
  header{background:linear-gradient(135deg,#2e7d32,#1b5e20);color:#fff;text-align:center;padding:16px;border-bottom-left-radius:14px;border-bottom-right-radius:14px;}
  h1{margin:0;font-size:24px;}
  .tabs{text-align:center;margin:16px auto;}
  .tab-btn{background:#e8f5e9;border:2px solid #2e7d32;color:#2e7d32;padding:8px 18px;margin:0 6px;
           font-size:16px;font-weight:600;border-radius:8px;cursor:pointer;transition:.2s;}
  .tab-btn.active{background:#2e7d32;color:#fff;}
  .container{max-width:900px;margin:20px auto;background:white;border-radius:16px;box-shadow:0 4px 18px rgba(0,0,0,0.1);padding:40px;}
  textarea{width:100%;height:150px;padding:14px;font-size:16px;border:1px solid #ccc;border-radius:10px;resize:none;}
  button{background:linear-gradient(135deg,#43a047,#2e7d32);color:white;padding:10px 20px;font-size:16px;border:none;border-radius:8px;margin:8px;cursor:pointer;}
  button:hover{background:linear-gradient(135deg,#388e3c,#1b5e20);}
  .clear-btn{background:#ef5350;} .clear-btn:hover{background:#c62828;}
  #status{text-align:center;color:#555;margin-top:10px;font-style:italic;}
  .output{background:#f1f8e9;border-left:5px solid #43a047;padding:16px;border-radius:10px;white-space:pre-wrap;font-size:15px;color:#2e7d32;}
  /* 模块二样式 */
  .bucket-card{background:#fff;border:1px solid #E0E0E0;border-radius:10px;padding:10px;box-shadow:0 6px 16px rgba(0,0,0,.04);}
  .bucket-title{font-weight:700;color:#2e7d32;margin-bottom:8px;font-size:14px}
  .dot-wrap{display:flex;gap:6px;flex-wrap:wrap;min-height:36px}
  .dot{width:10px;height:10px;border-radius:2px;background:#66bb6a;box-shadow:0 0 0 1px rgba(0,0,0,.06) inset;animation:pop .15s ease-out;}
  @keyframes pop { from { transform:scale(.6);opacity:.5 } to { transform:scale(1);opacity:1 } }
  .queue-item{background:#E8F5E9;border:1px solid #C8E6C9;border-radius:8px;padding:6px 10px;margin-bottom:6px;font-family:monospace;font-size:13px;color:#1b5e20;}
  .done-item{background:#F5F5F5;border:1px solid #E0E0E0;border-radius:8px;padding:6px 10px;margin-bottom:6px;font-family:monospace;font-size:13px;color:#424242;}
</style>
</head>
<body>
<header>
  <h1>🌿 BASS++ 推测推理演示系统</h1>
  <p style="margin:0;font-size:14px;opacity:.85;">模块一：推理对比　｜　模块二：动态分批可视化</p>
</header>

<div class="tabs">
  <button id="tab1" class="tab-btn active">模块一</button>
  <button id="tab2" class="tab-btn">模块二</button>
</div>

<!-- 模块一 -->
<div id="module1" style="display:block;">
  <div class="container">
    <h2>🧩 Baseline vs BASS 推理对比</h2>
    <textarea id="inputText" placeholder="例如：请解释 speculative decoding 的原理。"></textarea><br>
    <button onclick="generate()">🚀 生成</button>
    <button class="clear-btn" onclick="clearAll()">🧹 清除</button>
    <p id="status"></p>
    <div id="output" class="output"></div>
  </div>
</div>

<!-- 模块二 -->
<div id="module2" style="display:none;">
  <div class="container">
    <h2>📊 动态分批可视化</h2>
    <p>展示请求 → 分桶 → 批次形成 → 完成的全过程（同步 dynamic_batcher 输出）</p>
    <button id="startSimBtn">▶️ 开始模拟</button>
    <span id="simStatus" style="margin-left:10px;color:#555;">未开始</span>
    <div id="bucketPanel" style="display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:16px 0;"></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
      <div>
        <h3>📥 待推理队列</h3>
        <div id="queuePanel" style="min-height:120px;border:1px dashed #A5D6A7;border-radius:10px;padding:10px;background:#f6fbf2;"></div>
      </div>
      <div>
        <h3>✅ 已完成批次</h3>
        <div id="donePanel" style="min-height:120px;border:1px dashed #BDBDBD;border-radius:10px;padding:10px;background:#fafafa;"></div>
      </div>
    </div>
    <h3>🧾 日志输出</h3>
    <pre id="logBox" style="background:#f9f9f9;border:1px solid #ccc;border-radius:8px;padding:10px;height:260px;overflow-y:auto;font-size:13px;"></pre>
  </div>
</div>

<script>
// ======== 模块切换 ========
document.getElementById("tab1").onclick=()=>{switchTab("module1","tab1");};
document.getElementById("tab2").onclick=()=>{switchTab("module2","tab2");};
function switchTab(module,tab){
  document.getElementById("module1").style.display="none";
  document.getElementById("module2").style.display="none";
  document.getElementById("tab1").classList.remove("active");
  document.getElementById("tab2").classList.remove("active");
  document.getElementById(module).style.display="block";
  document.getElementById(tab).classList.add("active");
}

// ======== 模块一推理 ========
async function generate(){
  const text=document.getElementById("inputText").value.trim();
  const output=document.getElementById("output"),status=document.getElementById("status");
  if(!text){alert("请输入文本");return;}
  output.innerHTML="";status.textContent="⏳ 正在生成...";
  try{
    const res=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:text})});
    const result=await res.json();
    status.textContent="✅ 完成";
    output.innerHTML=`🧩 Baseline 延迟:${result.baseline.latency}s 吞吐:${result.baseline.throughput}tok/s
${result.baseline.text}
🌿 BASS 批大小:${result.bass.batch} 延迟:${result.bass.latency}s 吞吐:${result.bass.throughput}tok/s
${result.bass.text}
📈 吞吐提升:x${result.compare.throughput_gain} 延迟提升:x${result.compare.latency_gain}`;
  }catch(e){status.textContent="❌ 错误:"+e;}
}
function clearAll(){document.getElementById("inputText").value="";document.getElementById("output").textContent="";document.getElementById("status").textContent="";}

// ======== 模块二可视化 ========
const BUCKETS=["0-32","33-64","65-128","129-256","257-512","513-1024"];
const bucketDots={}; let queueState=[]; let evtSource=null;

function renderBuckets(){
  const panel=document.getElementById("bucketPanel");
  panel.innerHTML="";
  BUCKETS.forEach(b=>{
    const div=document.createElement("div");
    div.className="bucket-card";
    div.innerHTML=`<div class="bucket-title">${b}</div><div class="dot-wrap" id="dots-${b}"></div>`;
    panel.appendChild(div);
    bucketDots[b]=[];
  });
}

function addDot(bucket){
  const wrap=document.getElementById("dots-"+bucket);
  if(!wrap)return;
  const dot=document.createElement("div");
  dot.className="dot";
  wrap.appendChild(dot);
  bucketDots[bucket].push(dot);
}

function moveToQueue(bucket,batchSize){
  const qPanel=document.getElementById("queuePanel");
  const item=document.createElement("div");
  item.className="queue-item";
  item.textContent=`batch=${batchSize}, bucket=${bucket}`;
  qPanel.appendChild(item);
  queueState.push({bucket,size:batchSize,el:item});
}

function finishBatch(batchSize,line){
  const idx=queueState.findIndex(x=>x.size===batchSize);
  const donePanel=document.getElementById("donePanel");
  const done=document.createElement("div");
  done.className="done-item";
  if(idx>=0){
    done.textContent=`✅ ${queueState[idx].el.textContent}`;
    queueState[idx].el.remove();
    queueState.splice(idx,1);
  }else done.textContent=line;
  donePanel.appendChild(done);
}

document.getElementById("startSimBtn").onclick=async()=>{
  renderBuckets();
  queueState=[];
  document.getElementById("queuePanel").innerHTML="";
  document.getElementById("donePanel").innerHTML="";
  document.getElementById("logBox").textContent="";
  document.getElementById("simStatus").textContent="运行中...";
  await fetch("/start_simulation",{method:"POST"});
  if(evtSource)evtSource.close();
  evtSource=new EventSource("/log_stream");
  evtSource.onmessage=(e)=>{
    const data=JSON.parse(e.data);
    const line=data.log||"";
    const logBox=document.getElementById("logBox");
    logBox.textContent+=line+"\\n";
    logBox.scrollTop=logBox.scrollHeight;
    // 解析来自 dynamic_batcher 的日志
    if(line.includes("[REQ]")){
      const m=line.match(/len=(\\d+)/);
      const L=m?parseInt(m[1]):32;
      const bucket=BUCKETS.find(b=>{
        const[lo,hi]=b.split("-").map(Number);
        return L>=lo && L<=hi;
      })||"0-32";
      addDot(bucket);
    }
    if(line.includes("[ENQUEUE]")){
        const m=line.match(/batch=(\d+).*?avg_len=(\d+\.\d+)/);
        const size = m ? parseInt(m[1]) : 0;
        const avg = m ? m[2] : "?";
        const qPanel = document.getElementById("queuePanel");
        const item = document.createElement("div");
        item.className = "queue-item";
        item.textContent = `batch=${size}, avg_len=${avg}`;
        qPanel.appendChild(item);
        queueState.push({ size, el:item });
    }
    if(line.startsWith("[BATCH DONE]")){
      const m=line.match(/batch=(\\d+)/);
      finishBatch(m?parseInt(m[1]):0,line);
    }
    if(line.includes("结束")){
      document.getElementById("simStatus").textContent="✅ 已结束";
      evtSource.close();
    }
  };
};
</script>
</body>
</html>
""")

# ===========================================================
# 模型逻辑（模块一）
# ===========================================================
class Server:
    def __init__(self, approx_model_name, target_model_name, args=None):
        self._device = args.device if hasattr(args, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading models on {self._device} ...")
        dtype = torch.float16 if "cuda" in self._device else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
        self._small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, torch_dtype=dtype, device_map={"": self._device}, trust_remote_code=True)
        self._large_model = AutoModelForCausalLM.from_pretrained(target_model_name, torch_dtype=dtype, device_map={"": self._device}, trust_remote_code=True)
        self.num_tokens, self.top_k, self.top_p = 20, 40, 0.9

    def compare_baseline_vs_bass(self, prompt: str):
        input_ids = self._tokenizer.encode(prompt, return_tensors='pt').to(self._device)
        t0 = time.perf_counter()
        out_base = autoregressive_sampling(x=input_ids, model=self._large_model, N=self.num_tokens, top_k=self.top_k, top_p=self.top_p)
        t1 = time.perf_counter()
        base_text = self._tokenizer.decode(out_base[0], skip_special_tokens=True)
        base_time = t1 - t0
        base_tput = self.num_tokens / base_time if base_time > 0 else 0
        prefixes = input_ids.repeat(16, 1)
        t0 = time.perf_counter()
        out_bass, lengths = speculative_sampling_bass_pad(prefixes, self._small_model, self._large_model, max_new_tokens=self.num_tokens, gamma_init=4, top_k=self.top_k, top_p=self.top_p)
        t1 = time.perf_counter()
        bass_text = self._tokenizer.decode(out_bass[0, :int(lengths[0].item())], skip_special_tokens=True)
        bass_time = t1 - t0
        total_new = int(lengths.sum().item()) - prefixes.numel()
        bass_tput = total_new / bass_time if bass_time > 0 else 0
        return {
            "baseline": {"text": base_text, "latency": round(base_time, 3), "throughput": round(base_tput, 2)},
            "bass": {"text": bass_text, "latency": round(bass_time, 3), "throughput": round(bass_tput, 2), "batch": prefixes.size(0)},
            "compare": {
                "throughput_gain": round(bass_tput / base_tput, 2) if base_tput > 0 else 0,
                "latency_gain": round(base_time / bass_time, 2) if bass_time > 0 else 0
            }
        }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prompt = data.get('prompt', '')
    return jsonify(GLOBAL_SERVER.compare_baseline_vs_bass(prompt))

# ===========================================================
# 模块二：日志流接口
# ===========================================================
@app.route("/start_simulation", methods=["POST"])
def start_simulation():
    global simulation_running, process_thread
    if simulation_running:
        return jsonify({"status": "already_running"})
    simulation_running = True
    def run_and_capture():
        global simulation_running
        try:
            cmd = ["python", "sampling/dynamic_batcher.py", "--device", GLOBAL_SERVER._device, "--mode", "dynamic", "--max_samples", "120"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in iter(proc.stdout.readline, ''):
                if not line: continue
                log_queue.put(json.dumps({"log": line.strip()}))
            proc.stdout.close(); proc.wait()
        finally:
            simulation_running = False
            log_queue.put(json.dumps({"log": "[SIM] 结束"}))
    process_thread = threading.Thread(target=run_and_capture, daemon=True)
    process_thread.start()
    return jsonify({"status": "started"})

@app.route("/log_stream")
def log_stream():
    def event_stream():
        while True:
            try:
                msg = log_queue.get(timeout=1.0)
                yield f"data: {msg}\n\n"
            except:
                yield f": keep-alive {int(time.time())}\n\n"
    return app.response_class(event_stream(), mimetype="text/event-stream")

# ===========================================================
# 启动入口
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()
    GLOBAL_SERVER = Server(
        approx_model_name="../data/models/bloom-560m",
        target_model_name="../data/models/bloomz-7b1",
        args=args
    )
    app.run(host="0.0.0.0", port=5000)
