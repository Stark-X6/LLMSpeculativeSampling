from flask import Flask, request, jsonify, render_template_string
import numpy as np
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time
from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2, speculative_sampling_bass_pad
import argparse
import subprocess, threading, json
from queue import Queue

log_queue = Queue()
process_thread = None
simulation_running = False
app = Flask(__name__)
@app.route("/")
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>BASS++ 推测推理演示系统</title>
    <style>
      body {
          font-family: 'Segoe UI', Arial, sans-serif;
          background: linear-gradient(135deg, #e8f5e9, #f9fff5);
          margin: 0; padding: 0;
          color: #1b5e20;
          font-size: 18px;           /* ⬆️ 全局字体更大 */
          line-height: 1.6;          /* ⬆️ 行距更舒适 */
        }
      header{ background:linear-gradient(135deg,#2e7d32,#1b5e20);color:#fff;text-align:center;padding:18px 0;border-bottom-left-radius:14px;border-bottom-right-radius:14px;box-shadow:0 4px 10px rgba(0,0,0,0.1);}
      h1{margin:0;font-size:22px;} p.subtitle{margin:0;font-size:14px;opacity:.85;}
      .tabs{text-align:center;margin:16px auto;}
      .tab-btn{background:#e8f5e9;border:2px solid #2e7d32;color:#2e7d32;padding:8px 18px;margin:0 6px;font-size:15px;font-weight:600;border-radius:8px;cursor:pointer;transition:.2s;}
      .tab-btn.active{background:#2e7d32;color:#fff;}
      .tab-btn:hover{filter:brightness(1.1);}
      .container {
          max-width: 1100px;   /* ⬆️ 页面主框宽一些 */
          margin: 40px auto;
          background: white;
          border-radius: 20px;
          box-shadow: 0 6px 24px rgba(0,0,0,0.1);
          padding: 40px 50px;  /* ⬆️ 内边距更大 */
        }
      textarea {
          width: 100%;
          height: 160px;         /* ⬆️ 更高的输入框 */
          padding: 14px;
          font-size: 18px;       /* ⬆️ 字体更大 */
          border: 1px solid #ccc;
          border-radius: 10px;
          resize: none;
        }
      button {
          background: linear-gradient(135deg, #43a047, #2e7d32);
          color: white;
          padding: 14px 24px;   /* ⬆️ 按钮更大 */
          font-size: 17px;
          border: none;
          border-radius: 8px;
          margin: 8px;
          cursor: pointer;
        }
      button:hover{background:linear-gradient(135deg,#388e3c,#1b5e20);}
      .clear-btn{background:#ef5350;} .clear-btn:hover{background:#c62828;}
      .output {
          background: #f1f8e9;
          border-left: 5px solid #43a047;
          padding: 18px;
          border-radius: 10px;
          white-space: pre-wrap;
          color: #2e7d32;
          font-size: 18px;       /* 输出字体更大 */
        }
      #status{text-align:center;color:#555;margin-top:10px;font-style:italic;}
      #logBox{background:#f9f9f9;border:1px solid #ccc;border-radius:8px;padding:10px;height:400px;overflow-y:scroll;white-space:pre-wrap;font-size:14px;color:#2e7d32;}
    </style>
    </head>
    <body>
    <header>
      <h1>🌿 BASS++ 推测推理演示系统</h1>
      <p class="subtitle">模块一：交互推理对比　｜　模块二：动态分批日志可视化</p>
    </header>

    <div class="tabs">
      <button id="tab1" class="tab-btn active">模块一：交互推理对比</button>
      <button id="tab2" class="tab-btn">模块二：动态分批日志可视化</button>
    </div>

    <!-- 模块一 -->
    <div id="module1" style="display:block;">
      <div class="container">
        <h2>🧩 模型对比</h2>
        <p>输入文本，点击“生成”以比较 <b>Baseline（自回归）</b> 与 <b>BASS 批处理</b> 的性能</p>
        <textarea id="inputText" placeholder="例如：请解释推测解码（speculative decoding）的原理。"></textarea><br>
        <button onclick="generate()">🚀 生成</button>
        <button class="clear-btn" onclick="clearAll()">🧹 清除</button>
        <p id="status"></p>
        <div id="output" class="output"></div>
      </div>
    </div>

    <!-- 模块二 -->
    <div id="module2" style="display:none;">
      <div class="container">
        <h2>📊 动态分批日志可视化</h2>
        <p>点击下方按钮启动动态分批算法并查看实时日志输出：</p>
        <button id="startSimBtn">▶️ 开始模拟</button>
        <span id="simStatus" style="margin-left:10px;color:#555;">未开始</span>
        <pre id="logBox"></pre>
      </div>
    </div>

    <script>
      // 模块切换
      document.getElementById("tab1").onclick = () => {
        document.getElementById("module1").style.display = "block";
        document.getElementById("module2").style.display = "none";
        document.getElementById("tab1").classList.add("active");
        document.getElementById("tab2").classList.remove("active");
      };
      document.getElementById("tab2").onclick = () => {
        document.getElementById("module1").style.display = "none";
        document.getElementById("module2").style.display = "block";
        document.getElementById("tab2").classList.add("active");
        document.getElementById("tab1").classList.remove("active");
      };

      // 模块一：推理对比
      async function generate() {
        const text = document.getElementById("inputText").value.trim();
        const outputBox = document.getElementById("output");
        const status = document.getElementById("status");
        if (!text) { alert("请输入文本"); return; }
        outputBox.innerHTML = ""; status.textContent = "⏳ 正在生成中...";
        try {
          const res = await fetch("/predict", {
            method:"POST", headers:{ "Content-Type":"application/json" },
            body:JSON.stringify({prompt:text})
          });
          const result = await res.json();
          status.textContent = "✅ 对比完成";
          outputBox.innerHTML = `
          🧩 <b>Baseline（普通自回归）</b><br>
          延迟：${result.baseline.latency}s　吞吐：${result.baseline.throughput} tok/s<br>
          <div style="background:#f5f5f5;border-radius:6px;padding:8px;">${result.baseline.text}</div><hr>
          🌿 <b>BASS 批处理</b><br>
          批大小：${result.bass.batch}　延迟：${result.bass.latency}s　吞吐：${result.bass.throughput} tok/s<br>
          <div style="background:#e8f5e9;border-radius:6px;padding:8px;">${result.bass.text}</div><hr>
          📈 吞吐提升：x${result.compare.throughput_gain}　延迟提升：x${result.compare.latency_gain}`;
        } catch (err) {
          status.textContent = "❌ 出错：" + err;
        }
      }

      function clearAll(){
        document.getElementById("inputText").value="";
        document.getElementById("output").textContent="";
        document.getElementById("status").textContent="";
      }

      // 模块二：动态日志
      const startBtn=document.getElementById("startSimBtn");
      const logBox=document.getElementById("logBox");
      const simStatus=document.getElementById("simStatus");
      startBtn.addEventListener("click", async()=>{
        logBox.textContent=""; simStatus.textContent="运行中...";
        await fetch("/start_simulation",{method:"POST"});
        const evt=new EventSource("/log_stream");
        evt.onmessage=(e)=>{
          try{
            const data=JSON.parse(e.data);
            logBox.textContent+=data.log+"\\n";
            logBox.scrollTop=logBox.scrollHeight;
            if(data.log.includes("结束")){ simStatus.textContent="✅ 已结束"; evt.close(); }
          }catch{}
        };
      });
    </script>
    </body>
    </html>
    """)


pipeline = None

GLOBAL_SERVER = None

class Server:
    def __init__(self, approx_model_name, target_model_name, args = None) -> None:
        self._device = args.device if hasattr(args, "device") else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Loading models on {self._device} ...")

        torch_dtype = torch.float16 if "cuda" in self._device else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)

        # 小模型（approx）
        self._small_model = AutoModelForCausalLM.from_pretrained(
            approx_model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map={"": self._device},
        )

        # 大模型（target）
        self._large_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map={"": self._device},
        )

        print(f"[INFO] Models loaded successfully on {self._device}.")
        if "cuda" in self._device:
            print(
                f"[INFO] Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        self.num_tokens = 40
        self.top_k = 10
        self.top_p = 0.9
        
    def process_request(self, request : str) -> torch.Tensor:
        input_str = request['prompt']
        logging.info(f"recieve request {input_str}")
        input_ids = self._tokenizer.encode(input_str, return_tensors='pt').to(self._device)
        output = speculative_sampling(input_ids, 
                                      self._small_model, 
                                      self._large_model, self.num_tokens, 
                                      top_k = self.top_k, 
                                      top_p = self.top_p)
        generated_text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    def compare_baseline_vs_bass(self, prompt: str):
        """
        对比 Baseline (自回归) 和 BASS 批处理推理
        返回生成文本与性能指标
        """
        input_ids = self._tokenizer.encode(prompt, return_tensors='pt').to(self._device)
        results = {}

        # ===== Baseline: autoregressive decoding =====
        t0 = time.perf_counter()
        out_base = autoregressive_sampling(
            x=input_ids,
            model=self._large_model,
            N=self.num_tokens,
            top_k=self.top_k,
            top_p=self.top_p
        )
        t1 = time.perf_counter()
        base_text = self._tokenizer.decode(out_base[0], skip_special_tokens=True)
        base_time = t1 - t0
        base_tput = self.num_tokens / base_time if base_time > 0 else 0

        results["baseline"] = {
            "text": base_text,
            "latency": round(base_time, 3),
            "throughput": round(base_tput, 2)
        }

        # ===== BASS-PAD: speculative batched decoding =====
        prefixes = input_ids.repeat(4, 1)  # 比如一次批量 4，可自行调整
        t0 = time.perf_counter()
        out_bass, lengths = speculative_sampling_bass_pad(
            prefixes,
            self._small_model,
            self._large_model,
            max_new_tokens=self.num_tokens,
            gamma_init=4,
            top_k=self.top_k,
            top_p=self.top_p,
            verbose=False,
        )
        t1 = time.perf_counter()
        bass_text = self._tokenizer.decode(out_bass[0, :int(lengths[0].item())], skip_special_tokens=True)
        bass_time = t1 - t0
        total_new = int(lengths.sum().item()) - prefixes.numel()
        bass_tput = total_new / bass_time if bass_time > 0 else 0

        results["bass"] = {
            "text": bass_text,
            "latency": round(bass_time, 3),
            "throughput": round(bass_tput, 2),
            "batch": prefixes.size(0)
        }

        # ===== 对比指标 =====
        gain_tput = (results["bass"]["throughput"] / results["baseline"]["throughput"]) if results["baseline"][
                                                                                               "throughput"] > 0 else 0
        gain_lat = (results["baseline"]["latency"] / results["bass"]["latency"]) if results["bass"][
                                                                                        "latency"] > 0 else 0
        results["compare"] = {
            "throughput_gain": round(gain_tput, 2),
            "latency_gain": round(gain_lat, 2)
        }

        return results


# Set up a route to listen for inference requests
@app.route('/predict', methods=['POST'])
def predict():
    if request.headers['Content-Type'] != 'application/json':
        return jsonify({'error': 'Invalid content type'})
    data = request.json
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'Empty prompt'})
    result = GLOBAL_SERVER.compare_baseline_vs_bass(prompt)
    return jsonify(result)

@app.route("/start_simulation", methods=["POST"])
def start_simulation():
    """
    启动 dynamic_batcher.py 作为子进程，并异步读取日志
    """
    global simulation_running, process_thread
    if simulation_running:
        return jsonify({"status": "already_running"})
    simulation_running = True

    def run_and_capture():
        global simulation_running
        try:
            cmd = [
                "python", "sampling/dynamic_batcher.py",
                "--device", GLOBAL_SERVER._device,
                "--mode", "dynamic",
                "--max_samples", "200"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            print(f"[SIM] 启动命令: {' '.join(cmd)}")
            for line in iter(proc.stdout.readline, ''):
                if line:
                    log_queue.put(line.strip())
            proc.stdout.close()
            proc.wait()
        except Exception as e:
            log_queue.put(f"[ERROR] {e}")
        finally:
            simulation_running = False
            log_queue.put("[SIM] 结束。")

    process_thread = threading.Thread(target=run_and_capture, daemon=True)
    process_thread.start()
    return jsonify({"status": "started"})


@app.route("/log_stream")
def log_stream():
    """
    SSE 实时推送 dynamic_batcher.py 的日志
    """
    def event_stream():
        while True:
            try:
                line = log_queue.get(timeout=1.0)
                yield f"data: {json.dumps({'log': line})}\n\n"
            except Exception:
                yield f": keep-alive {int(time.time())}\n\n"
    return app.response_class(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:3', help='选择运行设备，如 cuda:0 或 cuda:3')
    args = parser.parse_args()

    GLOBAL_SERVER = Server(
        approx_model_name="../data/models/bloom-560m",
        target_model_name="../data/models/bloomz-7b1",
        args = args
    )
    # Start the Flask service
    app.run(host='0.0.0.0', port=5000)
