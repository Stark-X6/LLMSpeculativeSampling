from flask import Flask, request, jsonify, render_template_string
import numpy as np
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time
from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2, speculative_sampling_bass_pad

app = Flask(__name__)
@app.route("/")
def home():
    return render_template_string("""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>BASS++ 推测推理演示系统</title>
<style>
    body {
        font-family: 'Segoe UI', Arial, sans-serif;
        background: linear-gradient(135deg, #d7f0d1, #f1f8e9);
        margin: 0; padding: 0;
    }
    .container {
        max-width: 800px;
        margin: 60px auto;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        padding: 30px 40px;
    }
    h2 {
        text-align: center;
        color: #1b5e20;
        margin-bottom: 10px;
    }
    p.intro {
        text-align: center;
        color: #555;
        margin-bottom: 20px;
    }
    textarea {
        width: 100%;
        height: 120px;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #ccc;
        font-size: 16px;
        resize: none;
        outline: none;
        transition: all 0.2s;
    }
    textarea:focus {
        border-color: #2e7d32;
        box-shadow: 0 0 5px rgba(46,125,50,0.4);
    }
    .btn-group {
        margin-top: 15px;
        text-align: center;
    }
    button {
        background: linear-gradient(135deg, #43a047, #2e7d32);
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        margin: 0 5px;
        transition: 0.2s;
    }
    button:hover {
        background: linear-gradient(135deg, #388e3c, #1b5e20);
        transform: translateY(-1px);
    }
    .clear-btn {
        background: #ef5350;
    }
    .clear-btn:hover {
        background: #c62828;
    }
    #status {
        text-align: center;
        font-style: italic;
        color: #777;
        margin-top: 10px;
    }
    .output {
        margin-top: 20px;
        background: #f1f8e9;
        border-left: 4px solid #43a047;
        padding: 15px;
        border-radius: 8px;
        min-height: 60px;
        white-space: pre-wrap;
        font-size: 16px;
        color: #2e7d32;
    }
</style>
</head>
<body>
<div class="container">
    <h2>🌿 BASS++ 推测推理交互演示系统</h2>
    <p class="intro">输入文本后点击 <b>生成</b>，即可查看模型生成结果。</p>

    <textarea id="inputText" placeholder="请输入文本内容，例如：人工智能的推理机制包括哪些方面？"></textarea>

    <div class="btn-group">
        <button onclick="generate()">🚀 生成</button>
        <button class="clear-btn" onclick="clearAll()">🧹 清除</button>
    </div>

    <p id="status"></p>
    <div id="output" class="output"></div>
</div>

<script>
async function generate() {
    const text = document.getElementById("inputText").value.trim();
    const outputBox = document.getElementById("output");
    const status = document.getElementById("status");

    if (!text) {
        alert("请输入文本");
        return;
    }

    outputBox.innerHTML = "";
    status.textContent = "⏳ 正在生成中，请稍候...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ "prompt": text })
        });

        const result = await response.json();
        status.textContent = "✅ 对比完成";

        outputBox.innerHTML = `
        <h4>🧩 Baseline（普通自回归）</h4>
        <div><b>延迟：</b> ${result.baseline.latency}s</div>
        <div><b>吞吐：</b> ${result.baseline.throughput} tokens/s</div>
        <div style="background:#f5f5f5; padding:10px; border-radius:6px;">${result.baseline.text}</div>
        <hr>
        <h4>🌿 BASS 批处理模式</h4>
        <div><b>批大小：</b> ${result.bass.batch}</div>
        <div><b>延迟：</b> ${result.bass.latency}s</div>
        <div><b>吞吐：</b> ${result.bass.throughput} tokens/s</div>
        <div style="background:#e8f5e9; padding:10px; border-radius:6px;">${result.bass.text}</div>
        <hr>
        <h4>📈 对比结果</h4>
        <div><b>吞吐提升：</b> x${result.compare.throughput_gain}</div>
        <div><b>延迟提升：</b> x${result.compare.latency_gain}</div>
        `;
    } catch (err) {
        status.textContent = "❌ 发生错误：" + err;
        outputBox.textContent = "";
    }
}

function clearAll() {
    document.getElementById("inputText").value = "";
    document.getElementById("output").textContent = "";
    document.getElementById("status").textContent = "";
}
</script>
</body>
</html>
""")

pipeline = None

GLOBAL_SERVER = None

class Server:
    def __init__(self, approx_model_name, target_model_name) -> None:
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logging.info("begin load models")
        self._small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, trust_remote_code=True).to(self._device)
        self._large_model = AutoModelForCausalLM.from_pretrained(target_model_name, trust_remote_code=True).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(approx_model_name)
        logging.info("fininsh load models")
          
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

if __name__ == '__main__':
    GLOBAL_SERVER = Server(
        approx_model_name="../data/models/bloom-560m",
        target_model_name="../data/models/bloomz-7b1"
    )
    # Start the Flask service
    app.run(host='0.0.0.0', port=5000)
