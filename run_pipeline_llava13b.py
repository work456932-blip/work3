#!/usr/bin/env python3
# run_pipeline_llava13b.py
# 专门跑 LLAVA-13B 的 pipeline

import argparse
from pathlib import Path
import json
from tqdm import tqdm
from mm_rcna.agents import LLAVAAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--study-id", type=str, required=True, help="单个 study id")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vlm-model", type=str, required=True, help="LLAVA 模型路径")
    parser.add_argument("--cache-dir", type=str, required=True, help="embedding cache 目录")
    parser.add_argument("--output-json", type=str, required=True, help="输出 JSON 文件")
    parser.add_argument("--conformal-json", type=str, default=None, help="可选 conformal json")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 初始化 LLAVA agent
    agent = LLAVAAgent(
        vlm_model=args.vlm_model,
        device=args.device,
        reuse_embeddings=False  # 每次都重新生成
    )
    
    # 确保 embedding 目录存在
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 输出目录检查
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取 study
    study_id = args.study_id
    
    # 执行推理
    print(f"[START] {study_id}")
    try:
        predictions = agent.run_study(study_id)
    except Exception as e:
        print(f"[ERROR] {study_id}: {e}")
        predictions = []

    # 保存 JSON
    out_data = {
        "study_id": study_id,
        "agent_backbone": "llava",
        "vlm_model": args.vlm_model,
        "predictions": predictions,
    }

    if args.conformal_json:
        out_data["conformal_json"] = args.conformal_json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    
    print(f"[DONE] {study_id}, saved to {output_path}")

if __name__ == "__main__":
    main()