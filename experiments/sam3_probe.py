import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import Sam3Model, Sam3Processor

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

img_path = ROOT / "third_party/efficientsam3/sam3/assets/images/groceries.jpg"
prompts = sys.argv[1:] or ["car", "paper bag", "car seat", "taillight"]
dtype = torch.float16

t = time.time()
model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=dtype).to("cuda").eval()
processor = Sam3Processor.from_pretrained("facebook/sam3")
print(f"load {time.time() - t:.1f}s, vram after load {torch.cuda.memory_allocated() / 1e6:.0f} MB")

image = Image.open(img_path).convert("RGB")
for run in range(2):
    t = time.time()
    img_inputs = processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        vision = model.get_vision_features(pixel_values=img_inputs["pixel_values"].to(dtype))
    torch.cuda.synchronize()
    print(f"  image encode {time.time() - t:.2f}s")
    for p in prompts:
        txt = processor(text=p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(vision_embeds=vision, input_ids=txt["input_ids"], attention_mask=txt["attention_mask"])
        res = processor.post_process_instance_segmentation(
            out, threshold=0.3, mask_threshold=0.5, target_sizes=img_inputs["original_sizes"].tolist()
        )[0]
        scores = [round(x, 2) for x in res["scores"].tolist()]
        print(f"  [{run}] '{p}': {len(scores)} found", scores[:6], [[round(v) for v in b] for b in res["boxes"][:2].tolist()])
    torch.cuda.synchronize()
    print(f"run {run}: {time.time() - t:.2f}s total for {len(prompts)} prompts")

print(f"peak vram {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
