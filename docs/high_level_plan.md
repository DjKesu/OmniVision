Love it — let's turn **OmniVision** into something you can ship, evaluate, and grow into 3D splats. Below is a **detailed plan**, **curated datasets**, and a **repo/Hugging Face/Gradio blueprint** you can copy‑paste to get moving.

---

## 0) Core references we will stand on

* **DINOv3** release (models, tokens incl. registers, training data sizes, license): Meta blog + HF model card + official repo. ([Meta AI][1], [Hugging Face][2], [GitHub][3])
* **Transformers integration + int4 quantization snippet + how to split CLS/register/patch tokens**: HF DINOv3 docs. ([Hugging Face][4])
* **Why “register tokens” matter**: Registers paper (cleaner attention/dense tasks). ([arXiv][5], [OpenReview][6])
* **SAM 2** (video‑first, streaming memory, Apache‑2.0): official GitHub. ([GitHub][7])
* **3D Gaussian Splatting** and semantic/distilled‑feature splats (for your extension): original 3DGS + LangSplat + Feature‑3DGS. ([arXiv][8], [repo-sam.inria.fr][9], [ACM Digital Library][10], [CVF Open Access][11], [GitHub][12])

---

## 1) What OmniVision does (short)

* **Localize by visual example**: Take a user‑clicked patch or reference image → compute **DINOv3 patch‑token similarity** over a target image/frame to produce a heatmap → boxes. (Registers help stabilize the patch tokens.) ([Hugging Face][4], [arXiv][5])
* **Refine & track**: Feed boxes/points to **SAM 2** for crisp masks and **streaming video tracking**. ([GitHub][7])
* **Open‑vocabulary (optional)**: Train a tiny **Text→DINOv3** adapter (CLIP‑text → DINOv3 feature space) so phrases can act like patch prompts.

---

## 2) Datasets (ready-to-use packs for each capability)

**A. Visual‑example localization & correspondence (images)**
Use these to validate “find similar stuff by patch features” and correspondence robustness.

* **SPair‑71k** (70,958 paired images; strong viewpoint/scale changes). Great for dense matching & “where is the same part?” evaluation. ([cvlab.postech.ac.kr][13], [arXiv][14])
* **HPatches** (local patch matching under illumination/viewpoint changes). Good sanity checks for patch‑level similarity. ([CVF Open Access][15], [GitHub][16])

**Metrics**: PCK@α for correspondences; pixel‑wise AUC of similarity heatmaps; retrieval mAP over parts.

---

**B. “Phrase → region” (open‑vocabulary prompts for the adapter)**

* **RefCOCO / RefCOCO+ / RefCOCOg** (classic referring expression datasets; easy to load via `lichengunc/refer`). Use for training/evaluating the **Text→DINOv3** bridge. ([GitHub][17], [arXiv][18])
* **Localized Narratives** (849k images with word‑level mouse‑trace grounding across COCO/Flickr30k/ADE20K/OpenImages). Use as weakly supervised region–text pairs for scale. ([Google GitHub][19], [arXiv][20], [Hugging Face][21])
* **Flickr30k Entities** (phrase↔box annotations). Clean phrase–region supervision. ([GitHub][22], [bryanplummer.com][23])

**Metrics**: Referring expression pointing IoU (mIoU@τ), box AP\@50/75 (when using boxes).

---

**C. Segmentation & Video tracking (to prove SAM 2 + DINOv3 integration)**

* **DAVIS 2017** (semi‑supervised VOS; widely used for mask tracking quality). ([davischallenge.org][24], [arXiv][25])
* **YouTube‑VOS** (large‑scale VOS: semi‑supervised/VIS/RVOS). ([YouTube-VOS][26])
* **OVIS** (occluded video instance segmentation; hard cases for re‑ID). ([arXiv][27], [Song Bai - 柏 松][28])

**Metrics**: Jaccard (J), F‑boundary (F), combined J\&F on DAVIS; mAP/mAR on VIS; IDF1/ID metrics for tracking consistency.

---

**D. (Optional) General open‑vocabulary instance segmentation sanity checks**

* **LVIS** (large‑vocabulary instance segmentation). Use as a probing ground for “phrase → mask” without heavy training. ([arXiv][29], [Hugging Face][30])

---

## 3) Experiments you can run immediately

**Exp‑1: Detector‑free “segment by example” (images)**

* Input: image, user clicks a reference patch.
* Compute DINOv3 patch‑token features (ViT‑B/16 to start), **cosine heatmap**, pick top blobs → **SAM 2** boxes/points → masks.
* Report IoU on RefCOCO (treat referring phrases as “visual exemplar” by cropping the referenced region) + qualitative SPair‑71k correspondences.
* Notes: Use **HF DINOv3 models** (`facebook/dinov3-vitb16-pretrain-lvd1689m` etc.). These expose **CLS + 4 register + patch tokens**; the docs show how to split tensors. ([Hugging Face][2])

**Exp‑2: Video “track by example”**

* First frame click → DINOv3 similarity → SAM 2 prompt → propagate with **SAM 2 VideoPredictor**; re‑associate with nearest‑neighbor in DINOv3 feature space per frame.
* Benchmark on **DAVIS / YouTube‑VOS** with J\&F and VIS AP. ([GitHub][7], [davischallenge.org][24], [YouTube-VOS][26])

**Exp‑3: Text→DINOv3 bridging (adapter)**

* Train a tiny MLP (or low‑rank) that maps **CLIP text** embeddings into **DINOv3 feature dimension**.
* Supervision: (phrase, region) pairs from RefCOCO/Flickr30k Entities/Localized Narratives; loss = contrastive (InfoNCE) between projected text and **average of patch tokens** inside region vs. negatives (other regions/patches).
* Evaluate phrase→mask with SAM 2 refinement on RefCOCO splits. ([GitHub][17], [Google GitHub][19])

**Exp‑4 (Efficiency): Quantized large backbones**

* Load **ViT‑7B** with **torchao int4 weight‑only** to check latency/VRAM; HF docs show a working snippet. Compare ViT‑B/L vs. 7B (int4) for retrieval quality vs. time. ([Hugging Face][4])

> FYI: DINOv3 includes **web‑pretrained (LVD‑1689M)** and **satellite (SAT‑493M)** variants; license = **DINOv3 License** with access gating on HF. Keep your repo compliant and optionally support **DINOv2 (Apache‑2.0)** as a fallback. ([Hugging Face][2], [GitHub][31])

---

## 4) Repository sketch (Python)

```
omnivision/
  __init__.py
  configs/
    model.yaml            # dinov3 backbone id, fp16/bf16/int4, pooling rules
    sam2.yaml             # checkpoint name, memory opts
    eval.yaml             # datasets, splits, metrics
  data/
    builders/
      spair71k.py         # pairs loader + keypoint/warp GT
      hpatches.py
      refcoco.py          # via lichengunc/refer
      flickr30k_entities.py
      localized_narratives.py
      davis.py
      youtube_vos.py
      ovis.py
    utils.py              # common transforms, resizing↔patch grid mapping
  models/
    dinov3_backbone.py    # HF AutoModel wrapper, returns cls / registers / patch grid
    similarity.py         # cosine heatmaps, upsample, peak picking, NMS
    text_bridge.py        # CLIP text encoder + MLP/LoRA to DINOv3 dim
    sam2_wrapper.py       # prompt APIs (point/box/mask), video predictor
    tracker.py            # re-ID via DINOv3 features + Hungarian, SAM2 memory glue
  pipelines/
    localize_by_example.py    # image pipeline
    track_by_example.py       # video pipeline
    phrase_to_mask.py         # text→box → SAM2 masks
  training/
    train_text_bridge.py      # datasets, losses (InfoNCE), EMA, fp16
    losses.py
    schedulers.py
  eval/
    eval_referring.py         # mIoU, pointing game
    eval_spair.py             # PCK@α, AEPE
    eval_davis.py             # J, F, J&F
    eval_vis.py               # AP/AR (YouTube-VOS/OVIS)
  cli/
    __init__.py
    localize.py           # `omnivision localize --ref ref.jpg --target target.jpg`
    track.py              # `... track --video in.mp4 --click 320,200`
    phrase.py             # `... phrase "red backpack" --image in.jpg`
  demos/
    gradio_app.py         # web UI (image/video tabs)
  export/
    hf_pipeline.py        # push_to_hub: model card, tags, inference script
  utils/
    viz.py                # overlay heatmaps, masks
    boxes.py
    registry.py
    logging.py
tests/
  test_tokens.py
  test_similarity.py
  test_sam2_integration.py
  test_datasets.py
README.md
LICENSE
```

**Design notes**

* `dinov3_backbone.py` exposes a simple call that returns **(cls, registers, patch\_grid\[h,w,d])** consistent with the HF docs; registers are present and useful for stabilization. ([Hugging Face][4])
* Everything should run with **frozen DINOv3**; the only trainable part is `text_bridge.py`.

---

## 5) Gradio demo (Hugging Face Space)

**Tabs**

1. **Image: Segment by example**

   * Upload image → click 1–3 points defining an exemplar (or crop).
   * Controls: backbone (ViT‑B/L/7B‑int4), stride/threshold, top‑k, SAM 2 checkpoint, output masks/boxes.
2. **Video: Track by example**

   * Upload MP4 → click first frame → run SAM 2 video predictor; optionally “Add object” mid‑stream.
3. **Phrase → mask**

   * Enter text, optional negative phrase (“not sky”), choose adapter weights, get boxes + SAM 2 masks.

**Outputs**: overlays, downloadable **COCO JSON** and mask PNGs; in video tab, MP4 with masks.

**Implementation nits**

* Use the **SAM 2 `build_sam2_video_predictor`** for tracking; it supports per‑object inference and torch.compile for speed. ([GitHub][7])
* Provide a “**ViT‑7B (int4)**” toggle using HF **TorchAo** quantization to make large backbones practical. ([Hugging Face][4])

---

## 6) Hugging Face release plan

**Model repos (at minimum)**

* `your-handle/omnivision-text2dinov3-mlp`: tiny adapter weights + config.
* (Optional) pre‑exported **DINOv3 feature head** (a thin wrapper config) that loads HF backbones by name and exposes a standardized API.

**Space**

* `your-handle/omnivision` (Gradio demo).
* Add **model cards** with: purpose/use cases, **DINOv3 license note (gated)**, SAM 2 license (Apache‑2.0), datasets used for evaluation, and evaluation scripts. ([Hugging Face][2], [GitHub][7])

**Pipelines tag**: `image-segmentation`, `video-segmentation`, `open-vocabulary`, `visual-correspondence`.

---

## 7) Training details (Text→DINOv3 bridge)

* **Backbones**: `facebook/dinov3-vitb16-pretrain-lvd1689m` for dev, swap to L/7B later. (HF shows token layout and example code). ([Hugging Face][2])
* **Text encoder**: CLIP text (frozen).
* **Projection head**: 2‑layer MLP w/ LayerNorm to **DINOv3 dim**; train on (phrase, region) pairs.
* **Positives**: average DINOv3 patch tokens inside the GT region; **Negatives**: random patches, hard negatives from same image.
* **Loss**: InfoNCE + temperature; optional center loss to keep norms stable (registers reduce high‑norm artifacts already). ([arXiv][5])
* **Augmentations**: mild spatial jitter on regions, color jitter (ensure you re‑extract tokens post‑aug).
* **Eval**: RefCOCO(mIoU), and pointing accuracy.

---

## 8) Baselines to report

* **DINOv3 heatmap → box → SAM 2** vs. **DINOv3 heatmap alone** (no SAM 2).
* **Phrase → box/mask** using CLIP zero‑shot boxes (if you include a quick CLIP‑GradCAM probe) vs. **Text→DINOv3 adapter**.
* **Backbone scaling**: ViT‑B vs. ViT‑L vs. ViT‑7B‑int4 (report latency and J\&F). (Quantization example from HF docs.) ([Hugging Face][4])

> Tip: DINOv3’s own model card lists results on **DAVIS** and **SPair** for different backbones — useful as a sanity check for expected ranges. ([Hugging Face][2])

---

## 9) 3D Gaussian Splat extension (clear path)

**Goal**: Make your 2D OmniVision prompts usable **inside a 3DGS scene** for **language/patch‑guided selection, editing, and segmentation**.

**Phase A — “Feature splats” (distillation)**

* Follow **Feature‑3DGS**: during 3DGS training, distill 2D feature maps (from DINOv3 patch tokens and/or your adapter outputs) into a **per‑Gaussian feature vector**. Add a small MLP to predict 2D features at render time and train with multi‑view consistency losses. ([arXiv][32], [CVF Open Access][33])
* Camera poses from COLMAP; base 3DGS training from the public repo. ([arXiv][8], [repo-sam.inria.fr][9])

**Phase B — “Language in 3D”**

* Follow **LangSplat** idea: learn a **3D language field** (or store a language embedding per Gaussian) and enable **text queries that select Gaussians** (or render a mask from queried Gaussians). Your Text→DINOv3 adapter can supply language‑aligned supervision per view. ([arXiv][34], [CVF Open Access][11])

**Phase C — Interactive editing**

* With per‑Gaussian features, implement “**select splats by patch**” (click on a rendered view → backproject to Gaussians with high DINOv3 similarity) and “**select splats by text**” (cosine sim in feature space), then **edit** (visibility, color, deletion), as shown in Feature‑3DGS demos. ([Feature 3DGS][35])

**Comms angle**: tie to the ongoing “splats moment” (adoption across industry) — nice for your launch blog. ([The Verge][36])

---

## 10) Concrete “getting started” tasks (no blockers)

* ✅ **Data scripts**

  * `scripts/download_spair71k.sh`, `download_hpatches.sh`, `prepare_refcoco.sh`, `prepare_davis.sh`, `prepare_ytvos.sh`, `prepare_ovis.sh`. (Use the dataset homepages/APIs cited above.) ([cvlab.postech.ac.kr][13], [GitHub][16], [davischallenge.org][24], [YouTube-VOS][26], [Song Bai - 柏 松][28])
* ✅ **Backbone wrapper**

  * Implement `DinoV3Backbone(name)` → returns `(cls, regs, patch_grid, stride)` matching the **HF docs**; include the **TorchAo int4** load path for 7B. ([Hugging Face][4])
* ✅ **Similarity to boxes**

  * Cosine on normalized patch\_grid → bilinear upsample → morphological open/close → connected components → top‑k boxes.
* ✅ **SAM 2 glue**

  * `sam2_wrapper.py`: `predict_image(points|boxes)` and `predict_video(state, new_prompts)` using **SAM 2 video predictor** with memory. ([GitHub][7])
* ✅ **CLI**

  * `omnivision localize --ref ref.jpg --target img_or_vid`
  * `omnivision track --video in.mp4 --click 320,200`
  * `omnivision phrase --text "red backpack" --image in.jpg`
* ✅ **Eval scripts**

  * `eval_davis.py` (J, F, J\&F); `eval_referring.py` (mIoU); `eval_spair.py` (PCK).

---

## 11) Licenses & compliance (important)

* **DINOv3** weights on HF are gated and under **DINOv3 License** (link from model card). **Do not redistribute** the weights; require users to accept the terms via HF. ([Hugging Face][2])
* **SAM 2** code/checkpoints: **Apache‑2.0** (plus a BSD‑3‑Clause dep). Safe for OSS integration. ([GitHub][7])
* Optional **fallback**: **DINOv2** (now **Apache‑2.0**) for those who can’t accept DINOv3 terms. ([GitHub][31], [Hugging Face][37])

---

## 12) Quality bar & reporting

* Publish a **Results** section in the README mirroring **DINOv3 model card** task names (SPair, DAVIS) so readers know where you sit. ([Hugging Face][2])
* Include side‑by‑side GIFs for **Track by example** on DAVIS/YouTube‑VOS. ([davischallenge.org][24], [YouTube-VOS][26])
* Add a **Reproducibility** script that fetches small subsets of each dataset and runs a batch evaluation end‑to‑end.

---

## 13) Example code snippets you’ll need (pseudocode level)

**Extract patch features and produce a similarity heatmap (per HF docs):** ([Hugging Face][4])

```python
model = AutoModel.from_pretrained(name, device_map="auto", torch_dtype=torch.bfloat16)
processor = AutoImageProcessor.from_pretrained(name)
px = processor(images=PIL.Image.open(img), return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model(**px)
last = out.last_hidden_state                               # [B, 1 + R + Npatch, D]
num_regs = model.config.num_register_tokens                # 4
patch = last[:, 1 + num_regs:, :].unflatten(1, (H, W))     # [B, H, W, D]
patch = F.normalize(patch, dim=-1)
ref_vec = F.normalize(extract_ref_vector(...), dim=-1)     # click/crop pooled
sim = (patch * ref_vec[None, None, :]).sum(-1)             # [B, H, W]
```

**SAM 2 video tracking:** (build predictor, init state, add prompts) — the API mirrors the official examples. ([GitHub][7])

**Load 7B with int4:** use **TorchAo** `Int4WeightOnlyConfig` as shown in HF docs (works with Transformers’ `AutoModel`). ([Hugging Face][4])

---

## 14) Nice-to-haves (that stand out)

* **Adapters Zoo**: release multiple small heads (Text→DINOv3; optional Edge/Depth heads trained linearly on patch tokens).
* **Quantization notes**: a doc explaining FP16 vs. BF16 vs. **int4 weight‑only** tradeoffs, citing the **HF DINOv3** quantization example. ([Hugging Face][4])
* **Unreal sample** (hooks into your work): expose a Python/REST endpoint to receive a frame buffer and return mask tracks (plug to UE Sequencer or your digital‑twin pipeline).

---

If you want, I can generate the **initial repo scaffolding** (folders, empty modules, boilerplate configs, and a Gradio stub) and a **HF model card template** you can paste into `README.md`.

[1]: https://ai.meta.com/blog/dinov3-self-supervised-vision-model/?utm_source=chatgpt.com "DINOv3: Self-supervised learning for vision at ..."
[2]: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m "facebook/dinov3-vitb16-pretrain-lvd1689m · Hugging Face"
[3]: https://github.com/facebookresearch/dinov3?utm_source=chatgpt.com "Reference PyTorch implementation and models for DINOv3"
[4]: https://huggingface.co/docs/transformers/main/en/model_doc/dinov3 "DINOv3"
[5]: https://arxiv.org/html/2309.16588v2?utm_source=chatgpt.com "Vision Transformers Need Registers"
[6]: https://openreview.net/forum?id=2dnO3LLiJ1&utm_source=chatgpt.com "Vision Transformers Need Registers"
[7]: https://github.com/facebookresearch/sam2 "GitHub - facebookresearch/sam2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model."
[8]: https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
[9]: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/?utm_source=chatgpt.com "3D Gaussian Splatting for Real-Time Radiance Field ..."
[10]: https://dl.acm.org/doi/10.1145/3592433?utm_source=chatgpt.com "3D Gaussian Splatting for Real-Time Radiance Field ..."
[11]: https://openaccess.thecvf.com/content/CVPR2024/papers/Qin_LangSplat_3D_Language_Gaussian_Splatting_CVPR_2024_paper.pdf?utm_source=chatgpt.com "LangSplat: 3D Language Gaussian Splatting"
[12]: https://github.com/minghanqin/LangSplat?utm_source=chatgpt.com "3D Language Gaussian Splatting\" [CVPR2024 Highlight]"
[13]: https://cvlab.postech.ac.kr/research/SPair-71k/?utm_source=chatgpt.com "SPair-71k: A Large-scale Benchmark for Semantic ..."
[14]: https://arxiv.org/abs/1908.10543?utm_source=chatgpt.com "SPair-71k: A Large-scale Benchmark for Semantic Correspondence"
[15]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Balntas_HPatches_A_Benchmark_CVPR_2017_paper.pdf?utm_source=chatgpt.com "HPatches: A benchmark and evaluation of handcrafted and ..."
[16]: https://github.com/hpatches/hpatches-dataset?utm_source=chatgpt.com "HPatches: Homography-patches dataset."
[17]: https://github.com/lichengunc/refer?utm_source=chatgpt.com "lichengunc/refer: Referring Expression Datasets API"
[18]: https://arxiv.org/html/2406.16866v1?utm_source=chatgpt.com "Revisiting Referring Expression Comprehension ..."
[19]: https://google.github.io/localized-narratives/?utm_source=chatgpt.com "Localized Narratives - Google"
[20]: https://arxiv.org/abs/1912.03098?utm_source=chatgpt.com "Connecting Vision and Language with Localized Narratives"
[21]: https://huggingface.co/datasets/HuggingFaceM4/LocalizedNarratives?utm_source=chatgpt.com "HuggingFaceM4/LocalizedNarratives · Datasets at ..."
[22]: https://github.com/BryanPlummer/flickr30k_entities?utm_source=chatgpt.com "BryanPlummer/flickr30k_entities: Flickr30K Entities Dataset"
[23]: https://bryanplummer.com/Flickr30kEntities/?utm_source=chatgpt.com "Flickr30k Entities"
[24]: https://davischallenge.org/?utm_source=chatgpt.com "DAVIS: Densely Annotated VIdeo Segmentation"
[25]: https://arxiv.org/abs/1704.00675?utm_source=chatgpt.com "The 2017 DAVIS Challenge on Video Object Segmentation"
[26]: https://youtube-vos.org/dataset/?utm_source=chatgpt.com "YouTube-VOS Dataset"
[27]: https://arxiv.org/abs/2102.01558?utm_source=chatgpt.com "Occluded Video Instance Segmentation: A Benchmark"
[28]: https://songbai.site/ovis/?utm_source=chatgpt.com "Occluded Video Instance Segmentation (OVIS)"
[29]: https://arxiv.org/abs/1908.03195?utm_source=chatgpt.com "LVIS: A Dataset for Large Vocabulary Instance Segmentation"
[30]: https://huggingface.co/datasets/Voxel51/LVIS?utm_source=chatgpt.com "Voxel51/LVIS · Datasets at Hugging Face"
[31]: https://github.com/facebookresearch/dinov2/blob/main/LICENSE?utm_source=chatgpt.com "dinov2/LICENSE at main - GitHub"
[32]: https://arxiv.org/abs/2312.03203?utm_source=chatgpt.com "Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields"
[33]: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Feature_3DGS_Supercharging_3D_Gaussian_Splatting_to_Enable_Distilled_Feature_CVPR_2024_paper.pdf?utm_source=chatgpt.com "Feature 3DGS: Supercharging 3D Gaussian Splatting to ..."
[34]: https://arxiv.org/abs/2312.16084?utm_source=chatgpt.com "[2312.16084] LangSplat: 3D Language Gaussian Splatting"
[35]: https://feature-3dgs.github.io/?utm_source=chatgpt.com "Feature 3DGS: Supercharging 3D Gaussian Splatting to ..."
[36]: https://www.theverge.com/2025/1/19/24345491/gaussian-splats-3d-scanning-scaniverse-niantic?utm_source=chatgpt.com "The tech to build the holodeck"
[37]: https://huggingface.co/facebook/dinov2-base/commit/369b1309d0ea32e09fdc25ee029917e2e5a2db8f?utm_source=chatgpt.com "facebook/dinov2-base at 369b130 - Update license"
