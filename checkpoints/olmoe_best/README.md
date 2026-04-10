# Best BVH Router Checkpoints (OLMoE-1B-7B)

Unified directory with the best checkpoint for each of the 16 layers.
**PPL: 7.00 (+0.0% vs baseline) — ZERO DEGRADATION**

## Source

| Layer | Top-8 | Source | Training |
|-------|-------|--------|----------|
| L0  | 95.4% | olmoe_distill_layer0 | 100ep spectral dim=256 |
| L1  | **95.9%** | **olmoe_distill (retrained 2026-04-10)** | **200ep spectral dim=256** |
| L2  | 96.1% | olmoe_distill_layer2 | 100ep spectral dim=256 |
| L3  | 96.2% | olmoe_distill_layer3 | 100ep spectral dim=256 |
| L4  | 95.1% | olmoe_distill_layer4 | 100ep spectral dim=256 |
| L5  | 96.1% | olmoe_distill_layer5 | 100ep spectral dim=256 |
| L6  | 96.4% | olmoe_distill_layer6 | 100ep spectral dim=256 |
| L7  | 96.6% | olmoe_distill_layer7 | 100ep spectral dim=256 |
| L8  | **96.4%** | **olmoe_distill (retrained 2026-04-10)** | **200ep spectral dim=256** |
| L9  | 96.8% | olmoe_distill_layer9 | 100ep spectral dim=256 |
| L10 | 97.2% | olmoe_distill_layer10 | 100ep spectral dim=256 |
| L11 | 97.2% | olmoe_distill_layer11 | 100ep spectral dim=256 |
| L12 | 97.4% | olmoe_distill_layer12 | 100ep spectral dim=256 |
| L13 | 97.0% | olmoe_distill_layer13 | 100ep spectral dim=256 |
| L14 | 97.5% | olmoe_distill_layer14 | 100ep spectral dim=256 |
| L15 | 97.6% | olmoe_distill_layer15 | 100ep spectral dim=256 |

**Mean top-8: 96.56% | All layers > 95%**

## Usage

```bash
# Eval all 16 layers (hybrid mode, pre-filter 48 candidates)
python3 python/olmoe_e2e_eval.py \
  --model-dir /path/to/olmoe-1b-7b \
  --n-candidates 48 --max-tokens 20000 --hybrid \
  --multi-layer "0:checkpoints/olmoe_best/bvh_router_L0_best.pt,1:checkpoints/olmoe_best/bvh_router_L1_best.pt,2:checkpoints/olmoe_best/bvh_router_L2_best.pt,3:checkpoints/olmoe_best/bvh_router_L3_best.pt,4:checkpoints/olmoe_best/bvh_router_L4_best.pt,5:checkpoints/olmoe_best/bvh_router_L5_best.pt,6:checkpoints/olmoe_best/bvh_router_L6_best.pt,7:checkpoints/olmoe_best/bvh_router_L7_best.pt,8:checkpoints/olmoe_best/bvh_router_L8_best.pt,9:checkpoints/olmoe_best/bvh_router_L9_best.pt,10:checkpoints/olmoe_best/bvh_router_L10_best.pt,11:checkpoints/olmoe_best/bvh_router_L11_best.pt,12:checkpoints/olmoe_best/bvh_router_L12_best.pt,13:checkpoints/olmoe_best/bvh_router_L13_best.pt,14:checkpoints/olmoe_best/bvh_router_L14_best.pt,15:checkpoints/olmoe_best/bvh_router_L15_best.pt"
```

## Important Notes

- Use `--hybrid` flag for correct results (BVH pre-selects 48/64 candidates, original gate weights)
- Without `--hybrid`, mode is "pure" which gives much worse PPL
- L1 and L8 were retrained with 200 epochs (vs 100 for other layers) to fix accuracy bottlenecks
- The `olmoe_distill_perm` directory contains permuted checkpoints optimized for RT Core traversal (NOT for PPL)
