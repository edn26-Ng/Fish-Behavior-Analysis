# Research Notes (condensed)

Sources: team's *Research for Feeder* brief, *Materials Budget* sheet, and the FishNet paper. Summaries only; see original papers for details.

## Hunger signals
- **Clumping**: hungry schools tighten (facilitation + imitation). Delaunay-triangulation gap size used as an "aggregation index" — reported 0.945 correlation with expert hunger scoring. Primary CV signal in commercial feeding systems.
- **Surface-seeking**: fish rise toward surface/feeding zone when motivated; settle toward bottom/sides when not. Used industrially as a feeding-decision input.

## Camera options considered
| Option | Pros | Cons |
|---|---|---|
| Top-down, apparent size as depth proxy (monocular) | No waterproofing | Orientation changes apparent size; refraction → calibrate in water; occlusion in dense schools |
| Side-mounted underwater | Sees depth band directly | Housing, biofouling, turbidity; reliable range ~50–200 cm |
| **Chosen: top-down stereo** | Real depth, no housing | Needs synced cameras + calibration |

## Relevant published systems
### VLM feeder (commercial tilapia farm)
- CLIP-style VLM + optical flow + biomass/water-quality sensors decides when to trigger the feeder.
- 6-month case study: F1 0.95, 92% accuracy, **28% less feed waste** (~135 kg/day).

### Hybrid CNN + ViT + LSTM (industrial RAS)
- CNN = local features, ViT = global group context, LSTM = feeding intensity over time.
- Camera ~70 cm above pond, oblique angle to limit splash/glare.
- Recording starts ~2 min before feeding; extraneous footage trimmed.
- Frames sampled every 3 s; 80/20 train/test.
- Grades: **None** (still, calm surface) · **Weak** (a few fish rise, slight ripples) · **Medium** (some splashing) · **Strong** (aggressive feeding, keeps searching, large splash).
- **Takeaway:** single-frame classification isn't enough; use a temporal window.

### FishSort (RAS)
- YOLOv8-pose detection → two-stage association (fused IoU + embedding distance) → Kalman prediction + camera-motion compensation.
- Features: displacement, speed, acceleration, tail-beat frequency (individual + group).
- Lightweight MLP classifier, 96.61% accuracy on strong/medium/weak.
- IDs/boxes annotated with **DarkLabel** (semi-automatic).

### GNN school model
- Each fish = node; inter-fish distances = edge features; 3-layer GCN classifies feeding behavior.
- Node positions capture depth; edges capture clumping.

### FishNet (individual salmon re-ID)
- FaceNet-style triplet-loss embedding (128-d) on head crops, 175×175.
- Dataset bootstrapped: ~500 hand-labeled boxes → YOLOv3 head detector → crops clustered with DBSCAN using IoU + frame-distance metric, then manual fix-up of occlusion re-ID errors.
- MobileNetV2 (2.4M params) ≈ Inception-ResNet-v2 (55M): TPR ~96% @ 1% FPR. Small models are enough.
- Caveat: one short, clear-water video; frames very similar. Robustness under worse conditions is unproven.
- **Relevance:** re-ID could fix tracker ID switches after occlusion; ~500 labels is a realistic bootstrap size for our detector.

## Implications for this project
- Track kinematics (FishSort-style) + group geometry (clumping, depth) + temporal model.
- Record continuously, not just around feedings: the goal is *when* fish are hungry, which no reviewed paper optimizes.
- Keep full-FPS video; label sparsely.

## Hardware budget (as ordered)
| Item | Model | Qty | Line total |
|---|---|---|---|
| Compute | Raspberry Pi 5 8GB (SC1112) | 1 | $175.00 |
| Cameras | Camera Module 3 Wide (IMX708, 120°) | 2 | $70.00 |
| Cables | Pi 5 22→15 pin, 500 mm | 2 | $5.00 |
| Power | Official 27W USB-C (SC1153) | 1 | $11.99 |
| Cooling | Pi 5 Active Cooler (SC1148) | 1 | $10.95 |
| Feeder motor | 28BYJ-48 + ULN2003 (5-pack) | 1 | $4.14 |
| Load cell | 1 kg + HX711 | 1 | $17.46 |
| Storage | Lexar 128GB microSDXC | 1 | $16.49 |
| Wiring | 120pc Dupont jumper kit | 1 | $8.87 |
| Coverage | PVC (Lowe's), tarp (lab), shower curtain, jump net (from tank) | — | ~$6.61 |

Total (cheapest stores): ~$302 (excl. coverage).
