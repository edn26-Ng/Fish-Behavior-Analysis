# Smart Fish Feeder — Project Description (living doc)

## Goal
Build a hand-labeled video library of tilapia group behavior to train a model that classifies state, then drive an automated feeder that minimizes feed waste.

## Behavior states (labels)
1. Resting  2. Hungry  3. Feeding  4. Satiated
Plus event tag: Disturbance (human/lab activity) — used to catch false positives.

## Tank
- Solid concrete cylinder, open top (opaque → cameras top-down).
- Diameter ~48–52 in (~1.2–1.3 m), water depth ~4 ft (~1.2 m).
- Water clarity: good — bottom visible. Full-column stereo depth is feasible.
- Species: tilapia. Count: unknown, est. 50–100.
- Feeding: 2x/day (morning, late afternoon); adjustable for experiments.

## Key behavioral signals
- Vertical position: hungry school rises; satiated school sinks.
- Clumping/dispersion of the school.
- Individual fish (a few tracked): velocity, acceleration, vertical + horizontal displacement → separates resting vs hungry.

## Experimental protocol
- Micro-dosing: small repeated doses; measure response per dose until fish calm (satiation point).
- Continuous recording including between feeds (catch spontaneous activity / false positives).
- Lab-traffic segments flagged.

## Recording + labeling strategy
- Record continuously; hand-label sparsely (every Nth frame, e.g. every ~3 s as in published work).
- Keep full-frame-rate video: tracking/velocity/acceleration need consecutive frames even if labels are sparse.
- Labeling: manual.

## Hardware
- Raspberry Pi 4 + dual-camera extension (must be frame-synchronized stereo HAT).
- 2x Camera Module 3 Wide (IMX708), top-down stereo, ~60 cm above water, ~10 cm baseline.
- Circular polarizing filters for surface glare (to test).
- Feeder: purchased commercially — secondary project. Pi = capture only for now.

## Night recording (open)
- Tilapia detect red and near-IR (thresholds reported up to ~930 nm; strain-dependent). Red light likely visible to fish.
- Candidate: 940 nm IR illuminator + NoIR cameras; verify no behavior change.

## Open questions
- Feeder model + whether it can be triggered/logged externally.
- Night recording decision.
- Mount structure over tank.
- Pi 4 + stereo HAT (spec) vs Pi 5 with two native camera ports (order sheet).
