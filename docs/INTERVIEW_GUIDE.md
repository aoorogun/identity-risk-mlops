# Interview guide

“I designed this around Prembly’s public identity-verification workflow, but used fully synthetic events to avoid implying access to sensitive production data. The model predicts verification failure, an Isolation Forest detects unusual traffic, and an explicit cost policy turns calibrated risk into approve/review/reject recommendations. I use time-based validation because fraud patterns drift, report group slices because geographic risk features can create harm, and keep humans in the loop. A scenario engine shows how a market launch and stricter policy change review staffing and rejection rates. The repository includes an API, monitoring, AWS controls and a multi-task PyTorch challenger.”

Key discussion points: average precision over accuracy for rare failures; calibration before cost optimisation; analyst feedback without confirmation bias; delayed labels; adversarial drift; privacy-by-design; and why anomaly scores must never be treated as proof of fraud.

