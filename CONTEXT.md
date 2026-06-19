# RHC Causal Inference — CONTEXT
Slug: rhc-causal-inference
Dataset: SUPPORT Right Heart Catheterization (Connors et al. 1996, JAMA). Public, no DUA. Source: hbiostat.org/data/repo/rhc.html
Question: causal effect of RHC within 24h of ICU admission on 30-day mortality.
ENV: python = .venv/bin/python ; kernel = rhc-causal ; NEVER use system python/pip.
Structure: Part 1 data foundation -> Kali gate -> Part 2A estimation -> Part 2B delivery.
Benchmark gate (Part 2): RHC is HARMFUL. ATE on 30-day survival approx -0.05 to -0.06 (Hirano-Imbens 2001: -0.053 to -0.062; Crump 2009: -0.059 to -0.060). Wrong sign or |ATE| outside ~[0.03,0.08] = bug, STOP.
Content rules (portfolio): no em/en dashes; human-engineer voice; no agent/autonomous/agentic language; description 150-160 chars.
