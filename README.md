# genpark-adamw-decoupled-weight-decay-skill

[![Stars](https://img.shields.io/github/stars/alphaparkinc/genpark-adamw-decoupled-weight-decay-skill?style=social)](https://github.com/alphaparkinc/genpark-adamw-decoupled-weight-decay-skill/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pure Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-0%20External-brightgreen.svg)]()

> AdamW optimizer engine decoupling L2 weight decay regularization from adaptive gradient first and second moments.

---

## Architectural Overview

```mermaid
graph TD
    A[Loss & Gradients] -->|Compute| B[Optimizer Engine]
    B --> C[Momentum / Weight Decay]
    C --> D[Adaptive Update / Normalization]
    D --> E[Updated Model Weights]
```

## Features
- **Pure Python Standard Library**: Zero third-party dependencies required.
- **Model Context Protocol (MCP)**: Native JSON-RPC server ready for LLM integration.
- **Deterministic Verification**: End-to-end sandbox tested with 100% pass rate.

## Quickstart

```bash
python example_usage.py
```

## Running the MCP Server

```bash
python mcp_server.py
```
