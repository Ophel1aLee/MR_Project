# MR Project — 3D Shape Retrieval System

A 3D shape retrieval system built with Python.
Given a query mesh, the system finds visually and geometrically similar shapes from a database using extracted shape descriptors and nearest-neighbor search.

## Features

- **Mesh Preprocessing** — Fixes mesh errors, resamples to uniform vertex count, and normalizes scale/orientation
- **Shape Descriptor Extraction** — Computes global and shape-distribution descriptors (stored as CSV)
- **Querying** — Supports ANN and custom distance-based querying
- **Evaluation** — Precision-recall evaluation with per-class breakdown
- **Visualization** — Interactive interface for viewing meshes and matching results
