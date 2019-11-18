#!/usr/bin/env bash

train \
  --model_type="prediction" \
  --strokes_path="data/strokes.npy" \
  --texts_path="data/sentences.txt"
