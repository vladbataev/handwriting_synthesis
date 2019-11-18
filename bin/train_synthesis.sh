#!/usr/bin/env bash

train \
  --model_type="synthesis" \
  --strokes_path="data/strokes.npy" \
  --texts_path="data/sentences.txt"
