# Reliability-Aware Image Captioning with CLIP Reranking

This project builds an attention-based image captioning pipeline that generates visually grounded and reliability-aware image descriptions. The model pairs a convolutional image encoder with an autoregressive text decoder using the visual attention approach from *Show, Attend and Tell*. We further integrate CLIP-based reranking to select the candidate caption that best aligns with the input image.

The broader motivation is assistive visual understanding for blind and low-vision users. Since users may rely on generated captions to understand image content, the project focuses not only on caption fluency, but also on visual grounding, robustness, and reliability under degraded image conditions.

## Project Goals

- Implement an attention-based image captioning model.
- Generate multiple candidate captions using beam search.
- Apply CLIP-based reranking based on image-text alignment.
- Evaluate caption quality using standard captioning metrics.
- Analyze model robustness under degraded image conditions such as blur, darkness, cropping, and occlusion.
- Provide qualitative visualizations such as attention heatmaps and example failure cases.

## Key Outputs

1. A working Git repository containing:
   - data preprocessing code,
   - model training code,
   - caption generation and CLIP reranking code,
   - evaluation scripts,
   - demo or inference examples.

2. A quantitative evaluation and robustness analysis report, including:
   - comparison between captioning variants with and without CLIP reranking,
   - caption-quality metrics such as CIDEr, METEOR, SPICE, and CLIPScore,
   - robustness analysis under degraded image conditions.
