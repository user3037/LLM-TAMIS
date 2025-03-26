# LLM-TAMIS

## Overview
LLM-TAMIS is a large language model-based text-augmented medical image segmentation method. It integrates both visual and textual data. By leveraging large language models (LLMs) and multimodal data, our method enhances segmentation accuracy by incorporating relevant contextual information from clinical reports and annotations.

Additionally, we introduce a new Spatial Channel Driven Module (SCDM), which refines feature extraction by focusing on important spatial regions and relevant feature channels. This results in more accurate and robust segmentation. We evaluate our method on two publicly available medical image datasets, demonstrating that our approach outperforms state-of-the-art methods in segmentation accuracy, achieving superior results across multiple evaluation metrics.

## Features

- **Multimodal Integration:** LLM-TAMIS effectively combines features from medical images and contextual information from clinical text annotations, addressing the limitations of existing image-only segmentation methods.
- **Spatial Channel Driven Module (SCDM):** Introduces a novel module that enhances the extraction and fusion of multimodal features, improving segmentation accuracy.
- **State-of-the-Art Performance:** Our method is comprehensively evaluated on two diverse datasets, **MosMed-Data+** and **QaTa-COV19**, demonstrating superior performance and robustness compared to existing methods.
