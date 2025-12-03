---
title: "Masked Diffusion Language Models Latter Part"
customTheme: "https://comping-style.qihang-zhang.com/stylesheets/slides.css"
---

# Masked Diffusion Language Models

<div style="text-align: center;">

> [**Paper Link:** Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)

</div>

<div style="text-align: center;"> 

**Presenter:** Qihang Zhang

</div>
<br>
<div style="text-align: center;">Dec 3rd, 2025</div>

---

## Today's Agenda

1. Sampling/Inference Process
2. Experimental Results
3. Key Findings & Conclusions

---

## Sampling / Inference Process

---

## Experimental Setting: PPL

Perplexity evaluates how well the model predicts the reference tokens:

$$
\mathrm{PPL} = \exp \left(\mathbb{E} - \frac{1}{N} \sum_{i=1}^{N} \log P(x_i \mid x_{<i})\right)
$$

Lower PPL means the model assigns higher probability to the observed sequence (better predictive quality).

