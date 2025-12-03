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

## Experimental Setting: 

**PPL**: Perplexity evaluates how well the model predicts the reference tokens:

$$
\mathrm{PPL} = \exp \left(\mathbb{E} _{x \sim \mathbb{P} _{data}} - \frac{1}{N(x)} \sum _{i=1}^{N(x)} \log P(x _i \mid x _{<i})\right),
$$

Where:
+ $N(x)$ is the number of tokens of the sequence $x$.
+ PPL is indeed the exponential of the average negative log-likelihood per token.

---
## Calculate PPL for AR and Diffusion Models

- **AR Models**: We can directly compute PPL.
- **Diffusion Models**: We approximate PPL.
    <!-- TODO: estimate the bound of PPL-->

---
## 

---
## Table 1: Main Results on Language Modeling Benchmarks

---


