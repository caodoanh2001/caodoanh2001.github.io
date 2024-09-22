---
layout: post
title:  One paper has been accepted by ACCV2024
date:   2024-09-20 00:00:00
description: One paper has been accepted by ACCV2024
tags: formatting links
categories: sample-posts
inline: false
---

## MECFormer: Multi-task Whole Slide Image Classification with Expert Consultation Network
*Doanh C. Bui and Jin Tae Kwak*

**Abstract:** Whole slide image (WSI) classification is a crucial problem for cancer diagnostics in clinics and hospitals. A WSI, acquired at gigapixel size, is commonly tiled into patches and processed by multiple-instance learning (MIL) models. Previous MIL-based models designed for this problem have only been evaluated on individual tasks for specific organs, and the ability to handle multiple tasks within a single model has not been investigated. In this study, we propose MECFormer, a generative Transformer-based model designed to handle multiple tasks within one model. To leverage the power of learning multiple tasks simultaneously and to enhance the model's effectiveness in focusing on each individual task, we introduce an Expert Consultation Network, a projection layer placed at the beginning of the Transformer-based model. Additionally, to enable flexible classification, autoregressive decoding is incorporated by a language decoder for WSI classification. Through extensive experiments on five datasets involving four different organs, one cancer classification task, and four cancer subtyping tasks, MECFormer demonstrates superior performance compared to individual state-of-the-art multiple-instance learning models.

<img src="https://caodoanh2001.github.io/assets/img/mecformer.png" data-canonical-src="https://caodoanh2001.github.io/assets/img/mecformer.png" width="750" height="500" />

This work was conducted under the supervision of Prof. Jin Tae Kwak during my master's degree.
