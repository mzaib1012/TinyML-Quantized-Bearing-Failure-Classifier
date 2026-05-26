# TinyML Quantized Classifier for Real-Time Bearing Failure via Acceleration Data

## 📌 Project Overview

This repository contains an end-to-end TinyML post-training quantization pipeline engineered for edge-based predictive maintenance. In industrial automation, heavy rotating machinery relies on rolling-element bearings that are highly susceptible to localized structural degradation (including inner race, outer race, and ball element fatigues) due to mechanical stress. Continuous high-frequency vibrational or acoustic telemetry monitoring is crucial to prevent catastrophic failure. However, streaming raw sensor datasets to centralized cloud servers introduces high latency and massive bandwidth overhead.

This project solves this bottleneck by training a highly compact 1D Convolutional Neural Network (CNN) to ingest high-frequency raw acceleration data and identify early-stage structural micro-fractures directly at the sensor node. To make deployment possible on low-cost, resource-constrained edge hardware lacking dedicated Floating-Point Units (FPUs), the network undergoes rigorous 8-bit post-training integer quantization ($INT8$). This shrinks the final binary footprint well below a strict 50 KB flash ceiling, proving it can be flashed onto standard 32-bit microcontrollers.

## ⚡ Technical Architecture

The software and compilation framework executes across four main operational blocks:

* **Programmatic Telemetry Synthesis Engine:** Generates high-frequency acceleration data streams mapping distinct operating regimes, including healthy baselines, high-frequency inner race impacts, structural outer race periodic repetitions, and transient ball element bursts.
* **1D Convolutional Neural Network (CNN) Classifier:** Establishes a localized feature extraction baseline, feeding the normalized time-domain input windows into a sequence of a 1D convolutional layer, a max-pooling operation, and a dense classification head to map fault signatures.
* **Representative Data Calibration Engine:** Batches a subset of raw training variables without labels through a generator loop, capturing active internal layer activation distributions to extract accurate dynamic scalar ranges.
* **INT8 Tensor-Type Optimization Compiler:** Ingests the baseline graph parameters via the TensorFlow Lite Converter and enforces a strict integer-only weight calculation conversion across all underlying convolutional filters and inner tensor connections.

## 📊 Pipeline Compliance & Diagnostic Outputs

The quantization pipeline has been validated and compiled to a deployment-ready hardware binary. The core output is the **`bearing_fault_model.tflite`** file, which acts as the finalized, quantized inference engine ready for flash-loading onto edge hardware.

The project achieves high-fidelity signal characterization, building complex time-series windows that emulate standard mechanical datasets like the Case Western Reserve University (CWRU) bearing telemetry repository. The baseline $Float32$ model requires approximately **129.89 KB**, which exceeds typical microcontroller memory constraints. Through strict $INT8$ integer optimization, the structure is compressed by over **70%**, resulting in a final binary size of **38.88 KB**, successfully meeting the strict compliance target of $<50\text{ KB}$.

**TinyML Model Validation Report**
tinyml_bearing_performance_report.png


## 🛠️ How to Replicate

1. Open the file `notebooks/tinyml_bearing_fault_classifier.ipynb` inside [Google Colab](https://colab.research.google.com/).
2. Run the processing blocks sequentially to generate the vibrational telemetry, train the baseline deep classifier, and activate the $INT8$ quantization compiler.
3. The pipeline automatically exports the hardware-ready binary **`bearing_fault_model.tflite`** to your local workspace, alongside the **`tinyml_bearing_performance_report.png`** performance bar layout.

## 📂 Repository Structure

```text
├── notebooks/
│   └── tinyml_bearing_fault_classifier.ipynb   # Edge AI optimization and quantization script
├── models/
│   └── bearing_fault_model.tflite              # Compiled 38.88 KB INT8 hardware model file
├── assets/
│   └── tinyml_bearing_performance_report.png   # Signal layout and compression bar plots
└── README.md                                   # Targeted project documentation
