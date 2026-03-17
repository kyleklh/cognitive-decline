<a id="readme-top"></a>

[![Contributors][contributors-count-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<br />
<div align="center">
<h2 align="center">APSC103: Cognitive Decline Behavior Segmentation</h3>

<p align="center">
Video-based behavior feature extraction and HMM-based classification for normal, borderline, and exit-seeking movement patterns.
<br />

</div>

<details>
	<summary>Table of Contents</summary>
	<ol>
		<li>
			<a href="#about-the-project">About The Project</a>
			<ul>
				<li><a href="#built-with">Built With</a></li>
				<li><a href="#project-structure">Project Structure</a></li>
			</ul>
		</li>
		<li>
			<a href="#getting-started">Getting Started</a>
			<ul>
				<li><a href="#prerequisites">Prerequisites</a></li>
				<li><a href="#installation">Installation</a></li>
			</ul>
		</li>
		<li><a href="#usage">Usage</a></li>
		<li><a href="#roadmap">Roadmap</a></li>
		<li><a href="#top-contributors">Top contributors</a></li>
		<li><a href="#license">License</a></li>
	</ol>
</details>

## About The Project

This repository contains a behavior-analysis pipeline for cognitive-decline experiments, built around per-frame video segmentation and Hidden Markov Model (HMM) sequence modeling.

Pipeline summary:
* Segment subject movement from videos using SAM3.
* Extract temporal behavior features:
	* `d_t`: distance to exit center
	* `tau_t`: dwell time near exit
	* `n_t`: count of entries into near-exit zone
	* `v_t`: approach velocity toward exit
* Train a 3-state Gaussian HMM over extracted sequences.
* Classify clip-level behavior as `normal`, `borderline`, or `exit_seek`.

### Built With

* [Python](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [Ultralytics](https://docs.ultralytics.com/)
* [hmmlearn](https://hmmlearn.readthedocs.io/)
* [scikit-learn](https://scikit-learn.org/)
* [Pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)

### Project Structure

```text
cognitive-decline/
├── features/
├── models/
├── src/
│   ├── config.py
│   ├── hmm/
│   │   ├── train_hmm.py
│   │   ├── detect_behavior.py
│   │   └── visualize_hmm.py
│   └── sam/
│       ├── extract_features.py
│       └── exit_region_test.py
└── videos/
```

Directory guide:
* `features/`: per-video extracted feature CSV files
* `models/`: model artifacts (`sam3.pt`, `hmm_model.pkl`, `scaler.pkl`, `hmm_meta.pkl`)
* `src/config.py`: experiment paths, exit region, and thresholds
* `src/hmm/`: HMM training, inference, and visualization scripts
* `src/sam/`: SAM-based feature extraction and exit-region debug scripts
* `videos/`: input videos

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

Follow the steps below to run the full pipeline locally.

### Prerequisites

* Python 3.12+
* CUDA-capable GPU (recommended for SAM3 video segmentation)
* Git

### Installation

1. Clone the repository
	 ```sh
	 git clone https://github.com/kyleklh/cognitive-decline.git
	 cd cognitive-decline
	 ```

2. Create and activate a virtual environment
	 ```sh
	 python -m venv .venv
	 .venv\Scripts\activate
	 ```

3. Install dependencies
	 ```sh
	 pip install numpy pandas matplotlib scikit-learn hmmlearn joblib opencv-python ultralytics
	 ```

4. Ensure model files are available in `models/`
	 * `sam3.pt` for feature extraction
	 * `sam3.pt` is available from Hugging Face: [facebook/sam3](https://huggingface.co/facebook/sam3)
	 * `hmm_model.pkl`, `scaler.pkl`, `hmm_meta.pkl` after training

5. Configure experiment constants in `src/config.py`
	 * `EXIT_REGION`
	 * `NEAR_EXIT_RADIUS`
	 * folder/model paths if needed

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

1. Validate exit region overlay on a sample video
	 ```sh
	 python src/sam/exit_region_test.py
	 ```

2. Extract behavior features from videos in `videos/`
	 ```sh
	 python src/sam/extract_features.py
	 ```

3. Train HMM on CSV files in `features/`
	 ```sh
	 python src/hmm/train_hmm.py
	 ```

4. Run behavior detection on one feature file
	 ```sh
	 python src/hmm/detect_behavior.py features/IMG_8312.csv
	 ```

5. Visualize decoded state timeline and role scores
	 ```sh
	 python src/hmm/visualize_hmm.py features/IMG_8312.csv
	 ```

Expected output classes:
* `normal`
* `borderline`
* `exit_seek`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Implement per-video SAM3 feature extraction
- [x] Train sequence model with Gaussian HMM
- [x] Add behavior summary scoring (`normal`, `borderline`, `exit_seek`)
- [x] Add timeline and score visualization
- [ ] Add reproducible environment lockfile (`requirements.txt` / `environment.yml`)
- [ ] Add unit tests for feature and scoring helpers

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[contributors-count-shield]: https://img.shields.io/badge/contributors-5-1f6feb?style=for-the-badge
[contributors-url]: https://github.com/kyleklh/cognitive-decline/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/kyleklh/cognitive-decline.svg?style=for-the-badge
[forks-url]: https://github.com/kyleklh/cognitive-decline/network/members
[stars-shield]: https://img.shields.io/github/stars/kyleklh/cognitive-decline.svg?style=for-the-badge
[stars-url]: https://github.com/kyleklh/cognitive-decline/stargazers
[issues-shield]: https://img.shields.io/github/issues/kyleklh/cognitive-decline.svg?style=for-the-badge
[issues-url]: https://github.com/kyleklh/cognitive-decline/issues
