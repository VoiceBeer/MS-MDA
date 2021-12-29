# MS-MDA
Code of paper MS-MDA: Multisource Marginal Distribution Adaptation for Cross-subject and Cross-session EEG Emotion Recognition

## Datasets
The dataset files (SEED and SEED-IV) can be downloaded from the [BCMI official website](https://bcmi.sjtu.edu.cn/~seed/index.html)

To facilitate data retrieval, we divided both datasets into three folders according to the sessions, the file structure of the datasets should be like:
```
eeg_feature_smooth/
    1/
    2/
    3/
ExtractedFeatures/
    1/
    2/
    3/
```


## Usage
Run `python msmdaer.py`, and the results will be printed in the terminal.

## Contributing
Issues are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Citation
If you find our work useful for your research, please consider citing our paper as:

```bibtex
@article{chen2021ms,
  title={MS-MDA: Multisource Marginal Distribution Adaptation for Cross-subject and Cross-session EEG Emotion Recognition},
  author={Chen, Hao and Jin, Ming and Li, Zhunan and Fan, Cunhang and Li, Jinpeng and He, Huiguang},
  journal={arXiv preprint arXiv:2107.07740},
  year={2021}
}
```
## TODO
- [ ] LOSO experiments on SEED and SEED-IV, methods including DDC, DAN, DCORAL, MS-MDA, on two transfer scenarios (cross-subject, cross-session)

## LOSO Experiments
In our paper, Section III. B. Scenarios mentioned:

> we take the first 2 session data from one subject as the source domains for cross-session transfer, and take the first 14 subjects data from one session as the source domains for cross-subject transfer. The results of cross-session scenarios are averaged over 15 subjects, and the results of cross-subject are averaged over 3 sessions. Standard deviations are also calculated.

However, as descriped in ISSUE [3](https://github.com/VoiceBeer/MS-MDA/issues/3), LOSO (Leave-one-subject-out) is also required, we therefore additionally evaluated our method in the LOSO paradigm with compared works (In batch size of {16, 32, 64, 128, 256, 512}). Note that this LOSO experiments are not included in the original paper, and since other works have not yet made their code open source, thus we reproduced some of them. The results are shown below (csesn stands for cross-session, csub stands for cross-subject, the number next to it represents to batch size):

| Dataset | Method | csesn_16 | csub_16 | csesn_32 | csub_32 | csesn_64 | csub_64 | csesn_128 | csub_128 | csesn_256 | csub_256 | csesn_512 | csub_512 | 
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| **SEED** | DDC | 89.89±6.69 | 82.21±7.15 | 90.65±6.41 | 81.27±6.83 | 88.16±6.89 | 78.91±7.62 | 84.98±7.64 | 72.78±8.25 | 80.37±8.62 | 68.98±6.34 | 68.32±8.62 | 57.21±6.52 |
| | DAN | 88.88±7.02 | 82.22±7.09 | 88.57±7.60 | 81.10±6.63 | 87.12±7.20 | 79.03±7.07 | 83.15±7.30 | 71.95±6.55 | 79.84±9.42 | 68.48±6.74 | 67.67±8.41 | 57.11±6.57 |
| | DCORAL | 76.40±10.49 | 66.39±7.55 | 76.24±8.51 | 64.98±8.42 | 74.65±10.46 | 65.40±9.27 | 73.74±9.09 | 64.05±8.38 | 77.33±11.10 | 62.50±6.77 | 65.49±9.95 | 57.43±8.49 |
| | MS-MDA | | | | | | | | | 87.68±9.22 | 78.78±10.70 | 79.93±9.90 | 72.31±10.17 | 
| **SEED-IV** | DDC | | | | | | | | | | | | |
| | DAN | | | 71.51±11.98 | 63.57±9.07 | 67.12±13.47 | 59.11±7.99 | 58.63±11.77 | 47.50±8.80 | 52.62±11.91 | 41.47±8.02 | 27.79±8.43 | 21.30±4.98 |
| | DCORAL | | | | | | | | | | | | |
| | MS-MDA | | | | | | | | | | | | |



## License
This source code is licensed under the MIT license
