# MS-MDA
Code of paper MS-MDA: Multisource Marginal Distribution Adaptation for Cross-subject and Cross-session EEG Emotion Recognition

<div align="center">
  <img width="70%" alt="SimCLR Illustration" src="https://www.frontiersin.org/files/Articles/778488/fnins-15-778488-HTML/image_m/fnins-15-778488-g001.jpg">
</div>


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

## LOSO Experiments
> **Warning**
> ATTENTION! The results here are different from those in the paper due to variations in the experimental setup.

In our paper, Section III. B. Scenarios mentioned:

> we take the first 2 session data from one subject as the source domains for cross-session transfer, and take the first 14 subjects data from one session as the source domains for cross-subject transfer. The results of cross-session scenarios are averaged over 15 subjects, and the results of cross-subject are averaged over 3 sessions. Standard deviations are also calculated.

However, as described in ISSUE [3](https://github.com/VoiceBeer/MS-MDA/issues/3), LOSO (Leave-one-subject-out) is also required, we therefore additionally evaluated our method in the LOSO paradigm with compared works (In the batch size of {16, 32, 64, 128, 256, 512}). Note that these LOSO experiments are not included in the original paper, and since other works have not yet made their code open-source, we reproduced some of them. The results are shown below (csesn stands for cross-session, csub stands for cross-subject, the number next to it represents batch size, the best result for one transfer scenario is in bold):

| Dataset | Method | csesn_512 | csub_512 | csesn_256 | csub_256 | csesn_128 | csub_128 | csesn_64 | csub_64 | csesn_32 | csub_32 | csesn_16 | csub_16 |
|   :---  |  :---  |   :---:   |   :---:  |   :---:   |   :---:  |   :---:   |   :---:  |    :---: |   :---: |   :---:  |  :---:  |   :---:  |  :---:  | 
|**SEED** | DCORAL |65.49±9.95 |57.43±8.49|77.33±11.10|62.50±6.77|73.74±9.09 |64.05±8.38|74.65±10.46|65.40±9.27|76.24±8.51|64.98±8.42|76.40±10.49|66.39±7.55|
|         |   DAN  |67.67±8.41 |57.11±6.57|79.84±9.42 |68.48±6.74|83.15±7.30 |71.95±6.55|87.12±7.20|79.03±7.07|88.57±7.60|81.10±6.63|88.88±7.02|82.22±7.09|
|         |   DDC  |68.32±8.62 |57.21±6.52|80.37±8.62 |68.98±6.34|84.98±7.64 |72.78±8.25|88.16±6.89|78.91±7.62|**90.65±6.41**|81.27±6.83|89.89±6.69|82.21±7.15|
|         | MS-MDA |**79.93±9.90**|**72.31±10.17**|**87.68±9.22**|**78.78±10.70**|**87.20±10.76**|**80.33±10.00**|**89.76±9.03**|**80.91±9.38**|90.38±7.03|**81.57±9.81**|**90.65±8.01**|**82.67±9.51**|
|**SEED-IV**| DCORAL |24.80±7.18|20.12±5.31|42.71±9.47|39.77±8.27|48.60±12.53|41.48±7.72|54.39±10.90|45.95±7.19|57.48±11.55|48.12±6.35|59.61±10.03|51.85±7.30|
|           |   DAN  |27.79±8.43|21.30±4.98|52.62±11.91|41.47±8.02|58.63±11.77|47.50±8.80|67.12±13.47|59.11±7.99|**71.51±11.98**|63.57±9.07|74.40±12.67|69.68±9.24|
|           |   DDC  |27.18±6.83|21.94±5.89|53.30±11.42|42.67±9.81|59.00±11.48|48.50±8.44|**67.94±12.20**|58.18±8.20|71.00±13.25|64.00±8.52|**74.81±12.65**|**69.90±9.95**|
|           | MS-MDA |**37.75±10.26**|**36.03±8.99**|**62.01±13.43**|**56.20±12.85**|**64.04±15.27**|**61.06±12.69**|66.87±15.69|**62.77±11.23**|70.31±15.35|**65.12±13.85**|72.77±14.71|67.96±11.94|

## Usage
Run `python msmdaer.py`, and the results will be printed in the terminal.

## Contributing
Issues are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Citation
If you find our work useful for your research, please consider citing our paper as:

```bibtex
@article{chen2021ms,
  title={MS-MDA: Multisource marginal distribution adaptation for cross-subject and cross-session EEG emotion recognition},
  author={Chen, Hao and Jin, Ming and Li, Zhunan and Fan, Cunhang and Li, Jinpeng and He, Huiguang},
  journal={Frontiers in Neuroscience},
  volume={15},
  pages={778488},
  year={2021},
  publisher={Frontiers}
}

@inproceedings{chen2021meernet,
  title={MEERNet: Multi-source EEG-based Emotion Recognition Network for Generalization Across Subjects and Sessions},
  author={Chen, Hao and Li, Zhunan and Jin, Ming and Li, Jinpeng},
  booktitle={2021 43rd Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  pages={6094--6097},
  year={2021},
  organization={IEEE}
}
```

## TODO
- [x] LOSO experiments on SEED and SEED-IV, methods including DDC, DAN, DCORAL, MS-MDA, on two transfer scenarios (cross-subject, cross-session)
- [ ] ISSUE [6](https://github.com/VoiceBeer/MS-MDA/issues/6)

## License
This source code is licensed under the MIT license

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=VoiceBeer/MS-MDA&type=Date)](https://star-history.com/#VoiceBeer/MS-MDA&Date)

