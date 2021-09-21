# MS-MDAER
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

## License
