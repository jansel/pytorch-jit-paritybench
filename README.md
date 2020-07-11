A test suite to measure TorchScript parity with PyTorch on many `nn.Module`s
crawled from popular GitHub projects.


###  Running ParityBench

- [Install conda] with python>=3.8
and create/activate a [conda environment]

- Install requirements:
```
conda install pip
pip install -r requirements.txt
conda install pytorch torchvision cpuonly -c pytorch-nightly
```

- Run `python main.py`, you should see an output like:
```
TorchScript ParityBench:
          total  passing  score
projects   1172      346  29.5%
tests      8292     4734  57.1%
```
A file `errors.csv` is generated containing the top error messages and example
`generated/*` files to reproduce those errors.

[Install conda]: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
[conda environment]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


### Regenerate ParityBench

*WARNING*: this will download 10+ gigabytes of code from crawling github and
take days to complete.  It is likely not necessary for you to do this.
```
python main.py --download
python main.py --generate-all
```




