# Installation

This guide will walk through the installation process.

## Required Software

IMPORTANT: You must install correct version of chromedriver (matching with your chrome version)
[chromedriver.exe](https://chromedriver.chromium.org/downloads)

## Setup

Copy chromedriver.exe and paste into repo folder, next to t-rex-run.py.

## Run

Train:
```shell
python t-rex-train.py <version_tag> <save_gap>
```

Example of saved model: t-rex-v<version>-dqn<training episode>.h5

Argument version_tag indicates the version name of saving model, default value is 0

Argument save_gap indicates the frequency of saving training model, default value is 100.

We allow user to ignore both argument at same time to use default value, but user cannot only provide one of them

Run:
```shell
python t-rex-run.py <model_file>
```

Model files are saved in Original-Model and Improved-Model

## Terminate

Use `Ctrl+C` to terminate the application.
