# Compressed Text-to-speech (Tacotron 2 + WaveGlow)
**Authors:** Ivan Vovk, Grigoriy Sitnikov, Vladimir Gogoryan, Dmitriy Smorchkov

## **Compression results**
**Status:** in progress.
___
## **Installation**
To install all necessary libraries, open the terminal, `cd` to the project folder and run the command `pip install -r requirements.txt`.
___
## **References**
This project is based on the following repositories:
* NVIDIA's [Tacotron 2](https://github.com/NVIDIA/tacotron2)
* NVIDIA's [WaveGlow](https://github.com/NVIDIA/waveglow)

Trained models:
* [Tacotron 2](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view)
* [WaveGlow](https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ljs_256channels)

Put them all into the `checkpoints` directory.
___
## **Synthesis**
In order to perform synthesis, run `inference.ipynb` notebook from the project root folder and follow the intructions.
___
## **Training**
### Tacotron 2
To train the model go to the `tacotron2` folder and set the proper `config.json` file (you can use default), after run from the terminal from the `tacotron2` folder the command `python distributed.py -c config.json` (It will run training process on available GPUs).
### WaveGlow
Do the same things as for the Tacotron 2.
