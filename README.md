# Rendering Machine and Trans2k dataset

![](example.gif)

## Citation 

If you use Trans2k or Rendering Machine in a research project, please cite as follows:

```
add here
```

## Installation

### Git clone

Clone the repository:

```bash
git clone https://github.com/trojerz/Trans2k
```

### Environment

Create a new conda environment:

```bash
conda env create --name blender --file=environment.yml

conda activate blender
```

### Download objects

Download objects from [here](https://drive.google.com/drive/folders/1vX4Jf1Ej_wIdfaFyVgvno6_RbqMyrf8-?usp=sharing). Put all objects in the `objects` folder.

### Download GOT-10k
Download GOT-10k dataset from [here](http://got-10k.aitestunion.com/downloads_dataset/full_data) and put it into `RenderingMachine/GOT10k` folder.

### Data file structure
The downloaded and extracted train dataset should follow the file structure:

```
|--RenderingMachine/
|   |--GOT10k/
|   |   |-- GOT-10k_Train_000001/
|   |   |    ......
|   |   |-- GOT-10k_Train_009335/
```

### Pre-processing data
All images in GOT-10k should first be the resized to 1280x720 (otherwise, backgrounds will not be placed correctly in the scene):

```bash
python preprocess_dataset.py
```

## Usage

Rendering Machine has to be run inside the blender python environment, as only there we can access the blender API. 
Therefore, instead of running rendering machine with the usual python interpreter, the command line interface of BlenderProc has to be used.

```bash
blenderproc run rendeding_machine.py
```

In general, one run of the script first loads or constructs a 3D scene, then sets some camera poses inside this scene and renders a transparent object on background for each of those camera poses. Usually, the script is run multiple times, each time producing one sequence containing around 50 images (it is recommended to not increase the length, because of memory issues).

### Debugging in the Blender GUI

To understand how the scene in constructed, BlenderProc has the great feature of visualizing everything inside the blender UI.
To do so, call the script with the additional `debug` instead of `run` subcommand and add additional argument `1`:

```bash
blenderproc debug rendeding_machine.py 1
```

Now the Blender UI opens up, the scripting tab is selected and the correct script is loaded. To start the Rendering Machine pipeline, one now just has to press `Run BlenderProc`. Note that sequences will not be rendered, only the scene will be generated inside blender UI.

### Post-processing of data

When finished rendering sequences, additional files need to be created (for example, `groundtruth.txt`, `absence.label`, etc.). Go into `RenderingMachine/RenderedVideos/GOT10k` and run the following command:

```bash
bash postprocess.sh
```

The script will create all necessary files (`absence.label`, `cover.label`, `cut_by_image.label`, `groundtruth.txt`, `meta_info.ini`, `list.txt` and `got10k_train_full_split.txt`). It will delete sequences, which were rendered without groundtruth labels (it happens sometimes). 