# Trans2k Dataset and Rendering Engine

![](example.gif)


## Citation 

If you use Trans2k or Rendering Engine in a research project, please cite as follows:

```
@InProceedings{trans2k_bmvc2022,
  Title = {Trans2k: Unlocking the Power of Deep Models for Transparent Object Tracking},
  Author = {Lukezic, Alan and Trojer, Ziga and Matas, Jiri and Kristan, Matej},
  Booktitle = {In Proceedings of the British Machine Vision Conference (BMVC)},
  Year = {2022}
}
```

Paper - https://arxiv.org/abs/2210.03436.

**The paper received the [Best Paper Prize at the BMVC 2022](https://bmvc2022.org/programme/paper-awards/)** :trophy:

Visual object tracking has focused predominantly on opaque objects, while transparent object tracking received very little attention. Motivated by the uniqueness of transparent objects in that their appearance is directly affected by the background, the first dedicated evaluation dataset has emerged recently.
We contribute to this effort by proposing the first transparent object tracking training dataset Trans2k that consists of over 2k sequences with 104,343 images overall, annotated by bounding boxes and segmentation masks. Noting that transparent objects can be realistically rendered by modern renderers, we quantify domain-specific attributes and render the dataset containing visual attributes and tracking situations not covered in the existing object training datasets. We observe a consistent performance boost (up to 16%) across a diverse set of modern tracking architectures when trained using Trans2k, and show insights not previously possible due to the lack of appropriate training sets.


## Trans2k Dataset
Trans2k dataset contains 2,039 challenging sequences and 104,343 frames in total. We provide the ground truth in two standard forms, the widely accepted target enclosing axis-aligned bounding-box and the segmentation mask.

The dataset is available for download [here](https://go.vicos.si/trans2k). This is a compressed version of the dataset, which size is approximately 7.5 GB and uses a jpeg compression. The uncompressed version (approx. 125 GB), which could be used to exactly reproduce the results from the paper is available on request only. If you would like to get the uncompressed dataset, please contact alan.lukezic@fri.uni-lj.si. Note that the visual difference between frames from both versions is minor.


### Trained models with setting files
Trained models with Trans2k are available [here](https://drive.google.com/drive/folders/1EjXEqPa2WuQtixvFkTG9_kzAhVdTp2xC?usp=sharing). Original TOTB dataset can be found [here](https://hengfan2010.github.io/projects/TOTB/) and dataset, appropriate for Stark and Pytracking can be found [here](https://drive.google.com/drive/folders/1vkrWedoy5_VoRXUmmZwrAu7rv5tImrhl?usp=sharing). You can find additional files that can help you setting the environment for training/evaluating different trackers in the `Trans2k` folder.

## Installation of Rendering Engine

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

## Usage of Rendering Engine

### Examples

* [Basic scene](Examples/basic_scene/README.md): Basic example for construction of the scene, this is the ideal place to get an idea how sequences are generated.
* [Advanced  tutorial](Examples/advanced/README.md): for advanced user: how to change parameters, different settings for rendering engine.

### Generating dataset


Rendering Engine has to be run inside the blender python environment, as only there we can access the blender API. 
Therefore, instead of running rendering engine with the usual python interpreter, the command line interface of BlenderProc has to be used.

```bash
blenderproc run rendeding_machine.py
```

In general, one run of the script first loads or constructs a 3D scene, then sets some camera poses inside this scene and renders a transparent object on background for each of those camera poses. Usually, the script is run multiple times, each time producing one sequence containing around 50 images (it is recommended to not increase the length, because of memory issues).

### Debugging in the Blender UI

To understand how the scene in constructed, BlenderProc has the great feature of visualizing everything inside the blender UI.
To do so, call the script with the additional `debug` instead of `run` subcommand and add additional argument `1`:

```bash
blenderproc debug rendeding_machine.py 1
```

Now the Blender UI opens up, the scripting tab is selected and the correct script is loaded. To start the Rendering Engine pipeline, one now just has to press `Run BlenderProc`. Note that sequences will not be rendered, only the scene will be generated inside blender UI.

### Post-processing of data

When finished rendering sequences, additional files need to be created (for example, `groundtruth.txt`, `absence.label`, etc.). Go into `RenderingMachine/RenderedVideos/GOT10k` and run the following command:

```bash
bash postprocess.sh
```

The script will create all necessary files (`absence.label`, `cover.label`, `cut_by_image.label`, `groundtruth.txt`, `meta_info.ini`, `list.txt` and `got10k_train_full_split.txt`). It will delete sequences, which were rendered without groundtruth labels (it happens sometimes). 
