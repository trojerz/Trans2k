# Trans2k and Rendering Machine



## Citation 

If you use BlenderProc in a research project, please cite as follows:

```
add here
```


## Installation

### Git clone

Clone the repository:

```bash
git clone https://github.com/trojerz/Trans2k
```

### Download objects

Download objects from [here](https://drive.google.com/drive/folders/1vX4Jf1Ej_wIdfaFyVgvno6_RbqMyrf8-?usp=sharing). Put all objects in the `objects` folder.

### Environment

Create a new conda environment:

```bash
conda env create --name blender --file=environment.yml
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
To do so, set the debug_mode in line 870 and call your script with the `debug` instead of `run` subcommand:
```bash
debug_mode = 1
```

```bash
blenderproc debug rendeding_machine.py
```

Now the Blender UI opens up, the scripting tab is selected and the correct script is loaded. To start the Rendering Machine pipeline, one now just has to press `Run BlenderProc`.

