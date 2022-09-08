# Advanced tutorial

In this example we show, which parameters can be set to generate appropriate sequences. This tutorial is divided into two sections: [The first section](#Parameters-for-Trans2k) describes the parameters, used for generating Trans2k dataset, while [the second one](#Non-standard-setup) describes non-standard setup to generate sequences, containing objects with "glass-like" appearance.

## Parameters for Trans2k
This section describes the parameters, that were used when generating Trans2k dataset. The described parameters can be changed. For those settings, set the parameter `render_material="Plastic"`.

### Transparency profiles
Each transparent profile is defined with 6 parameters: `Transmission`, `Base Color`, `Alpha`, `IOR`, `Clearcoat` and `Roughness`. Additional parameters can be added, but they need to be added in `set_plastic_material` function. More information on parameters is available [here](https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html).

```
{"transparency_profile_1": {"Transmission": value,
			    "Base Color": value,
			    "Alpha": value,
			    "IOR": value,
			    "Clearcoat": value,
			    "Roughness": value},
			    ...
"transparency_profile_n": {"Transmission": value,
			    "Base Color": value,
			    "Alpha": value,
			    "IOR": value,
			    "Clearcoat": value,
			    "Roughness": value}
}
```

Parameter `profiles` then contains transparency profile names, which will be used for rendering:

```
profiles = ['transparency_profile_1', 'transparency_profile_2']
```

### Occlusion
Level of occlusion is parameterized as the number of non-transparent stripes, rendered in front of the object. Parameter `occlusion_types` defines the spacing between each stripe and `occlusion_names` defines the names of the occlusion (more space between the stripes, less occlusion):

```
occlusion_types = [0.01, 0.02, 0.04]

occlusion_names = ['high_occlusion', 'medium_occlusion', 'low_occlusion']
```

### Motion blur
Parameter `blur_intensity_vec` defines the level of motion blur, where 0 means no blur and 1 means very high blur.

### Objects
You can download objects, that were used in Trans2k [here](https://drive.google.com/drive/folders/1vX4Jf1Ej_wIdfaFyVgvno6_RbqMyrf8-?usp=sharing). You can add additional object models, only constraint is that is needs to be in the `.obj` format.

### Length of the sequence
Parameter `k` defines the length of the rendered sequence. First `k` backgrounds will be imported to the scene. It is recommended to keep this parameter 50, but not more than 100 (you can increase it if you don't experience memory issues).

### Other settings

One can change other settings in the script (after the line `902`). For example, the probability for including distractors can be changed in line `903` and for motion blur in line `973`. Description of parameters is available the script. Before rendering it is recommended to view scene with different settings in the debug mode.

## Non-standard setup

If you wish to experiment with different setup, where objects are of different transparent material (advanced settings for material), set the parameter `render_material="Glass"`. Values for this transparent material are already provided in glass_profiles. More information, how to set those values is available in this [tutorial](https://www.youtube.com/watch?v=JYyUMMboZFk&ab_channel=RyanKingArt).


