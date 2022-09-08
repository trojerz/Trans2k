import blenderproc as bproc
import argparse
import os
import math
import numpy as np
from scipy import interpolate
import random
import bpy
import time
import sys

class Preprocessing:
    def __init__(self, got_directory, result_directory, choose_n, only_new_videos, already_rendered):
        self.got_directory = got_directory
        self.result_directory = result_directory
        self.choose_n = choose_n
        self.chosen_videos = []
        self.original_videos = None
        self.only_new_videos = only_new_videos
        self.already_rendered = already_rendered

    def create_subdirectories(self):
        # Check if GOT videos even exist
        exists_ = os.path.exists(self.got_directory)
        # if GOT videos exists
        if exists_:
            # read which videos are available
            video_dirs = list(sorted([os.path.join(self.got_directory, p) for p in os.listdir(self.got_directory)]))
            # choose n random videos to render
            self.original_videos = random.sample(video_dirs, self.choose_n)
            # for each video, create new directory
            exclude_videos = list()
            for i, dir_ in enumerate(self.original_videos):
                video_name = dir_.split('/')[1]
                new_dir = str(self.result_directory) + '/' + dir_
                if video_name in self.already_rendered:
                    print(f'Video {new_dir} already rendered in previous batch! Skipping.')
                    continue
                exists_result_dir = os.path.exists(new_dir)
                if exists_result_dir:
                    if self.only_new_videos:
                        exclude_videos.append(i)
                        del self.original_videos[i]
                        print(f'Video {new_dir} already rendered! Skipping.')
                    else:
                        string_ = new_dir.split('_')
                        if len(string_[-1]) > 3:
                            new_dir_cand = new_dir + '_' + str(1)
                            lo = 1
                            while os.path.exists(new_dir_cand):
                                lo += 1
                                new_dir_cand = new_dir + '_' + str(lo)
                            new_dir = new_dir + '_' + str(lo)
                        self.chosen_videos.append(new_dir)
                        os.makedirs(new_dir, exist_ok=False)
                else:
                    self.chosen_videos.append(new_dir)
                    os.makedirs(new_dir, exist_ok=False)
        else:
            raise FileExistsError

    def get_rendered_video_path(self):
        return self.chosen_videos

    def get_original_video_path(self):
        return self.original_videos


class TransparentObjectRender:
    def __init__(self,
                 object_path,
                 output_path,
                 background_path,
                 resolution_x,
                 resolution_y,
                 dim_x,
                 dim_y,
                 dim_z,
                 x_position,
                 z_position,
                 k,
                 room_illumination,
                 materials,
                 rotation_x,
                 rotation_y,
                 rotation_z,
                 light_energy,
                 light_type,
                 light_position,
                 light_position_noise,
                 include_distractor,
                 distractor_objects,
                 distractor_position,
                 rotation_dist_x,
                 rotation_dist_y,
                 rotation_dist_z,
                 rotation_noise_lower,
                 rotation_noise_upper,
                 dim_dist_x,
                 dim_dist_y,
                 dim_dist_z,
                 overwrite_k,
                 random_scale_distractor,
                 random_scale_object,
                 add_occlusion,
                 add_occlusion_vertical,
                 distractor_follow_object,
                 over_the_edge,
                 blur_intensity,
                 line_width,
                 occlusion_color,
                 material_type,
                 debug_mode):

        self.x_position, self.z_position = x_position, z_position
        self.k, self.overwrite_k = k, overwrite_k
        self.material_dict = materials
        self.dimension_vec, self.dimension_dist_vec = [dim_x, dim_y, dim_z], [dim_dist_x, dim_dist_y, dim_dist_z]
        self.room_illumination, self.light_energy = room_illumination, light_energy
        self.light_type, self.light_position = light_type, light_position
        self.rot_x, self.rot_y, self.rot_z = rotation_x, rotation_y, rotation_z
        self.random_light_x_min, self.random_light_x_max = light_position_noise[0], light_position_noise[3]
        self.random_light_y_min, self.random_light_y_max = light_position_noise[1], light_position_noise[4]
        self.random_light_z_min, self.random_light_z_max = light_position_noise[2], light_position_noise[5]
        self.include_distractors, self.distractors_list = include_distractor, distractor_objects
        self.positions_x, self.positions_z = distractor_position[0], distractor_position[1]
        self.rot_dist_x, self.rot_dist_y, self.rot_dist_z = rotation_dist_x, rotation_dist_y, rotation_dist_z
        self.rotation_noise_lower, self.rotation_noise_upper = rotation_noise_lower, rotation_noise_upper
        self.object_path = object_path
        self.output_path = output_path
        self.background_path = background_path
        self.random_scale_distractor, self.random_scale_object = random_scale_distractor, random_scale_object
        self.possible_end = ('jpg', 'jpeg', 'png')
        self.add_occlusion = add_occlusion
        self.add_occlusion_vertical = add_occlusion_vertical
        self.random_color_occ = None
        self.random_position_occ = None
        self.distractor_follow_object = distractor_follow_object
        self.over_the_edge = over_the_edge
        self.glass_material = None
        self.first_image = None
        self.blur_intensity = blur_intensity
        self.line_width = line_width
        self.occlusion_color = occlusion_color
        self.new_scaling_dist = None
        self.new_scale_object = None
        self.original_object_dimension = None
        self.random_radius = random.uniform(1, 2)
        self.material_type = material_type
        self.debug_mode = debug_mode

        self.previous_object_size = (0, 0, 0)
        self.previous_distractor_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
        self.original_distractor_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]

        bproc.init(compute_device='GPU')
        bproc.camera.set_resolution(resolution_x, resolution_y)
        bpy.data.scenes['Scene'].cycles.device = 'GPU'

    @staticmethod
    def get_random_path(points, tangents, n_samples):
        """
        source: https://stackoverflow.com/questions/36644259/cubic-hermit-spline-interpolation-python
        :param points: Points for the trajectory go through
        :param tangents: Tangents the trajectory will not cross
        :param n_samples:  Number of points in the trajectory
        :return: Random trajectory
        """
        points = np.asarray(points)
        nPoints, dim = points.shape
        dp = np.diff(points, axis=0)
        dp = np.linalg.norm(dp, axis=1)
        d = np.cumsum(dp)
        d = np.hstack([[0], d])
        s, r = np.linspace(0, d[-1], n_samples, retstep=True)
        assert (len(points) == len(tangents))
        data = np.empty([nPoints, dim], dtype=object)
        for i, p in enumerate(points):
            t = tangents[i]
            assert (t is None or len(t) == dim)
            fuse = list(zip(p, t) if t is not None else zip(p, ))
            data[i, :] = fuse
        samples = np.zeros([n_samples, dim])
        for i in range(dim):
            poly = interpolate.BPoly.from_derivatives(d, data[:, i])
            samples[:, i] = poly(s)
        return samples

    @staticmethod
    def normalize(vec_):
        """
        :param vec_: vector to normalize
        :return: min max normalized vector
        """
        return (vec_ - np.min(vec_)) / (np.max(vec_) - np.min(vec_))

    @staticmethod
    def calculate_displacement(x, y, z, x_d, y_d, z_d):
        """
        :param x: position x
        :param y: position y
        :param z: position z
        :param x_d: step x
        :param y_d: step y
        :param z_d: step z
        :return: new position, defined by a step
        """
        return x + x_d, y + y_d, z + z_d

    def define_trajectory_with_noise(self, n_images, m=20):
        x_arry = np.arange(start=-1, stop=1, step=2 / n_images)
        z_arry = np.sin(x_arry * math.pi * 2)
        noise = np.random.uniform(-1 / n_images / m, 1 / n_images / m, n_images)
        return (2 * self.normalize(x_arry) - 1) * self.x_position, self.normalize(z_arry + noise) - 0.5

    def define_trajectory(self):
        """
        param n_images: Length of the trajectory (number of points)
        :return: Sine trajectory
        """
        if self.blur_intensity > 0:
            n_images = self.k * 2
        else:
            n_images = self.k
        x_arry = np.arange(start=-1, stop=1, step=2 / n_images)
        z_arry = np.sin(x_arry * math.pi * 2)
        return (2 * self.normalize(x_arry) - 1) * self.x_position / 3.5, (
                    2 * self.normalize(z_arry) - 1) * self.z_position / 3.5 * self.over_the_edge

    def define_circle(self):
        """
        param n_images: Length of the trajectory (number of points)
        :return: Sine trajectory
        """
        if self.blur_intensity > 0:
            n_images = self.k * 2
        else:
            n_images = self.k
        x_arry = np.arange(start=-1, stop=1, step=2 / n_images)
        z_arry = np.sin(x_arry * math.pi * 2)
        return (2 * self.normalize(x_arry) - 1) * self.x_position / 3.5, (
                    2 * self.normalize(z_arry) - 1) * self.z_position / 3.5 * self.over_the_edge

    def define_random_trajectory(self):
        """
        :return: numpy array of points that defines trajectory
        """
        points = []
        tangents = []
        if self.blur_intensity > 0:
            n_samples = self.k * 2
        else:
            n_samples = self.k
        for _ in range(4):
            points.append([random.randint(1, 9),
                           random.randint(1, 9)])
            tangents.append([random.randint(-1, 1),
                             random.randint(-1, 1)])
        points.sort()
        points = np.asarray(points)
        self.random_points = points
        tangents = np.asarray(tangents)
        tangents2 = np.dot(tangents, 2. * np.eye(2))
        samples2 = self.get_random_path(points, tangents2, n_samples)
        xarry = np.transpose(samples2)[0]
        zarry = np.transpose(samples2)[1]
        return (2 * self.normalize(xarry) - 1) * self.x_position / 3.5, (
                    2 * self.normalize(zarry) - 1) * self.z_position / 3.5

    def get_dimension_object(self):
        """
        :return: create a dictionary of dimensions
        """
        size_dict = dict()
        all_obj = [obj for obj in bpy.context.scene.objects if
                   not obj.name.startswith(('0', 'Camera', 'PointLight', 'Plane', 'light'))]
        for obj_ in all_obj:
            size_dict[obj_.name] = (obj_.dimensions.x, obj_.dimensions.y, obj_.dimensions.z)
        return size_dict

    @staticmethod
    def get_size_object(object_):
        """
        param object_: MeshObject
        :return: Dimensions of an object
        """
        bb = np.array(object_.get_bound_box())
        xmax, ymax, zmax = bb.max(axis=0)
        xmin, ymin, zmin = bb.min(axis=0)
        xmin, ymin, zmin = abs(xmin), abs(ymin), abs(zmin)
        return xmax + xmin, ymax + ymin, zmax + zmin

    @staticmethod
    def get_diagonal_object(object_):
        """
        param object_: MeshObject
        :return: Dimensions of an object
        """
        bb = np.array(object_.get_bound_box())
        xmax, ymax, zmax = bb.max(axis=0)
        xmin, ymin, zmin = bb.min(axis=0)
        return max(xmax - xmin, ymax - ymin, zmax - zmin)

    def get_dimension_dist_obj(self, k):
        """
        :return: dimension of the object and distractor, so we can set the same size
        """
        dimension_dict = self.get_dimension_object()
        obj_name = self.object_path.split('/')[1].split('.')[0]
        object_dimension = 0
        distractor_dimension = 0
        for key_ in dimension_dict.keys():
            val_ = dimension_dict[key_]
            if key_.split('.')[0] == obj_name and key_.split('.')[-1] == key_ and k == 0:
                object_dimension = val_
            if key_.split('.')[0] == obj_name and key_.split('.')[-1] == str(k).zfill(3):
                object_dimension = val_
            if key_.split('.')[0] != obj_name and key_.split('.')[-1] == key_ and k == 0:
                distractor_dimension = val_
            if key_.split('.')[0] != obj_name and key_.split('.')[-1] == str(k).zfill(3):
                distractor_dimension = val_
        return object_dimension, distractor_dimension

    def resize_object(self, object_, new_dimension_vec, random_scale, type_, k, frame):
        """
        param object_: MeshObjects
        :param new_dimension_vec: Vector of new dimensions, e.g. [2, 1, 0.5]
        :return: scales the object to the new dimension
        """
        if self.previous_object_size == (0, 0, 0) and type_ == 'object':
            object_dimension_init, _ = self.get_dimension_dist_obj(0)
            size_x, size_y, size_z = self.get_size_object(object_)
            ratio_x = (new_dimension_vec[0] / size_x)
            ratio_y = (new_dimension_vec[1] / size_y)
            ratio_z = (new_dimension_vec[2] / size_z)
            object_.set_scale([ratio_x, ratio_y, ratio_z])
            object_dimension_last, _ = self.get_dimension_dist_obj(0)
            self.previous_object_size = (ratio_x, ratio_y, ratio_z)
            self.object_dimensions = (size_x, size_y, size_z)
            self.original_object_dimension = object_dimension_init


        elif self.previous_object_size != (0, 0, 0) and type_ == 'object':
            new_size_x = self.previous_object_size[0]
            new_size_y = self.previous_object_size[1]
            new_size_z = self.previous_object_size[2]
            random_size = random.uniform(-random_scale / 2, random_scale)
            noise_x = random_size * new_size_x
            noise_y = random_size * new_size_y
            noise_z = random_size * new_size_z
            ratio_x = new_size_x + noise_x
            ratio_y = new_size_y + noise_y
            ratio_z = new_size_z + noise_z
            object_.set_scale([ratio_x, ratio_y, ratio_z])
            object_dimension_init, _ = self.get_dimension_dist_obj(0)
            self.new_scale_object = (self.original_object_dimension[0] / object_dimension_init[0],
                                     self.original_object_dimension[1] / object_dimension_init[1],
                                     self.original_object_dimension[2] / object_dimension_init[2])
            self.previous_object_size = (ratio_x, ratio_y, ratio_z)
            self.object_dimensions = (new_size_x, new_size_y, new_size_z)

        elif self.previous_distractor_size[k] == (0, 0, 0) and type_ == 'distractor':
            size_x, size_y, size_z = self.get_size_object(object_)
            ratio_x = (new_dimension_vec[0] / size_x)
            ratio_y = (new_dimension_vec[1] / size_y)
            ratio_z = (new_dimension_vec[2] / size_z)
            object_.set_scale([ratio_x, ratio_y, ratio_z])
            self.original_distractor_size[k] = (ratio_x, ratio_y, ratio_z)
            self.previous_distractor_size[k] = (ratio_x, ratio_y, ratio_z)
            if self.distractor_follow_object:
                object_dimension, distractor_dimension = self.get_dimension_dist_obj(0)
                ratio_x = object_dimension[0] / distractor_dimension[0]
                ratio_y = object_dimension[1] / distractor_dimension[1]
                ratio_z = object_dimension[2] / distractor_dimension[2]
                self.new_scaling_dist = [ratio_x, ratio_y, ratio_z]
                object_.set_scale([ratio_x, ratio_y, ratio_z])
        else:
            if self.distractor_follow_object:
                object_.set_scale(self.new_scale_object)
                object_dimension, distractor_dimension = self.get_dimension_dist_obj(frame)
                ratio_x = object_dimension[0] / distractor_dimension[0]
                ratio_y = object_dimension[1] / distractor_dimension[1]
                ratio_z = object_dimension[2] / distractor_dimension[2]
                self.new_scaling_dist = [ratio_x, ratio_y, ratio_z]
                object_.set_scale([ratio_x, ratio_y, ratio_z])
            else:

                new_size_x = self.previous_distractor_size[k][0]
                new_size_y = self.previous_distractor_size[k][1]
                new_size_z = self.previous_distractor_size[k][2]
                noise_x = random.uniform(-random_scale * self.previous_distractor_size[k][0],
                                         random_scale * self.previous_distractor_size[k][0])
                noise_y = random.uniform(-random_scale * self.previous_distractor_size[k][1],
                                         random_scale * self.previous_distractor_size[k][1])
                noise_z = random.uniform(-random_scale * self.previous_distractor_size[k][2],
                                         random_scale * self.previous_distractor_size[k][2])
                ratio_x = new_size_x + noise_x
                ratio_y = new_size_y + noise_y
                ratio_z = new_size_z + noise_z
                object_.set_scale([ratio_x, ratio_y, ratio_z])

                self.previous_distractor_size[k] = (ratio_x, ratio_y, ratio_z)


    def set_glass_material(self):
        """
        Sets the material glass
        """
        name = "Glass"
        overwrite = True

        if (overwrite is True) and (name in bpy.data.materials):
            blenderMat = bpy.data.materials[name]
        else:
            blenderMat = bpy.data.materials.new(name)
            name = blenderMat.name
        blenderMat.use_nodes = True
        nodes = blenderMat.node_tree.nodes
        for node in nodes:
            nodes.remove(node)
        links = blenderMat.node_tree.links
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        node_output.location = 400, 0
        node_pbsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        node_pbsdf.location = 400, 207
        node_pbsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)
        node_pbsdf.inputs['Alpha'].default_value = self.material_dict["Alpha"]
        node_pbsdf.inputs['Roughness'].default_value = 0
        node_pbsdf.inputs['Specular'].default_value = 0.5
        node_pbsdf.inputs['Transmission'].default_value = 1

        node_transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        node_transparent.location = 220, 279
        node_mix = nodes.new(type='ShaderNodeMixShader')
        node_mix.location = 730, 315
        fresnel_node = nodes.new(type='ShaderNodeFresnel')
        fresnel_node.inputs[0].default_value = 1.15
        fresnel_node.location = 222, 438
        color_ramp = nodes.new(type='ShaderNodeValToRGB')
        color_ramp.color_ramp.elements[0].color = (self.material_dict["ColorRamp"],
                                                   self.material_dict["ColorRamp"],
                                                   self.material_dict["ColorRamp"],
                                                   1)
        color_ramp.location = 575, 379
        links.new(node_pbsdf.outputs['BSDF'], node_mix.inputs[2])
        links.new(node_transparent.outputs['BSDF'], node_mix.inputs[1])
        links.new(fresnel_node.outputs['Fac'], color_ramp.inputs[0])
        links.new(color_ramp.outputs['Color'], node_mix.inputs[0])
        links.new(node_mix.outputs['Shader'], node_output.inputs['Surface'])

        bpy.data.materials[name].node_tree.nodes["Principled BSDF"].subsurface_method = 'BURLEY'
        bpy.data.materials[name].blend_method = 'HASHED'
        bpy.data.materials[name].shadow_method = 'NONE'
        bpy.data.materials[name].use_screen_refraction = True
        bpy.data.materials[name].use_sss_translucency = True
        self.glass_material = bpy.data.materials['Glass']

    def set_plastic_material(self):
        """
        Sets the material glass
        """
        name = "Plastic"
        overwrite = True
        if (overwrite is True) and (name in bpy.data.materials):
            blenderMat = bpy.data.materials[name]
        else:
            blenderMat = bpy.data.materials.new(name)
            name = blenderMat.name
        blenderMat.use_nodes = True
        nodes = blenderMat.node_tree.nodes
        for node in nodes:
            nodes.remove(node)
        links = blenderMat.node_tree.links
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        node_output.location = 400, 0
        node_pbsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        node_pbsdf.location = 400, 207
        node_pbsdf.inputs['Base Color'].default_value = self.material_dict["Base Color"]
        node_pbsdf.inputs['Alpha'].default_value = self.material_dict["Alpha"]
        node_pbsdf.inputs['Roughness'].default_value = self.material_dict["Roughness"]
        node_pbsdf.inputs['IOR'].default_value = self.material_dict["IOR"]
        node_pbsdf.inputs['Transmission'].default_value = self.material_dict["Transmission"]
        node_pbsdf.inputs['Clearcoat'].default_value = self.material_dict["Clearcoat"]
        links.new(node_pbsdf.outputs['BSDF'], node_output.inputs['Surface'])
        self.plastic_material = bpy.data.materials['Plastic']

    def set_material(self, material_type='Plastic'):
        """
        set the material glass to all objects
        """
        obj_name = self.object_path.split('/')[1].split('.')[0]
        distractor_names = list()
        for dist_ in self.distractors_list:
            distractor_names.append(dist_.split('/')[1].split('.')[0].split('_')[0])
        distractor_names.append(obj_name)

        scene = bpy.context.scene
        print(f"all object names: {distractor_names}, all names: {scene.objects}")
        all_obj = [obj for obj in scene.objects if not obj.name.startswith(('0', 'Camera', 'PointLight', 'Plane'))]
        for obj in all_obj:
            if material_type == 'Glass':
                try:
                    obj.data.materials[0] = self.glass_material
                except:
                    pass
            else:
                try:
                    obj.data.materials[0] = self.plastic_material
                except:
                    pass

    def set_light(self, position_vec, include_noise=True):
        """
        Sets the light to the frame
        :param position_vec: position of the light
        :param include_noise: do we include noise
        :return: sets the lights
        """
        if include_noise:
            position_vec = position_vec + np.random.uniform([self.random_light_x_min,
                                                             self.random_light_y_min,
                                                             self.random_light_z_min],
                                                            [self.random_light_x_max,
                                                             self.random_light_y_max,
                                                             self.random_light_z_max])
        light = bproc.types.Light()
        light.set_type(self.light_type)
        light.set_location(position_vec)
        light.set_energy(self.light_energy)

    def cicle_path(self, object, position_of_object, i):
        radius = self.get_diagonal_object(object) * self.random_radius
        ns = np.linspace(0, 360, num=self.k)
        n = ns[i]
        x = position_of_object[0] + math.cos(n) * radius
        z = position_of_object[2] + math.sin(n) * radius
        return x, z

    def include_distractors_blender(self, i, position_of_object):
        """
        param i: frame number
        :return:  each frame will have included distractors
        """
        # check if we need to include distractors
        if self.include_distractors:
            # position each distractor
            if self.distractor_follow_object:
                range_ = 1
            else:
                range_ = len(self.distractors_list)
            for k in range(range_):
                # load each distractor - path is in the list
                dist = bproc.loader.load_obj(self.distractors_list[k])[0]
                # resize each distractor to the appropriate size
                if i != 0:
                    self.resize_object(dist, self.dimension_dist_vec, self.random_scale_distractor, 'distractor', k, i)
                else:
                    self.resize_object(dist, self.dimension_dist_vec, 0, 'distractor', k, i)
                # get the current location of the distractor
                rot_x, rot_y, rot_z = dist.get_rotation()
                # set new rotation
                dist.set_rotation_euler([rot_x + i * self.rot_dist_x + random.uniform(0, self.rotation_noise_upper),
                                         rot_y + i * self.rot_dist_y + random.uniform(0, self.rotation_noise_upper),
                                         rot_z + i * self.rot_dist_z + random.uniform(0, self.rotation_noise_upper)])
                # if self.distractor_follow_object:
                #    loc_x, loc_y, loc_z = position_of_object[0] + 0.06 + random.uniform(-0, 0.05),\
                #                          position_of_object[1], \
                #                          position_of_object[2] - 0.06 - random.uniform(-0, 0.05)
                if self.distractor_follow_object:
                    x_, z_ = self.cicle_path(dist, position_of_object, i)
                    loc_x, loc_y, loc_z = x_, position_of_object[1], z_

                else:
                    loc_x, loc_y, loc_z = self.calculate_displacement(self.positions_x[k], -0.5, self.positions_z[k],
                                                                      i * self.x_position, 0, 0)
                dist.set_location([loc_x, loc_y, loc_z])
                dist.set_cp("category_id", 10 + k)

    def include_horizontal_lines(self, color, number_of_lines, sign):
        bpy.ops.mesh.primitive_plane_add(size=2, align='WORLD',
                                         location=(0, - 2, - sign * 0.38 - sign * self.line_width * number_of_lines))
        ov = bpy.context.copy()
        ov['area'] = [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"][0]
        bpy.ops.transform.rotate(ov, value=math.radians(90), orient_axis='X')
        bpy.ops.transform.rotate(ov, value=math.radians(sign * 4), orient_axis='Y')
        bpy.context.object.scale[1] = 0.001
        bpy.context.object.scale[0] = 400
        activeObject = bpy.context.active_object  # Set active object to variable
        mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
        activeObject.data.materials.append(mat)  # add the material to the object
        bpy.context.object.active_material.diffuse_color = (color[0], color[1], color[2], 1)  # change color

    def initialize_world(self):
        """
        adding the background image to the scene
        """
        scn = bpy.context.scene
        node_tree = scn.world.node_tree
        tree_nodes = node_tree.nodes
        tree_nodes.clear()
        node_background = tree_nodes.new(type='ShaderNodeBackground')
        node_math = tree_nodes.new(type='ShaderNodeMath')
        node_math.inputs[0].default_value = 0.25
        node_math.inputs[1].default_value = 0.25
        node_math.location = -300, 50
        node_background.inputs[1].default_value = 0.4
        node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
        node_environment.image = bpy.data.images.load(self.first_image)  # Relative path
        node_environment.location = -300, 0
        node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
        node_output.location = 200, 0
        links = node_tree.links
        links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
        links.new(node_background.outputs["Background"], node_output.inputs["Surface"])
        links.new(node_math.outputs["Value"], node_background.inputs["Strength"])
        bpy.context.scene.cycles.diffuse_bounces = 8
        bpy.context.scene.cycles.max_bounces = 8
        bpy.context.scene.cycles.glossy_bounces = 8
        bpy.context.scene.cycles.transmission_bounces = 8
        bpy.context.scene.cycles.volume_bounces = 8
        if self.material_type == 'Plastic':
            self.set_plastic_material()
        else:
            self.set_glass_material()

    def construct_scene(self):
        # create a list of background names and sort it ascending, so the video will be correct
        background_img = list(sorted([os.path.join(self.background_path, p) for p in os.listdir(self.background_path) if
                                      p.lower().endswith(self.possible_end)]))
        self.first_image = background_img[0]
        # overwrite number of backgrounds
        if self.overwrite_k:
            len_seq = len(background_img)
            self.k = min(len_seq, self.k)
        # initialize world background
        self.initialize_world()
        # define random trajectory for the object (on each call, different trajectory)
        x_arry, z_arry = self.define_random_trajectory()  # self.define_random_trajectory() # self.define_trajectory()#self.define_random_trajectory()
        # create empty list
        list_of_files = list()
        # for each background name
        for im_ in background_img[:self.k + 1]:
            # define new dictionary - for loading
            list_of_files.append({"name": im_, "name": im_})
        # load the object
        obj = bproc.loader.load_obj(self.object_path)[0]
        # resize the object
        self.resize_object(obj, self.dimension_vec, 0, 'object', 0, 0)
        # get the current rotation
        rot_x, rot_y, rot_z = obj.get_rotation()
        # set the new rotation of the object
        obj.set_rotation_euler([rot_x + self.rot_x, rot_y + self.rot_y, rot_z + self.rot_z])
        # set category id for the annotations
        obj.set_cp("category_id", 1)
        # calculate displacement
        loc_x, loc_y, loc_z = self.calculate_displacement(x_arry[0], -0.5, z_arry[0], 0 * self.x_position, 0, 0)
        # move the object
        obj.set_location([loc_x, loc_y, loc_z])
        if self.blur_intensity > 0:
            # get name of the object
            object_name = self.object_path.split('/')[1].split('.')[0]
            # get the object to blur
            obj_to_blur = bpy.data.objects[object_name]
            # set the location
            obj_to_blur.location = (loc_x, loc_y, loc_z)
            # set keyframe location, frame = 0
            obj_to_blur.keyframe_insert(data_path="location", frame=0)
            obj_to_blur.location = (loc_x, loc_y, loc_z)
            # set keyframe location, frame = 1
            obj_to_blur.keyframe_insert(data_path="location", frame=1)
            # construct distractors
            self.include_distractors_blender(0, [loc_x, loc_y, loc_z])
            # calculate the matrix for camera position
            matrix_world = bproc.math.build_transformation_mat([0, -2.47, 0], [math.radians(90), 0, 0])  # -2.473
            # need new pose for blur
            bproc.camera.add_camera_pose(matrix_world)
        else:
            # construct distractors
            self.include_distractors_blender(0, [loc_x, loc_y, loc_z])
            # calculate the matrix for camera position
            matrix_world = bproc.math.build_transformation_mat([0, -2.47, 0], [math.radians(90), 0, 0])  # -2.473
        # add new pose to the scene
        bproc.camera.add_camera_pose(matrix_world)
        # set constant illumination
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = self.room_illumination
        # construct the world by placing all the backgrounds on X-axis
        bpy.ops.import_image.to_plane(files=list_of_files, directory="", offset_amount=0, fill_mode='FILL',
                                      relative=True)
        # set the light
        self.set_light([self.light_position[0], self.light_position[1], self.light_position[2]])

        if self.add_occlusion:
            color_ = self.occlusion_color  # (0, 204, 204)
            n_times = 600  # some big number
            for r in range(1, n_times):
                self.include_horizontal_lines(color_, r, 1)

        if self.add_occlusion_vertical:
            self.random_position_occ = random.uniform(-0.3, 0.3)
            bpy.ops.mesh.primitive_plane_add(size=2, align='WORLD', location=(self.random_position_occ, -0.65, 0))
            # bpy.ops.transform.rotate(value=math.radians(90), orient_axis='X')
            ov = bpy.context.copy()
            ov['area'] = [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"][0]
            bpy.ops.transform.rotate(ov, value=math.radians(90), orient_axis='X')
            bpy.ops.transform.rotate(ov, value=math.radians(1), orient_axis='Y')
            bpy.context.object.scale[0] = min(0.02, min(self.previous_object_size[0],
                                                        self.previous_object_size[1],
                                                        self.previous_object_size[2]))
            bpy.context.object.scale[1] = 250
            activeObject = bpy.context.active_object  # Set active object to variable
            mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
            activeObject.data.materials.append(mat)  # add the material to the object
            self.random_color_occ = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1)
            bpy.context.object.active_material.diffuse_color = (self.random_color_occ[0],
                                                                self.random_color_occ[1],
                                                                self.random_color_occ[2],
                                                                1)  # change color
        # for each new frame
        for i in range(1, self.k):
            # load the object
            obj = bproc.loader.load_obj(self.object_path)[0]
            # set category id for the annotations
            obj.set_cp("category_id", 1)
            # resize the object
            self.resize_object(obj, self.dimension_vec, self.random_scale_object, 'object', 0, i)
            # get the current rotation
            rot_x, rot_y, rot_z = obj.get_rotation()
            # set the new rotation of the object
            obj.set_rotation_euler([rot_x + i * self.rot_x, rot_y + i * self.rot_y, rot_z + i * self.rot_z])
            # calculate displacement
            loc_x, loc_y, loc_z = self.calculate_displacement(x_arry[i], -0.5, z_arry[i], i * self.x_position, 0, 0)
            if self.blur_intensity > 0:
                loc_x_next, loc_y_next, loc_z_next = self.calculate_displacement(x_arry[i + 1], -0.5, z_arry[i + 1],
                                                                                 i * self.x_position, 0, 0)
                move_x = loc_x_next - loc_x
                move_z = loc_z_next - loc_z
            # move the object
            obj.set_location([loc_x, loc_y, loc_z])
            # construct distractors
            self.include_distractors_blender(i, [loc_x, loc_y, loc_z])
            if self.add_occlusion_vertical:
                bpy.ops.mesh.primitive_plane_add(size=2, align='WORLD',
                                                 location=(self.random_position_occ + i * self.x_position, -0.65, 0))
                # bpy.ops.transform.rotate(value=math.radians(90), orient_axis='X')
                ov = bpy.context.copy()
                ov['area'] = [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"][0]
                bpy.ops.transform.rotate(ov, value=math.radians(90), orient_axis='X')
                bpy.ops.transform.rotate(ov, value=math.radians(1), orient_axis='Y')
                bpy.context.object.scale[0] = min(0.02, min(self.previous_object_size[0],
                                                            self.previous_object_size[1],
                                                            self.previous_object_size[2]))
                bpy.context.object.scale[1] = 250
                activeObject = bpy.context.active_object  # Set active object to variable
                mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
                activeObject.data.materials.append(mat)  # add the material to the object
                bpy.context.object.active_material.diffuse_color = (self.random_color_occ[0],
                                                                    self.random_color_occ[1],
                                                                    self.random_color_occ[2],
                                                                    1)  # change color
            if self.blur_intensity > 0:
                # define the correct object to blur (example: vase_8.001)
                object_name_num = str(i).zfill(3)
                # object to blur
                obj_i = bpy.data.objects[object_name + '.' + object_name_num]
                obj_i.location = (loc_x, loc_y, loc_z)
                obj_i.keyframe_insert(data_path="location", frame=2 * i)
                obj_i.location = (loc_x + move_x * self.blur_intensity, loc_y, loc_z + move_z * self.blur_intensity)
                obj_i.keyframe_insert(data_path="location", frame=2 * i + 1)
                # calculate the matrix for camera position
                matrix_world = bproc.math.build_transformation_mat([i * self.x_position, -2.47, 0],
                                                                   [math.radians(90), 0, 0])  # -2.473
                # set the light
                self.set_light(
                    [(i * self.x_position) + self.light_position[0], self.light_position[1], self.light_position[2]])
                # additional pose
                bproc.camera.add_camera_pose(matrix_world)
            else:
                # calculate the matrix for camera position
                matrix_world = bproc.math.build_transformation_mat([i * self.x_position, -2.47, 0],
                                                                   [math.radians(90), 0, 0])  # -2.473
                # set the light
                self.set_light(
                    [(i * self.x_position) + self.light_position[0], self.light_position[1], self.light_position[2]])
            # add new pose to the scene
            bproc.camera.add_camera_pose(matrix_world)
        # set glass material
        self.set_material(self.material_type)
        # enable transparency, so background will be visible
        bproc.renderer.set_output_format(enable_transparency=True)
        # bpy.context.scene.cycles.samples = 2048 / 2
        # with step 2 if blur
        if self.blur_intensity > 0:
            bpy.context.scene.frame_step = 2
            # use motion blur
            bpy.context.scene.render.use_motion_blur = True
            # motion from start frame to next
            bpy.context.scene.cycles.motion_blur_position = 'START'

        if self.debug_mode == 0:
            data = bproc.renderer.render()
            ##data = bpy.ops.render.render(use_viewport = True, write_still=True)
            seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
            bproc.writer.write_coco_annotations(self.output_path,
                                                instance_segmaps=seg_data["instance_segmaps"],
                                                instance_attribute_maps=seg_data["instance_attribute_maps"],
                                                colors=data["colors"],
                                                append_to_existing_output=False)








# define parameters for different plastic profiles
plastic_profiles =  {"high_transparent": {"Roughness": 0.001,
                                           "Base Color": [1, 1, 1, 1],
                                           "Transmission": 0.99,
                                           "Alpha": 0.02,
                                           "Clearcoat": 0.01,
                                           "IOR": 1.4},
                         "med_high_transparent": {"Roughness": 0.001,
                                           "Base Color": [1, 1, 1, 1],
                                           "Transmission": 0.97,
                                           "Alpha": 0.05,
                                           "Clearcoat": 0.01,
                                           "IOR": 1.4},
                         "med_low_transparent": {"Roughness": 0.001,
                                           "Base Color": [1, 1, 1, 1],
                                           "Transmission": 0.95,
                                           "Alpha": 0.10,
                                           "Clearcoat": 0.01,
                                           "IOR": 1.4},
                         "low_transparent": {"Roughness": 0.001,
                                           "Base Color": [1, 1, 1, 1],
                                           "Transmission": 0.90,
                                           "Alpha": 0.15,
                                           "Clearcoat": 0.01,
                                           "IOR": 1.4}
                      }

# define different parameteres for glass profiles
glass_profiles = {"med_low_transparent": {"Alpha": 0.95,
                                           "ColorRamp": 0.38643},
                  "low_transparent":      {"Alpha": 1,
                                           "ColorRamp": 1},
                  "high_transparent":     {"Alpha": 0.3,
                                           "ColorRamp": 0},
                  "med_high_transparent": {"Alpha": 0.6,
                                           "ColorRamp": 0.2}}
# which profiles to render
profiles = ['low_transparent', 'med_low_transparent', 'high_transparent', 'med_high_transparent']

# define the line spacing for occlusion
occlusion_types = [0.01, 0.02, 0.04]
# define the names of occlusion
occlusion_names = ['high_occlusion', 'medium_occlusion', 'low_occlusion']
# define the blur intensity
blur_intensity_vec = [1 / 8, 1 / 4, 1 / 2]  # 1/8 # between 0 and 1, where 0 = No blur, 1 = very high blur
# define the blur names
blur_names = ["low_blur", "medium_blur", "high_blur"]
# where backgrounds sequences are saved
background_dir = "GOT10k"
# where to save rendered sequences
save_dir = "RenderedVideos"
# file where objects models are saved, all objects need to be in .obj format
objects_dir = 'objects'
# we suggest to keep this at 1 and render only 1 sequences per one call, otherwise there may be some memory issues
sequences_to_render = 1
# if you want to use same backgrounds only once, keep True
render_only_new_sequences = True
# list of sequences not to render
skip_sequences = list()
# length of the sequence
k = 5
# set Plastic or Glass
render_material = 'Plastic'

if len(sys.argv) == 2:
    debug_mode = sys.argv
else:
    debug_mode = 0



if __name__ == '__main__':
    prepare = Preprocessing(background_dir, save_dir, sequences_to_render, render_only_new_sequences, skip_sequences)
    prepare.create_subdirectories()
    rendered_video_dirs_ = prepare.get_rendered_video_path()
    original_video_dirs_ = prepare.get_original_video_path()
    objects = list(sorted([os.path.join(objects_dir + '/', p) for p in os.listdir(objects_dir + '/') if p.lower().endswith('obj')]))
    random.shuffle(objects)


    for m, vid_dir_ in enumerate(original_video_dirs_):
        resolution_x = 1280  # X resolution of a camera - DO NOT CHANGE
        resolution_y = 720  # Y resolution of a camera - DO NOT CHANGE
        x_position = 1.7777777  # X position of a camera - DO NOT CHANGE
        z_position = 1  # Z position of a camera - DO NOT CHANGE

        # with 0.2 probability include distractors
        if random.random() <= 0.2:
            distractor_include = 1
        else:
            distractor_include = 0
        material_choose = random.choice(profiles)
        distractor_positions_ = [[-0.6, 0.6, 0, 0.3], [0, 0.3, -0.1, -0.1]]
        if distractor_include:
            object_path = objects[m]
            to_exclude_dist = list()
            for obj in objects:
                if obj.split('/')[-1].split('_')[0] == object_path.split('/')[-1].split('_')[0]:
                    to_exclude_dist.append(obj)
            to_exclude_dist.append(object_path)
            distractors = random.sample(list(set(objects) - set(to_exclude_dist)),
                                        len(distractor_positions_[0]))  # objects[m]
        else:
            object_path = objects[m]
            distractors = list()
        additional_info = list()
        object_path = object_path
        output_path = rendered_video_dirs_[m]  # path to the folder to save results
        background_path = vid_dir_  # path to the folder with background images
        if render_material == 'Plastic':
            materials = plastic_profiles[material_choose]  # material settings
        elif render_material == 'Glass':
            materials = glass_profiles[material_choose]
        else:
            raise Exception('Choose between Plastic or Glass material')
        light_type = 'POINT'
        room_illumination = random.uniform(4, 7)  # constant illumination
        dim_x = random.uniform(0.03, 0.06)  # size of object in X direction
        dim_y = random.uniform(0.03, 0.06)  # size of object in Y direction
        dim_z = random.uniform(0.03, 0.06)  # size of object in Z direction
        rotation_x = math.radians(
            random.uniform(0, 1))  # rotation of moving object in each step - X axis (write in degrees)
        rotation_y = math.radians(
            random.uniform(0, 1))  # rotation of moving object in each step - Y axis (write in degrees)
        rotation_z = math.radians(
            random.uniform(0, 1))  # rotation of moving object in each step - Z axis (write in degrees)
        light_energy = random.uniform(15, 25)  # how strong can be 'SUN'
        light_position = [0, -1.4, 0.3]  # position of the light at the first frame
        light_position_noise = [-0.2, -0.4, -0.1, 0.2, 0, 0.1]  # for how much can position of the light change
        include_distractor = distractor_include  # 0 or 1 - do we include distractors
        distractor_objects = distractors  # list of distractor objects
        distractor_position = distractor_positions_  # [x1, x2, x3, x4], [z1, z2, z3, z4] - positions of distractors
        rotation_dist_x = math.radians(
            random.uniform(0, 1))  # rotation of distractions in each step - X axis (write in degrees)
        rotation_dist_y = math.radians(
            random.uniform(0, 1))  # rotation of distractions in each step - Y axis (write in degrees)
        rotation_dist_z = math.radians(
            random.uniform(0, 1))  # rotation of distractions in each step - Z axis (write in degrees)
        rotation_noise_lower = -random.uniform(0, 0.1)  # uniform noise rotation lower
        rotation_noise_upper = random.uniform(0, 0.1)  # uniform noise rotation upper
        dim_dist_x = random.uniform(0.03, 0.05)  # size of distractors in X direction
        dim_dist_y = random.uniform(0.03, 0.05)  # size of distractors in Y direction
        dim_dist_z = random.uniform(0.03, 0.06)  # size of distractors in Z direction
        overwrite_k = True
        random_scale_distractor = random.uniform(0, 0.1)  # 0.05
        random_scale_object = random.uniform(0, 0.1)  # 0.05
        add_occlusion_vertical = False  # True - add vertical occlusion
        distractor_follow_object = True  # True - one distractor following
        over_the_edge = 0.9

        blur_idx = random.randint(0, 2)
        occlusion_idx = random.randint(0, 2)
        line_width = occlusion_types[occlusion_idx]
        line_type = occlusion_names[occlusion_idx]
        occlusion_color = (random.random(), random.random(), random.random())

        # probability 0.15 to include blur
        if random.random() <= 0.15:
            blur_include = True
            blur_intensity = blur_intensity_vec[blur_idx]
            blur_name = blur_names[blur_idx]
        else:
            blur_include = False
            blur_intensity = 0
            blur_name = 'no_blur'
        # probability 0.2 to include occlusion
        if random.random() <= 0.2:
            add_occlusion = True
        elif blur_include:
            add_occlusion = False
            line_type = 'no_occlusion'
        else:
            add_occlusion = False
            line_type = 'no_occlusion'

        additional_info.append("General information (objects name, backgrounds name):")
        additional_info.append(f"object_path: {object_path}")
        additional_info.append(f"output_path: {output_path}")
        additional_info.append(f"background_path: {background_path}")
        additional_info.append(f"length: {k}")
        additional_info.append("Camera information:")
        additional_info.append(f"resolution_x: {resolution_x}")
        additional_info.append(f"resolution_y: {resolution_y}")
        additional_info.append(f"x_position: {x_position}")
        additional_info.append(f"z_position: {z_position}")
        additional_info.append("Light information:")
        additional_info.append(f"room_illumination: {room_illumination}")
        additional_info.append(f"light_energy: {light_energy}")
        additional_info.append(f"light_type: {light_type}")
        additional_info.append(f"light_position: {light_position}")
        additional_info.append(f"light_position_noise: {light_position_noise}")
        additional_info.append("Material information:")
        additional_info.append(f"materials: {material_choose}")
        additional_info.append("Object information:")
        additional_info.append(f"dim_x: {dim_x}")
        additional_info.append(f"dim_y: {dim_y}")
        additional_info.append(f"dim_z: {dim_z}")
        additional_info.append(f"rotation_x: {rotation_x}")
        additional_info.append(f"rotation_y: {rotation_y}")
        additional_info.append(f"rotation_z: {rotation_z}")
        additional_info.append(f"random_scale_object: {random_scale_object}")
        additional_info.append(f"over_the_edge: {over_the_edge}")
        additional_info.append("Distractors information:")
        additional_info.append(f"include_distractor: {include_distractor}")
        additional_info.append(f"distractor_objects: {distractor_objects}")
        additional_info.append(f"distractor_position: {distractor_position}")
        additional_info.append(f"dim_dist_x: {dim_dist_x}")
        additional_info.append(f"dim_dist_y: {dim_dist_y}")
        additional_info.append(f"dim_dist_z: {dim_dist_z}")
        additional_info.append(f"rotation_dist_x: {rotation_dist_x}")
        additional_info.append(f"rotation_dist_y: {rotation_dist_y}")
        additional_info.append(f"rotation_dist_z: {rotation_dist_z}")
        additional_info.append(f"rotation_noise_lower: {rotation_noise_lower}")
        additional_info.append(f"rotation_noise_upper: {rotation_noise_upper}")
        additional_info.append(f"rotation_dist_z: {rotation_dist_z}")
        additional_info.append(f"random_scale_distractor: {random_scale_distractor}")
        additional_info.append("Occlusion information:")
        additional_info.append(f"add_occlusion: {add_occlusion}")
        additional_info.append(f"occlusion_type: {line_type}")
        additional_info.append(f"add_occlusion_vertical: {add_occlusion_vertical}")
        additional_info.append(f"line_width: {line_width}")
        additional_info.append(f"occlusion_color: {occlusion_color}")
        additional_info.append("Blur information:")
        additional_info.append(f"blur_include: {blur_include}")
        additional_info.append(f"blur_type: {blur_name}")
        additional_info.append(f"blur_intensity: {blur_intensity}")

        with open(output_path + "/additional_information.txt", "w") as f:
            for line in additional_info:
                f.write(line)
                f.write('\n')

        render = TransparentObjectRender(object_path=object_path, output_path=output_path,
                                         background_path=background_path,
                                         resolution_x=resolution_x, resolution_y=resolution_y, dim_x=dim_x,
                                         dim_y=dim_y, dim_z=dim_z, x_position=x_position, z_position=z_position,
                                         k=k, room_illumination=room_illumination, materials=materials,
                                         rotation_x=rotation_x, rotation_y=rotation_y, rotation_z=rotation_z,
                                         light_energy=light_energy, light_type=light_type,
                                         light_position=light_position,
                                         light_position_noise=light_position_noise,
                                         include_distractor=include_distractor,
                                         distractor_objects=distractor_objects, distractor_position=distractor_position,
                                         rotation_dist_x=rotation_dist_x, rotation_dist_y=rotation_dist_y,
                                         rotation_dist_z=rotation_dist_z, rotation_noise_lower=rotation_noise_lower,
                                         rotation_noise_upper=rotation_noise_upper, dim_dist_x=dim_dist_x,
                                         dim_dist_y=dim_dist_y, dim_dist_z=dim_dist_z, overwrite_k=overwrite_k,
                                         random_scale_distractor=random_scale_distractor,
                                         random_scale_object=random_scale_object,
                                         add_occlusion=add_occlusion, add_occlusion_vertical=add_occlusion_vertical,
                                         distractor_follow_object=distractor_follow_object, over_the_edge=over_the_edge,
                                         blur_intensity=blur_intensity, line_width=line_width,
                                         occlusion_color=occlusion_color, material_type=render_material, debug_mode=debug_mode)
        render.construct_scene()
    print('Rendering complete!')
