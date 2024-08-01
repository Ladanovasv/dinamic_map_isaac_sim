# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from multiprocessing import Process
import omni
# from pxr import Sdf
from omni.isaac.kit import SimulationApp
import os
gpu_to_use = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 1920, "height": 1080, "multi_gpu": False, "active_gpu": gpu_to_use}
import argparse
import glob
import os
import re

"""
Standalone script to schedule people sdg jobs in a local env.
"""


class PeopleSDG:
    def __init__(self, num_runs, sim_app):
        self.num_runs = num_runs
        self.config_dict = None
        self._sim_manager = None
        self._data_generator = None
        self._settings = None
        self._sim_app = sim_app

    def set_config(self, config_file):
        import carb
        from omni.replicator.character.core.data_generation import DataGeneration
        from omni.replicator.character.core.simulation import SimulationManager

        self._sim_manager = SimulationManager()
        
        self.config_dict, is_modified = self._sim_manager.load_config_file(config_file)
        if not self.config_dict:
            carb.log_error("Loading config file ({0}) fails. Data generation will not start.".format(config_file))
            return

        try:
            folder_path = self.config_dict["replicator"]["parameters"]["output_dir"]
            self.config_dict["replicator"]["parameters"]["output_dir"] = PeopleSDG._get_output_folder_by_index(
                folder_path, index=self.num_runs
            )
        except:
            carb.log_warn("'output_dir' does not exists in config file. Will not auto increase output path")

        data_generation_config = {}
        data_generation_config["writer_name"] = self.config_dict["replicator"]["writer"]
        data_generation_config["num_cameras"] = self.config_dict["global"]["camera_num"]
        data_generation_config["num_lidars"] = self.config_dict["global"]["lidar_num"]
        data_generation_config["num_frames"] = self.config_dict["global"]["simulation_length"] * 30
        data_generation_config["writer_params"] = self.config_dict["replicator"]["parameters"]
        self._data_generator = DataGeneration(data_generation_config)

    def set_simulation_settings(self):
        import carb
        import omni.replicator.core as rep

        rep.settings.carb_settings("/omni/replicator/backend/writeThreads", 16)
        self._settings = carb.settings.get_settings()
        self._settings.set("/rtx/rtxsensor/coordinateFrameQuaternion", "0.5,-0.5,-0.5,-0.5")
        self._settings.set("/app/scripting/ignoreWarningDialog", True)
        self._settings.set("/persistent/exts/omni.anim.navigation.core/navMesh/viewNavMesh", False)
        self._settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
        self._settings.set(
            "/exts/omni.replicator.character/default_robot_command_file_path", "default_robot_command.txt"
        )
        self._settings.set("/persistent/exts/omni.replicator.character/aim_camera_to_character", True)
        self._settings.set("/persistent/exts/omni.replicator.character/min_camera_distance", 6.5)
        self._settings.set("/persistent/exts/omni.replicator.character/max_camera_distance", 14.5)
        self._settings.set("/persistent/exts/omni.replicator.character/max_camera_look_down_angle", 60)
        self._settings.set("/persistent/exts/omni.replicator.character/min_camera_look_down_angle", 0)
        self._settings.set("/persistent/exts/omni.replicator.character/min_camera_height", 2)
        self._settings.set("/persistent/exts/omni.replicator.character/max_camera_height", 3)
        self._settings.set("/persistent/exts/omni.replicator.character/character_focus_height", 0.7)
        self._settings.set("/persistent/exts/omni.replicator.character/frame_write_interval", 1)

    def save_commands_to_file(self, file_path, commands):
        from omni.replicator.character.core.file_util import TextFileUtil

        command_str = ""
        for cmd in commands:
            command_str += cmd
            command_str += "\n"
        result = TextFileUtil.write_text_file(file_path, command_str)
        return result

    def generate_data(self, config_file):
        import carb
        from omni.isaac.core.utils.stage import open_stage

        # Set simulation settings
        self.set_simulation_settings()

        # Load from config file
        self.set_config(config_file)

        # Open stage with blocking call
        stage_open_result = open_stage(self.config_dict["scene"]["asset_path"])

        if not stage_open_result:
            carb.log_error("Unable to open stage {}".format(self.config_dict["scene"]["asset_path"]))
            self._sim_app.close()

        self._sim_app.update()

        # Create character and cameras
        self._sim_manager.load_agents_cameras_from_config_file()
        self._sim_app.update()

        # Create random character actions (when character section exists)
        if "character" in self.config_dict:
            commands = self._sim_manager.generate_random_commands()
            # Write commands to file
            self.save_commands_to_file(self.config_dict["character"]["command_file"], commands)
            self._sim_app.update()

        import omni.replicator.core as rep
        from omni.isaac.sensor import Camera
        camera_path = "/World/Cameras/Camera"
        # camera = rep.create.camera(position=(0, 0, 1000))
        camera1_pos = []
        for i in range(100):
            camera1_pos.append((i/100-5, 0, 5))
        # with rep.new_layer():
        #     with rep.trigger.on_frame(interval=1):
        #         with camera:
        #             rep.modify.pose(look_at=(0,0,0), position=rep.distribution.sequence(camera1_pos))
        #             print(camera)
        #     rep.orchestrator.run()
            # camera.set_world_pose(position=camera1_pos[i])
            # print(camera.get_world_pose())
            # i+=1
               


                 
        # Run data generation
        from omni.isaac.core import World
        my_world = World(stage_units_in_meters=1.0)

        self._data_generator._init_recorder()
        camera = Camera(
            camera_path, "camera"
        )
        width, height = 1920, 1200
        camera_matrix = [[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]

        # Pixel size in microns, aperture and focus distance from the camera sensor specification
        # Note: to disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
        pixel_size = 3 * 1e-3   # in mm, 3 microns is a common pixel size for high resolution cameras
        f_stop = 1.8            # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
        focus_distance = 0.6    # in meters, the distance from the camera to the object plane

        # Calculate the focal length and aperture size from the camera matrix
        ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
        horizontal_aperture =  pixel_size * width                   # The aperture size in mm
        vertical_aperture =  pixel_size * height
        focal_length_x  = fx * pixel_size
        focal_length_y  = fy * pixel_size
        focal_length = (focal_length_x + focal_length_y) / 2         # The focal length in mm

        i = 1
        flag = 1

        
        from omni.isaac.core.prims import XFormPrim
        import numpy as np
        prim_for_teleport = XFormPrim("/Root/SM_BarelPlastic_C_01_25/SM_BarelPlastic_C_01")
        prim_for_hide = XFormPrim("/Root/S_TrafficCone_2/S_TrafficCone")
        
        def teleport(prim, translation):
            pose, quant = prim.get_world_pose()
            prim.set_world_pose(position = pose + translation)

        def hide_prim(prim):
            """Hide a prim

            Args:
                prim_path (str, required): The prim path of the prim to hide
            """
            prim.set_visibility(False)


        def show_prim(prim):
            """Show a prim

            Args:
                prim_path (str, required): The prim path of the prim to show
            """
            prim.set_visibility(True)


        # def set_prim_visibility_attribute(prim_path: str, value: str):
        #     """Set the prim visibility attribute at prim_path to value

        #     Args:
        #         prim_path (str, required): The path of the prim to modify
        #         value (str, required): The value of the visibility attribute
        #     """
        #     # You can reference attributes using the path syntax by appending the
        #     # attribute name with a leading `.`
        #     prop_path = f"{prim_path}.visibility"
        #     omni.kit.commands.execute(
        #         "ChangeProperty", prop_path=Sdf.Path(prop_path), value=value, prev=None
        #     )  
        # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
        # camera.set_focal_length(focal_length / 10.0)                # Convert from mm to cm (or 1/10th of a world unit)
        # camera.set_focus_distance(focus_distance)                   # The focus distance in meters
        # camera.set_lens_aperture(f_stop * 100.0)                    # Convert the f-stop to Isaac Sim units
        # camera.set_horizontal_aperture(horizontal_aperture / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
        # camera.set_vertical_aperture(vertical_aperture / 10.0)

        # camera.set_clipping_range(0.05, 1.0e5)
        
        while self._sim_app.is_running():
            my_world.step(render=True)
            
            camera.set_world_pose(position=camera1_pos[i])
            print(camera.get_world_pose())
            
            # import omni
            # import math
            # width = 1920
            # height = 1080
            # aspect_ratio = width / height
            # # get camera prim attached to viewport
            # focal_length = camera.GetAttribute("focalLength").Get()
            # horiz_aperture = camera.GetAttribute("horizontalAperture").Get()
            # # Pixels are square so we can do:
            # vert_aperture = height/width * horiz_aperture
            # near, far = camera.GetAttribute("clippingRange").Get()
            # fov = 2 * math.atan(horiz_aperture / (2 * focal_length))

            # # compute focal point and center
            # focal_x = height * focal_length / vert_aperture
            # focal_y = width * focal_length / horiz_aperture
            # center_x = height * 0.5
            # center_y = width * 0.5
            # print(focal_x, focal_y, center_x, center_y)
            i += flag
            if i % 100 == 0:
                translation = np.array([0,-1,0])
                teleport(prim_for_teleport, translation)
                if flag == 1:
                    hide_prim(prim_for_hide)
                else:
                    show_prim(prim_for_hide)
                flag *= -1
                i += flag



        self._data_generator.run_until_complete()
        self._sim_app.update()

  
            # with rep.trigger.on_frame(num_frames=30):
        # Clear State after completion
        self._data_generator._clear_recorder()
        self._sim_app.update()
        self._sim_app.close()

    def _get_output_folder_by_index(path, index):
        """
        Get the next output_folder following naming convention '_d' where d is digit string.
        If file name dose not follow naming convention, append '_d' at the end.
        """
        if index == 0:
            return path
        m = re.search(r"_\d+$", path)
        if m:
            cur_index = int(m.group()[1:])
            if cur_index:
                index = cur_index + index
                path = path[: m.start()]
        return path + "_" + str(index)


def enable_extensions():
    # Enable extensions
    from omni.isaac.core.utils.extensions import enable_extension

    enable_extension("omni.kit.window.viewport")
    enable_extension("omni.kit.manipulator.prim")
    enable_extension("omni.kit.property.usd")
    enable_extension("omni.anim.navigation.bundle")
    enable_extension("omni.anim.timeline")
    enable_extension("omni.anim.graph.bundle")
    enable_extension("omni.anim.graph.core")
    enable_extension("omni.anim.graph.ui")
    enable_extension("omni.anim.retarget.bundle")
    enable_extension("omni.anim.retarget.core")
    enable_extension("omni.anim.retarget.ui")
    enable_extension("omni.kit.scripting")
    enable_extension("omni.anim.people")
    enable_extension("omni.replicator.character.core")
    enable_extension("omni.kit.mesh.raycast")


def launch_data_generation(num_runs, config_file):

    # Initalize kit app
    kit = SimulationApp(launch_config=CONFIG)

    # Enable extensions
    enable_extensions()
    kit.update()

    # Load modules from extensions
    import carb
    import omni.kit.loop._loop as omni_loop

    loop_runner = omni_loop.acquire_loop_interface()
    loop_runner.set_manual_step_size(1.0 / 30.0)
    loop_runner.set_manual_mode(True)
    carb.settings.get_settings().set("/app/player/useFixedTimeStepping", False)

    # set config app
    sdg = PeopleSDG(num_runs, kit)
    sdg.generate_data(config_file)


def get_args():
    parser = argparse.ArgumentParser("People SDG")
    parser.add_argument("-c", "--config_file", required=True, help="Path to config file or a folder of config files")
    parser.add_argument(
        "-n",
        "--num_runs",
        required=False,
        type=int,
        nargs="?",
        default=1,
        const=1,
        help="Number or run. After each run, the output path index will increase. If not provided, the default run is 1.",
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    config_path = args.config_file
    num_runs = args.num_runs
    files = []

    # Check for config file or folder
    if os.path.isdir(config_path):
        files = glob.glob("{}/*.yaml".format(config_path))
    elif os.path.isfile(config_path):
        files.append(config_path)
    else:
        print("Invalid config path passed. Path must be a file or a folder containing config files.")
    print("Total SDG jobs - {}".format(len(files)))

    # Launch jobs
    for run in range(num_runs):
        for idx, config_file in enumerate(files):
            print("{} round: Starting SDG job number - {} with config file {}".format(run, idx, config_file))
            p = Process(
                target=launch_data_generation,
                args=(
                    run,
                    config_file,
                ),
            )
            p.start()
            p.join()


if __name__ == "__main__":
    main()
# "rgb": True,
            # "bounding_box_2d_tight": False,
            # "bounding_box_2d_loose": False,
            # "semantic_segmentation": False,
            # "colorize_semantic_segmentation": False,
            # "instance_id_segmentation": False,
            # "colorize_instance_id_segmentation": False,
            # "instance_segmentation": False,
            # "colorize_instance_segmentation": False,
            # "distance_to_camera": False,
            # "distance_to_image_plane": False,
            # "bounding_box_3d": False,
            # "occlusion": False,
            # "normals": False,
            # "motion_vectors": False,
            # "camera_params": False,
            # "pointcloud": False,
            # "pointcloud_include_unlabelled": False,
            # "skeleton_data": False, 
