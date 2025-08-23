import open3d as o3d
import matplotlib.pyplot as plt
from open3d.visualization import gui
import typing
import numpy as np
import copy
import cv2
import os.path
import glob
from . import ios, misc, transformation as trans


class Open3DWindow:
    def __init__(self,
                 baseline: float,
                 intrinsics: typing.Union[typing.List[float], np.ndarray],
                 dataset: list[dict[str, str | None]],
                 width_range: list[float],
                 height_range: list[float],
                 depth_range: list[float],
                 stereo: bool = False,
                 mean_offset: float = 10,
                 detect_height: float = 10,
                 resolution: int = 10,
                 win_name: str = 'open3d',
                 width: int = 1920,
                 height: int = 1280):
        self.stereo = stereo
        self.rotation_matrix = None
        self.mean = None
        self.dataset = dataset
        # self.width = width
        # self.height = height
        self.index = -1
        self.resolution = resolution
        self.pointcloud = None
        self.image = None
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(win_name, width, height)

        self.widget = o3d.visualization.gui.SceneWidget()
        self.window.add_child(self.widget)
        self.widget.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.widget.enable_scene_caching(True)

        # visualize point cloud
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = 'defaultUnlit'
        material.point_size = 2.0

        # visualize pivot point
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
        self.widget.scene.add_geometry('mesh_frame', mesh_frame, material)

        # setup camera
        self.widget.setup_camera(60, self.widget.scene.bounding_box, [0, 0, 0])

        em = self.window.theme.font_size
        self.layout = gui.Vert(em, gui.Margins(em, em, em, em))
        self.layout.preferred_width = int(360 * self.window.scaling)
        self.window.add_child(self.layout)

        collapse_image = gui.CollapsableVert("image", 20)
        self.layout.add_child(collapse_image)
        self.image_widget = gui.ImageWidget()
        collapse_image.add_child(self.image_widget)

        collapse_depth = gui.CollapsableVert("depth map", 20)
        self.layout.add_child(collapse_depth)
        self.depth_widget = gui.ImageWidget()
        collapse_depth.add_child(self.depth_widget)

        collapse_obstacle = gui.CollapsableVert("obstacle map", 20)
        self.layout.add_child(collapse_obstacle)
        self.obstacle_widget = gui.ImageWidget()
        collapse_obstacle.add_child(self.obstacle_widget)

        self.width_range = [-50, 50] if width_range is None else width_range
        self.height_range = [-50, 50] if height_range is None else height_range
        self.depth_range = [0, 100] if depth_range is None else depth_range

        xcollapse = gui.CollapsableVert("width range", 20)
        self.layout.add_child(xcollapse)
        self.xslider_left = gui.Slider(gui.Slider.DOUBLE)
        self.xslider_left.set_limits(-500, 500)
        self.xslider_left.double_value = self.width_range[0]
        self.xslider_left.set_on_value_changed(self.draw_bbox)
        xcollapse.add_child(self.xslider_left)
        self.xslider_right = gui.Slider(gui.Slider.DOUBLE)
        self.xslider_right.double_value = self.width_range[1]
        self.xslider_right.set_limits(-500, 500)
        self.xslider_right.set_on_value_changed(self.draw_bbox)
        xcollapse.add_child(self.xslider_right)

        ycollapse = gui.CollapsableVert("height range", 20)
        self.layout.add_child(ycollapse)
        self.yslider_up = gui.Slider(gui.Slider.DOUBLE)
        self.yslider_up.set_limits(-50, 500)
        self.yslider_up.double_value = self.height_range[0]
        self.yslider_up.set_on_value_changed(self.draw_bbox)
        ycollapse.add_child(self.yslider_up)
        self.yslider_down = gui.Slider(gui.Slider.DOUBLE)
        self.yslider_down.set_limits(-50, 500)
        self.yslider_down.double_value = self.height_range[1]
        self.yslider_down.set_on_value_changed(self.draw_bbox)
        ycollapse.add_child(self.yslider_down)

        zcollapse = gui.CollapsableVert("depth range", 20)
        self.layout.add_child(zcollapse)
        self.zslider_near = gui.Slider(gui.Slider.DOUBLE)
        self.zslider_near.set_limits(0, 1000)
        self.zslider_near.double_value = self.depth_range[0]
        self.zslider_near.set_on_value_changed(self.draw_bbox)
        zcollapse.add_child(self.zslider_near)
        self.zslider_far = gui.Slider(gui.Slider.DOUBLE)
        self.zslider_far.set_limits(0, 1000)
        self.zslider_far.double_value = self.depth_range[1]
        self.zslider_far.set_on_value_changed(self.draw_bbox)
        zcollapse.add_child(self.zslider_far)

        ocollapse = gui.CollapsableVert("mean offset & detect height", 20)
        self.layout.add_child(ocollapse)
        self.mean_offset_slider = gui.Slider(gui.Slider.DOUBLE)
        self.mean_offset_slider.set_limits(0, 500)
        self.mean_offset_slider.double_value = mean_offset
        self.mean_offset_slider.set_on_value_changed(self.draw_obstacle_box)
        ocollapse.add_child(self.mean_offset_slider)

        self.detect_height_slider = gui.Slider(gui.Slider.DOUBLE)
        self.detect_height_slider.set_limits(0, 500)
        self.detect_height_slider.double_value = detect_height
        self.detect_height_slider.set_on_value_changed(self.draw_obstacle_box)
        ocollapse.add_child(self.detect_height_slider)

        button_horiz = gui.Horiz(em, gui.Margins(em, em, em, em))
        self.layout.add_child(button_horiz)
        next_button = gui.Button("Next")
        button_horiz.add_child(next_button)
        next_button.set_on_clicked(self.next_data)

        fit_button = gui.Button("Fit")
        button_horiz.add_child(fit_button)
        fit_button.set_on_clicked(self.fit)

        process_button = gui.Button("Process")
        button_horiz.add_child(process_button)
        process_button.set_on_clicked(self.process)

        prev_button = gui.Button("Prev")
        button_horiz.add_child(prev_button)
        prev_button.set_on_clicked(self.prev_data)

        quit_button = gui.Button("Quit")
        button_horiz.add_child(quit_button)
        quit_button.set_on_clicked(self.destroy_window)

        param_collapse = gui.CollapsableVert("camera params", 20)
        self.layout.add_child(param_collapse)
        # baseline
        baseline_horiz = gui.Horiz(em, gui.Margins(em, em, em, em))
        param_collapse.add_child(baseline_horiz)
        baseline_label = gui.Label('baseline')
        baseline_horiz.add_child(baseline_label)
        self.baseline_textedit = gui.TextEdit()
        self.baseline_textedit.text_value = '{}'.format(baseline)
        baseline_horiz.add_child(self.baseline_textedit)
        # fx
        f_horiz = gui.Horiz(em, gui.Margins(em, em, em, em))
        param_collapse.add_child(f_horiz)
        fx_label = gui.Label('fx')
        f_horiz.add_child(fx_label)
        self.fx_textedit = gui.TextEdit()
        self.fx_textedit.text_value = '{}'.format(intrinsics[0])
        f_horiz.add_child(self.fx_textedit)
        # fy
        fy_label = gui.Label('fy')
        f_horiz.add_child(fy_label)
        self.fy_textedit = gui.TextEdit()
        self.fy_textedit.text_value = '{}'.format(intrinsics[1])
        f_horiz.add_child(self.fy_textedit)
        # cx
        c_horiz = gui.Horiz(em, gui.Margins(em, em, em, em))
        param_collapse.add_child(c_horiz)
        cx_label = gui.Label('cx')
        c_horiz.add_child(cx_label)
        self.cx_textedit = gui.TextEdit()
        self.cx_textedit.text_value = '{}'.format(intrinsics[2])
        c_horiz.add_child(self.cx_textedit)
        # cy
        cy_label = gui.Label('cy')
        c_horiz.add_child(cy_label)
        self.cy_textedit = gui.TextEdit()
        self.cy_textedit.text_value = '{}'.format(intrinsics[3])
        c_horiz.add_child(self.cy_textedit)

        self.window.set_on_layout(self.on_layout)

        # visualize bounding box
        self.draw_bbox(10)

        self.next_data()

    def draw_bbox(self, value):
        # visualize 3D bounding-box
        self.width_range[0] = self.xslider_left.double_value
        self.width_range[1] = self.xslider_right.double_value
        self.height_range[0] = self.yslider_up.double_value
        self.height_range[1] = self.yslider_down.double_value
        self.depth_range[0] = self.zslider_near.double_value
        self.depth_range[1] = self.zslider_far.double_value
        fitpoints = [
            [self.width_range[0], self.height_range[0], self.depth_range[0]],
            [self.width_range[1], self.height_range[0], self.depth_range[0]],
            [self.width_range[0], self.height_range[1], self.depth_range[0]],
            [self.width_range[1], self.height_range[1], self.depth_range[0]],
            [self.width_range[0], self.height_range[0], self.depth_range[1]],
            [self.width_range[1], self.height_range[0], self.depth_range[1]],
            [self.width_range[0], self.height_range[1], self.depth_range[1]],
            [self.width_range[1], self.height_range[1], self.depth_range[1]],
        ]
        fitlines = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        color = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0],
                 [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0],
                 [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
        fitline_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(fitpoints),
            lines=o3d.utility.Vector2iVector(fitlines)
        )
        fitline_set.colors = o3d.utility.Vector3dVector(color)
        self.widget.scene.remove_geometry('fitbbox')
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = 'unlitLine'
        material.point_size = 10.0
        self.widget.scene.add_geometry('fitbbox', fitline_set, material)

        self.widget.force_redraw()

    def on_layout(self, layout_context):
        frame = self.window.content_rect
        self.widget.frame = frame
        panel_size = self.layout.calc_preferred_size(layout_context,
                                                     self.layout.Constraints())
        self.layout.frame = gui.Rect(frame.get_right() - panel_size.width,
                                     frame.y, panel_size.width,
                                     panel_size.height)
        self.widget.force_redraw()

    def preprocess(self, index: int):
        data = self.dataset[index]
        pred_name = data['pred']
        image_name = data['image']
        pred = ios.load_disp(pred_name, 1.0).squeeze()
        image = None
        if image_name is not None:
            image = ios.load_others(image_name)
            if self.stereo:
                image = np.split(image, 2, axis=1)[0]
            if image.shape[0] != pred.shape[0] or image.shape[1] != pred.shape[1]:
                pred = cv2.resize(pred, (image.shape[1], image.shape[0])) * image.shape[1] / pred.shape[1]
        # grid map
        baseline = float(self.baseline_textedit.text_value)
        intrinsics = [float(self.fx_textedit.text_value), float(self.fy_textedit.text_value),
                      float(self.cx_textedit.text_value), float(self.cy_textedit.text_value)]
        pred_pc = trans.disparity_to_pointcloud(pred, baseline, intrinsics=intrinsics)

        depth_image = misc.visualize_map(pred_pc[..., 2])
        depth_o3d = o3d.geometry.Image(depth_image)
        self.depth_widget.update_image(depth_o3d)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred_pc.reshape(-1, 3))
        if image is not None:
            # ratio = image.shape[1] * depth_image.shape[0] / (image.shape[0] * depth_image.shape[1])
            # if ratio != 1:
            #     image = np.split(image, ratio)
            # if image.shape[0] != depth_image.shape[0] or image.shape[1] != depth_image.shape[1]:
            #     image = cv2.resize(image, (depth_image.shape[1], depth_image.shape[0]))
            image_o3d = o3d.geometry.Image(image)
            self.image_widget.update_image(image_o3d)
            pcd.colors = o3d.utility.Vector3dVector(image.reshape(-1, 3) / 255.0)
        self.add_geometry('pcd', pcd)
        self.pointcloud = pred_pc
        self.image = image

        self.rotation_matrix = None
        self.mean = None

    def postprocess(self):
        pass

    def draw_obstacle_box(self, value=None):
        # draw obstacle area as box
        # visualize 3D bounding-box
        self.width_range[0] = self.xslider_left.double_value
        self.width_range[1] = self.xslider_right.double_value
        self.depth_range[0] = self.zslider_near.double_value
        self.depth_range[1] = self.zslider_far.double_value
        height_down = self.mean - self.mean_offset_slider.double_value
        height_up = -self.detect_height_slider.double_value
        height_range = [height_up, height_down]
        points = [
            [self.width_range[0], height_range[0], self.depth_range[0]],
            [self.width_range[1], height_range[0], self.depth_range[0]],
            [self.width_range[0], height_range[1], self.depth_range[0]],
            [self.width_range[1], height_range[1], self.depth_range[0]],
            [self.width_range[0], height_range[0], self.depth_range[1]],
            [self.width_range[1], height_range[0], self.depth_range[1]],
            [self.width_range[0], height_range[1], self.depth_range[1]],
            [self.width_range[1], height_range[1], self.depth_range[1]],
        ]
        lines = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        color = [[0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1],
                 [0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 1],
                 [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(color)
        self.widget.scene.remove_geometry('obsbbox')
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = 'unlitLine'
        material.point_size = 10.0
        self.widget.scene.add_geometry('obsbbox', line_set, material)

        self.widget.force_redraw()

    def fit(self):
        height_mask = (self.pointcloud[..., 1] > self.height_range[0]) & (
                self.pointcloud[..., 1] < self.height_range[1])
        depth_mask = (self.pointcloud[..., 2] > self.depth_range[0]) & (self.pointcloud[..., 2] < self.depth_range[1])
        mask = height_mask & depth_mask
        self.rotation_matrix, self.mean, _, _ = misc.fit_plane(self.pointcloud, mask)

        rotated_pc = (self.rotation_matrix @ self.pointcloud.reshape(-1, 3).T).T
        geo = o3d.geometry.PointCloud()
        geo.points = o3d.utility.Vector3dVector(rotated_pc)
        if self.image is not None:
            geo.colors = o3d.utility.Vector3dVector(self.image.reshape(-1, 3) / 255.0)
        self.add_geometry('pcd', geo)
        self.draw_obstacle_box()

    def process(self):
        if self.mean is None:
            height_range = self.height_range
            rotated_pc = copy.deepcopy(self.pointcloud)
        else:
            height_down = self.mean - self.mean_offset_slider.double_value
            height_up = - self.detect_height_slider.double_value
            height_range = [height_up, height_down]
            shape = self.pointcloud.shape
            rotated_pc = (self.rotation_matrix @ self.pointcloud.reshape(-1, 3).T).T.reshape(shape)

        width_mask = (self.width_range[1] > rotated_pc[..., 0]) & (rotated_pc[..., 0] > self.width_range[0])
        # a little big higher than the floor
        # y-axis is negative, i.e., pointing to the floor
        height_mask = (rotated_pc[..., 1] > height_range[0]) & (rotated_pc[..., 1] < height_range[1])
        depth_mask = (rotated_pc[..., 2] > self.depth_range[0]) & (rotated_pc[..., 2] < self.depth_range[1])

        valid_mask = (width_mask & height_mask & depth_mask)
        rotated_pc[..., 1] = rotated_pc[..., 1] - self.mean
        fig = plt.gcf()
        fig.clear()
        ax = fig.gca()
        scaled_width_range = np.asarray(self.width_range) / self.resolution
        scaled_height_range = np.asarray(self.height_range) / self.resolution
        scaled_depth_range = np.asarray(self.depth_range) / self.resolution

        xbins = np.arange(scaled_width_range[0], scaled_width_range[1] + 1, step=1)
        zbins = np.arange(scaled_depth_range[0], scaled_depth_range[1] + 1, step=1)
        if len(rotated_pc[valid_mask]) > 0:
            # move origin from center to top-left
            # then divide by 10 x 10 (downscale), where each pixel (grid) in grid map has the size of 10mm x 10mm
            rotated_pc[..., 0] = rotated_pc[..., 0] / self.resolution
            rotated_pc[..., 1] = -rotated_pc[..., 1] / self.resolution
            rotated_pc[..., 2] = rotated_pc[..., 2] / self.resolution

            discrete_obstacle_pc = rotated_pc.astype(np.int32)
            xmask = ((discrete_obstacle_pc[..., 0] >= scaled_width_range[0]) &
                     (discrete_obstacle_pc[..., 0] <= scaled_width_range[1]))
            ymask = ((discrete_obstacle_pc[..., 1] >= 0) &
                     (discrete_obstacle_pc[..., 1] <= scaled_height_range[1]))
            zmask = ((discrete_obstacle_pc[..., 2] >= scaled_depth_range[0]) &
                     (discrete_obstacle_pc[..., 2] <= scaled_depth_range[1]))
            valid_mask = (xmask & ymask & zmask & valid_mask)

            masked_discrete_obstacle_pc = discrete_obstacle_pc[valid_mask]

            hist, _, _ = np.histogram2d(masked_discrete_obstacle_pc[..., 0],
                                        masked_discrete_obstacle_pc[..., 2], bins=(xbins, zbins))
            obstacle_hist = hist.T

            grid = ax.imshow(obstacle_hist, cmap='jet', origin='lower')
            fig.colorbar(grid, ax=ax, orientation='horizontal', fraction=0.1)

        xticks = np.linspace(0, xbins.shape[0] - 1, xbins.shape[0])
        yticks = np.linspace(0, zbins.shape[0] - 1, zbins.shape[0])
        ax.set_xticks(xticks[::5], labels=(xbins[::5] * self.resolution).astype(np.int32),
                      rotation='vertical')
        ax.set_yticks(yticks[::5], labels=(zbins[::5] * self.resolution).astype(np.int32))
        ax.grid(which='major', axis='both', linestyle='--', linewidth=0.5)
        plt.axis('equal')
        canvas = fig.canvas
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        width, height = fig.get_size_inches() * fig.get_dpi()
        obstacle_map = o3d.geometry.Image(image_flat.reshape(int(height), int(width), 3))
        self.obstacle_widget.update_image(obstacle_map)

    def next_data(self):
        if self.index > len(self.dataset) - 1:
            return
        self.index += 1
        self.preprocess(self.index)

    def prev_data(self):
        if self.index <= 0:
            return
        self.index -= 1
        self.preprocess(self.index)

    def run(self):
        gui.Application.instance.run()

    def add_geometry(self, name: str, geometry: o3d.geometry.Geometry):
        geom_mat = o3d.visualization.rendering.MaterialRecord()
        geom_mat.shader = 'defaultUnlit'
        geom_mat.point_size = 2.0
        self.widget.scene.remove_geometry(name)
        self.widget.scene.add_geometry(name, geometry, geom_mat)

    def destroy_window(self):
        gui.Application.instance.quit()


def colorize(disp):
    image = ((disp - disp.min()) / (disp.max() - disp.min()) * 255).astype(np.uint8)
    viz_map = cv2.applyColorMap(image, 2)
    return viz_map


class DisparityViewer:
    MENU_OPEN_DIR = 1
    MENU_UPDATE = 2
    MENU_QUIT = 3

    def __init__(self,
                 baseline: float,
                 intrinsics: typing.Union[typing.List[float], np.ndarray[np.float64]],
                 path: str = None,
                 image_path: str = None,
                 win_name: str = 'disparity viewer',
                 win_width: int = 1920,
                 win_height: int = 1280):
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(win_name, win_width, win_height)

        em = self.window.theme.font_size

        if gui.Application.instance.menubar is None:
            menubar = gui.Menu()
            file_menu = gui.Menu()
            file_menu.add_item("Load", DisparityViewer.MENU_OPEN_DIR)
            file_menu.add_item("Update", DisparityViewer.MENU_UPDATE)
            file_menu.add_item("Quit", DisparityViewer.MENU_QUIT)
            # On macOS the first menu item is the application menu item and will
            # always be the name of the application (probably "Python"),
            # regardless of what you pass in here. The application menu is
            # typically where About..., Preferences..., and Quit go.
            menubar.add_menu("File", file_menu)
            gui.Application.instance.menubar = menubar

        # Each window needs to know what to do with the menu items, so we need
        # to tell the window how to handle menu items.
        self.window.set_on_menu_item_activated(DisparityViewer.MENU_OPEN_DIR,
                                               self._on_filedlg_button)
        self.window.set_on_menu_item_activated(DisparityViewer.MENU_UPDATE,
                                               self._on_pointcloud_update)
        self.window.set_on_menu_item_activated(DisparityViewer.MENU_QUIT,
                                               self._on_menu_quit)

        self.scene = gui.SceneWidget()
        self.window.add_child(self.scene)
        self.scene.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.scene.enable_scene_caching(True)
        # visualize point cloud
        self.material = o3d.visualization.rendering.MaterialRecord()
        self.material.shader = 'defaultUnlit'
        self.material.point_size = 2.0
        # visualize pivot point
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
        self.scene.scene.add_geometry('mesh_frame', self.mesh_frame, self.material)
        # setup camera
        self.scene.setup_camera(60, self.scene.scene.bounding_box, [0, 0, 0])

        # self.layout = gui.Vert(em, gui.Margins(em, em, em, em))
        self.layout = gui.CollapsableVert('settings', 10)
        self.layout.preferred_width = int(360 * self.window.scaling)
        self.window.add_child(self.layout)

        fileedit_layout = gui.CollapsableVert('directory')
        fileedit_layout.add_fixed(0.25 * em)
        self.layout.add_child(fileedit_layout)

        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button("disparity")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)

        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_child(filedlgbutton)

        collapse_images = gui.CollapsableVert('preview', 10)
        self.layout.add_child(collapse_images)

        collapse_disparity = gui.CollapsableVert('disparity', 10)
        collapse_images.add_child(collapse_disparity)
        self.widget_disparity = gui.ImageWidget()
        collapse_disparity.add_child(self.widget_disparity)

        collapse_image = gui.CollapsableVert('image', 10)
        collapse_images.add_child(collapse_image)
        self.widget_image = gui.ImageWidget()
        collapse_image.add_child(self.widget_image)

        file_collapse = gui.CollapsableVert('files', 10)
        self.layout.add_child(file_collapse)
        self.combo = gui.Combobox()
        self.combo.set_on_selection_changed(self._on_combo_selection_changed)
        file_collapse.add_child(self.combo)

        param_collapse = gui.CollapsableVert('camera params', 10)
        self.layout.add_child(param_collapse)

        # baseline
        baseline_horiz = gui.Horiz(em, gui.Margins(em, em, em, em))
        param_collapse.add_child(baseline_horiz)
        baseline_label = gui.Label('baseline')
        baseline_horiz.add_child(baseline_label)
        self.baseline_textedit = gui.TextEdit()
        self.baseline_textedit.text_value = '{}'.format(baseline)
        baseline_horiz.add_child(self.baseline_textedit)
        # fx
        f_horiz = gui.Horiz(em, gui.Margins(em, em, em, em))
        param_collapse.add_child(f_horiz)
        fx_label = gui.Label('fx')
        f_horiz.add_child(fx_label)
        self.fx_textedit = gui.TextEdit()
        self.fx_textedit.text_value = '{}'.format(intrinsics[0])
        f_horiz.add_child(self.fx_textedit)
        # fy
        fy_label = gui.Label('fy')
        f_horiz.add_child(fy_label)
        self.fy_textedit = gui.TextEdit()
        self.fy_textedit.text_value = '{}'.format(intrinsics[1])
        f_horiz.add_child(self.fy_textedit)
        # cx
        c_horiz = gui.Horiz(em, gui.Margins(em, em, em, em))
        param_collapse.add_child(c_horiz)
        cx_label = gui.Label('cx')
        c_horiz.add_child(cx_label)
        self.cx_textedit = gui.TextEdit()
        self.cx_textedit.text_value = '{}'.format(intrinsics[2])
        c_horiz.add_child(self.cx_textedit)
        # cy
        cy_label = gui.Label('cy')
        c_horiz.add_child(cy_label)
        self.cy_textedit = gui.TextEdit()
        self.cy_textedit.text_value = '{}'.format(intrinsics[3])
        c_horiz.add_child(self.cy_textedit)

        # checkbox
        self.pivot = gui.Checkbox('show pivot')
        self.pivot.set_on_checked(self._on_pivot_checked)
        self.pivot.checked = True
        self.layout.add_child(self.pivot)

        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)

        self.pcd = None

        self.image_path = image_path
        if path is not None:
            if os.path.isdir(path):
                self._fileedit.text_value = path
                filelist = os.listdir(path)
                for f in filelist:
                    self.combo.add_item(f)
                if len(filelist) > 0:
                    filename = os.path.join(self._fileedit.text_value,
                                            self.combo.selected_text)
                    self.visualize_pointcloud(filename)
            else:
                print('warning: {} is either not exists or not a directory'.format(path))

    def visualize_pointcloud(self, filename):
        data = np.load(filename)
        if isinstance(data, (dict, np.lib.npyio.NpzFile)):
            data = data['disp']
        if data.ndim == 3:
            data = data[0]
        disp = data.squeeze()
        assert len(disp.shape) == 2, '{}, {}'.format(data.shape, disp.shape)
        disp_map = colorize(disp)
        disp_o3d = o3d.geometry.Image(disp_map)
        self.widget_disparity.update_image(disp_o3d)
        baseline = float(self.baseline_textedit.text_value)
        fx = float(self.fx_textedit.text_value)
        fy = float(self.fy_textedit.text_value)
        cx = float(self.cx_textedit.text_value)
        cy = float(self.cy_textedit.text_value)
        pc = trans.disparity_to_pointcloud(disp, baseline, fx=fx, fy=fy, cx=cx, cy=cy)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3))
        if self.image_path is None:
            self.pcd.colors = o3d.utility.Vector3dVector(disp_map.reshape(-1, 3) / 255.0)
        else:
            basename = os.path.basename(filename)
            image_name = glob.glob(os.path.join(self.image_path, basename)[:-3] + '*')[0]
            image = copy.deepcopy(cv2.imread(image_name)[:, :, ::-1])
            image_o3d = o3d.geometry.Image(image)
            self.widget_image.update_image(image_o3d)
            self.pcd.colors = o3d.utility.Vector3dVector(image.reshape(-1, 3) / 255.0)
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = 'defaultUnlit'
        material.point_size = 2.0
        self.scene.scene.remove_geometry('pcd')
        self.scene.scene.add_geometry('pcd', self.pcd, material)
        self.scene.force_redraw()

    def _on_key(self, e):
        if e.key == gui.KeyName.ESCAPE:
            self.window.close()
        elif e.key == gui.KeyName.S:
            if e.type == gui.KeyEvent.UP and self.pcd is not None:
                # filename = os.path.join(self._fileedit.text_value,
                #                         self.combo.selected_text)
                o3d.io.write_point_cloud('pcd.ply', self.pcd)
        return True

    def _on_pivot_checked(self, checked):
        if checked:
            self.scene.scene.add_geometry('mesh_frame', self.mesh_frame, self.material)
        else:
            self.scene.scene.remove_geometry('mesh_frame')
        self.scene.force_redraw()

    def _on_pointcloud_update(self):
        filename = os.path.join(self._fileedit.text_value, self.combo.selected_text)
        if os.path.exists(filename):
            self.visualize_pointcloud(filename)
        else:
            print('no file selected or file not exists')

    def _on_combo_selection_changed(self, new_val, new_idx):
        filename = os.path.join(self._fileedit.text_value, new_val)
        self.visualize_pointcloud(filename)

    def _on_filedlg_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Select Directory",
                                 self.window.theme)
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self.window.show_dialog(filedlg)

    def _on_filedlg_cancel(self):
        self.window.close_dialog()

    def _on_filedlg_done(self, path):
        self._fileedit.text_value = path
        filelist = sorted(os.listdir(path))
        for f in filelist:
            self.combo.add_item(f)
        self.window.close_dialog()
        if len(filelist) > 0:
            filename = os.path.join(self._fileedit.text_value,
                                    self.combo.selected_text)
            self.visualize_pointcloud(filename)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_layout(self, layout_context):
        frame = self.window.content_rect
        self.scene.frame = frame
        panel_size = self.layout.calc_preferred_size(layout_context,
                                                     self.layout.Constraints())
        self.layout.frame = gui.Rect(frame.get_right() - panel_size.width,
                                     frame.y, panel_size.width,
                                     panel_size.height)
        self.scene.force_redraw()

    def run(self):
        gui.Application.instance.run()
