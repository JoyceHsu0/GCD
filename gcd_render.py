import sys
import vtk, vtk.util.numpy_support
import numpy as np
import imageio


class VTKRenderer:
    def __init__(self, vtk_widget, main_window=None):
        self.vtk_widget = vtk_widget
        self.main_window = main_window
        self.renderer = vtk.vtkRenderer()
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = self.render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.volume = None
        
        # Initialize as an empty list, waiting for external setup
        self.color_settings = []
        self.opacity_settings = []
        
        self.rotating = False
        self.timer_id = None
        self.rotation_speed = 0.5  # Default rotation speed
        
        self.initial_camera = None # Store initial camera state
        #self.setup_default_camera()  
        
        self.frame_count = 0  # Frame count for saving screenshots
        
        self.observer_tag = None  # New: Store the observer tag
        
        # Add axes indicator
        self.add_axes_indicator()

    def setup_default_camera(self):
        """Setup a default camera state with Y-axis upward"""
        self.initial_camera = vtk.vtkCamera()
        self.initial_camera.SetPosition(0, 0, 500)  
        self.initial_camera.SetFocalPoint(0, 0, 0)  
        self.initial_camera.SetViewUp(0, 1, 0)      
        self.initial_camera.SetViewAngle(30.0)      
        
    def add_axes_indicator(self):
        """Add a 3D axes indicator to the renderer"""
        # Create axes actor
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)  # Set the total length of the axes
        axes.SetShaftTypeToCylinder()       # Set the shaft type to cylinder
        axes.SetCylinderRadius(0.02)        # Set the cylinder radius
        axes.SetConeRadius(0.1)             # Set the cone radius

        # Set the labels
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")

        # Create an orientation marker widget
        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(axes)
        self.orientation_widget.SetInteractor(self.interactor)
        
        # Set the orientation marker to the lower right corner
        self.orientation_widget.SetViewport(0.8, 0.0, 1.0, 0.2)  # (xmin, ymin, xmax, ymax)
        
        # Set the background color of the orientation marker to match the renderer
        self.orientation_widget.SetEnabled(1)
        self.orientation_widget.InteractiveOff()  # Make the orientation marker non-interactive
        
    def setup_volume_data(self, data, spacing):
        np_array = np.ascontiguousarray(data.numpy())
        vtk_array = vtk.util.numpy_support.numpy_to_vtk(np_array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(np_array.shape[::-1])
        image_data.SetSpacing(spacing)
        image_data.GetPointData().SetScalars(vtk_array)

        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(image_data)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.SetAmbient(0.4)
        volume_property.SetDiffuse(0.6)
        volume_property.SetSpecular(0.4)

        if self.volume:
            self.renderer.RemoveVolume(self.volume)
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(volume_mapper)
        self.volume.SetProperty(volume_property)

        self.renderer.AddVolume(self.volume)
        self.renderer.SetBackground(0.1, 0.1, 0.1)

        camera = self.renderer.GetActiveCamera()
        
        bounds = image_data.GetBounds()
        #print(f"Bounds: {bounds}")
        center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
        #print(f"Center: {center}")
        distance = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 2
        #print(f"Distance: {distance}")
        
        camera.SetPosition(center[0], center[1], center[2] + distance)
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 1, 0)
        #print(f"Set Position: {camera.GetPosition()}")
        self.renderer.ResetCameraClippingRange()

        if self.color_settings and self.opacity_settings:
            self.apply_transfer_functions()

        self.store_initial_camera()
        self.render_window.Render()
        #print(f"Final Position: {camera.GetPosition()}, ViewUp: {camera.GetViewUp()}")
        
    def apply_transfer_functions(self):
        if not self.volume:
            return
        volume_property = self.volume.GetProperty()
        
        # Set the opacity function
        opacity_function = vtk.vtkPiecewiseFunction()
        for value, opacity in self.opacity_settings:
            opacity_function.AddPoint(value, opacity)

        # Set the color function
        color_function = vtk.vtkColorTransferFunction()
        for value, r, g, b in self.color_settings:
            color_function.AddRGBPoint(value, r, g, b)

        volume_property.SetScalarOpacity(opacity_function)
        volume_property.SetColor(color_function)

    def clear_volumes(self):
        """Clear all volume data from the renderer"""
        if self.renderer and self.volume:
            self.renderer.RemoveVolume(self.volume)
            self.volume = None
        self.render()

    def render(self):
        """Render the scene"""
        if self.render_window:
            self.render_window.Render()
        else:
            print("Warning: Render window not available")
            
    def reset_camera(self):
        """Reset the camera to its initial angle with Y-axis upward"""
        camera = self.renderer.GetActiveCamera()
        if self.initial_camera is not None:
            camera.DeepCopy(self.initial_camera)  
            camera.SetViewUp(0, 1, 0)           
        else:
            self.renderer.ResetCamera()  
            camera.SetViewUp(0, 1, 0)    
            self.store_initial_camera()
        self.renderer.ResetCameraClippingRange()
        #print(f"Camera after setup/reset - Position: {camera.GetPosition()}, ViewUp: {camera.GetViewUp()}")
        self.render_window.Render()
    
    def store_initial_camera(self):
        """Store the current camera state as the initial camera state"""
        camera = self.renderer.GetActiveCamera()
        self.initial_camera = vtk.vtkCamera()
        self.initial_camera.DeepCopy(camera)
                
    def render(self):
        self.render_window.Render()

    def set_rotation_speed(self, speed):
        """Set the rotation speed of the renderer"""
        self.rotation_speed = speed
        
    def start_rotation(self):
        """Start rotating the renderer, ensuring a clean start"""
        if self.rotating:
            self.stop_rotation()
        
        if self.rotation_speed > 0:
            base_speed = 30.0
            timer_interval = 30
            def rotate_callback(obj, event):
                degrees_per_second = base_speed * self.rotation_speed
                angle_step = degrees_per_second * (timer_interval / 1000.0)
                camera = self.renderer.GetActiveCamera()
                camera.Azimuth(angle_step)
                self.renderer.ResetCameraClippingRange()
                self.render_window.Render()
            
            # Store the observer tag when adding it
            self.observer_tag = self.interactor.AddObserver("TimerEvent", rotate_callback)
            self.timer_id = self.interactor.CreateRepeatingTimer(timer_interval)
            self.rotating = True
            if self.main_window:
                self.main_window.start_button.setEnabled(False)
                self.main_window.stop_button.setEnabled(True)

    def stop_rotation(self):
        """Stop rotating the renderer"""
        if self.rotating and self.interactor:
            self.rotating = False
            if self.timer_id is not None:
                self.interactor.DestroyTimer(self.timer_id)
                self.timer_id = None
            if self.observer_tag is not None:
                self.interactor.RemoveObserver(self.observer_tag)
                self.observer_tag = None
            if self.main_window:
                self.main_window.start_button.setEnabled(True)
                self.main_window.stop_button.setEnabled(False)

    def set_rotation_speed(self, speed):
        self.rotation_speed = speed
        
    def set_color_settings(self, color_settings, opacity_settings):
        """Set the color and opacity settings for the volume data"""
        self.color_settings = color_settings  # [(value, r, g, b), ...]
        self.opacity_settings = opacity_settings  # [(value, opacity), ...]
        self.apply_transfer_functions()
        self.render()

    def save_screenshot(self, filename):
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(f"{filename}.png")
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

    def record_rotation_video(self, filename, rotation_speed):
        """Record a 360° rotation video of the volume rendering"""
        if not self.volume:
            print("No volume data to record.", flush=True)
            return
        if rotation_speed <= 0:
            print("Rotation speed is 0 or negative, no rotation will occur.", flush=True)
            return

        total_rotation = 360
        fps = 30
        base_speed = 60.0
        degrees_per_second = base_speed * rotation_speed
        recording_time = total_rotation / degrees_per_second
        total_frames = max(int(recording_time * fps), 30)  # Ensure at least 30 frames
        angle_step = total_rotation / total_frames

        # Print recording details as requested
        print(f"Recording 360° rotation: speed={rotation_speed:.1f}, "
            f"{total_frames} frames, {recording_time:.2f} seconds", flush=True)

        if not filename.endswith('.mp4'):
            filename += '.mp4'
        writer = imageio.get_writer(filename, fps=fps, codec='libx264', quality=8)

        self.render_window.SetSize(1328, 960)
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.SetScale(1)

        camera = self.renderer.GetActiveCamera()
        initial_position = camera.GetPosition()

        print("Starting video recording...", flush=True)
        for frame in range(total_frames):
            camera.Azimuth(angle_step)
            self.render_window.Render()

            window_to_image_filter.Modified()
            window_to_image_filter.Update()
            vtk_image = window_to_image_filter.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            numpy_array = np.flipud(
                vtk.util.numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, 3)
            )
            writer.append_data(numpy_array)
            
            # Optional: Print progress every 10% of the frames
            if total_frames > 10 and frame % (total_frames // 10) == 0:
                print(f"Progress: {frame}/{total_frames} frames ({(frame/total_frames)*100:.0f}%)", flush=True)

        writer.close()
        camera.SetPosition(*initial_position)  # Reset camera to initial position
        self.render()

        print(f"Video recording completed. Saved as {filename}", flush=True)


if __name__ == "__main__":
    import torch
    from PyQt6.QtWidgets import QApplication
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    app = QApplication(sys.argv)
    vtk_widget = QVTKRenderWindowInteractor()
    renderer = VTKRenderer(vtk_widget)
    dummy_data = torch.zeros(128, 256, 256)
    renderer.setup_volume_data(dummy_data, (0.7, 0.7, 1.0))
    vtk_widget.Initialize()
    vtk_widget.Start()
    renderer.render()
    renderer.start_rotation()
    vtk_widget.show()
    app.exec()
