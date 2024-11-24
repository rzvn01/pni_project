import os
import platform
from PIL import Image, ImageTk
from algorithm import sketch
import customtkinter
import threading
from tkinter import filedialog, Canvas

# Set up the theme for the customtkinter application
customtkinter.set_default_color_theme("dark-blue")

# Constants for application
APP_NAME = "Sketch Image Retrieval"
WIDTH = 1200
HEIGHT = 800


# Main Application Class
class GUI (customtkinter.CTk):

    def __init__(self, *args, **kwargs):
        """
        Initializes the GUI application.
        Sets up the window, frames, and UI elements.
        """
        super().__init__(*args, **kwargs)
        self.leaderboard_images = ["loc1.png", "loc2.png", "loc3.png"]  # Overlay images for leaderboard
        self.auto = False  # Default to auto-select parameters
        self.K = None  # K parameter (number of bins)
        self.W = None  # W parameter (number of blocks)
        self.image_refs = []  # Keep references to images displayed on the canvas
        self.k_parameters_list = [str(round(x, 1)) for x in range(20, 90, 5)]  # K options
        self.w_parameters_list = [str(round(x, 1)) for x in range(20, 90, 5)]  # W options

        # Set up the GUI window and layout
        self.setup_window()
        self.create_frames()
        self.add_elements_top_frame()
        self.add_elements_bottom_frame()
        self.update_canvas_background()

    def create_frames(self):
        """
        Creates the top and bottom frames for the application.
        """
        self.grid_rowconfigure(0, weight=9)  # Top frame takes up 90% of the window
        self.grid_rowconfigure(1, weight=1)  # Bottom frame takes up 10%
        self.grid_columnconfigure(0, weight=1)

        # Top frame for image and leaderboard display
        self.frame_top = customtkinter.CTkFrame(master=self, corner_radius=0)
        self.frame_top.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        # Bottom frame for control elements
        self.frame_bottom = customtkinter.CTkFrame(master=self, corner_radius=0)
        self.frame_bottom.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")

    def add_elements_top_frame(self):
        """
        Adds a canvas for displaying images and leaderboard to the top frame.
        """
        self.canvas = customtkinter.CTkCanvas(self.frame_top, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

    def add_elements_bottom_frame(self):
        """
        Adds controls (buttons, parameter selectors) to the bottom frame with customized colors.
        """
        # Appearance mode options
        appearance_frame = customtkinter.CTkFrame(self.frame_bottom, fg_color="transparent")
        appearance_frame.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="w")

        self.appearance_mode_label = customtkinter.CTkLabel(
            appearance_frame, text="Appearance Mode:", anchor="w", text_color="cyan"
        )
        self.appearance_mode_label.grid(row=0, column=0, padx=(0, 5), sticky="w")

        self.appearance_mode_option_menu = customtkinter.CTkOptionMenu(
            appearance_frame,
            values=["Dark", "Light"],
            command=self.change_appearance_mode,
            fg_color="blue",  # Button color
            button_color="cyan",  # Dropdown button color
            button_hover_color="darkblue",  # Dropdown hover color
            text_color="white"  # Text color
        )
        self.appearance_mode_option_menu.grid(row=0, column=1, padx=(5, 0), sticky="w")

        # File selection (Choose Sketch Image button and path entry)
        sketch_frame = customtkinter.CTkFrame(self.frame_bottom, fg_color="transparent")
        sketch_frame.grid(row=0, column=1, padx=(10, 10), pady=20, sticky="w")

        self.choose_button = customtkinter.CTkButton(
            sketch_frame,
            text="Choose Sketch Image",
            command=self.choose_sketch_image,
            fg_color="green",  # Button color
            hover_color="darkgreen",  # Hover color
            text_color="white"  # Text color
        )
        self.choose_button.grid(row=0, column=0, padx=(0, 5), sticky="w")

        self.path_entry = customtkinter.CTkEntry(
            sketch_frame,
            width=400,
            placeholder_text="Sketch Path",
            fg_color="lightgray",  # Entry background
            text_color="black"  # Text color
        )
        self.path_entry.grid(row=0, column=1, padx=(5, 0), sticky="w")

        # Action buttons (Go and Reset)
        action_frame = customtkinter.CTkFrame(self.frame_bottom, fg_color="transparent")
        action_frame.grid(row=0, column=2, padx=(10, 20), pady=20, sticky="e")

        self.go_button = customtkinter.CTkButton(
            action_frame,
            text="Go",
            command=self.start_similarity_search,
            fg_color="blue",  # Button color
            hover_color="darkblue",  # Hover color
            text_color="white"  # Text color
        )
        self.go_button.grid(row=0, column=0, padx=(0, 5), sticky="e")
        self.disable_button(self.go_button)  # Initially disabled

        self.reset_button = customtkinter.CTkButton(
            action_frame,
            text="Reset",
            command=self.reset_canvas_and_entry,
            fg_color="red",  # Button color
            hover_color="darkred",  # Hover color
            text_color="white"  # Text color
        )
        self.reset_button.grid(row=0, column=1, sticky="e")

        # Parameters row (Auto Select, K, and W options)
        parameters_frame = customtkinter.CTkFrame(self.frame_bottom, fg_color="transparent")
        parameters_frame.grid(row=1, column=0, columnspan=3, padx=(20, 20), pady=(20, 0), sticky="w")

        self.auto_select = customtkinter.CTkCheckBox(
            parameters_frame,
            text="Auto Select",
            command=self.set_auto_parameters,
            fg_color="purple",  # Check box color
            hover_color="violet",  # Hover color
            text_color="white"  # Text color
        )
        self.auto_select.grid(row=0, column=0, padx=(0, 10), pady=(0, 0), sticky="w")

        # K parameter label and combobox
        k_frame = customtkinter.CTkFrame(parameters_frame, fg_color="transparent")
        k_frame.grid(row=0, column=1, padx=(0, 20), pady=(0, 0), sticky="w")

        self.k_parameters_label = customtkinter.CTkLabel(
            k_frame, text="K parameter", text_color="orange"
        )
        self.k_parameters_label.grid(row=0, column=0, padx=(0, 5), sticky="e")

        self.k_parameters = customtkinter.CTkComboBox(
            k_frame,
            values=self.k_parameters_list,
            width=80,
            fg_color="gray",  # Background color
            button_color="orange",  # Dropdown button color
            button_hover_color="darkorange",  # Hover color
            text_color="black"  # Text color
        )
        self.k_parameters.grid(row=0, column=1, padx=(5, 0), sticky="w")

        # W parameter label and combobox
        w_frame = customtkinter.CTkFrame(parameters_frame, fg_color="transparent")
        w_frame.grid(row=0, column=2, padx=(0, 20), pady=(0, 0), sticky="w")

        self.w_parameters_label = customtkinter.CTkLabel(
            w_frame, text="W parameter", text_color="orange"
        )
        self.w_parameters_label.grid(row=0, column=0, padx=(0, 5), sticky="e")

        self.w_parameters = customtkinter.CTkComboBox(
            w_frame,
            values=self.w_parameters_list,
            width=80,
            fg_color="gray",  # Background color
            button_color="orange",  # Dropdown button color
            button_hover_color="darkorange",  # Hover color
            text_color="black"  # Text color
        )
        self.w_parameters.grid(row=0, column=1, padx=(5, 0), sticky="w")

    def set_auto_parameters(self):
        """
        Enables or disables manual selection of parameters K and W based on the Auto Select checkbox.
        """
        if self.auto_select.get():  # Auto-select enabled
            self.k_parameters.configure(state=customtkinter.DISABLED)
            self.w_parameters.configure(state=customtkinter.DISABLED)
        else:  # Manual selection enabled
            self.k_parameters.configure(state=customtkinter.NORMAL)
            self.w_parameters.configure(state=customtkinter.NORMAL)

    def set_k_parameter(self, value):
        """
        Sets the K parameter and disables auto mode.
        """
        self.K = value
        self.auto = False

    def set_w_parameter(self, value):
        """
        Sets the W parameter and disables auto mode.
        """
        self.W = value
        self.auto = False

    def choose_sketch_image(self):
        """
        Opens a file dialog to select a sketch image.
        """
        file_path = filedialog.askopenfilename(
            title="Select Sketch Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")],
            initialdir="../"
        )
        if file_path:
            self.path_entry.delete(0, "end")  # Clear previous path
            self.path_entry.insert(0, file_path)  # Add new path
            self.enable_button(self.go_button)  # Enable Go button

    def add_loading_circle(self):
        """
        Adds a spinning transparent circle to the canvas.
        """
        # Create a new Canvas overlay for the loading circle
        appearance_mode = customtkinter.get_appearance_mode()
        if appearance_mode == "Dark":
           color = "#23272d"  # Dark mode background
        else:
            color = "white"  # Light mode background

        self.loading_canvas = Canvas(
            self.canvas,  # Parent widget
            width=150, height=150,
            bg=color,  # Match parent widget's background to mimic transparency
            highlightthickness=0  # Remove border highlight
        )
        self.loading_canvas.place(relx=0.5, rely=0.5, anchor="center")  # Center it on the canvas

        # Create a circular arc (partial circle)
        self.loading_arc = self.loading_canvas.create_arc(
            10, 10, 140, 140,  # Bounding box for the circle
            start=0, extent=90,  # Start angle and extent of the arc
            outline="blue", width=4,  # Circle color and line width
            style="arc"  # Render as an arc
        )

        # Start the animation
        self.animate_circle()

    def animate_circle(self, angle=0):
        """
        Animates the spinning circle by incrementally rotating the arc.

        Args:
            angle (int): Current rotation angle.
        """
        # Safeguard: Check if the canvas and arc still exist
        if hasattr(self, "loading_canvas") and self.loading_canvas:
            try:
                # Update the arc's start angle
                self.loading_canvas.itemconfig(self.loading_arc, start=angle)
                # Schedule the next frame of the animation
                self.loading_animation = self.after(50, self.animate_circle, (angle + 10) % 360)
            except Exception as e:
                print(f"Animation stopped due to error: {e}")

    def remove_loading_circle(self):
        """
        Stops the spinning animation and removes the circle from the canvas.
        """
        # Stop the animation by cancelling the scheduled after callback
        if hasattr(self, "loading_animation"):
            self.after_cancel(self.loading_animation)

        # Destroy the loading canvas
        if hasattr(self, "loading_canvas"):
            self.loading_canvas.destroy()
            del self.loading_canvas  # Remove the reference

    def start_similarity_search(self):
        """
        Starts the similarity search by displaying the loading circle and processing results.
        """
        self.canvas.delete("all")  # Clear all drawings and images on the canvas
        self.image_refs.clear()  # Clear stored image references
        query_image_path = self.path_entry.get()
        if query_image_path and os.path.exists(query_image_path):
            # Add and display the loading circle
            self.add_loading_circle()

            # Determine the values for K, W, and auto based on the current settings
            auto = self.auto_select.get()  # Get the value of the "Auto Select" checkbox
            K = int(self.k_parameters.get()) if not auto else None  # Get the default or user-selected K
            W = int(self.w_parameters.get()) if not auto else None  # Get the default or user-selected W

            # Run similarity search in a separate thread
            def process_similarity():
                # Perform the similarity search with the selected parameters
                self.similarity_scores = sketch.rank_images(
                    query_image_path=query_image_path,
                    auto=auto,
                    K=K,
                    W=W,
                    test_image_dir="../dataset/"  # Adjust as per your dataset location
                )

                # Once done, remove the loading circle and display results
                self.remove_loading_circle()
                self.display_leaderboard()

            # Start the thread for background processing
            threading.Thread(target=process_similarity).start()
        else:
            print("Please select a valid sketch image path.")

    def display_leaderboard(self):
        """
        Displays the top-ranked images on the canvas, overlaying them with leaderboard graphics.
        """
        self.remove_loading_circle()
        global x_pos, y_pos

        # Check if similarity scores exist and have enough results
        if not self.similarity_scores or len(self.similarity_scores) < 3:
            print("Not enough results to display leaderboard.")
            return

        # Clear existing content on the canvas
        self.canvas.delete("all")

        # Loop through the top 3 results and create composite images
        for rank, (image_path, _) in enumerate(self.similarity_scores[:3]):
            overlay_path = self.leaderboard_images[rank]  # Get overlay for the specific rank
            composite_image = self.composite_images(image_path, overlay_path, rank)

            # Resize composite image for display
            composite_image = composite_image.resize((400, 250), Image.Resampling.LANCZOS)

            # Convert the composite image to a format compatible with Tkinter
            composite_image_tk = ImageTk.PhotoImage(composite_image)

            # Store the reference to prevent garbage collection
            self.image_refs.append(composite_image_tk)

            # Position images differently for rank 1, 2, and 3
            if rank == 0:  # First place
                x_pos = 550
                y_pos = 50
            elif rank == 1:  # Second place
                x_pos = 50
                y_pos = 250
            elif rank == 2:  # Third place
                x_pos = 1000
                y_pos = 250

            # Place the composite image on the canvas
            self.canvas.create_image(x_pos, y_pos, image=composite_image_tk, anchor="nw")

        # Add the original sketch image to the canvas for reference
        image_path = self.path_entry.get()
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                image.thumbnail((300, 250))  # Resize for display
                image_tk = ImageTk.PhotoImage(image)

                # Display the image on the canvas
                self.canvas.create_image(600, 550, image=image_tk, anchor="nw")
                self.image_refs.append(image_tk)

                # Add a label below the sketch image
                self.canvas.create_text(
                    750, 525,  # Position for the label
                    text="The Sketch",  # Label text
                    fill="Brown",  # Label color
                    font=("Helvetica", 20, "bold")  # Font style
                )
            except Exception as e:
                print(f"Error loading image: {e}")
        else:
            print("Invalid file path. Please enter a valid image path.")

    def update_canvas_background(self):
        """
        Updates the canvas background color based on the selected appearance mode (dark or light).
        """
        appearance_mode = customtkinter.get_appearance_mode()
        if appearance_mode == "Dark":
            self.canvas.configure(bg="#23272d")  # Dark mode background
        else:
            self.canvas.configure(bg="white")  # Light mode background

    def change_appearance_mode(self, new_appearance_mode):
        """
        Changes the appearance mode of the application and updates the canvas accordingly.

        Args:
            new_appearance_mode (str): Selected mode ("Dark", "Light",").
        """
        customtkinter.set_appearance_mode(new_appearance_mode)
        self.update_canvas_background()  # Dynamically update canvas background

    def reset_canvas_and_entry(self):
        """
        Resets the canvas, clears the image references, and resets the file path entry.
        """
        self.canvas.delete("all")  # Clear all drawings and images on the canvas
        self.image_refs.clear()  # Clear stored image references

        # Reset the path entry field
        self.path_entry.delete(0, "end")
        self.path_entry.insert(0, "Please select a sketch.")

        # Disable the "Go" button to prevent accidental actions
        self.disable_button(self.go_button)

    @staticmethod
    def composite_images(background_path, overlay_path, rank=1, output_path="output.png"):
        """
        Creates a composite image by overlaying a leaderboard graphic over the background image.

        Args:
            background_path (str): Path to the background image (e.g., test image).
            overlay_path (str): Path to the overlay image (e.g., leaderboard graphic).
            rank (int): Rank of the image, used to adjust scaling.
            output_path (str): Path to save the composite image (optional).

        Returns:
            PIL.Image: Composite image with the overlay applied.
        """
        # Open the background and overlay images
        background = Image.open(background_path)
        overlay = Image.open(overlay_path).convert("RGBA")  # Convert to RGBA for transparency

        # Resize the background to fit within the overlay's container area
        container_width, container_height = overlay.size[0] * 0.5, overlay.size[1] * (0.6 + 0.1 * rank)
        resized_background = background.resize((int(container_width), int(container_height)))

        # Create a blank RGBA image with the same size as the overlay
        composite_image = Image.new("RGBA", overlay.size, (0, 0, 0, 0))

        # Center the background image within the overlay
        position_x = (overlay.size[0] - resized_background.size[0]) // 2
        position_y = (overlay.size[1] - resized_background.size[1]) // 2
        composite_image.paste(resized_background, (position_x, position_y))

        # Overlay the leaderboard graphic on top of the background
        final_image = Image.alpha_composite(composite_image, overlay)
        return final_image

    def setup_window(self):
        """
        Configures the main application window (title, size, and close event behavior).
        Disables resizing of the window.
        """
        self.title(APP_NAME)  # Set application title
        self.geometry(f"{WIDTH}x{HEIGHT}")  # Set window dimensions
        self.minsize(WIDTH, HEIGHT)  # Minimum window size

        # Disable resizing of the window
        self.resizable(False, False)  # Disable both horizontal and vertical resizing
        customtkinter.set_appearance_mode("Dark")
        # Handle closing behavior for different platforms
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        if platform.system() == "Darwin":  # macOS
            self.bind("<Command-q>", self.on_closing)
            self.createcommand("tk::mac::Quit", self.on_closing)
        else:
            self.bind("<Control-q>", self.on_closing)

    @staticmethod
    def disable_button(button: customtkinter.CTkButton):
        """
        Disables a button to prevent user interaction.

        Args:
            button (customtkinter.CTkButton): Button to disable.
        """
        button.configure(state=customtkinter.DISABLED)

    @staticmethod
    def enable_button(button: customtkinter.CTkButton):
        """
        Enables a button to allow user interaction.

        Args:
            button (customtkinter.CTkButton): Button to enable.
        """
        button.configure(state=customtkinter.NORMAL)

    def on_closing(self, key=None):
        """
        Handles the window close event and terminates the application.
        """
        self.destroy()

    def start(self):
        """
        Starts the application's main loop.
        """
        self.mainloop()


if __name__ == "__main__":
    """
    Main execution entry point.
    Initializes and starts the application.
    """
    app =GUI()  # Instantiate the application
    app.start()  # Start the main loop
