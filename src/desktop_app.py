import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from hybrid_predictor import HybridPredictor
import json

class FakeAccountDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Instagram Fake Account Detector")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Set theme colors - Instagram inspired
        self.bg_color = "#FAFAFA"
        self.primary_color = "#E1306C"  # Instagram pink
        self.secondary_color = "#405DE6"  # Instagram blue
        self.fake_color = "#E74C3C"
        self.real_color = "#27AE60"
        self.card_bg = "#FFFFFF"
        
        self.root.configure(bg=self.bg_color)
        
        # Initialize predictor
        try:
            self.predictor = HybridPredictor('models')
            self.model_loaded = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}\n\nMake sure model files are in 'models' folder.")
            self.model_loaded = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Header with gradient effect
        header_frame = tk.Frame(self.root, bg=self.primary_color, height=100)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="📷 Instagram Fake Account Detector",
            font=("Segoe UI", 28, "bold"),
            bg=self.primary_color,
            fg="white"
        )
        title_label.pack(pady=25)
                
        # Main container with scrolling
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Create canvas for scrolling
        canvas = tk.Canvas(main_container, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.bg_color)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Main content grid
        main_frame = tk.Frame(scrollable_frame, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid
        main_frame.columnconfigure(0, weight=1, minsize=600)
        main_frame.columnconfigure(1, weight=1, minsize=500)
        main_frame.rowconfigure(0, weight=1)
        
        # Left side - Input Section
        self.create_input_section(main_frame)
        
        # Right side - Results Section
        self.create_results_section(main_frame)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bottom button frame
        self.create_button_section()
        
        # Bind mouse wheel for scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def create_input_section(self, parent):
        """Create input section with better layout"""
        input_frame = tk.Frame(
            parent,
            bg=self.card_bg,
            relief=tk.RAISED,
            bd=2
        )
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)
        
        # Title
        title = tk.Label(
            input_frame,
            text="📊 Account Information",
            font=("Segoe UI", 16, "bold"),
            bg=self.card_bg,
            fg=self.secondary_color
        )
        title.pack(pady=15, padx=20, anchor="w")
        
        # Scrollable content
        canvas = tk.Canvas(input_frame, bg=self.card_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        content_frame = tk.Frame(canvas, bg=self.card_bg)
        
        content_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.entries = {}
        
        # Section 1: Numerical Features
        section1 = tk.LabelFrame(
            content_frame,
            text="Numerical Features",
            font=("Segoe UI", 12, "bold"),
            bg=self.card_bg,
            fg=self.primary_color,
            padx=15,
            pady=10
        )
        section1.pack(fill=tk.X, padx=20, pady=10)
        
        fields = [
            ("Has Profile Picture? (0=No, 1=Yes)", "profile_pic", "1"),
            ("Username Number Ratio (0.0-1.0)", "nums_username", "0.0"),
            ("Fullname Word Count", "fullname_words", "2"),
            ("Fullname Number Ratio (0.0-1.0)", "nums_fullname", "0.0"),
            ("Name equals Username? (0=No, 1=Yes)", "name_username", "0"),
            ("Bio Description Length", "desc_length", "50"),
            ("Has External URL? (0=No, 1=Yes)", "external_url", "0"),
            ("Is Private Account? (0=No, 1=Yes)", "private", "0"),
            ("Total Number of Posts", "num_posts", "100"),
            ("Total Followers Count", "num_followers", "500"),
            ("Total Following Count", "num_follows", "300")
        ]
        
        for idx, (label_text, key, default) in enumerate(fields):
            row_frame = tk.Frame(section1, bg=self.card_bg)
            row_frame.pack(fill=tk.X, pady=5)
            
            label = tk.Label(
                row_frame,
                text=label_text,
                font=("Segoe UI", 10),
                bg=self.card_bg,
                anchor="w"
            )
            label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            entry = tk.Entry(
                row_frame,
                font=("Segoe UI", 10),
                width=15,
                relief=tk.SOLID,
                bd=1
            )
            entry.pack(side=tk.RIGHT)
            entry.insert(0, default)  # Add placeholder
            self.entries[key] = entry
        
        # Section 2: Text Features
        section2 = tk.LabelFrame(
            content_frame,
            text="Text Content Features",
            font=("Segoe UI", 12, "bold"),
            bg=self.card_bg,
            fg=self.primary_color,
            padx=15,
            pady=10
        )
        section2.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Bio
        bio_label = tk.Label(
            section2,
            text="Bio / Description:",
            font=("Segoe UI", 10, "bold"),
            bg=self.card_bg,
            anchor="w"
        )
        bio_label.pack(anchor="w", pady=(5, 2))
        
        self.bio_text = tk.Text(
            section2,
            height=4,
            font=("Segoe UI", 9),
            wrap=tk.WORD,
            relief=tk.SOLID,
            bd=1
        )
        self.bio_text.pack(fill=tk.X, pady=(0, 10))
        self.bio_text.insert('1.0', "Enter account bio here...")
        
        # Captions
        captions_label = tk.Label(
            section2,
            text="Recent Post Captions (one per line):",
            font=("Segoe UI", 10, "bold"),
            bg=self.card_bg,
            anchor="w"
        )
        captions_label.pack(anchor="w", pady=(5, 2))
        
        self.captions_text = scrolledtext.ScrolledText(
            section2,
            height=6,
            font=("Segoe UI", 9),
            wrap=tk.WORD,
            relief=tk.SOLID,
            bd=1
        )
        self.captions_text.pack(fill=tk.X, pady=(0, 10))
        self.captions_text.insert('1.0', "Enter captions here, one per line...")
        
        # Comments
        comments_label = tk.Label(
            section2,
            text="Recent Comments (one per line):",
            font=("Segoe UI", 10, "bold"),
            bg=self.card_bg,
            anchor="w"
        )
        comments_label.pack(anchor="w", pady=(5, 2))
        
        self.comments_text = scrolledtext.ScrolledText(
            section2,
            height=6,
            font=("Segoe UI", 9),
            wrap=tk.WORD,
            relief=tk.SOLID,
            bd=1
        )
        self.comments_text.pack(fill=tk.X, pady=(0, 10))
        self.comments_text.insert('1.0', "Enter comments here, one per line...")
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
    
    def create_results_section(self, parent):
        """Create results display section"""
        results_frame = tk.Frame(
            parent,
            bg=self.card_bg,
            relief=tk.RAISED,
            bd=2
        )
        results_frame.grid(row=0, column=1, sticky="nsew", pady=5)
        
        # Title
        title = tk.Label(
            results_frame,
            text="🎯 Detection Results",
            font=("Segoe UI", 16, "bold"),
            bg=self.card_bg,
            fg=self.secondary_color
        )
        title.pack(pady=15, padx=20, anchor="w")
        
        # Result indicator
        self.result_frame = tk.Frame(results_frame, bg=self.bg_color, height=120)
        self.result_frame.pack(fill=tk.X, padx=20, pady=10)
        self.result_frame.pack_propagate(False)
        
        self.result_icon = tk.Label(
            self.result_frame,
            text="🔍",
            font=("Arial", 40),
            bg=self.bg_color
        )
        self.result_icon.pack(pady=5)
        
        self.result_label = tk.Label(
            self.result_frame,
            text="Ready to Analyze",
            font=("Segoe UI", 18, "bold"),
            bg=self.bg_color,
            fg=self.secondary_color
        )
        self.result_label.pack()
        
        # Metrics
        metrics_container = tk.Frame(results_frame, bg=self.card_bg)
        metrics_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.metrics_labels = {}
        
        metrics = [
            ("Overall Confidence", "confidence", "💯"),
            ("Hybrid Probability", "hybrid", "🔄"),
            ("SVM Score", "svm", "📊"),
            ("Neural Network Score", "nn", "🧠")
        ]
        
        for label_text, key, icon in metrics:
            frame = tk.Frame(
                metrics_container,
                bg="white",
                relief=tk.SOLID,
                bd=1,
                highlightbackground=self.secondary_color,
                highlightthickness=1
            )
            frame.pack(fill=tk.X, pady=6)
            
            icon_label = tk.Label(
                frame,
                text=icon,
                font=("Arial", 14),
                bg="white"
            )
            icon_label.pack(side=tk.LEFT, padx=10, pady=12)
            
            text_label = tk.Label(
                frame,
                text=label_text,
                font=("Segoe UI", 11, "bold"),
                bg="white",
                anchor="w"
            )
            text_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=12)
            
            value_label = tk.Label(
                frame,
                text="--",
                font=("Segoe UI", 12, "bold"),
                bg="white",
                fg=self.secondary_color,
                anchor="e"
            )
            value_label.pack(side=tk.RIGHT, padx=15, pady=12)
            
            self.metrics_labels[key] = value_label
        
        # Interpretation
        interp_label = tk.Label(
            results_frame,
            text="📝 Detailed Analysis",
            font=("Segoe UI", 12, "bold"),
            bg=self.card_bg,
            fg=self.primary_color,
            anchor="w"
        )
        interp_label.pack(fill=tk.X, padx=20, pady=(10, 5))
        
        self.interpretation_text = scrolledtext.ScrolledText(
            results_frame,
            height=12,
            font=("Segoe UI", 10),
            wrap=tk.WORD,
            bg="white",
            relief=tk.SOLID,
            bd=1
        )
        self.interpretation_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        self.interpretation_text.insert('1.0', "📋 Analysis results will appear here after clicking 'Analyze Account'...\n\n"
                                             "The system will:\n"
                                             "• Analyze numerical account features using SVM\n"
                                             "• Process text content using Neural Networks\n"
                                             "• Combine results for final prediction\n"
                                             "• Provide confidence scores and recommendations")
        self.interpretation_text.config(state=tk.DISABLED)
    
    def create_button_section(self):
        """Create bottom button section"""
        button_frame = tk.Frame(self.root, bg=self.bg_color, height=80)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=15)
        button_frame.pack_propagate(False)
        
        # Analyze button
        analyze_btn = tk.Button(
            button_frame,
            text="🔍 Analyze Account",
            font=("Segoe UI", 14, "bold"),
            bg=self.primary_color,
            fg="white",
            command=self.analyze_account,
            relief=tk.FLAT,
            cursor="hand2",
            activebackground="#C13584",
            activeforeground="white"
        )
        analyze_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear All Fields",
            font=("Segoe UI", 14, "bold"),
            bg=self.secondary_color,
            fg="white",
            command=self.clear_all,
            relief=tk.FLAT,
            cursor="hand2",
            activebackground="#3949AB",
            activeforeground="white"
        )
        clear_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 5))
        
        # Load Example button
        example_btn = tk.Button(
            button_frame,
            text="📄 Load Example",
            font=("Segoe UI", 14, "bold"),
            bg="#95a5a6",
            fg="white",
            command=self.load_example,
            relief=tk.FLAT,
            cursor="hand2",
            activebackground="#7f8c8d",
            activeforeground="white"
        )
        example_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
    
    def analyze_account(self):
        """Analyze the account with entered data"""
        if not self.model_loaded:
            messagebox.showerror("Error", "Models not loaded. Please check model files in 'models' folder.")
            return
        
        try:
            # Collect structured features (fixed key names)
            structured_features = []
            keys_order = ['profile_pic', 'nums_username', 'fullname_words', 'nums_fullname',
                         'name_username', 'desc_length', 'external_url', 'private',
                         'num_posts', 'num_followers', 'num_follows']
            
            for key in keys_order:
                value = self.entries[key].get().strip()
                if not value:
                    messagebox.showwarning("Missing Data", f"Please fill in: {key}\n\nAll numerical fields are required.")
                    return
                try:
                    structured_features.append(float(value))
                except ValueError:
                    messagebox.showerror("Invalid Input", f"Invalid number for {key}: '{value}'\n\nPlease enter a valid number.")
                    return
            
            # Collect text data
            bio = self.bio_text.get('1.0', tk.END).strip()
            if bio == "Enter account bio here...":
                bio = ""
            
            captions_raw = self.captions_text.get('1.0', tk.END).strip()
            if captions_raw == "Enter captions here, one per line...":
                captions = []
            else:
                captions = [c.strip() for c in captions_raw.split('\n') if c.strip()]
            
            comments_raw = self.comments_text.get('1.0', tk.END).strip()
            if comments_raw == "Enter comments here, one per line...":
                comments = []
            else:
                comments = [c.strip() for c in comments_raw.split('\n') if c.strip()]
            
            text_data = {
                'bio': bio,
                'captions': captions,
                'comments': comments
            }
            
            # Show processing message
            self.result_label.config(text="Analyzing...")
            self.root.update()
            
            # Make prediction
            result = self.predictor.predict_hybrid(structured_features, text_data)
            
            # Display results
            self.display_results(result)
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid numeric values.\n\nError: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n\n{str(e)}\n\nPlease check your input and try again.")
    
    def display_results(self, result):
        """Display analysis results"""
        is_fake = result['is_fake']
        classification = result['classification']
        
        # Update colors and icons
        if is_fake:
            color = self.fake_color
            icon = "⚠️"
            text_color = "white"
        else:
            color = self.real_color
            icon = "✅"
            text_color = "white"
        
        self.result_frame.configure(bg=color)
        self.result_icon.configure(text=icon, bg=color)
        self.result_label.configure(
            text=f"{classification} ACCOUNT",
            bg=color,
            fg=text_color
        )
        
        # Update metrics with colors
        confidence_color = self.fake_color if is_fake else self.real_color
        
        self.metrics_labels['confidence'].config(
            text=f"{result['confidence']:.1%}",
            fg=confidence_color
        )
        self.metrics_labels['hybrid'].config(text=f"{result['hybrid_probability']:.1%}")
        self.metrics_labels['svm'].config(text=f"{result['svm_probability']:.1%}")
        self.metrics_labels['nn'].config(text=f"{result['nn_probability']:.1%}")
        
        # Update interpretation
        interpretation = self.generate_interpretation(result)
        self.interpretation_text.config(state=tk.NORMAL)
        self.interpretation_text.delete('1.0', tk.END)
        self.interpretation_text.insert('1.0', interpretation)
        self.interpretation_text.config(state=tk.DISABLED)
    
    def generate_interpretation(self, result):
        """Generate human-readable interpretation"""
        interpretation = f"{'='*60}\n"
        interpretation += f"   ANALYSIS COMPLETE\n"
        interpretation += f"{'='*60}\n\n"
        
        if result['is_fake']:
            interpretation += "⚠️ WARNING: HIGH PROBABILITY OF FAKE ACCOUNT\n\n"
            interpretation += "This account exhibits multiple characteristics commonly\n"
            interpretation += "associated with fake or suspicious Instagram accounts.\n\n"
            interpretation += "KEY FINDINGS:\n"
            interpretation += f"  • Detection Confidence: {result['confidence']:.1%}\n"
            interpretation += f"  • SVM Analysis: {result['svm_probability']:.1%} suspicious\n"
            interpretation += f"  • Neural Network: {result['nn_probability']:.1%} suspicious\n"
            interpretation += f"  • Combined Score: {result['hybrid_probability']:.1%}\n\n"
            interpretation += "RISK INDICATORS:\n"
            
            if result['svm_probability'] > 0.7:
                interpretation += "  ⚠️ Profile metrics show bot-like patterns\n"
            if result['nn_probability'] > 0.7:
                interpretation += "  ⚠️ Text content appears automated/copied\n"
            if result['hybrid_probability'] > 0.8:
                interpretation += "  ⚠️ Very high confidence - likely fake\n"
            
            interpretation += "\nRECOMMENDATIONS:\n"
            interpretation += "  • Avoid sharing personal information\n"
            interpretation += "  • Do not click on suspicious links\n"
            interpretation += "  • Consider reporting to Instagram\n"
            interpretation += "  • Verify through other channels if important\n"
        else:
            interpretation += "✅ ACCOUNT APPEARS LEGITIMATE\n\n"
            interpretation += "This account shows characteristics typical of\n"
            interpretation += "authentic Instagram profiles.\n\n"
            interpretation += "KEY FINDINGS:\n"
            interpretation += f"  • Authenticity Confidence: {result['confidence']:.1%}\n"
            interpretation += f"  • SVM Analysis: {result['svm_probability']:.1%} suspicious\n"
            interpretation += f"  • Neural Network: {result['nn_probability']:.1%} suspicious\n"
            interpretation += f"  • Combined Score: {result['hybrid_probability']:.1%}\n\n"
            interpretation += "POSITIVE INDICATORS:\n"
            
            if result['svm_probability'] < 0.3:
                interpretation += "  ✓ Profile metrics look natural\n"
            if result['nn_probability'] < 0.3:
                interpretation += "  ✓ Content appears genuine\n"
            if result['hybrid_probability'] < 0.2:
                interpretation += "  ✓ Very high confidence in authenticity\n"
            
            interpretation += "\nNOTE:\n"
            interpretation += "  While indicators suggest this is a real account,\n"
            interpretation += "  always exercise caution when interacting online.\n"
            interpretation += "  No automated system is 100% accurate.\n"
        
        interpretation += f"\n{'='*60}\n"
        interpretation += "Analysis Method: Hybrid SVM + Neural Network\n"
        interpretation += "Model Weights: 60% SVM + 40% NN\n"
        interpretation += f"{'='*60}"
        
        return interpretation
    
    def load_example(self):
        """Load example data for testing"""
        # Fake account example
        example = messagebox.askyesno(
            "Load Example",
            "Load FAKE account example?\n\n"
            "Click 'Yes' for Fake Account Example\n"
            "Click 'No' for Real Account Example"
        )
        
        if example:  # Fake account
            self.entries['profile_pic'].delete(0, tk.END)
            self.entries['profile_pic'].insert(0, "0")
            self.entries['nums_username'].delete(0, tk.END)
            self.entries['nums_username'].insert(0, "0.48")
            self.entries['fullname_words'].delete(0, tk.END)
            self.entries['fullname_words'].insert(0, "1")
            self.entries['nums_fullname'].delete(0, tk.END)
            self.entries['nums_fullname'].insert(0, "0.25")
            self.entries['name_username'].delete(0, tk.END)
            self.entries['name_username'].insert(0, "0")
            self.entries['desc_length'].delete(0, tk.END)
            self.entries['desc_length'].insert(0, "0")
            self.entries['external_url'].delete(0, tk.END)
            self.entries['external_url'].insert(0, "0")
            self.entries['private'].delete(0, tk.END)
            self.entries['private'].insert(0, "0")
            self.entries['num_posts'].delete(0, tk.END)
            self.entries['num_posts'].insert(0, "12")
            self.entries['num_followers'].delete(0, tk.END)
            self.entries['num_followers'].insert(0, "291")
            self.entries['num_follows'].delete(0, tk.END)
            self.entries['num_follows'].insert(0, "2284")
            
            self.bio_text.delete('1.0', tk.END)
            self.bio_text.insert('1.0', "")
            
            self.captions_text.delete('1.0', tk.END)
            self.captions_text.insert('1.0', "Follow for follow!\nLike for like\nDM for shoutout")
            
            self.comments_text.delete('1.0', tk.END)
            self.comments_text.insert('1.0', "Nice pic\nAmazing\nGreat post")
        else:  # Real account
            self.entries['profile_pic'].delete(0, tk.END)
            self.entries['profile_pic'].insert(0, "1")
            self.entries['nums_username'].delete(0, tk.END)
            self.entries['nums_username'].insert(0, "0.0")
            self.entries['fullname_words'].delete(0, tk.END)
            self.entries['fullname_words'].insert(0, "2")
            self.entries['nums_fullname'].delete(0, tk.END)
            self.entries['nums_fullname'].insert(0, "0.0")
            self.entries['name_username'].delete(0, tk.END)
            self.entries['name_username'].insert(0, "0")
            self.entries['desc_length'].delete(0, tk.END)
            self.entries['desc_length'].insert(0, "82")
            self.entries['external_url'].delete(0, tk.END)
            self.entries['external_url'].insert(0, "0")
            self.entries['private'].delete(0, tk.END)
            self.entries['private'].insert(0, "0")
            self.entries['num_posts'].delete(0, tk.END)
            self.entries['num_posts'].insert(0, "679")
            self.entries['num_followers'].delete(0, tk.END)
            self.entries['num_followers'].insert(0, "414")
            self.entries['num_follows'].delete(0, tk.END)
            self.entries['num_follows'].insert(0, "651")
            
            self.bio_text.delete('1.0', tk.END)
            self.bio_text.insert('1.0', "Digital creator | Photographer | Travel enthusiast | Living life one adventure at a time 🌍📸")
            
            self.captions_text.delete('1.0', tk.END)
            self.captions_text.insert('1.0', "Beautiful sunset at the beach today 🌅\nCoffee and conversations ☕\nWeekend vibes with friends!")
            
            self.comments_text.delete('1.0', tk.END)
            self.comments_text.insert('1.0', "This is gorgeous! Where was this taken?\nLove your photography style!\nCan't wait to see more!")
    
    def clear_all(self):
        """Clear all input fields and results"""
        # Clear entries
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        
        # Clear text fields
        self.bio_text.delete('1.0', tk.END)
        self.bio_text.insert('1.0', "Enter account bio here...")
        
        self.captions_text.delete('1.0', tk.END)
        self.captions_text.insert('1.0', "Enter captions here, one per line...")
        
        self.comments_text.delete('1.0', tk.END)
        self.comments_text.insert('1.0', "Enter comments here, one per line...")
        
        # Reset results
        self.result_frame.configure(bg=self.bg_color)
        self.result_icon.configure(text="🔍", bg=self.bg_color)
        self.result_label.configure(
            text="Ready to Analyze",
            bg=self.bg_color,
            fg=self.secondary_color
        )
        
        for label in self.metrics_labels.values():
            label.config(text="--", fg=self.secondary_color)
        
        self.interpretation_text.config(state=tk.NORMAL)
        self.interpretation_text.delete('1.0', tk.END)
        self.interpretation_text.insert('1.0', "📋 Analysis results will appear here after clicking 'Analyze Account'...\n\n"
                                             "The system will:\n"
                                             "• Analyze numerical account features using SVM\n"
                                             "• Process text content using Neural Networks\n"
                                             "• Combine results for final prediction\n"
                                             "• Provide confidence scores and recommendations")
        self.interpretation_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = FakeAccountDetectorApp(root)
    root.mainloop()