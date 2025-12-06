#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline Control Center v2.0
- Fixed paths for pipeline1/ structure
- Cross-platform support
- Better error handling
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import sys
import os
import json
import yaml
import shutil
import threading
import queue
import platform
from pathlib import Path
from datetime import datetime

# ============================================================================
# Path Configuration
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()      # main_control/
PIPELINE_DIR = SCRIPT_DIR.parent                   # pipeline1/
PROJECT_ROOT = PIPELINE_DIR.parent                 # AAI6610_FALL2025/

# Config paths
CONFIG_PATH = PIPELINE_DIR / "config.yaml"
STATE_FILE = PIPELINE_DIR / "pipeline_state.json"

# Script paths
CLUSTERING_CODES = PIPELINE_DIR / "clustering" / "codes"
SCRAPER_CODES = PIPELINE_DIR / "scrapers" / "codes"

# Output paths
SCRAPER_OUTPUTS = PIPELINE_DIR / "scrapers" / "outputs"
CLUSTERING_OUTPUTS = PIPELINE_DIR / "clustering" / "outputs"

# ============================================================================
# GUI Application
# ============================================================================
class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Uncertainty Pipeline Control Center v2.0")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 800)
        
        # Validate paths
        if not CONFIG_PATH.exists():
            # Try alternate location
            alt_config = PROJECT_ROOT / "config.yaml"
            if alt_config.exists():
                self.config_path = alt_config
            else:
                messagebox.showerror("Error", f"config.yaml not found!\nExpected: {CONFIG_PATH}")
                raise FileNotFoundError("Config not found")
        else:
            self.config_path = CONFIG_PATH
        
        # State
        self.is_running = False
        self.should_stop = False
        self.current_process = None
        self.output_queue = queue.Queue()
        
        # Build UI
        self.create_ui()
        self.load_config()
        self.update_status()
        self.process_output_queue()
    
    def create_ui(self):
        """Create the user interface"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky='nsew')
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # ========== Left Panel: Controls ==========
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, rowspan=3, sticky='nsew', padx=(0, 10))
        
        # --- Step 1: Scrapers ---
        scraper_frame = ttk.LabelFrame(control_frame, text="Step 1: Collect Data", padding="8")
        scraper_frame.pack(fill='x', pady=(0, 10))
        
        self.scraper_vars = {}
        scrapers = [
            ('arxiv', 'arXiv Papers', True),
            ('openalex', 'OpenAlex/OpenReview', True),
            ('reddit', 'Reddit/StackExchange', True),
            ('news', 'News Articles', True),
            ('biorxiv', 'bioRxiv Papers', False),
            ('linkedin', 'LinkedIn (Paid)', False),
        ]
        
        for name, label, default in scrapers:
            var = tk.BooleanVar(value=default)
            self.scraper_vars[name] = var
            ttk.Checkbutton(scraper_frame, text=label, variable=var).pack(anchor='w', pady=1)
        
        tk.Button(
            scraper_frame, text="▶ Run Selected Scrapers",
            command=self.run_scrapers,
            bg="#9C27B0", fg="white", font=("Arial", 10, "bold")
        ).pack(fill='x', pady=(8, 0))
        
        # --- Step 2: Processing ---
        process_frame = ttk.LabelFrame(control_frame, text="Step 2: Process Data", padding="8")
        process_frame.pack(fill='x', pady=(0, 10))
        
        steps = [
            ("1. Semantic Filtering", "#FF9800", self.run_filtering),
            ("2. Clustering", "#4CAF50", self.run_clustering),
            ("3. Visualization", "#2196F3", self.run_visualization),
        ]
        
        for text, color, cmd in steps:
            tk.Button(
                process_frame, text=text, command=cmd,
                bg=color, fg="white", font=("Arial", 9)
            ).pack(fill='x', pady=2)
        
        # --- Step 3: Auto Run ---
        auto_frame = ttk.LabelFrame(control_frame, text="Step 3: Auto Pipeline", padding="8")
        auto_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(
            auto_frame, 
            text="Runs: Filter → Cluster → Visualize",
            font=("Arial", 8)
        ).pack(anchor='w')
        
        self.run_btn = tk.Button(
            auto_frame, text="▶ Run Full Pipeline",
            command=self.run_full_pipeline,
            bg="#00BCD4", fg="white", font=("Arial", 10, "bold"), height=2
        )
        self.run_btn.pack(fill='x', pady=(5, 2))
        
        self.stop_btn = tk.Button(
            auto_frame, text="■ Stop",
            command=self.stop_pipeline,
            bg="#F44336", fg="white", state='disabled'
        )
        self.stop_btn.pack(fill='x')
        
        # --- System ---
        sys_frame = ttk.LabelFrame(control_frame, text="System", padding="8")
        sys_frame.pack(fill='x')
        
        ttk.Button(sys_frame, text="📁 View Results", command=self.view_results).pack(fill='x', pady=1)
        ttk.Button(sys_frame, text="🔄 Check Status", command=self.check_data_status).pack(fill='x', pady=1)
        ttk.Button(sys_frame, text="🗑 Reset All Data", command=self.reset_pipeline).pack(fill='x', pady=1)
        
        # ========== Right Panel: Status & Log ==========
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, rowspan=3, sticky='nsew')
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # Status
        status_frame = ttk.LabelFrame(right_frame, text="Pipeline Status", padding="10")
        status_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready", font=("Arial", 12, "bold"))
        self.status_label.pack(anchor='w')
        
        self.progress_label = ttk.Label(status_frame, text="Progress: Waiting...")
        self.progress_label.pack(anchor='w', pady=5)
        
        self.progress_bar = ttk.Progressbar(status_frame, mode='determinate', length=400)
        self.progress_bar.pack(fill='x', pady=5)
        
        # Log
        log_frame = ttk.LabelFrame(right_frame, text="Output Log", padding="10")
        log_frame.grid(row=1, column=0, sticky='nsew')
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap='word', width=70, height=25,
            font=("Consolas", 9), bg="#1E1E1E", fg="#D4D4D4"
        )
        self.log_text.pack(fill='both', expand=True)
        
        # Log buttons
        log_btn_frame = ttk.Frame(log_frame)
        log_btn_frame.pack(fill='x', pady=(5, 0))
        ttk.Button(log_btn_frame, text="Clear", command=self.clear_log).pack(side='left', padx=2)
        ttk.Button(log_btn_frame, text="Save", command=self.save_log).pack(side='left', padx=2)
    
    # ========== Config & Logging ==========
    def load_config(self):
        """Load configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            topic = config.get('topic', 'ML Uncertainty')
            self.log(f"Loaded config: {self.config_path.name}")
            self.log(f"Topic: {topic}")
        except Exception as e:
            self.log(f"Config error: {e}")
    
    def log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{timestamp}] {message}\n")
        self.log_text.see('end')
        self.root.update_idletasks()
    
    def clear_log(self):
        self.log_text.delete('1.0', 'end')
    
    def save_log(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get('1.0', 'end'))
            self.log(f"Log saved to {filepath}")
    
    # ========== Script Paths ==========
    def get_script_path(self, name: str) -> Path:
        """Get correct script path"""
        if name in ['semantic_filter.py', 'cluster_engine.py', 'visualizer.py']:
            return CLUSTERING_CODES / name
        return SCRAPER_CODES / name
    
    # ========== Script Execution ==========
    def run_script_in_terminal(self, script_path: Path, title: str = "Script"):
        """Run script in new terminal window"""
        if not script_path.exists():
            self.log(f"ERROR: Script not found: {script_path.name}")
            messagebox.showerror("Error", f"Script not found:\n{script_path}")
            return False
        
        self.log(f"Starting: {script_path.name}")
        
        try:
            if platform.system() == "Windows":
                cmd = f'start "{title}" cmd /k "cd /d {PIPELINE_DIR} && python {script_path} && pause"'
                subprocess.Popen(cmd, shell=True)
            elif platform.system() == "Darwin":  # macOS
                cmd = f'osascript -e \'tell app "Terminal" to do script "cd {PIPELINE_DIR} && python3 {script_path}"\''
                subprocess.Popen(cmd, shell=True)
            else:  # Linux
                cmd = f'gnome-terminal -- bash -c "cd {PIPELINE_DIR} && python3 {script_path}; read -p \'Press Enter...\'"'
                subprocess.Popen(cmd, shell=True)
            
            return True
        except Exception as e:
            self.log(f"ERROR: {e}")
            return False
    
    def run_script_background(self, script_path: Path) -> bool:
        """Run script in background, capture output"""
        if not script_path.exists():
            self.log(f"ERROR: Script not found: {script_path.name}")
            return False
        
        try:
            self.current_process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(PIPELINE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )
            
            # Read output
            while True:
                if self.should_stop:
                    self.current_process.terminate()
                    return False
                
                line = self.current_process.stdout.readline()
                if not line and self.current_process.poll() is not None:
                    break
                
                if line:
                    self.output_queue.put(line.rstrip())
            
            rc = self.current_process.wait()
            self.current_process = None
            return rc == 0
            
        except Exception as e:
            self.log(f"ERROR: {e}")
            return False
    
    def process_output_queue(self):
        """Process queued output messages"""
        try:
            while True:
                line = self.output_queue.get_nowait()
                self.log(f"  {line}")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_output_queue)
    
    # ========== Scraper Actions ==========
    def run_scrapers(self):
        """Run selected scrapers"""
        selected = [name for name, var in self.scraper_vars.items() if var.get()]
        
        if not selected:
            messagebox.showwarning("Warning", "Please select at least one scraper!")
            return
        
        self.log(f"\nStarting {len(selected)} scraper(s)...")
        
        for name in selected:
            script = self.get_script_path(f"scraper_{name}.py")
            self.run_script_in_terminal(script, f"Scraper: {name.upper()}")
            self.log(f"  Launched: {name}")
            
    # ========== Processing Actions ==========
    def run_filtering(self):
        script = self.get_script_path("semantic_filter.py")
        self.run_script_in_terminal(script, "Semantic Filter")
    
    def run_clustering(self):
        script = self.get_script_path("cluster_engine.py")
        self.run_script_in_terminal(script, "Clustering")
    
    def run_visualization(self):
        script = self.get_script_path("visualizer.py")
        self.run_script_in_terminal(script, "Visualization")
    
    # ========== Full Pipeline ==========
    def run_full_pipeline(self):
        """Run complete processing pipeline"""
        if self.is_running:
            messagebox.showwarning("Warning", "Pipeline is already running!")
            return
        
        # Check for data
        if not self._has_scraped_data():
            if not messagebox.askyesno(
                "No Data Found",
                "No scraped data found!\n\nRun scrapers first to collect data.\n\nContinue anyway?"
            ):
                return
        
        if not messagebox.askyesno(
            "Run Pipeline",
            "This will run:\n1. Semantic Filtering\n2. Clustering Analysis\n3. Visualization\n\nContinue?"
        ):
            return
        
        self.is_running = True
        self.should_stop = False
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="Status: Running...")
        
        self.log("\n" + "=" * 60)
        self.log("Starting Full Pipeline")
        self.log("=" * 60)
        
        threading.Thread(target=self._run_pipeline_thread, daemon=True).start()
    
    def _run_pipeline_thread(self):
        """Pipeline execution thread"""
        stages = [
            ("Semantic Filtering", "semantic_filter.py"),
            ("Clustering Analysis", "cluster_engine.py"),
            ("Visualization", "visualizer.py"),
        ]
        
        success = True
        
        for idx, (name, script_name) in enumerate(stages, 1):
            if self.should_stop:
                self.root.after(0, lambda: self.log("\n⚠ Pipeline stopped by user"))
                break
            
            # Update progress
            self.root.after(0, lambda n=name, i=idx: self._update_progress(n, i, len(stages)))
            self.root.after(0, lambda n=name: self.log(f"\n▶ Stage {idx}: {n}"))
            
            script_path = self.get_script_path(script_name)
            
            if not self.run_script_background(script_path):
                self.root.after(0, lambda n=name: self.log(f"✗ {n} FAILED"))
                success = False
                break
            
            self.root.after(0, lambda n=name: self.log(f"✓ {n} completed"))
        
        # Finish
        if success and not self.should_stop:
            self.root.after(0, self._pipeline_success)
        else:
            self.root.after(0, self._pipeline_failed)
    
    def _update_progress(self, stage: str, current: int, total: int):
        self.progress_label.config(text=f"Progress: {current}/{total} - {stage}")
        self.progress_bar['value'] = (current / total) * 100
    
    def _pipeline_success(self):
        self.is_running = False
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Status: Complete ✓")
        self.progress_bar['value'] = 100
        
        self.log("\n" + "=" * 60)
        self.log("✓ PIPELINE COMPLETED SUCCESSFULLY")
        self.log("=" * 60)
        
        messagebox.showinfo("Complete", "Pipeline finished!\n\nClick 'View Results' to see outputs.")
    
    def _pipeline_failed(self):
        self.is_running = False
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Status: Failed ✗")
        
        self.log("\n✗ Pipeline failed - check log for details")
        messagebox.showerror("Error", "Pipeline failed!\n\nCheck the log for details.")
    
    def stop_pipeline(self):
        if messagebox.askyesno("Stop", "Stop the running pipeline?"):
            self.should_stop = True
            if self.current_process:
                self.current_process.terminate()
            self.log("Stopping pipeline...")
    
    # ========== Utility Functions ==========
    def _has_scraped_data(self) -> bool:
        """Check if any scraped data exists"""
        if not SCRAPER_OUTPUTS.exists():
            return False
        
        for folder in SCRAPER_OUTPUTS.iterdir():
            if folder.is_dir():
                files = list(folder.glob("*.txt")) + list(folder.glob("*.pdf"))
                if files:
                    return True
        return False
    
    def check_data_status(self):
        """Check and display data status"""
        self.log("\n" + "=" * 50)
        self.log("DATA STATUS")
        self.log("=" * 50)
        
        # Scraped data
        self.log("\nScraped Data:")
        if SCRAPER_OUTPUTS.exists():
            for folder in sorted(SCRAPER_OUTPUTS.iterdir()):
                if folder.is_dir():
                    txt = len(list(folder.rglob("*.txt")))
                    pdf = len(list(folder.rglob("*.pdf")))
                    if txt + pdf > 0:
                        self.log(f"  {folder.name}: {txt} TXT, {pdf} PDF")
        
        # Processed data
        self.log("\nProcessed Data:")
        
        filtered = CLUSTERING_OUTPUTS / "filtered_posts_all_sources"
        if filtered.exists():
            count = len(list(filtered.glob("*.txt")))
            self.log(f"  Filtered texts: {count}")
        
        clusters = CLUSTERING_OUTPUTS / "cluster_output"
        if clusters.exists():
            csvs = len(list(clusters.glob("*.csv")))
            self.log(f"  Cluster files: {csvs}")
        
        viz = CLUSTERING_OUTPUTS / "cluster_visualizations"
        if viz.exists():
            pngs = len(list(viz.glob("*.png")))
            self.log(f"  Visualizations: {pngs}")
        
        self.log("=" * 50)
    
    def view_results(self):
        """Open results folder"""
        paths_to_try = [
            CLUSTERING_OUTPUTS / "cluster_visualizations",
            CLUSTERING_OUTPUTS / "cluster_output",
            CLUSTERING_OUTPUTS,
        ]
        
        for path in paths_to_try:
            if path.exists():
                if platform.system() == "Windows":
                    os.startfile(path)
                elif platform.system() == "Darwin":
                    subprocess.run(["open", str(path)])
                else:
                    subprocess.run(["xdg-open", str(path)])
                return
        
        messagebox.showinfo("No Results", "No results found.\n\nRun the pipeline first!")
    
    def reset_pipeline(self):
        """Reset all pipeline data"""
        if not messagebox.askyesno(
            "Reset",
            "⚠ This will DELETE all data!\n\n• Scraped articles\n• Filtered texts\n• Cluster results\n• Visualizations\n\nContinue?"
        ):
            return
        
        deleted = 0
        
        try:
            # Clear scraper outputs
            if SCRAPER_OUTPUTS.exists():
                for folder in SCRAPER_OUTPUTS.iterdir():
                    if folder.is_dir():
                        shutil.rmtree(folder)
                        deleted += 1
            
            # Clear clustering outputs
            if CLUSTERING_OUTPUTS.exists():
                for folder in CLUSTERING_OUTPUTS.iterdir():
                    if folder.is_dir():
                        shutil.rmtree(folder)
                        deleted += 1
            
            self.log(f"\n✓ Cleared {deleted} folders")
            messagebox.showinfo("Reset", f"Cleared {deleted} data folders.")
            
        except Exception as e:
            self.log(f"Reset error: {e}")
            messagebox.showerror("Error", f"Reset failed:\n{e}")
    
    def update_status(self):
        """Periodic status update"""
        if not self.is_running:
            # Could add status checks here
            pass
        self.root.after(5000, self.update_status)

# ============================================================================
# Main
# ============================================================================
def main():
    print(f"\n{'='*60}")
    print("ML Uncertainty Pipeline Control Center v2.0")
    print(f"{'='*60}")
    print(f"Pipeline directory: {PIPELINE_DIR}")
    print(f"{'='*60}\n")
    
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()