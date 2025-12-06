#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline Control Center - Final Version
Sequence: 1.Scrapers -> 2.Manual Operations -> 3.One-Click Run -> 4.System
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
import sys
import os
import threading
import time
import queue


class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Uncertainty Pipeline Control Center")
        self.root.geometry("1200x1000")
        self.root.minsize(1000, 850)
        
        # Find project root directory
        current_file = Path(__file__).resolve()
        search_path = current_file.parent
        found = False
        
        while search_path.parent != search_path:
            if (search_path / "config.yaml").exists():
                self.project_root = search_path
                found = True
                break
            search_path = search_path.parent
        
        if not found:
            messagebox.showerror("Error", "Cannot find project root directory!")
            raise FileNotFoundError("Project root not found")
        
        self.config_path = self.project_root / "config.yaml"
        self.state_file = self.project_root / "pipeline_state.json"
        
        # Running state
        self.is_running = False
        self.should_stop = False
        self.current_process = None
        self.output_queue = queue.Queue()
        
        self.create_ui()
        self.load_config()
        self.update_status()
        self.process_output_queue()
    
    def create_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # ========== Left Control Panel ==========
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Step 1: Scrapers
        scraper_frame = ttk.LabelFrame(control_frame, text="Step 1: Collect Data", padding="8")
        scraper_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(scraper_frame, text="Required:", font=("Arial", 8, "bold")).pack(anchor=tk.W, pady=(0, 2))
        
        self.scraper_vars = {}
        for scraper in ['arxiv', 'biorxiv']:
            var = tk.BooleanVar(value=True)
            self.scraper_vars[scraper] = var
            ttk.Checkbutton(scraper_frame, text=scraper.upper(), variable=var).pack(anchor=tk.W, pady=1, padx=8)
        
        ttk.Separator(scraper_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        ttk.Label(scraper_frame, text="Optional:", font=("Arial", 8, "bold")).pack(anchor=tk.W, pady=(0, 2))
        
        for scraper in ['news', 'reddit', 'openalex', 'linkedin']:
            default = (scraper in ['news', 'reddit'])
            var = tk.BooleanVar(value=default)
            self.scraper_vars[scraper] = var
            label = scraper.upper() + (" (Paid)" if scraper == 'linkedin' else "")
            ttk.Checkbutton(scraper_frame, text=label, variable=var).pack(anchor=tk.W, pady=1, padx=8)
        
        tk.Button(scraper_frame, text="Run Selected Scrapers", command=self.run_scrapers,
                 bg="#9C27B0", fg="white", font=("Arial", 9, "bold"), cursor="hand2").pack(fill=tk.X, pady=(6, 0))
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        # Step 2: Manual Operations
        manual_frame = ttk.LabelFrame(control_frame, text="Step 2: Step-by-Step Processing", padding="8")
        manual_frame.pack(fill=tk.X, pady=(0, 8))
        
        for i, (text, color, cmd) in enumerate([
            ("[1] Semantic Filtering", "#FF9800", self.run_filtering),
            ("[2] Clustering Analysis", "#4CAF50", self.run_clustering),
            ("[3] Visualization Generation", "#2196F3", self.run_visualization),
            ("[4] View Results", "#E91E63", self.view_results)
        ]):
            tk.Button(manual_frame, text=text, command=cmd, bg=color, fg="white",
                     font=("Arial", 9), cursor="hand2").pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        # Step 3: One-Click Run
        auto_frame = ttk.LabelFrame(control_frame, text="Step 3: Automation", padding="8")
        auto_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(auto_frame, text="WARNING: Run scrapers first to collect data",
                 font=("Arial", 7), foreground="#F44336").pack(anchor=tk.W, pady=(0, 4))
        
        self.run_button = tk.Button(auto_frame, text="Auto Execute Full Pipeline\n(Filter -> Cluster -> Visualize)",
                                    command=self.run_full_pipeline, bg="#00BCD4", fg="white",
                                    font=("Arial", 10, "bold"), height=2, cursor="hand2")
        self.run_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = tk.Button(auto_frame, text="Stop Running", command=self.stop_pipeline,
                                     bg="#F44336", fg="white", font=("Arial", 9, "bold"), state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        ttk.Label(auto_frame, text="Will execute in order, auto-stop on error",
                 font=("Arial", 7), foreground="gray").pack(anchor=tk.W, pady=(4, 0))
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        # System
        system_frame = ttk.LabelFrame(control_frame, text="System", padding="8")
        system_frame.pack(fill=tk.X)
        
        # Topic setting - compact layout
        topic_subframe = ttk.Frame(system_frame)
        topic_subframe.pack(fill=tk.X, pady=(0, 4))
        
        ttk.Label(topic_subframe, text="Topic:", font=("Arial", 8)).pack(side=tk.LEFT, padx=(0, 3))
        self.topic_var = tk.StringVar(value="")
        self.topic_entry = ttk.Entry(topic_subframe, textvariable=self.topic_var, width=18, font=("Arial", 8))
        self.topic_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(topic_subframe, text="Update", command=self.update_topic, width=5).pack(side=tk.LEFT)
        
        ttk.Separator(system_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        
        btn_frame1 = ttk.Frame(system_frame)
        btn_frame1.pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame1, text="Reset State", command=self.reset_pipeline).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(btn_frame1, text="Close CMD", command=self.close_all_cmd).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        btn_frame2 = ttk.Frame(system_frame)
        btn_frame2.pack(fill=tk.X, pady=1)
        ttk.Button(btn_frame2, text="Force Kill", command=self.force_kill_all).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(btn_frame2, text="Refresh Status", command=self.update_status).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        # ========== Right Status Display ==========
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        status_frame = ttk.LabelFrame(right_frame, text="Pipeline Status", padding="10")
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready", font=("Arial", 12, "bold"))
        self.status_label.pack(anchor=tk.W)
        
        self.progress_label = ttk.Label(status_frame, text="Progress: Waiting to start")
        self.progress_label.pack(anchor=tk.W, pady=5)
        
        self.progress_bar = ttk.Progressbar(status_frame, mode='determinate', length=300)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        log_frame = ttk.LabelFrame(right_frame, text="Output Log", padding="10")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=70, height=30,
                                                   font=("Consolas", 9), bg="#1E1E1E", fg="#D4D4D4")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        log_btn_frame = ttk.Frame(log_frame)
        log_btn_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(log_btn_frame, text="Clear", command=self.clear_log).pack(side=tk.LEFT, padx=2)
        ttk.Button(log_btn_frame, text="Save", command=self.save_log).pack(side=tk.LEFT, padx=2)
    
    def load_config(self):
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                topic = config.get('topic', 'uncertainty estimation in machine learning')
                self.topic_var.set(topic)
                self.log(f"Loaded topic: {topic}")
        except Exception as e:
            self.log(f"Config loading failed: {e}")
    
    def update_topic(self):
        """Update research topic"""
        new_topic = self.topic_var.get().strip()
        
        if not new_topic:
            messagebox.showwarning("Warning", "Topic cannot be empty!")
            return
        
        try:
            # Read existing config
            config = {}
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            
            # Update topic
            config['topic'] = new_topic
            
            # Save config
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            self.log(f"Topic updated: {new_topic}")
            messagebox.showinfo("Success", f"Research topic updated to:\n\n{new_topic}\n\nNOTE: Query words in semantic filter need manual synchronization!")
            
        except Exception as e:
            self.log(f"Update failed: {e}")
            messagebox.showerror("Error", f"Topic update failed:\n{e}")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
    
    def save_log(self):
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(defaultextension=".txt")
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self.log("Log saved!")
    
    def run_full_pipeline(self):
        if self.is_running:
            messagebox.showwarning("Warning", "Pipeline is already running!")
            return
        
        warning_msg = self._check_existing_data()
        if warning_msg:
            if not messagebox.askyesno("Existing Data Detected", warning_msg + "\n\nContinue?"):
                return
        
        if not messagebox.askyesno("Confirm Run", "Will auto-execute:\n1. Semantic Filtering\n2. Clustering Analysis\n3. Visualization Generation\n\nContinue?"):
            return
        
        self.is_running = True
        self.should_stop = False
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Running...")
        
        self.log("\n" + "="*60)
        self.log("Start auto-execution")
        self.log("="*60 + "\n")
        
        threading.Thread(target=self._run_pipeline_sequence, daemon=True).start()
    
    def _check_existing_data(self):
        warnings = []
        filtered = self.project_root / "clustering" / "outputs" / "filtered_posts_all_sources"
        if filtered.exists() and len(list(filtered.glob("*.txt"))) > 0:
            warnings.append(f"Found {len(list(filtered.glob('*.txt')))} filtered texts")
        
        cluster = self.project_root / "clustering" / "outputs" / "cluster_output"
        if cluster.exists() and len(list(cluster.glob("*.csv"))) > 0:
            warnings.append(f"Found {len(list(cluster.glob('*.csv')))} cluster files")
        
        viz = self.project_root / "clustering" / "outputs" / "cluster_visualizations"
        if viz.exists() and len(list(viz.glob("*.png"))) > 0:
            warnings.append(f"Found {len(list(viz.glob('*.png')))} charts")
        
        return "WARNING: Existing data detected:\n\n" + "\n".join(warnings) + "\n\nWill be overwritten!" if warnings else None
    
    def stop_pipeline(self):
        if messagebox.askyesno("Stop Confirmation", "Are you sure to stop?"):
            self.should_stop = True
            self.log("\nStopping...")
            if self.current_process:
                try:
                    self.current_process.terminate()
                except:
                    pass
    
    def _run_pipeline_sequence(self):
        try:
            stages = [("Semantic Filtering", "semantic_filter.py"),
                     ("Clustering Analysis", "cluster_engine.py"),
                     ("Visualization Generation", "visualizer.py")]
            
            for idx, (name, script) in enumerate(stages, 1):
                if self.should_stop:
                    self.root.after(0, lambda: self.log("\nInterrupted\n"))
                    break
                
                self.root.after(0, lambda n=name, i=idx: self._update_progress(n, i, len(stages)))
                
                script_path = self._get_script_path(script)
                if not script_path.exists():
                    self.root.after(0, lambda s=script: self.log(f"Script not found: {s}"))
                    break
                
                self.root.after(0, lambda n=name: self.log(f"\n{n}..."))
                
                if not self._run_script(script_path):
                    self.root.after(0, lambda n=name: self.log(f"\n[{n}] Failed!"))
                    break
                
                self.root.after(0, lambda n=name: self.log(f"\n[{n}] Completed"))
                time.sleep(1)
            else:
                self.root.after(0, self._pipeline_completed)
                return
            
            self.root.after(0, self._pipeline_error)
        except Exception as e:
            self.root.after(0, lambda e=e: self.log(f"\nException: {e}"))
            self.root.after(0, self._pipeline_error)
    
    def _run_script(self, script_path):
        try:
            self.current_process = subprocess.Popen([sys.executable, str(script_path)],
                                                    cwd=str(self.project_root),
                                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                    text=True, encoding='utf-8', errors='replace', bufsize=1)
            
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
            self.root.after(0, lambda e=e: self.log(f"Error: {e}"))
            return False
    
    def process_output_queue(self):
        try:
            while True:
                line = self.output_queue.get_nowait()
                if 'Success' in line or 'Complete' in line:
                    self.log(f"   {line}")
                elif 'Error' in line or 'Failed' in line:
                    self.log(f"   {line}")
                elif '%|' in line:
                    self.log(f"   {line}")
                else:
                    self.log(f"   {line}")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_output_queue)
    
    def _get_script_path(self, name):
        if name in ['semantic_filter.py', 'cluster_engine.py', 'visualizer.py']:
            return self.project_root / "clustering" / "codes" / name
        return self.project_root / "scrapers" / "codes" / name
    
    def _update_progress(self, stage, current, total):
        self.progress_label.config(text=f"Progress: {current}/{total} - {stage}")
        self.progress_bar['value'] = (current / total) * 100
    
    def _pipeline_completed(self):
        self.is_running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Completed!")
        self.progress_bar['value'] = 100
        self.log("\n" + "="*60)
        self.log("Pipeline completed!")
        self.log("="*60 + "\n")
        messagebox.showinfo("Complete", "Execution finished!\n\nClick 'View Results' to see outputs.")
    
    def _pipeline_error(self):
        self.is_running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Failed")
        messagebox.showerror("Error", "Pipeline failed, please check log.")
    
    def run_filtering(self):
        if self._is_script_running("semantic_filter.py"):
            messagebox.showwarning("Running Conflict", "Semantic filtering is running!\n\nPlease wait for completion.")
            return
        
        script = self.project_root / "clustering" / "codes" / "semantic_filter.py"
        subprocess.Popen(f'start "Filter" cmd /k "cd /d {self.project_root} && python {script} && pause"', shell=True)
        self.log("Semantic filtering started")
    
    def run_clustering(self):
        if self._is_script_running("cluster_engine.py"):
            messagebox.showwarning("Running Conflict", "Clustering is running!\n\nPlease wait for completion.")
            return
        
        script = self.project_root / "clustering" / "codes" / "cluster_engine.py"
        subprocess.Popen(f'start "Clustering" cmd /k "cd /d {self.project_root} && python {script} && pause"', shell=True)
        self.log("Clustering started")
    
    def run_visualization(self):
        if self._is_script_running("visualizer.py"):
            messagebox.showwarning("Running Conflict", "Visualization is running!\n\nPlease wait for completion.")
            return
        
        script = self.project_root / "clustering" / "codes" / "visualizer.py"
        subprocess.Popen(f'start "Viz" cmd /k "cd /d {self.project_root} && python {script} && pause"', shell=True)
        self.log("Visualization started")
    
    def run_scrapers(self):
        selected = [n for n, v in self.scraper_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("Warning", "Please select at least one scraper!")
            return
        
        running = self._check_running_scrapers(selected)
        if running:
            messagebox.showwarning("Running Conflict", f"Currently running:\n\n" + "\n".join([f"- {s}" for s in running]))
            return
        
        self.log(f"Starting {len(selected)} scraper(s)...")
        for s in selected:
            script = self.project_root / "scrapers" / "codes" / f"scraper_{s}.py"
            if script.exists():
                subprocess.Popen(f'start "{s.upper()}" cmd /k "cd /d {self.project_root} && python {script} && pause"', shell=True)
                self.log(f"   {s}")
    
    def _check_running_scrapers(self, scrapers):
        try:
            result = subprocess.run('wmic process where "name=\'python.exe\'" get commandline /format:list',
                                   shell=True, capture_output=True, text=True, encoding='gbk', errors='ignore', timeout=5)
            return [s.upper() for s in scrapers if f"scraper_{s}.py" in result.stdout]
        except:
            return []
    
    def _is_script_running(self, script):
        try:
            result = subprocess.run('wmic process where "name=\'python.exe\'" get commandline /format:list',
                                   shell=True, capture_output=True, text=True, encoding='gbk', errors='ignore', timeout=5)
            return script in result.stdout
        except:
            return False
    
    def _check_scraper_data(self, scrapers):
        existing = []
        folder_map = {'arxiv': 'arxiv_papers', 'biorxiv': 'biorxiv_papers', 'reddit': 'reddit_posts',
                     'linkedin': 'linkedin_posts', 'news': 'news_articles', 'openalex': 'openalex_openreview_papers/openalex'}
        
        for s in scrapers:
            if s in folder_map:
                folder = self.project_root / "scrapers" / "outputs" / folder_map[s]
                if folder.exists():
                    total = len(list(folder.glob("*.txt"))) + len(list(folder.glob("*.pdf")))
                    if total > 0:
                        existing.append(f"{s.upper()}: {total} files")
        return existing
    
    def close_all_cmd(self):
        """Close all CMD windows and scraper processes"""
        if not messagebox.askyesno("Confirm", "Close all script windows?\n\nWill also terminate related Python processes."):
            return
        
        try:
            self.log("Closing all scripts...")
            
            killed_cmd = 0
            killed_python = 0
            
            # 1. Close CMD windows
            result = subprocess.run('tasklist /V /FI "IMAGENAME eq cmd.exe" /FO CSV',
                                   shell=True, capture_output=True, text=True, errors='ignore')
            
            keywords = ['Clustering', 'Visualization', 'Filter', 'ARXIV', 'REDDIT', 'NEWS', 'BIORXIV', 'OPENALEX', 'LINKEDIN']
            for line in result.stdout.split('\n'):
                for kw in keywords:
                    if kw.lower() in line.lower():
                        try:
                            pid = line.split(',')[1].strip('"')
                            if pid.isdigit():
                                subprocess.run(f'taskkill /PID {pid} /F /T', shell=True, capture_output=True)
                                killed_cmd += 1
                        except:
                            pass
                        break
            
            # 2. Terminate scraper Python processes
            result2 = subprocess.run('wmic process where "name=\'python.exe\'" get commandline,processid /format:list',
                                    shell=True, capture_output=True, text=True, encoding='gbk', errors='ignore', timeout=10)
            
            current_pid = os.getpid()
            scraper_keywords = ['scraper_', 'semantic_filter', 'cluster_engine', 'visualizer']
            
            lines = result2.stdout.split('\n\n')
            for block in lines:
                if any(kw in block.lower() for kw in scraper_keywords):
                    # Extract PID
                    for line in block.split('\n'):
                        if 'ProcessId=' in line:
                            try:
                                pid = line.split('=')[1].strip()
                                if pid.isdigit() and int(pid) != current_pid:
                                    subprocess.run(f'taskkill /PID {pid} /F /T', shell=True, capture_output=True)
                                    killed_python += 1
                            except:
                                pass
            
            self.log(f"Closed {killed_cmd} CMD windows")
            self.log(f"Terminated {killed_python} Python processes")
            
        except Exception as e:
            self.log(f"Error: {e}")
    
    def force_kill_all(self):
        if not messagebox.askyesno("WARNING", "Terminate all Python processes?\n\nWill close all Python programs!"):
            return
        
        try:
            current_pid = os.getpid()
            killed = 0
            result = subprocess.run('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH',
                                   shell=True, capture_output=True, text=True)
            
            for line in result.stdout.split('\n'):
                try:
                    pid = line.split(',')[1].strip('"')
                    if pid.isdigit() and int(pid) != current_pid:
                        subprocess.run(f'taskkill /PID {pid} /F /T', shell=True, capture_output=True)
                        killed += 1
                except:
                    pass
            
            self.log(f"WARNING: Terminated {killed} processes")
        except Exception as e:
            self.log(f"Error: {e}")
    
    def reset_pipeline(self):
        if not messagebox.askyesno("Reset Confirmation", "WARNING: Will delete all data!\n\nContinue?"):
            return
        
        try:
            deleted = 0
            for folder in (self.project_root / "scrapers" / "outputs").iterdir():
                if folder.is_dir() and folder.name != ".gitkeep":
                    shutil.rmtree(folder)
                    deleted += 1
            
            for folder in (self.project_root / "clustering" / "outputs").iterdir():
                if folder.is_dir():
                    shutil.rmtree(folder)
                    deleted += 1
            
            self.log(f"Cleaned {deleted} folders")
            messagebox.showinfo("Success", "All data cleared!")
        except Exception as e:
            self.log(f"Error: {e}")
    
    def view_results(self):
        viz = self.project_root / "clustering" / "outputs" / "cluster_visualizations"
        cluster = self.project_root / "clustering" / "outputs" / "cluster_output"
        
        if viz.exists():
            os.startfile(viz)
        elif cluster.exists():
            os.startfile(cluster)
        else:
            messagebox.showinfo("No Results", "Results not found!\n\nPlease run the full pipeline first.")
    
    def update_status(self):
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                stages = state.get('stages', {})
                completed = sum(1 for s in stages.values() if s['status'] == 'completed')
                total = len(stages)
                
                if not self.is_running:
                    self.progress_label.config(text=f"Progress: {completed}/{total}")
                    self.progress_bar['value'] = (completed / total) * 100 if total > 0 else 0
        except:
            pass
        
        self.root.after(5000, self.update_status)


def check_dependencies():
    """Check required dependencies"""
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'hdbscan': 'hdbscan',
        'umap-learn': 'umap',
        'sentence-transformers': 'sentence_transformers',
        'openai': 'openai',
        'nltk': 'nltk',
        'matplotlib': 'matplotlib',
        'requests': 'requests',
        'beautifulsoup4': 'bs4',
        'playwright': 'playwright',
        'praw': 'praw',
        'pypdf': 'pypdf',
        'pyyaml': 'yaml',
    }
    
    missing = []
    
    print("\n" + "="*70)
    print("[CHECK] Checking dependencies...")
    print("="*70 + "\n")
    
    for pip_name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"[ OK ] {pip_name}")
        except ImportError:
            print(f"[MISS] {pip_name}")
            missing.append(pip_name)
    
    print("\n" + "="*70)
    
    if missing:
        print(f"[WARN] Missing {len(missing)} dependencies!")
        print("="*70 + "\n")
        print("Please run the following command to install:\n")
        print(f"pip install {' '.join(missing)}")
        print("\nOr use China mirror:")
        print(f"pip install {' '.join(missing)} -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("\n" + "="*70 + "\n")
        
        response = input("Install automatically now? (y/N): ").strip().lower()
        if response == 'y':
            try:
                cmd = [sys.executable, '-m', 'pip', 'install'] + missing + ['-i', 'https://pypi.tuna.tsinghua.edu.cn/simple']
                subprocess.run(cmd, check=True)
                print("\n[ OK ] Installation complete!\n")
                return True
            except:
                print("\n[ERROR] Installation failed, please install manually.\n")
                return False
        else:
            return False
    else:
        print("[ OK ] All dependencies installed!")
        print("="*70 + "\n")
        return True


def main():
    # Check dependencies
    if not check_dependencies():
        print("[WARN] Please install dependencies first!\n")
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Start GUI
    print("[START] Starting Pipeline GUI...\n")
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
