#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Backend Server for Research Fetcher GUI
Place this file in the same directory as multisearchfinal.py and index.html
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import threading
import json
from pathlib import Path
from datetime import datetime
import time
import traceback
import os

# Import your research fetcher
try:
    from multisearchfinal import MultiTopicResearchFetcherWithClustering
    FETCHER_AVAILABLE = True
except ImportError:
    print("WARNING: multisearchfinal.py not found. Running in demo mode.")
    FETCHER_AVAILABLE = False

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent

app = Flask(__name__)
CORS(app)

# Store active jobs in memory
jobs = {}

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        index_file = BASE_DIR / 'index.html'
        if not index_file.exists():
            return jsonify({
                'error': 'index.html not found',
                'message': f'Please create index.html in: {BASE_DIR}',
                'current_directory': str(BASE_DIR),
                'files_present': [f.name for f in BASE_DIR.glob('*.html')]
            }), 404
        
        return send_file(index_file)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'fetcher_available': FETCHER_AVAILABLE,
        'base_directory': str(BASE_DIR),
        'index_html_exists': (BASE_DIR / 'index.html').exists()
    })

@app.route('/api/start', methods=['POST'])
def start_fetching():
    """Start a new research fetching job"""
    
    if not FETCHER_AVAILABLE:
        return jsonify({
            'error': 'multisearchfinal.py not found',
            'message': 'Please ensure multisearchfinal.py is in the same directory'
        }), 500
    
    try:
        data = request.json
        
        # Extract configuration from request
        topics = data.get('topics', [])
        sources_config = data.get('sources', {})
        clustering_config = data.get('clustering', {})
        linkedin_key = data.get('linkedInKey', '')
        
        # Validation
        if not topics:
            return jsonify({'error': 'No topics provided'}), 400
        
        enabled_sources = [k for k, v in sources_config.items() if v.get('enabled')]
        if not enabled_sources:
            return jsonify({'error': 'No sources enabled'}), 400
        
        # Convert UI config to fetcher config format
        config = {}
        for source, settings in sources_config.items():
            if settings.get('enabled'):
                config[source] = {
                    'enabled': True,
                    'limit': settings.get('limit', 1000)
                }
                
                # Add source-specific settings
                if source == 'arxiv':
                    config[source]['from_year'] = settings.get('fromYear', 2020)
                    config[source]['to_year'] = settings.get('toYear', 2025)
                elif source == 'pubmed':
                    config[source]['years'] = settings.get('years', 5)
                elif source == 'biorxiv':
                    config[source]['days'] = settings.get('days', 1825)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = BASE_DIR / f"research_output_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create job ID
        job_id = timestamp
        jobs[job_id] = {
            'status': 'running',
            'progress': {},
            'logs': [],
            'started': datetime.now().isoformat(),
            'topics': topics,
            'output_dir': str(output_dir),
            'config': config
        }
        
        # Add initial log
        jobs[job_id]['logs'].append({
            'type': 'info',
            'message': f'Starting research fetch for {len(topics)} topics',
            'timestamp': time.strftime('%H:%M:%S')
        })
        
        # Initialize fetcher
        fetcher = MultiTopicResearchFetcherWithClustering(
            str(output_dir),
            topics
        )
        
        # Set LinkedIn API key if provided
        if linkedin_key and linkedin_key.strip():
            try:
                fetcher.set_linkedin_api_key(linkedin_key)
                jobs[job_id]['logs'].append({
                    'type': 'info',
                    'message': 'LinkedIn API key configured',
                    'timestamp': time.strftime('%H:%M:%S')
                })
            except Exception as e:
                jobs[job_id]['logs'].append({
                    'type': 'warning',
                    'message': f'LinkedIn API key error: {str(e)}',
                    'timestamp': time.strftime('%H:%M:%S')
                })
        
        # Convert clustering config
        clustering_params = {
            'algorithm': clustering_config.get('algorithm', 'kmeans'),
            'n_clusters': None if clustering_config.get('autoClusters') else clustering_config.get('nClusters'),
            'visualize': clustering_config.get('visualize', True)
        }
        
        # Run fetcher in background thread
        def run_fetcher():
            try:
                jobs[job_id]['logs'].append({
                    'type': 'info',
                    'message': f'Enabled sources: {", ".join(enabled_sources)}',
                    'timestamp': time.strftime('%H:%M:%S')
                })
                
                jobs[job_id]['logs'].append({
                    'type': 'info',
                    'message': f'Clustering algorithm: {clustering_params["algorithm"].upper()}',
                    'timestamp': time.strftime('%H:%M:%S')
                })
                
                # Execute the fetcher
                fetcher.run_all(config, clustering_params)
                
                # Update job status on completion
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['completed'] = datetime.now().isoformat()
                jobs[job_id]['results'] = {
                    'summary': fetcher.results_summary,
                    'output_dir': str(output_dir)
                }
                
                jobs[job_id]['logs'].append({
                    'type': 'success',
                    'message': '🎉 Research fetching completed successfully!',
                    'timestamp': time.strftime('%H:%M:%S')
                })
                
                jobs[job_id]['logs'].append({
                    'type': 'success',
                    'message': f'Results saved to: {output_dir}',
                    'timestamp': time.strftime('%H:%M:%S')
                })
                
            except Exception as e:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = str(e)
                jobs[job_id]['logs'].append({
                    'type': 'error',
                    'message': f'Error: {str(e)}',
                    'timestamp': time.strftime('%H:%M:%S')
                })
                print(f"Error in fetcher thread: {traceback.format_exc()}")
        
        # Start background thread
        thread = threading.Thread(target=run_fetcher, daemon=True)
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'Research fetching started successfully'
        }), 200
        
    except Exception as e:
        print(f"Error starting job: {traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to start job',
            'message': str(e)
        }), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get status of a specific job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(jobs[job_id])

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    return jsonify({
        'jobs': list(jobs.keys()),
        'count': len(jobs)
    })

@app.route('/api/stop/<job_id>', methods=['POST'])
def stop_job(job_id):
    """Stop a running job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    if jobs[job_id]['status'] == 'running':
        jobs[job_id]['status'] = 'stopped'
        jobs[job_id]['logs'].append({
            'type': 'warning',
            'message': 'Job stopped by user',
            'timestamp': time.strftime('%H:%M:%S')
        })
    
    return jsonify({'message': 'Job stopped', 'job_id': job_id})

@app.route('/api/download/<job_id>/<filename>', methods=['GET'])
def download_file(job_id, filename):
    """Download a result file"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    output_dir = Path(jobs[job_id]['output_dir'])
    file_path = output_dir / filename
    
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True)

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found',
        'hint': 'Make sure index.html is in the same directory as app.py'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 RESEARCH FETCHER API SERVER")
    print("="*70)
    print(f"✅ Server Status: Running")
    print(f"🌐 URL: http://localhost:5000")
    print(f"📡 API Base: http://localhost:5000/api")
    print(f"📂 Working Directory: {BASE_DIR}")
    print(f"📊 Fetcher Available: {FETCHER_AVAILABLE}")
    
    # Check for index.html
    index_path = BASE_DIR / 'index.html'
    if index_path.exists():
        print(f"✅ index.html found")
    else:
        print(f"⚠️  WARNING: index.html NOT found!")
        print(f"   Please create index.html in: {BASE_DIR}")
    
    if not FETCHER_AVAILABLE:
        print("\n⚠️  WARNING: multisearchfinal.py not found!")
        print("   Please ensure multisearchfinal.py is in the same directory")
    
    print("\n📖 Available Endpoints:")
    print("   GET  /              - Web interface")
    print("   GET  /api/health    - Health check")
    print("   POST /api/start     - Start fetching")
    print("   GET  /api/status/<job_id> - Check job status")
    print("   GET  /api/jobs      - List all jobs")
    print("   POST /api/stop/<job_id>   - Stop a job")
    
    print("\n⌨️  Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    try:
        app.run(
            debug=True,
            port=5000,
            host='0.0.0.0',
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
