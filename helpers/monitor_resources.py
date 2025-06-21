
import psutil
import threading
import time
import streamlit as st

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class SystemMonitor:
    def __init__(self):
        self.stats = {
            'cpu_percent': 0,
            'ram_percent': 0,
            'ram_used_gb': 0,
            'ram_total_gb': 0,
            'gpu_percent': 0,
            'gpu_memory_percent': 0,
            'gpu_memory_used': 0,
            'gpu_memory_total': 0,
            'gpu_name': "No GPU detected"
        }
        self.running = False
        self.thread = None
        
    def _monitor_loop(self):
        psutil.cpu_percent()
        
        while self.running:
            try:
                self.stats['cpu_percent'] = psutil.cpu_percent(interval=None)
                
                memory = psutil.virtual_memory()
                self.stats['ram_percent'] = memory.percent
                self.stats['ram_used_gb'] = memory.used / (1024**3)
                self.stats['ram_total_gb'] = memory.total / (1024**3)
                
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            self.stats['gpu_percent'] = gpu.load * 100
                            self.stats['gpu_memory_percent'] = gpu.memoryUtil * 100
                            self.stats['gpu_memory_used'] = gpu.memoryUsed
                            self.stats['gpu_memory_total'] = gpu.memoryTotal
                            self.stats['gpu_name'] = gpu.name
                    except:
                        pass
                        
            except Exception:
                pass
                
            time.sleep(2)
    
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def get_stats(self):
        return self.stats.copy()

def init_monitoring():
    if 'system_monitor' not in st.session_state:
        st.session_state.system_monitor = SystemMonitor()
        st.session_state.system_monitor.start()

def display_system_monitoring_lightweight():
    init_monitoring()
    
    stats = st.session_state.system_monitor.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💻 CPU", f"{stats['cpu_percent']:.0f}%")
    
    with col2:
        st.metric("🧠 RAM", f"{stats['ram_percent']:.0f}%", 
                 delta=f"{stats['ram_used_gb']:.1f}/{stats['ram_total_gb']:.1f} GB")
    
    with col3:
        st.metric("🎮 GPU", f"{stats['gpu_percent']:.0f}%")
    
    with col4:
        st.metric("📊 VRAM", f"{stats['gpu_memory_percent']:.0f}%",
                 delta=f"{stats['gpu_memory_used']:.0f}/{stats['gpu_memory_total']:.0f} MB")

def display_system_monitoring_detailed():
    init_monitoring()
    

    
    col0,col1, col2, col3, col4 = st.columns(5)
    with col0:
        if st.button("🔄 Refresh Stats (automatically during calculations)", type = 'tertiary'):
            pass
    
    stats = st.session_state.system_monitor.get_stats()

    with col1:
        st.metric("💻 CPU Usage", f"{stats['cpu_percent']:.1f}%")
        st.progress(stats['cpu_percent'] / 100)
    
    with col2:
        st.metric("🧠 RAM Usage", f"{stats['ram_percent']:.1f}%", 
                 help=f"{stats['ram_used_gb']:.1f} / {stats['ram_total_gb']:.1f} GB")
        st.progress(stats['ram_percent'] / 100)
    
    with col3:
        st.metric("🎮 GPU Usage", f"{stats['gpu_percent']:.1f}%",
                 help=f"GPU: {stats['gpu_name'][:30]}")
        st.progress(stats['gpu_percent'] / 100)
    
    with col4:
        st.metric("📊 GPU Memory", f"{stats['gpu_memory_percent']:.1f}%",
                 help=f"{stats['gpu_memory_used']:.0f} / {stats['gpu_memory_total']:.0f} MB")
        st.progress(stats['gpu_memory_percent'] / 100)
    
    #st.divider()

def display_system_monitoring_minimal():
    init_monitoring()
    
    stats = st.session_state.system_monitor.get_stats()
    
    st.caption(f"💻 CPU: {stats['cpu_percent']:.0f}% | 🧠 RAM: {stats['ram_percent']:.0f}% ({stats['ram_used_gb']:.1f}GB) | 🎮 GPU: {stats['gpu_percent']:.0f}% | 📊 VRAM: {stats['gpu_memory_percent']:.0f}%")
