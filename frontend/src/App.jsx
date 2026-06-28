import { useEffect, useState, useRef } from 'react';
import { createClient } from '@supabase/supabase-js';
import './index.css';

// Initialize Supabase Client
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY;
const supabase = (supabaseUrl && supabaseKey) ? createClient(supabaseUrl, supabaseKey) : null;

export default function App() {
  // Telemetry View State
  const [data, setData] = useState({
    current: { grade: "Grade 0", confidence: 100, timestamp: "--:--:--", engine: "Scanning..." },
    logs: []
  });
  const [error, setError] = useState(false);

  // App Configuration States
  const [activeTab, setActiveTab] = useState('telemetry'); // 'telemetry' or 'assessment'
  const [isAcademicOpen, setIsAcademicOpen] = useState(false);

  // Mobile Assessment Hub States
  const [assessmentMode, setAssessmentMode] = useState('upload'); // 'upload' or 'camera'
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadPreview, setUploadPreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [assessmentResult, setAssessmentResult] = useState(null);

  // Webcam stream states
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);
  const [frameIntervalMs, setFrameIntervalMs] = useState(800); // 800ms between frames
  const [cameraError, setCameraError] = useState(null);

  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  // --- Telemetry Effect (Supabase Cloud Mode or Local API Fallback) ---
  useEffect(() => {
    if (supabase) {
      console.log("☁️ Supabase Cloud Mode Active");
      
      const fetchInitial = async () => {
        const { data: logs, error } = await supabase
          .from('detections')
          .select('*')
          .order('created_at', { ascending: false })
          .limit(10);
        
        if (!error && logs && logs.length > 0) {
          setData({
            current: {
              grade: logs[0].grade,
              confidence: logs[0].confidence,
              timestamp: new Date(logs[0].created_at).toLocaleTimeString(),
              engine: logs[0].engine
            },
            logs: logs.map(l => ({
              grade: l.grade,
              confidence: l.confidence,
              timestamp: new Date(l.created_at).toLocaleTimeString(),
              engine: l.engine
            }))
          });
        }
      };

      fetchInitial();

      // Subscribe to real-time changes
      const channel = supabase
        .channel('schema-db-changes')
        .on('postgres_changes', 
          { event: 'INSERT', schema: 'public', table: 'detections' }, 
          (payload) => {
            const newLog = {
              grade: payload.new.grade,
              confidence: payload.new.confidence,
              timestamp: new Date(payload.new.created_at).toLocaleTimeString(),
              engine: payload.new.engine
            };
            setData(prev => ({
              current: newLog,
              logs: [newLog, ...prev.logs].slice(0, 10)
            }));
          }
        )
        .subscribe();

      return () => { supabase.removeChannel(channel); };
    } 
    
    // Local API Fallback Mode
    else {
      const fetchStatus = async () => {
        try {
          const response = await fetch("/api/live");
          if (!response.ok) throw new Error("API Offline");
          const json = await response.json();
          setData(json);
          setError(false);
        } catch (err) {
          setError(true);
        }
      };
      const interval = setInterval(fetchStatus, 800);
      return () => clearInterval(interval);
    }
  }, []);

  // --- Browser Camera Frame Streaming Inference Effect ---
  useEffect(() => {
    let active = true;
    let timerId = null;

    const captureFrame = async () => {
      if (!isCameraActive || !videoRef.current || !active) return;
      
      const video = videoRef.current;
      if (video.readyState === video.HAVE_ENOUGH_DATA) {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const imgB64 = canvas.toDataURL('image/jpeg', 0.7);
        
        try {
          const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_b64: imgB64 })
          });
          const result = await response.json();
          if (active && result.success) {
            setAssessmentResult(result);
          }
        } catch (err) {
          console.error("Frame inference streaming error:", err);
        }
      }
      
      if (active && isCameraActive) {
        timerId = setTimeout(captureFrame, frameIntervalMs);
      }
    };

    if (isCameraActive) {
      timerId = setTimeout(captureFrame, frameIntervalMs);
    }

    return () => {
      active = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [isCameraActive, frameIntervalMs]);

  // --- Helper Methods ---
  const getSeverityClass = (grade) => {
    if (!grade) return "severity-0";
    if (grade.includes("3")) return "severity-3";
    if (grade.includes("2")) return "severity-2";
    if (grade.includes("1")) return "severity-1";
    return "severity-0";
  };

  const getBadgeClass = (grade) => {
    if (!grade) return "badge-0";
    if (grade.includes("3")) return "badge-3";
    if (grade.includes("2")) return "badge-2";
    if (grade.includes("1")) return "badge-1";
    return "badge-0";
  };

  // Trigger file selection dialog
  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Handle uploaded image file
  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadFile(file);
    setUploadPreview(URL.createObjectURL(file));
    setAssessmentResult(null);
    setIsAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error("API analysis failed");
      const result = await response.json();
      
      if (result.success) {
        setAssessmentResult(result);
      }
    } catch (err) {
      console.error("Assessment Upload Error:", err);
      alert("Error running inference on image. Verify backend API server is online.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Reset uploaded file and state
  const resetUpload = () => {
    setUploadFile(null);
    setUploadPreview(null);
    setAssessmentResult(null);
  };

  // Start device webcam
  const startCamera = async () => {
    setCameraError(null);
    setAssessmentResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }
      });
      setCameraStream(stream);
      setIsCameraActive(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err) {
      console.error("Webcam Access Error:", err);
      setCameraError("Camera access denied or device not found. Ensure permissions are granted.");
    }
  };

  // Stop device webcam
  const stopCamera = () => {
    setIsCameraActive(false);
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setAssessmentResult(null);
  };

  // Toggle active assessment sub-mode and stop camera if active
  const handleAssessmentModeToggle = (mode) => {
    setAssessmentMode(mode);
    if (isCameraActive) {
      stopCamera();
    }
  };

  return (
    <div className="app-container">
      {/* 1. Header Section */}
      <header>
        <div className="logo-section">
          <h1>CORROSION SENSE AI</h1>
          <p>Cascaded YOLO Deep Learning Corrosion Monitoring & Severity Grading</p>
        </div>

        <div className="system-stats">
          <div className="stat-item">
            <span className="stat-label">Network Mode</span>
            <span className="stat-value">
              {supabase ? 'CLOUD SYNC' : 'LOCAL MESH'}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Inference Model</span>
            <span className="stat-value">YOLOv12 + YOLOv8</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Inference Engine</span>
            <span className="stat-value">
              {activeTab === 'telemetry' ? data.current.engine : (isCameraActive ? 'Real-Time WebCam' : 'Static Upload')}
            </span>
          </div>
        </div>
      </header>

      {/* 2. Academic Origination Panel (Drawer) */}
      <div className={`academic-drawer ${isAcademicOpen ? 'open' : ''}`}>
        <div className="academic-header" onClick={() => setIsAcademicOpen(!isAcademicOpen)}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 10v6M2 10l10-5 10 5-10 5z"></path>
              <path d="M6 12v5c0 2 2 3 6 3s6-1 6-3v-5"></path>
            </svg>
            PROJECT ORIGINATION & UNIVERSITY METADATA
          </span>
          <svg 
            className="academic-toggle-icon" 
            width="16" height="16" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="2.5"
            style={{ transform: isAcademicOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.3s ease' }}
          >
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </div>
        
        {isAcademicOpen && (
          <div className="academic-content">
            <div className="academic-info-block">
              <span>Researcher / Author</span>
              <p>Adeniji Yusuf Joseph (Matric: 20201777)</p>
            </div>
            <div className="academic-info-block">
              <span>Supervisor</span>
              <p>Prof. O. R. Adetunji</p>
            </div>
            <div className="academic-info-block">
              <span>Department & Institution</span>
              <p>Mechanical Engineering, Federal University of Agriculture, Abeokuta (FUNAAB)</p>
            </div>
            <div className="academic-info-block">
              <span>Calibration Standard</span>
              <p>ASTM D610 Standards for Steel Rust & Corrosion Area Percentage</p>
            </div>
          </div>
        )}
      </div>

      {/* 3. Primary Mode Navigation Tabs */}
      <div className="tabs-container">
        <button 
          className={`tab-btn ${activeTab === 'telemetry' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('telemetry');
            if (isCameraActive) stopCamera();
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
            <line x1="8" y1="21" x2="16" y2="21"></line>
            <line x1="12" y1="17" x2="12" y2="21"></line>
          </svg>
          Live CCTV Telemetry
        </button>
        <button 
          className={`tab-btn ${activeTab === 'assessment' ? 'active' : ''}`}
          onClick={() => setActiveTab('assessment')}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
            <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
            <line x1="12" y1="22.08" x2="12" y2="12"></line>
          </svg>
          Mobile Assessment Hub
        </button>
      </div>

      {/* 4. Active Tab Panels */}
      <main>
        {activeTab === 'telemetry' ? (
          /* CCTV Telemetry Grid Panel */
          <div className="dashboard-grid">
            {/* Live Monitoring Card */}
            <div className="panel">
              <h2 className="panel-title">Real-Time CCTV Telemetry</h2>
              
              {error ? (
                <div style={{ textAlign: 'center', padding: '3rem 0', color: 'var(--grade-3)' }}>
                  <p style={{ fontWeight: '700' }}>Local Telemetry Hub Offline</p>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '0.5rem' }}>
                    Ensure the Flask server (`src/api_server.py`) is running or Supabase is connected.
                  </p>
                </div>
              ) : (
                <>
                  <div className={`severity-card ${getSeverityClass(data.current.grade)}`}>
                    <span className="grade-text">{data.current.grade}</span>
                    <span className="conf-text">Confidence: {data.current.confidence}%</span>
                  </div>
                  <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'space-between', color: 'var(--text-dim)', fontSize: '0.8rem', fontWeight: '500' }}>
                    <span>Last Sync: {data.current.timestamp}</span>
                    <span>Source Feed: CCTV_RTSP_STREAM_01</span>
                  </div>
                </>
              )}
            </div>

            {/* Log History Panel */}
            <div className="panel">
              <h2 className="panel-title">Detection History</h2>
              <div className="log-table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Severity Grade</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.logs.length > 0 ? (
                      data.logs.map((log, i) => (
                        <tr key={i}>
                          <td>{log.timestamp}</td>
                          <td>
                            <span className={`badge ${getBadgeClass(log.grade)}`}>
                              {log.grade}
                            </span>
                          </td>
                          <td>{log.confidence}%</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="3" style={{ textAlign: 'center', padding: '5rem 0', color: 'var(--text-dim)' }}>
                          Waiting for live CCTV telemetry updates...
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : (
          /* Mobile Assessment Hub Panel */
          <div className="dashboard-grid">
            
            {/* Control & Input Side Panel */}
            <div className="panel">
              <div style={{ display: 'flex', gap: '0.8rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--panel-border)', paddingBottom: '1rem' }}>
                <button 
                  className={`btn-secondary ${assessmentMode === 'upload' ? 'active' : ''}`}
                  style={{ flex: 1, background: assessmentMode === 'upload' ? '#eae3db' : 'transparent', border: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
                  onClick={() => handleAssessmentModeToggle('upload')}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
                  </svg>
                  Image Upload
                </button>
                <button 
                  className={`btn-secondary ${assessmentMode === 'camera' ? 'active' : ''}`}
                  style={{ flex: 1, background: assessmentMode === 'camera' ? '#eae3db' : 'transparent', border: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
                  onClick={() => handleAssessmentModeToggle('camera')}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                    <circle cx="12" cy="13" r="4"></circle>
                  </svg>
                  Camera Feed
                </button>
              </div>

              {assessmentMode === 'upload' ? (
                /* Mode A: Image upload controls */
                <div className="assessment-container">
                  <h3 style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--text-main)' }}>DIRECT FILE CLASSIFIER</h3>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-dim)', marginBottom: '0.5rem' }}>
                    Select an image from your gallery or snap a photo using your smartphone's camera to run instantaneous multi-stage YOLO severity diagnostics.
                  </p>
                  
                  {!uploadPreview ? (
                    <div className="upload-zone" onClick={handleUploadClick}>
                      <div className="upload-icon" style={{ marginBottom: '0.2rem', display: 'flex', justifyContent: 'center' }}>
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                          <polyline points="17 8 12 3 7 8"></polyline>
                          <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                      </div>
                      <p>Tap to Choose Image / Camera</p>
                      <span>Supports PNG, JPG, JPEG</span>
                      <input 
                        ref={fileInputRef}
                        type="file" 
                        accept="image/*"
                        className="hidden-file-input" 
                        onChange={handleFileChange}
                      />
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      <div className="camera-preview-container">
                        <img src={uploadPreview} alt="Upload Preview" />
                        {isAnalyzing && (
                          <div className="loading-overlay">
                            <div className="spinner"></div>
                            <span>Analyzing Steel Surface...</span>
                          </div>
                        )}
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                        <button 
                          className="btn-secondary reset-btn" 
                          style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.6rem' }} 
                          onClick={resetUpload} 
                          disabled={isAnalyzing}
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M23 4v6h-6"></path>
                            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                          </svg>
                          Reset Image
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                /* Mode B: Camera stream controls */
                <div className="camera-wrapper">
                  <h3 style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--text-main)' }}>LIVE IN-BROWSER CAMERA INFERENCE</h3>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-dim)' }}>
                    Actively streams live frames from your smartphone camera to the Flask API. Perfect for walking around physical structures and grading corrosion on-the-go.
                  </p>

                  <div className="camera-preview-container">
                    {/* HTML Video tag for browser stream */}
                    <video 
                      ref={videoRef} 
                      playsInline 
                      muted 
                      style={{ display: isCameraActive ? 'block' : 'none' }}
                    />
                    
                    {/* Overlay showing annotated output if available */}
                    {isCameraActive && assessmentResult?.annotated_image && (
                      <img src={assessmentResult.annotated_image} alt="Live Bounding Box Inference" />
                    )}

                    {!isCameraActive && (
                      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-dim)', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <div style={{ marginBottom: '0.8rem', color: 'var(--text-dim)' }}>
                          <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M23 7l-7 5 7 5V7z"></path>
                            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                          </svg>
                        </div>
                        <p style={{ fontWeight: '600', color: 'var(--text-main)' }}>Camera in Standby Mode</p>
                        <p style={{ fontSize: '0.8rem', marginTop: '0.2rem' }}>Click below to grant permission and start the stream.</p>
                      </div>
                    )}

                    {cameraError && (
                      <div className="loading-overlay" style={{ background: '#ffebee', color: '#c62828' }}>
                        <div style={{ color: '#c62828', marginBottom: '0.5rem' }}>
                          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="8" x2="12" y2="12"></line>
                            <line x1="12" y1="16" x2="12.01" y2="16"></line>
                          </svg>
                        </div>
                        <span style={{ fontSize: '0.85rem', textAlign: 'center', padding: '0 1rem', fontWeight: '600' }}>{cameraError}</span>
                      </div>
                    )}
                  </div>

                  <div className="camera-controls">
                    {!isCameraActive ? (
                      <button className="btn-primary" onClick={startCamera} style={{ width: '100%' }}>
                        Start Device Camera
                      </button>
                    ) : (
                      <>
                        <button className="btn-primary" style={{ background: 'var(--grade-3)', flex: 1 }} onClick={stopCamera}>
                          Stop Camera
                        </button>
                        <div className="speed-slider">
                          <span>FPS Interval:</span>
                          <input 
                            type="range" 
                            min="300" 
                            max="2000" 
                            step="100"
                            value={frameIntervalMs}
                            onChange={(e) => setFrameIntervalMs(parseInt(e.target.value))}
                          />
                          <span>{frameIntervalMs}ms</span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Results Side Panel */}
            <div className="panel">
              <h2 className="panel-title">Diagnostic Results</h2>
              
              {assessmentResult ? (
                <div className="result-image-panel">
                  {/* Summary Metric Cards */}
                  <div className="result-summary-row">
                    <div className="summary-metric-card">
                      <span>Overall Severity</span>
                      <p style={{ color: `var(--grade-${assessmentResult.max_grade.replace('Grade ', '')})` }}>
                        {assessmentResult.max_grade} ({assessmentResult.severity})
                      </p>
                    </div>
                    <div className="summary-metric-card">
                      <span>Rusted Patches</span>
                      <p>{assessmentResult.total_spots} Spots Found</p>
                    </div>
                  </div>

                  {/* Dynamic Severity Color Card */}
                  <div className={`severity-card ${getSeverityClass(assessmentResult.max_grade)}`} style={{ height: '140px' }}>
                    <span className="grade-text" style={{ fontSize: '2.5rem' }}>{assessmentResult.max_grade}</span>
                    <span className="conf-text" style={{ fontSize: '0.95rem' }}>
                      Status: {assessmentResult.total_spots > 0 ? `${assessmentResult.severity} Corrosion` : 'Clear Steel'}
                    </span>
                  </div>

                  {/* Render the static annotated output if not in live camera loop */}
                  {assessmentMode === 'upload' && assessmentResult?.annotated_image && (
                    <div>
                      <span style={{ fontSize: '0.72rem', textTransform: 'uppercase', color: 'var(--text-dim)', fontWeight: '700', display: 'block', marginBottom: '0.4rem' }}>
                        Graded Output Overlay
                      </span>
                      <div className="annotated-output-view">
                        <img src={assessmentResult.annotated_image} alt="Inference Bounding Boxes" />
                      </div>
                    </div>
                  )}

                  {/* List of individual detected rusted regions */}
                  {assessmentResult.spots.length > 0 && (
                    <div style={{ marginTop: '0.5rem' }}>
                      <span style={{ fontSize: '0.72rem', textTransform: 'uppercase', color: 'var(--text-dim)', fontWeight: '700' }}>
                        Localised Rust Spots (ASTM D610 Calibration)
                      </span>
                      <div className="spot-items-list">
                        {assessmentResult.spots.map((spot, i) => (
                          <div key={i} className="spot-item-card">
                            <div className="spot-grade-info">
                              <div className={`spot-dot spot-dot-${spot.grade.replace('Grade ', '')}`}></div>
                              <div>
                                <p style={{ fontWeight: '700' }}>{spot.grade} ({spot.severity})</p>
                                <p style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                                  Patch Bounding Box: [{spot.box.join(', ')}]
                                </p>
                              </div>
                            </div>
                            <span className="spot-conf">{spot.confidence}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '6rem 0', color: 'var(--text-dim)', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <div style={{ marginBottom: '1rem', color: 'var(--text-dim)' }}>
                    <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="9" r="6"></circle>
                      <path d="M12 15v6M9 18h6"></path>
                      <path d="M16.5 13.5L21 18"></path>
                    </svg>
                  </div>
                  <p style={{ fontWeight: '600', color: 'var(--text-main)' }}>Awaiting Diagnostic Input</p>
                  <p style={{ fontSize: '0.8rem', marginTop: '0.2rem', padding: '0 2rem' }}>
                    Upload a picture or initiate a camera feed to run localized YOLO severity annotations.
                  </p>
                </div>
              )}
            </div>

          </div>
        )}
      </main>

      {/* 5. Academic Footer */}
      <footer>
        <div className="footer-content">
          <div className="footer-col">
            <span className="footer-title">Academic Thesis Calibration</span>
            <p>Development of a Smart Corrosion Monitoring System for Steel Structures Using CCTV</p>
            <p>Calibrated in accordance with ASTM D610 Standards</p>
          </div>
          <div className="footer-col right">
            <span className="footer-title">Research Credit & Origination</span>
            <p>Author: Adeniji Yusuf Joseph (Matric: 20201777)</p>
            <p>Supervisor: Prof. O. R. Adetunji</p>
            <p className="footer-copyright">Dept of Mechanical Engineering, FUNAAB &copy; 2026</p>
          </div>
        </div>
      </footer>
    </div>
  );
}