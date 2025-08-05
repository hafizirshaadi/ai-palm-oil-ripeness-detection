import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Dimensions,
  Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { Camera, CameraView } from 'expo-camera';
import * as FileSystem from 'expo-file-system';
import { StatusBar } from 'expo-status-bar';
import { detectPalmFruit } from '../../utils/palmDetection';

export default function RealTimeScanScreen() {
  const router = useRouter();
  const cameraRef = useRef(null);
  const scanTimerRef = useRef(null);
  const [hasPermission, setHasPermission] = useState(null);
  const [cameraType, setCameraType] = useState('back');
  const [detections, setDetections] = useState([]);
  const [continuousScanMode, setContinuousScanMode] = useState(false);
  const [processingFrame, setProcessingFrame] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [flashMode, setFlashMode] = useState('off');
  const [isMounted, setIsMounted] = useState(true);
  const [isCapturing, setIsCapturing] = useState(false);
  const errorCountRef = useRef(0);
  const [screenDimensions, setScreenDimensions] = useState({
    width: Dimensions.get('window').width,
    height: Dimensions.get('window').height,
  });
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  const [imageQuality, setImageQuality] = useState(0.8);
  const [lastCaptureTime, setLastCaptureTime] = useState(0);

  useEffect(() => {
    setIsMounted(true);
    return () => {
      setIsMounted(false);
      if (scanTimerRef.current) {
        clearTimeout(scanTimerRef.current);
        scanTimerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
      
      if (status !== 'granted') {
        Alert.alert(
          'Camera Permission Required',
          'We need access to your camera for real-time scanning to work.'
        );
      }
    })();

    const subscription = Dimensions.addEventListener('change', () => {
      setScreenDimensions({
        width: Dimensions.get('window').width,
        height: Dimensions.get('window').height,
      });
    });

    return () => {
      subscription.remove();
      setContinuousScanMode(false);
    };
  }, []);

  // Function to handle continuous scanning (real-time mode)
  const startContinuousScanning = () => {
    if (scanTimerRef.current) {
      clearTimeout(scanTimerRef.current);
    }
    
    if (!processingFrame && continuousScanMode && cameraReady && isMounted) {
      scanTimerRef.current = setTimeout(captureAndProcess, 50);
    }
  };

  // Function to capture and process a frame
  const captureAndProcess = async () => {
    if (!isMounted || isCapturing) {
      return;
    }

    try {
      setProcessingFrame(true);
      setIsCapturing(true);
      
      if (!cameraRef.current) {
        console.log('Camera ref not available');
        return;
      }
      
      const photo = await cameraRef.current.takePictureAsync({
        quality: imageQuality,
        base64: false,
        skipProcessing: false,
        exif: false,
      });
      
      setCapturedPhoto(photo.uri);
      setLastCaptureTime(Date.now());
      
      if (!isMounted) return;
      
      errorCountRef.current = 0;
      
      console.log('Processing camera frame...');
      
      const results = await detectPalmFruit(photo.uri);
      
      if (!isMounted) return;
      
      if (results && results.detections && results.detections.length > 0) {
        console.log(`Found ${results.detections.length} detections`);
        
        const updatedDetections = results.detections.map(detection => {
          console.log(`Detection: ${detection.class}, Confidence: ${detection.confidence}`);
          console.log(`Normalized box: ${JSON.stringify(detection.boundingBox.normalized)}`);
          
          return { ...detection };
        });
        
        setDetections(updatedDetections);
      } else {
        console.log('No detections found in the image');
        setDetections([]);
      }
      
    } catch (error) {
      console.error('Scan error:', error);
      errorCountRef.current += 1;
      
      if (errorCountRef.current > 3) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        errorCountRef.current = 0;
      }
    } finally {
      if (isMounted) {
        setProcessingFrame(false);
        setIsCapturing(false);
        
        // Only schedule another scan if in continuous mode
        if (continuousScanMode && isMounted) {
          const nextDelay = errorCountRef.current > 0 ? 1500 : 1000;
          scanTimerRef.current = setTimeout(captureAndProcess, nextDelay);
        }
      }
    }
  };

  // Single tap to scan functionality
  const takeSingleScan = async () => {
    // Don't allow too frequent taps
    const now = Date.now();
    if (now - lastCaptureTime < 1000 || isCapturing) {
      return;
    }
    
    await captureAndProcess();
  };

  // Toggle continuous scanning mode
  const toggleContinuousScan = () => {
    const newMode = !continuousScanMode;
    setContinuousScanMode(newMode);
    
    if (!newMode) {
      // Stop the scanning cycle
      if (scanTimerRef.current) {
        clearTimeout(scanTimerRef.current);
        scanTimerRef.current = null;
      }
    } else {
      // Start continuous scanning
      if (cameraReady && isMounted && !processingFrame && !isCapturing) {
        startContinuousScanning();
      }
    }
  };

  // Effect to start continuous scanning if enabled
  useEffect(() => {
    if (continuousScanMode && cameraReady && isMounted && !processingFrame && !isCapturing) {
      startContinuousScanning();
    }
    
    return () => {
      if (scanTimerRef.current) {
        clearTimeout(scanTimerRef.current);
        scanTimerRef.current = null;
      }
    };
  }, [continuousScanMode, cameraReady, isMounted, processingFrame, isCapturing]);

  const toggleCameraType = () => {
    if (isCapturing) return;
    
    if (scanTimerRef.current) {
      clearTimeout(scanTimerRef.current);
      scanTimerRef.current = null;
    }
    
    setCameraType(current => current === 'back' ? 'front' : 'back');
  };

  const toggleFlash = () => {
    if (isCapturing) return;
    
    setFlashMode(current => current === 'off' ? 'torch' : 'off');
  };

  const getColorForClass = (classId) => {
    const colors = {
      1: '#6c757d',
      2: '#ffc107',
      3: '#dc3545',
      4: '#28a745',
      5: '#17a2b8',
      6: '#fd7e14'
    };
    return colors[classId] || '#6c757d';
  };

  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#2e8b57" />
        <Text style={styles.statusText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Ionicons name="camera-off" size={60} color="#e74c3c" />
        <Text style={styles.statusText}>No access to camera</Text>
        <TouchableOpacity style={styles.button} onPress={() => router.back()}>
          <Text style={styles.buttonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar style="light" />
      
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing={cameraType}
        flash={flashMode}
        onCameraReady={() => {
          console.log('Camera is ready');
          setCameraReady(true);
        }}
        autoFocus={true}
        zoom={0}
      />
      
      {/* This is our transparent overlay to detect taps */}
      <TouchableOpacity 
        style={styles.cameraOverlay}
        activeOpacity={1}
        onPress={takeSingleScan}
        disabled={processingFrame || isCapturing}
      />
      
      <View style={styles.detectionOverlay} pointerEvents="none">
        {detections.length > 0 ? (
          detections.map((detection, index) => {
            const { normalized } = detection.boundingBox;
            
            const boxX = normalized.xmin * screenDimensions.width;
            const boxY = normalized.ymin * screenDimensions.height;
            const boxWidth = (normalized.xmax - normalized.xmin) * screenDimensions.width;
            const boxHeight = (normalized.ymax - normalized.ymin) * screenDimensions.height;
            
            const color = getColorForClass(detection.classId);
            
            return (
              <View key={index}>
                <View style={[styles.boundingBox, {
                  left: boxX,
                  top: boxY,
                  width: boxWidth,
                  height: boxHeight,
                  borderColor: color,
                }]} />
                
                <View style={[styles.detectionLabel, {
                  left: boxX,
                  top: boxY - 24,
                  backgroundColor: color,
                }]}>
                  <Text style={styles.detectionLabelText}>
                    {detection.class}: {Math.round(detection.confidence * 100)}%
                  </Text>
                </View>
              </View>
            );
          })
        ) : (
          <View style={styles.noDetectionPlaceholder}>
            {!processingFrame && (
              <Text style={styles.noDetectionText}>
                No palm fruits detected
              </Text>
            )}
          </View>
        )}
      </View>

      <View style={styles.topControls} pointerEvents="box-none">
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => {
            console.log('Back button pressed');
            router.back();
          }}
          activeOpacity={0.7}
          hitSlop={{top: 10, bottom: 10, left: 10, right: 10}}
        >
          <Ionicons name="arrow-back" size={28} color="white" />
        </TouchableOpacity>
        
        <Text style={styles.scanTitle}>
          Palm Fruit Scanner
        </Text>
        
        <TouchableOpacity 
          style={[styles.cameraTypeButton, isCapturing && styles.disabledButton]}
          onPress={toggleCameraType}
          disabled={isCapturing}
          activeOpacity={0.7}
          hitSlop={{top: 10, bottom: 10, left: 10, right: 10}}
        >
          <Ionicons name="camera-reverse" size={28} color="white" />
        </TouchableOpacity>
      </View>

      {processingFrame && (
        <View style={styles.processingIndicator} pointerEvents="none">
          <ActivityIndicator color="white" size="small" />
          <Text style={styles.processingText}>Processing</Text>
        </View>
      )}

      <View style={styles.bottomControls} pointerEvents="box-none">
        <TouchableOpacity 
          style={[styles.controlButton, isCapturing && styles.disabledButton]}
          onPress={toggleFlash}
          disabled={isCapturing}
          activeOpacity={0.7}
          hitSlop={{top: 10, bottom: 10, left: 10, right: 10}}
        >
          <Ionicons 
            name={flashMode === 'torch' ? "flash" : "flash-off"} 
            size={24} 
            color="white" 
          />
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.scanButton, !isCapturing && styles.scanButtonActive]}
          onPress={takeSingleScan}
          disabled={isCapturing || processingFrame}
          activeOpacity={0.7}
        >
          <Ionicons name="scan" size={30} color="white" />
          <Text style={styles.scanButtonText}>
            Tap to Scan
          </Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={[styles.controlButton, continuousScanMode && styles.continuousModeActive]}
          onPress={toggleContinuousScan}
          activeOpacity={0.7}
          disabled={isCapturing && !continuousScanMode}
        >
          <Ionicons name="refresh" size={24} color="white" />
        </TouchableOpacity>
      </View>

      <View style={styles.scanStatus} pointerEvents="none">
        <Text style={styles.scanStatusText}>
          {!cameraReady ? "Camera initializing..." : 
           continuousScanMode ? "Real-time mode active" : 
           "Tap to scan palm fruits"}
        </Text>
      </View>

      <View style={styles.hintContainer} pointerEvents="none">
        <Text style={styles.hintText}>
          Position palm fruit in frame and tap screen to scan
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  camera: {
    width: '100%',
    height: '100%',
    zIndex: 1,
  },
  cameraOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 2,
    backgroundColor: 'transparent',
  },
  statusText: {
    marginTop: 20,
    fontSize: 16,
    color: 'white',
    textAlign: 'center',
  },
  button: {
    backgroundColor: '#2e8b57',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 10,
    marginTop: 20,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  topControls: {
    position: 'absolute',
    top: 50,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    zIndex: 10,
  },
  backButton: {
    padding: 12,
    borderRadius: 20,
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  scanTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    textShadowColor: 'rgba(0,0,0,0.7)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 2,
  },
  cameraTypeButton: {
    padding: 12,
    borderRadius: 20,
    backgroundColor: 'rgba(0,0,0,0.5)',
  },
  bottomControls: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 20,
    zIndex: 10,
  },
  controlButton: {
    padding: 15,
    borderRadius: 30,
    backgroundColor: 'rgba(0,0,0,0.5)',
    width: 60,
    height: 60,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scanButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 30,
    backgroundColor: '#2e8b57',
    width: 160,
    height: 60,
    gap: 8,
  },
  scanButtonActive: {
    backgroundColor: '#2e8b57',
  },
  continuousModeActive: {
    backgroundColor: '#e4a84c',
  },
  scanButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  detectionOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 5,
  },
  boundingBox: {
    position: 'absolute',
    borderWidth: 3,
    borderRadius: 3,
  },
  detectionLabel: {
    position: 'absolute',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    zIndex: 20,
  },
  detectionLabelText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  processingIndicator: {
    position: 'absolute',
    top: 100,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    zIndex: 15,
  },
  processingText: {
    color: 'white',
    fontSize: 14,
  },
  scanStatus: {
    position: 'absolute',
    bottom: 120,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 8,
    zIndex: 10,
  },
  scanStatusText: {
    color: 'white',
    fontSize: 14,
  },
  disabledButton: {
    opacity: 0.5,
  },
  noDetectionPlaceholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  noDetectionText: {
    color: 'white',
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 10,
    borderRadius: 5,
    fontSize: 16,
  },
  hintContainer: {
    position: 'absolute',
    top: 150,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 8,
    zIndex: 10,
  },
  hintText: {
    color: 'white',
    fontSize: 14,
    textAlign: 'center',
  },
});
