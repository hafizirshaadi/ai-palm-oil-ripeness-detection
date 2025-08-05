import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
  Alert,
  Platform,
  Dimensions,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { initPalmDetection, detectPalmFruit } from '../../utils/palmDetection';
import { StatusBar } from 'expo-status-bar';
import { LABELS } from '../../utils/tensorflow';
import { generatePalmAnalysisPDF } from '../../utils/pdfService';
import { LinearGradient } from 'expo-linear-gradient';

export default function ScanScreen() {
  const router = useRouter();
  const [image, setImage] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [detections, setDetections] = useState([]);
  const [modelReady, setModelReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

  // Initialize the model
  useEffect(() => {
    const initModel = async () => {
      try {
        setLoading(true);
        const success = await initPalmDetection();
        setModelReady(success);
        if (!success) {
          setError('Could not initialize the palm detection model');
        }
      } catch (err) {
        console.error('Error initializing model:', err);
        setError('Error loading the detection model');
      } finally {
        setLoading(false);
      }
    };

    initModel();
  }, []);

  // Request permissions
  useEffect(() => {
    (async () => {
      // Request permission to access the camera and photo library
      if (Platform.OS !== 'web') {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== 'granted') {
          Alert.alert(
            'Permission Required',
            'Sorry, we need camera roll permissions to make this work!'
          );
        }
      }
    })();
  }, []);

  // Calculate image dimensions on screen
  const calculateImageDimensions = (imageUri) => {
    return new Promise((resolve) => {
      Image.getSize(imageUri, (width, height) => {
        // Get device screen dimensions
        const screenWidth = Dimensions.get('window').width - 40; // Adjusted for padding
        const screenHeight = 300; // Fixed height of image container
        
        // Calculate ratio to maintain aspect ratio within the container
        const widthRatio = screenWidth / width;
        const heightRatio = screenHeight / height;
        const ratio = Math.min(widthRatio, heightRatio);
        
        // Calculate scaled dimensions
        const scaledWidth = width * ratio;
        const scaledHeight = height * ratio;
        
        resolve({
          width: scaledWidth,
          height: scaledHeight,
          offsetX: (screenWidth - scaledWidth) / 2,
          offsetY: (screenHeight - scaledHeight) / 2,
          originalWidth: width,
          originalHeight: height,
          ratio: ratio
        });
      });
    });
  };

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });

      if (!result.canceled) {
        const imageUri = result.assets[0].uri;
        setImage(imageUri);
        setResults(null); // Reset results when a new image is selected
        setDetections([]);
        
        // Calculate and store image dimensions for accurate overlay
        const dimensions = await calculateImageDimensions(imageUri);
        setImageSize(dimensions);
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to pick image. Please try again.');
    }
  };

  const takePhoto = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      
      if (status !== 'granted') {
        Alert.alert('Permission Required', 'Camera permission is required to take photos.');
        return;
      }
      
      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [4, 3],
        quality: 1,
      });

      if (!result.canceled) {
        const imageUri = result.assets[0].uri;
        setImage(imageUri);
        setResults(null); // Reset results when a new image is taken
        setDetections([]);
        
        // Calculate and store image dimensions for accurate overlay
        const dimensions = await calculateImageDimensions(imageUri);
        setImageSize(dimensions);
      }
    } catch (error) {
      console.error('Error taking photo:', error);
      Alert.alert('Error', 'Failed to take photo. Please try again.');
    }
  };

  const analyzeImage = async () => {
    if (!image) {
      Alert.alert('No Image', 'Please select or take a photo first.');
      return;
    }

    if (!modelReady) {
      Alert.alert('Model Not Ready', 'Palm detection model is not yet initialized. Please wait or restart the app.');
      return;
    }

    setAnalyzing(true);

    try {
      // Use our TensorFlow model to detect palm fruit
      const detectionResults = await detectPalmFruit(image);
      
      if (detectionResults.detections.length > 0) {
        // Set primary result for the result card
        setResults(detectionResults.primaryResult);
        
        // Store all detections for drawing bounding boxes
        setDetections(detectionResults.detections);
      } else {
        Alert.alert('No Palm Fruit Detected', 'No palm fruit was detected in the image. Please try with another image.');
      }
    } catch (error) {
      console.error('Error analyzing image:', error);
      Alert.alert('Analysis Failed', 'Failed to analyze the image. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };

  // Function to navigate to real-time scan
  const goToRealTimeScan = () => {
    router.push('/real-time-scan');
  };

  // Function to generate and download PDF report
  const generateReport = async () => {
    try {
      if (!results || !image) {
        Alert.alert('Error', 'Analysis results or image not found.');
        return;
      }
      
      const success = await generatePalmAnalysisPDF(results, image, detections);
      
      if (success) {
        Alert.alert('Success', 'PDF report generated successfully.');
      } else {
        Alert.alert('Error', 'Failed to generate PDF report.');
      }
    } catch (error) {
      console.error('Error generating report:', error);
      Alert.alert('Error', 'An error occurred while generating the PDF report.');
    }
  };

  // Get color for a specific class
  const getColorForClass = (classId) => {
    const colors = {
      1: '#6c757d', // Empty Bunch - gray
      2: '#ffc107', // Underripe - yellow
      3: '#dc3545', // Abnormal - red
      4: '#28a745', // Ripe - green
      5: '#17a2b8', // Unripe - blue
      6: '#fd7e14'  // Overripe - orange
    };
    return colors[classId] || '#6c757d';
  };

  // Show loading state
  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <StatusBar style="auto" />
        <ActivityIndicator size="large" color="#2e8b57" />
        <Text style={styles.loadingText}>Loading palm detection model...</Text>
      </View>
    );
  }

  // Show error state
  if (error) {
    return (
      <View style={styles.errorContainer}>
        <StatusBar style="auto" />
        <Ionicons name="alert-circle-outline" size={60} color="#e74c3c" />
        <Text style={styles.errorTitle}>Error</Text>
        <Text style={styles.errorMessage}>{error}</Text>
        <TouchableOpacity 
          style={styles.retryButton}
          onPress={() => {
            setError(null);
            setLoading(true);
            initPalmDetection()
              .then(success => {
                setModelReady(success);
                if (!success) {
                  setError('Could not initialize the palm detection model');
                }
              })
              .catch(err => {
                setError('Error loading the detection model');
              })
              .finally(() => {
                setLoading(false);
              });
          }}
        >
          <Text style={styles.retryButtonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <StatusBar style="light" />
      
      {/* Updated Header with Gradient - Matching Palm Oil Price page styling */}
      <LinearGradient
        colors={['#2e8b57', '#1a5733']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
        style={styles.headerGradient}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="white" />
          </TouchableOpacity>
          <Text style={styles.title}>Scan Palm Fruit</Text>
          <View style={styles.placeholder} />
        </View>
      </LinearGradient>

      <View style={styles.content}>
        {!modelReady && (
          <View style={styles.modelWarning}>
            <Ionicons name="warning-outline" size={24} color="#e4a84c" />
            <Text style={styles.modelWarningText}>
              Detection model not ready. Some features may be limited.
            </Text>
          </View>
        )}

        <Text style={styles.instructionText}>
          Take a photo or select an image of palm fruit to analyze its ripeness and quality
        </Text>

        {/* Add Real-Time Scan Button */}
        <TouchableOpacity 
          style={[styles.realTimeButton, !modelReady && styles.disabledButton]} 
          onPress={goToRealTimeScan}
          disabled={!modelReady}
        >
          <Ionicons name="videocam" size={24} color="white" />
          <Text style={styles.buttonText}>Real-Time Scan</Text>
        </TouchableOpacity>

        <View style={styles.imageContainer}>
          {image ? (
            <View style={styles.imageWrapper}>
              <Image source={{ uri: image }} style={[styles.previewImage, {
                width: imageSize.width,
                height: imageSize.height,
                marginLeft: imageSize.offsetX,
                marginTop: imageSize.offsetY,
              }]} />
              
              {/* Detection overlays */}
              {detections.length > 0 && (
                <View style={[styles.detectionOverlay, {
                  width: imageSize.width,
                  height: imageSize.height,
                  marginLeft: imageSize.offsetX,
                  marginTop: imageSize.offsetY,
                }]}>
                  {detections.map((detection, index) => {
                    // Calculate position based on normalized coordinates and display size
                    const { normalized } = detection.boundingBox;
                    const boxX = normalized.xmin * imageSize.width;
                    const boxY = normalized.ymin * imageSize.height;
                    const boxWidth = (normalized.xmax - normalized.xmin) * imageSize.width;
                    const boxHeight = (normalized.ymax - normalized.ymin) * imageSize.height;
                    
                    const color = getColorForClass(detection.classId);
                    
                    return (
                      <View key={index}>
                        {/* Bounding box */}
                        <View style={[styles.boundingBox, {
                          left: boxX,
                          top: boxY,
                          width: boxWidth,
                          height: boxHeight,
                          borderColor: color,
                        }]} />
                        
                        {/* Label */}
                        <View style={[styles.detectionLabel, {
                          left: boxX,
                          top: boxY - 20,
                          backgroundColor: color,
                        }]}>
                          <Text style={styles.detectionLabelText}>
                            {detection.class}: {Math.round(detection.confidence * 100)}%
                          </Text>
                        </View>
                      </View>
                    );
                  })}
                </View>
              )}
            </View>
          ) : (
            <View style={styles.placeholderImage}>
              <Ionicons name="image-outline" size={80} color="#ccc" />
            </View>
          )}
        </View>

        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={takePhoto}>
            <Ionicons name="camera" size={24} color="white" />
            <Text style={styles.buttonText}>Take Photo</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.button} onPress={pickImage}>
            <Ionicons name="images" size={24} color="white" />
            <Text style={styles.buttonText}>Gallery</Text>
          </TouchableOpacity>
        </View>

        {image && (
          <TouchableOpacity 
            style={[styles.analyzeButton, (analyzing || !modelReady) && styles.disabledButton]} 
            onPress={analyzeImage}
            disabled={analyzing || !modelReady}
          >
            {analyzing ? (
              <ActivityIndicator color="white" />
            ) : (
              <>
                <Ionicons name="scan" size={24} color="white" />
                <Text style={styles.buttonText}>Analyze</Text>
              </>
            )}
          </TouchableOpacity>
        )}

        {results && (
          <View style={styles.resultsContainer}>
            <Text style={styles.resultsTitle}>Analysis Results</Text>
            <View style={styles.resultCard}>
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Classification:</Text>
                <Text style={styles.resultValue}>{results.class}</Text>
              </View>
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Confidence:</Text>
                <Text style={styles.resultValue}>{(results.confidence * 100).toFixed(1)}%</Text>
              </View>
              <View style={styles.confidenceBar}>
                <View 
                  style={[
                    styles.confidenceFill, 
                    { width: `${results.confidence * 100}%` },
                    results.confidence > 0.8 ? styles.highConfidence : 
                    results.confidence > 0.5 ? styles.mediumConfidence : 
                    styles.lowConfidence
                  ]} 
                />
              </View>
              
              <Text style={styles.recommendationText}>
                {results.class === 'Ripe' 
                  ? 'This palm fruit is at optimal ripeness for harvesting.' 
                  : results.class === 'Underripe' || results.class === 'Unripe'
                  ? 'This palm fruit is not yet at optimal ripeness.' 
                  : results.class === 'Overripe'
                  ? 'This palm fruit has passed optimal ripeness.'
                  : results.class === 'Abnormal'
                  ? 'This palm fruit shows abnormal characteristics.'
                  : results.class === 'Empty Bunch'
                  ? 'This appears to be an empty fruit bunch.'
                  : 'Assessment unclear. Please try another image.'}
              </Text>
              
              {/* PDF Download Button */}
              <TouchableOpacity 
                style={styles.pdfButton}
                onPress={() => generateReport()}
              >
                <Ionicons name="document-text" size={20} color="white" />
                <Text style={styles.pdfButtonText}>Download PDF Report</Text>
              </TouchableOpacity>
            </View>
            
            {detections.length > 1 && (
              <View style={styles.additionalDetections}>
                <Text style={styles.additionalDetectionsTitle}>
                  Additional Detections ({detections.length - 1})
                </Text>
                {detections.slice(1).map((detection, index) => (
                  <View key={index} style={styles.miniDetection}>
                    <View style={[styles.miniLabel, { backgroundColor: getColorForClass(detection.classId) }]}>
                      <Text style={styles.miniLabelText}>{detection.class}</Text>
                    </View>
                    <Text style={styles.miniConfidence}>
                      {(detection.confidence * 100).toFixed(1)}% confidence
                    </Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f7f9fc',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f7f9fc',
    padding: 20,
  },
  loadingText: {
    marginTop: 20,
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f7f9fc',
    padding: 20,
  },
  errorTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#e74c3c',
    marginVertical: 10,
  },
  errorMessage: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 20,
  },
  retryButton: {
    backgroundColor: '#2e8b57',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 10,
  },
  retryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  headerGradient: {
    paddingTop: 60,
    paddingBottom: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'white',
  },
  backButton: {
    padding: 5,
  },
  placeholder: {
    width: 34,
    height: 34,
  },
  content: {
    padding: 20,
  },
  modelWarning: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff3cd',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#ffeeba',
  },
  modelWarningText: {
    color: '#856404',
    marginLeft: 10,
    fontSize: 14,
    flex: 1,
  },
  instructionText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 20,
  },
  imageContainer: {
    width: '100%',
    height: 300,
    marginBottom: 20,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#f0f0f0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  imageWrapper: {
    position: 'relative',
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  previewImage: {
    resizeMode: 'contain',
    position: 'absolute',
  },
  detectionOverlay: {
    position: 'absolute',
    zIndex: 10,
  },
  boundingBox: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 2,
  },
  detectionLabel: {
    position: 'absolute',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
    zIndex: 20,
  },
  detectionLabelText: {
    color: 'white',
    fontSize: 10,
    fontWeight: 'bold',
  },
  placeholderImage: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 10,
    marginBottom: 20,
  },
  button: {
    flexDirection: 'row',
    backgroundColor: '#2e8b57',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    flex: 1,
  },
  buttonText: {
    color: 'white',
    fontWeight: '600',
    fontSize: 16,
  },
  analyzeButton: {
    flexDirection: 'row',
    backgroundColor: '#e4a84c',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  disabledButton: {
    opacity: 0.7,
  },
  realTimeButton: {
    flexDirection: 'row',
    backgroundColor: '#3498db',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginBottom: 20,
  },
  resultsContainer: {
    marginTop: 30,
    marginBottom: 30,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  resultCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  resultLabel: {
    fontSize: 16,
    color: '#666',
  },
  resultValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  confidenceBar: {
    height: 10,
    backgroundColor: '#f0f0f0',
    borderRadius: 5,
    marginBottom: 20,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 5,
  },
  highConfidence: {
    backgroundColor: '#2e8b57',
  },
  mediumConfidence: {
    backgroundColor: '#e4a84c',
  },
  lowConfidence: {
    backgroundColor: '#e74c3c',
  },
  recommendationText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  pdfButton: {
    flexDirection: 'row',
    backgroundColor: '#4a6fa5',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 20,
    gap: 8,
  },
  pdfButtonText: {
    color: 'white',
    fontWeight: '600',
    fontSize: 14,
  },
  additionalDetections: {
    marginTop: 15,
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  additionalDetectionsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  miniDetection: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  miniLabel: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 4,
    marginRight: 10,
  },
  miniLabelText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  miniConfidence: {
    fontSize: 14,
    color: '#666',
  },
});