import * as FileSystem from 'expo-file-system';

// Server configuration
const API_SERVER = 'http://172.20.10.4:8000'; // Use this for local server
// const API_SERVER = 'http://10.0.2.2:8000'; // Use this for Android emulator
// const API_SERVER = 'http://localhost:8000'; // Use this for iOS simulator

/**
 * Service for interacting with the Python inference API
 */
export default class PalmDetectionService {
  /**
   * Sends an image to the TensorFlow Lite model server for inference
   * @param {string} imageUri - The URI of the image to process
   * @returns {Promise<Object>} The detection results
   */
  static async detectPalmFruit(imageUri) {
    try {
      console.log('Sending image to inference server:', imageUri);
      
      // Check for valid image URI
      if (!imageUri) {
        throw new Error('Invalid image URI');
      }
      
      // Create form data for file upload
      const formData = new FormData();
      const fileInfo = await FileSystem.getInfoAsync(imageUri);
      
      if (!fileInfo.exists) {
        throw new Error('Image file does not exist');
      }
      
      // Get file extension from URI
      const fileExtension = imageUri.split('.').pop();
      const mimeType = fileExtension === 'jpg' || fileExtension === 'jpeg' 
        ? 'image/jpeg' 
        : fileExtension === 'png' ? 'image/png' : 'application/octet-stream';
      
      // Log file info for debugging
      console.log(`Image type: ${mimeType}, size: ${fileInfo.size} bytes`);
      
      formData.append('file', {
        uri: imageUri,
        type: mimeType,
        name: `palm_image.${fileExtension}`,
      });
      
      // Send request to inference server
      const response = await fetch(`${API_SERVER}/detect`, {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json',
        },
      });
      
      // Check for successful response
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`API Error (${response.status}): ${errorData.message || 'Unknown error'}`);
      }
      
      // Parse response data
      const resultData = await response.json();
      console.log('Detection results:', resultData);
      
      return resultData;
    } catch (error) {
      console.error('Error in palm detection service:', error);
      throw error;
    }
  }
  
  /**
   * Checks if the inference server is reachable and ready
   * @returns {Promise<boolean>} True if server is healthy and model is loaded
   */
  static async checkServerHealth() {
    try {
      const response = await fetch(`${API_SERVER}/health`, {
        method: 'GET',
      });
      
      if (!response.ok) {
        return false;
      }
      
      const healthData = await response.json();
      return healthData.status === 'healthy' && healthData.model_loaded;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }
  
  /**
   * Sends a base64-encoded image for inference
   * @param {string} base64Image - Base64 encoded image data
   * @returns {Promise<Object>} The detection results
   */
  static async detectWithBase64(base64Image) {
    try {
      console.log('Sending base64 image to inference server...');
      console.log(`Base64 image length: ${base64Image.length} characters`);
      
      // Remove any data URL prefix if present
      let imageData = base64Image;
      if (base64Image.startsWith('data:image')) {
        imageData = base64Image.split(',')[1];
      }
      
      const response = await fetch(`${API_SERVER}/detect-base64`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`API Error (${response.status}): ${errorData.message || 'Unknown error'}`);
      }
      
      const results = await response.json();
      
      // Add detailed logging for debugging
      console.log('Base64 detection successful:', results.success);
      console.log('Number of detections:', results.detections ? results.detections.length : 0);
      
      if (results.detections && results.detections.length > 0) {
        console.log('Primary detection class:', results.detections[0].class);
        console.log('Primary detection confidence:', results.detections[0].confidence);
        console.log('Primary detection bounding box:', results.detections[0].boundingBox);
      } else {
        console.log('No detections found in the image');
      }
      
      return results;
    } catch (error) {
      console.error('Error in base64 detection:', error);
      throw error;
    }
  }
  
  /**
   * Try to detect using both methods (for reliability)
   * @param {string} imageUri - URI of the image to process
   * @param {string} base64Image - Base64 encoded image data (optional)
   * @returns {Promise<Object>} The detection results
   */
  static async detectWithFallback(imageUri, base64Image = null) {
    try {
      // First try with URI for better quality
      const results = await this.detectPalmFruit(imageUri);
      return results;
    } catch (error) {
      console.log('URI detection failed, trying with base64');
      
      // Fallback to base64 if URI method fails
      if (base64Image) {
        return await this.detectWithBase64(base64Image);
      } else {
        // Convert URI to base64 if needed
        try {
          const base64 = await FileSystem.readAsStringAsync(imageUri, {
            encoding: FileSystem.EncodingType.Base64,
          });
          return await this.detectWithBase64(base64);
        } catch (e) {
          console.error('Failed to convert URI to base64:', e);
          throw error; // Throw original error if conversion fails
        }
      }
    }
  }
}