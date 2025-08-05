import * as Print from 'expo-print';
import { shareAsync } from 'expo-sharing';
import * as FileSystem from 'expo-file-system';
import { formatDate } from './helpers';

/**
 * Generates a PDF report for palm fruit analysis
 * @param {object} results - The analysis results
 * @param {string} imageUri - The URI of the analyzed image
 * @param {Array} detections - All detected objects in the image
 * @returns {Promise<boolean>} - Returns true if PDF generation was successful
 */
export const generatePalmAnalysisPDF = async (results, imageUri, detections) => {
  try {
    // Get the image as base64
    const base64Image = await fileToBase64(imageUri);
    
    // Current date and time for the report
    const reportDate = formatDate(new Date());
    
    // Generate recommendations based on the results
    const recommendation = getRecommendationText(results.class);
    
    // Generate detection table rows
    const detectionRows = detections.map((detection, index) => `
      <tr style="${index === 0 ? 'background-color: #f2f2f2;' : ''}">
        <td>${index + 1}</td>
        <td>${detection.class}</td>
        <td>${(detection.confidence * 100).toFixed(1)}%</td>
      </tr>
    `).join('');    // Build HTML overlays for bounding boxes over the image
    const boxColors = {
      1: '#6c757d', // Empty Bunch
      2: '#ffc107', // Underripe
      3: '#dc3545', // Abnormal
      4: '#28a745', // Ripe
      5: '#17a2b8', // Unripe
      6: '#fd7e14'  // Overripe
    };
    
    const overlayDivs = detections.map(d => {
      const { normalized } = d.boundingBox;
      const left = (normalized.xmin * 100).toFixed(2);
      const top = (normalized.ymin * 100).toFixed(2);
      const width = ((normalized.xmax - normalized.xmin) * 100).toFixed(2);
      const height = ((normalized.ymax - normalized.ymin) * 100).toFixed(2);
      // Get the correct color from the classId
      const color = boxColors[d.classId] || '#6c757d';
      return `
      <div style="position:absolute; left:${left}%; top:${top}%; width:${width}%; height:${height}%; border:2px solid ${color}; box-sizing:border-box;"></div>
      <div style="position:absolute; left:${left}%; top:${Math.max(0, parseFloat(top) - 5)}%; background-color:${color} !important; color:white; font-size:10px; padding:2px 4px; border-radius:2px; font-weight:bold; border:1px solid ${color}; z-index:999;">
        ${d.class}: ${Math.round(d.confidence * 100)}%
      </div>
      `;
    }).join('');
    
    // Create HTML content for the PDF with overlay boxes
    const htmlContent = `
      <!DOCTYPE html>
      <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0, user-scalable=no" />
          <style>
            body {
              font-family: 'Helvetica', sans-serif;
              padding: 20px;
              color: #333;
            }
            .header {
              text-align: center;
              margin-bottom: 30px;
            }
            .title {
              font-size: 24px;
              font-weight: bold;
              color: #2e8b57;
              margin-bottom: 5px;
            }
            .date {
              font-size: 14px;
              color: #666;
            }
            .image-container {
              text-align: center;
              margin-bottom: 30px;
              position: relative;
              display: flex;
              justify-content: center;
              align-items: center;
            }
            .analysis-image {
              max-width: 100%;
              max-height: 300px;
              border-radius: 8px;
              display: block;
              margin: 0 auto;
            }
            .results-section {
              margin-bottom: 30px;
            }
            .section-title {
              font-size: 18px;
              font-weight: bold;
              color: #2e8b57;
              margin-bottom: 15px;
              padding-bottom: 5px;
              border-bottom: 1px solid #eee;
            }
            .result-row {
              display: flex;
              justify-content: space-between;
              margin-bottom: 10px;
            }
            .result-label {
              font-weight: bold;
              color: #666;
            }
            .recommendation {
              background-color: #f9f9f9;
              border-left: 4px solid #2e8b57;
              padding: 15px;
              margin-top: 15px;
              font-style: italic;
              color: #555;
            }
            table {
              width: 100%;
              border-collapse: collapse;
              margin-top: 15px;
            }
            th, td {
              border: 1px solid #ddd;
              padding: 8px;
              text-align: left;
            }
            th {
              background-color: #2e8b57;
              color: white;
            }
            .footer {
              margin-top: 40px;
              text-align: center;
              font-size: 12px;
              color: #999;
            }
          </style>
        </head>
        <body>
          <div class="header">
            <div class="title">Palm Fruit Analysis Report</div>
            <div class="date">Generated on: ${reportDate}</div>
          </div>
            <div class="image-container">
            <div style="position:relative; display:inline-block;">
              <img class="analysis-image" src="data:image/jpeg;base64,${base64Image}" />
              ${overlayDivs}
            </div>
          </div>
          
          <div class="results-section">
            <div class="section-title">Primary Result</div>
            <div class="result-row">
              <span class="result-label">Classification:</span>
              <span>${results.class}</span>
            </div>
            <div class="result-row">
              <span class="result-label">Confidence:</span>
              <span>${(results.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="recommendation">
              ${recommendation}
            </div>
          </div>
          
          <div class="results-section">
            <div class="section-title">All Detections (${detections.length})</div>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Classification</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                ${detectionRows}
              </tbody>
            </table>
          </div>
          
          <div class="footer">
            &copy; 2025 Palm Fruit Analysis System
          </div>
        </body>
      </html>
    `;
    
    // Generate PDF file
    const { uri } = await Print.printToFileAsync({ html: htmlContent });
    
    // Get the filename from the results
    const filename = `palm-analysis-${results.class.toLowerCase()}-${Date.now()}.pdf`;
    
    // Save to a more permanent location if needed
    // const pdfDestination = FileSystem.documentDirectory + filename;
    // await FileSystem.moveAsync({
    //   from: uri,
    //   to: pdfDestination
    // });
    
    // Share the PDF file
    await shareAsync(uri, { UTI: '.pdf', mimeType: 'application/pdf' });
    
    return true;
  } catch (error) {
    console.error('Error generating PDF:', error);
    return false;
  }
};

/**
 * Converts a file to base64 string
 * @param {string} uri - File URI
 * @returns {Promise<string>} - Base64 string
 */
const fileToBase64 = async (uri) => {
  try {
    const base64 = await FileSystem.readAsStringAsync(uri, {
      encoding: FileSystem.EncodingType.Base64,
    });
    return base64;
  } catch (error) {
    console.error('Error converting file to base64:', error);
    throw error;
  }
};

/**
 * Get recommendation text based on fruit class
 * @param {string} fruitClass - The classification of the fruit
 * @returns {string} - Recommendation text
 */
const getRecommendationText = (fruitClass) => {
  switch (fruitClass) {
    case 'Ripe':
      return 'This palm fruit is at optimal ripeness for harvesting. It can be harvested immediately for the best quality and oil content.';
    case 'Underripe':
    case 'Unripe':
      return 'This palm fruit is not yet at optimal ripeness. It is recommended to wait before harvesting to maximize oil yield and quality.';
    case 'Overripe':
      return 'This palm fruit has passed optimal ripeness. It should be harvested as soon as possible to prevent further quality degradation.';
    case 'Abnormal':
      return 'This palm fruit shows abnormal characteristics. Further inspection is recommended to determine the cause.';
    case 'Empty Bunch':
      return 'This appears to be an empty fruit bunch. It does not contain viable fruit for processing.';
    default:
      return 'Assessment unclear. Please try another image or consult with an agricultural specialist.';
  }
};
