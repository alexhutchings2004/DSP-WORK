document.addEventListener('DOMContentLoaded', function() {
    const initializeBtn = document.getElementById('initialize-btn');
    const clearBtn = document.getElementById('clear-btn');
    const statusElement = document.getElementById('status');
    const selectedImage = document.getElementById('selected-image');
    const imageUrlElement = document.getElementById('image-url');
    const selectedImageContainer = document.getElementById('selected-image-container');
    const loaderElement = document.getElementById('loader');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analysisContainer = document.getElementById('analysis-container');
    const analysisLoader = document.getElementById('analysis-loader');
    const predictionResult = document.getElementById('prediction-result');
    const confidenceResult = document.getElementById('confidence-result');
    const modelResult = document.getElementById('model-result');
    const heatmapContainer = document.getElementById('heatmap-container');
    const heatmapImage = document.getElementById('heatmap-image');
    const openReactAppBtn = document.getElementById('open-react-app-btn');
    const facesContainer = document.getElementById('faces-container');
    const faceResults = document.getElementById('face-results');
    
    // Backend server URL
    const BACKEND_URL = 'http://localhost:5000';
    // React app URL
    const REACT_APP_URL = 'http://localhost:3000';
    
    // Store the last analysis data
    let lastAnalysisData = null;
    
    // Remove model selection element
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        modelSelect.style.display = 'none';
    }
    
    // Check if we're on a Twitter page
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      const url = tabs[0].url;
      if (!url.includes('twitter.com') && !url.includes('x.com')) {
        statusElement.textContent = 'Please navigate to Twitter to use this extension.';
        initializeBtn.disabled = true;
      }
    });
    
    // First, check if there's a stored image
    chrome.storage.local.get('selectedImage', function(data) {
      if (data.selectedImage) {
        // Display the stored image
        selectedImage.src = data.selectedImage.url;
        selectedImageContainer.style.display = 'flex';
        imageUrlElement.textContent = data.selectedImage.url;
        statusElement.textContent = 'Your selected image:';
        
        // If there's a saved analysis, show it
        chrome.storage.local.get('analysisResults', function(analysisData) {
          if (analysisData.analysisResults && 
              analysisData.analysisResults.imageUrl === data.selectedImage.url) {
            displayAnalysisResults(analysisData.analysisResults);
          }
        });
      }
    });
    
    // Initialize button click handler
    initializeBtn.addEventListener('click', function() {
      loaderElement.style.display = 'block';
      statusElement.textContent = 'Initializing...';
      
      chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        // Inject the content script to handle image selection
        chrome.scripting.executeScript({
          target: {tabId: tabs[0].id},
          function: injectImageSelector
        });
        
        // Tell the user what to do next
        statusElement.textContent = 'Now click on any image on Twitter...';
        
        // Close the popup (this is key - we close it AFTER sending instructions)
        window.close();
      });
    });
    
    // Open React App button handler
    openReactAppBtn.addEventListener('click', function() {
      if (selectedImage.src) {
        // Show loading indicator
        statusElement.textContent = 'Transferring image to React app...';
        
        // Convert the image to blob for sending to the backend
        fetch(selectedImage.src)
          .then(response => {
            if (!response.ok) {
              throw new Error(`Error fetching image: ${response.status} ${response.statusText}`);
            }
            return response.blob();
          })
          .then(blob => {
            // Create form data for the API request
            const formData = new FormData();
            formData.append('image', blob, 'extension_image.png');
            formData.append('source', 'chrome_extension');
            
            // Add timestamp to prevent caching issues with re-analysis
            formData.append('timestamp', new Date().toISOString());
            
            // If we have previous analysis, include it
            if (lastAnalysisData) {
              formData.append('analysis', JSON.stringify(lastAnalysisData));
            }
            
            // Send the image to the backend temporary storage endpoint
            return fetch(`${BACKEND_URL}/store_extension_image`, {
              method: 'POST',
              body: formData
            });
          })
          .then(response => {
            if (!response.ok) {
              throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            return response.json();
          })
          .then(data => {
            console.log("Image stored successfully:", data);
            // Get the temporary ID assigned by the backend
            const imageId = data.id;
            
            // Format the React app URL correctly (remove trailing slash if present)
            let reactUrl = REACT_APP_URL;
            if (reactUrl.endsWith('/')) {
              reactUrl = reactUrl.slice(0, -1);
            }
            
            // Open the React app in a new tab and pass the ID parameter
            const fullUrl = `${reactUrl}?source=extension&imageId=${imageId}`;
            console.log("Opening React app URL:", fullUrl);
            
            chrome.tabs.create({
              url: fullUrl
            });
          })
          .catch(error => {
            console.error('Error transferring image:', error);
            statusElement.textContent = 'Error transferring image to React app';
            
            // Even if there's an error, try to open the React app
            setTimeout(() => {
              chrome.tabs.create({
                url: `${REACT_APP_URL}?source=extension&error=transfer_failed`
              });
            }, 1000);
          });
      } else {
        // No image selected, just open the React app
        chrome.tabs.create({
          url: `${REACT_APP_URL}`
        });
      }
    });
    
    // Analyze button click handler
    analyzeBtn.addEventListener('click', function() {
      // Show the analysis container and loader
      analysisContainer.style.display = 'block';
      analysisLoader.style.display = 'block';
      facesContainer.style.display = 'none';
      
      // Clear previous results
      predictionResult.textContent = '';
      predictionResult.className = 'result-value';
      confidenceResult.textContent = '';
      modelResult.textContent = '';
      heatmapContainer.style.display = 'none';
      heatmapImage.src = '';
      faceResults.innerHTML = '';
      
      // Get the image URL
      const imageUrl = selectedImage.src;
      
      // Convert the image to a blob for sending to the backend
      fetch(imageUrl)
        .then(response => {
          if (!response.ok) {
            throw new Error(`Error fetching image: ${response.status} ${response.statusText}`);
          }
          return response.blob();
        })
        .then(blob => {
          // Create form data for the API request
          const formData = new FormData();
          formData.append('image', blob, 'twitter_image.png');
          formData.append('use_ensemble', 'true'); // Ensure ensemble analysis is used
          formData.append('extract_all_features', 'true'); // Request all available features
          
          // Send the image to the backend for analysis with comprehensive options
          return fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            body: formData
          });
        })
        .then(response => {
          if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
          }
          return response.json();
        })
        .then(data => {
          console.log("API Response:", data); // Log the response data
          
          // Hide the loader
          analysisLoader.style.display = 'none';
          
          // Save the analysis results with the image URL for future reference
          chrome.storage.local.set({
            'analysisResults': {
              ...data,
              imageUrl: imageUrl,
              model: 'EfficientNet'
            }
          });
          
          // Store the last analysis data
          lastAnalysisData = data;
          
          // Display the results
          displayAnalysisResults(data);
        })
        .catch(error => {
          console.error('Error analyzing image:', error);
          analysisLoader.style.display = 'none';
          predictionResult.textContent = 'Error: Could not analyze image';
          predictionResult.className = 'result-value error';
          
          // Add more descriptive error message
          confidenceResult.textContent = '';
          modelResult.textContent = error.message || 'Connection error';
          
          // Make the analysis container visible even with error
          analysisContainer.style.display = 'block';
        });
    });
    
    // Clear button click handler
    clearBtn.addEventListener('click', function() {
      // Add a subtle animation
      selectedImageContainer.style.opacity = '0.5';
      
      chrome.storage.local.remove(['selectedImage', 'analysisResults'], function() {
        setTimeout(() => {
          selectedImage.src = '';
          selectedImageContainer.style.display = 'none';
          selectedImageContainer.style.opacity = '1';
          imageUrlElement.textContent = '';
          statusElement.textContent = 'Image cleared. Click the button to select a new image.';
          
          // Hide analysis container
          analysisContainer.style.display = 'none';
          
          // Show success notification
          const notification = document.createElement('div');
          notification.className = 'success-notification';
          notification.textContent = 'Image successfully removed';
          document.querySelector('.content').appendChild(notification);
          
          // Remove notification after 3 seconds
          setTimeout(() => {
            notification.remove();
          }, 3000);
        }, 300);
      });
    });
    
    // Function to display analysis results
    function displayAnalysisResults(data) {
      // Always show EfficientNet as the model
      modelResult.textContent = 'EfficientNet';
      
      // Check if there are faces detected in the image
      if (data.face_regions && data.face_regions.length > 0) {
        // Multiple face detection found
        facesContainer.style.display = 'block';
        
        // Display overall result based on the most prevalent label among faces
        const fakeCount = data.face_regions.filter(face => face.label === "Fake").length;
        const realCount = data.face_regions.length - fakeCount;
        
        if (fakeCount > 0) {
          // At least one fake face detected
          predictionResult.textContent = `Fake Detected (${fakeCount}/${data.face_regions.length})`;
          predictionResult.className = 'result-value fake';
        } else {
          predictionResult.textContent = 'Real';
          predictionResult.className = 'result-value real';
        }
        
        // Display the average confidence
        const avgConfidence = data.face_regions.reduce((sum, face) => sum + face.confidence, 0) / data.face_regions.length;
        confidenceResult.textContent = `${(avgConfidence * 100).toFixed(1)}% avg`;
        
        // Create face cards for each face detected
        faceResults.innerHTML = ''; // Clear previous results
        
        data.face_regions.forEach((face, index) => {
          const faceCard = document.createElement('div');
          faceCard.className = `face-card ${face.label.toLowerCase()}`;
          
          let faceImageUrl = '';
          if (face.face_path) {
            faceImageUrl = `${BACKEND_URL}/static/${face.face_path}`;
          }
          
          let gradcamHtml = '';
          if (face.gradcam_path) {
            gradcamHtml = `
              <div class="face-heatmap">
                <img src="${BACKEND_URL}/static/${face.gradcam_path}" alt="Face heatmap" />
              </div>
            `;
          }
          
          // Add ensemble results if available
          let ensembleHtml = '';
          if (face.method_results) {
            ensembleHtml = `
              <div class="ensemble-results">
                <div class="ensemble-item">
                  <span>Primary: ${(face.method_results.primary.fake_prob * 100).toFixed(1)}%</span>
                </div>
                <div class="ensemble-item">
                  <span>Noise: ${(face.method_results.noise.fake_prob * 100).toFixed(1)}%</span>
                </div>
                <div class="ensemble-item">
                  <span>Frequency: ${(face.method_results.freq.fake_prob * 100).toFixed(1)}%</span>
                </div>
              </div>
            `;
          }
          
          faceCard.innerHTML = `
            <div class="face-header">
              <h4>Face #${index + 1}</h4>
              <span class="face-label ${face.label.toLowerCase()}">${face.label}</span>
            </div>
            <div class="face-content">
              <div class="face-image">
                <img src="${faceImageUrl}" alt="Detected face" />
              </div>
              ${gradcamHtml}
              <div class="face-details">
                <div class="face-detail-item">
                  <span class="detail-label">Confidence:</span>
                  <span class="detail-value">${(face.confidence * 100).toFixed(1)}%</span>
                </div>
                ${face.agreement_factor ? `
                <div class="face-detail-item">
                  <span class="detail-label">Agreement:</span>
                  <span class="detail-value">${(face.agreement_factor * 100).toFixed(1)}%</span>
                </div>` : ''}
                ${ensembleHtml}
              </div>
            </div>
          `;
          
          faceResults.appendChild(faceCard);
        });
        
        // If heatmap is available for the whole image, still show it
        if (data.gradcam) {
          heatmapContainer.style.display = 'block';
          heatmapImage.src = `${BACKEND_URL}/static/${data.gradcam}`;
        }
        
      } else {
        // Single/no face detection - show overall result
        predictionResult.textContent = data.label || 'Unknown';
        predictionResult.className = `result-value ${data.label ? data.label.toLowerCase() : ''}`;
        
        // Set confidence percentage
        if (data.confidence !== undefined) {
          const confidencePercent = (data.confidence * 100).toFixed(1);
          confidenceResult.textContent = `${confidencePercent}%`;
        } else {
          confidenceResult.textContent = 'N/A';
        }
        
        // If there's a gradcam (heatmap) visualization, display it
        if (data.gradcam) {
          heatmapContainer.style.display = 'block';
          heatmapImage.src = `${BACKEND_URL}/static/${data.gradcam}`;
        }
      }
      
      // Add additional details display if available
      const additionalResultsElement = document.getElementById('additional-results');
      if (additionalResultsElement && data.ensemble_results) {
        // Display ensemble results if available
        let additionalHTML = `
            <div class="additional-result-item">
                <div class="result-label">Frequency Analysis:</div>
                <div class="result-value">${(data.ensemble_results.freq.fake_prob * 100).toFixed(1)}%</div>
            </div>
            <div class="additional-result-item">
                <div class="result-label">Noise Analysis:</div>
                <div class="result-value">${(data.ensemble_results.noise.fake_prob * 100).toFixed(1)}%</div>
            </div>
            <div class="additional-result-item">
                <div class="result-label">Agreement Factor:</div>
                <div class="result-value">${data.agreement_factor ? (data.agreement_factor * 100).toFixed(1) : 'N/A'}%</div>
            </div>
        `;
        additionalResultsElement.innerHTML = additionalHTML;
        additionalResultsElement.style.display = 'block';
      } else if (additionalResultsElement) {
        additionalResultsElement.style.display = 'none';
      }
      
      // Show the faces details if there are multiple faces
      if (data.image_with_faces) {
        const imageWithFacesContainer = document.getElementById('image-with-faces-container');
        if (imageWithFacesContainer) {
          imageWithFacesContainer.style.display = 'block';
          const imageWithFaces = document.getElementById('image-with-faces');
          if (imageWithFaces) {
            imageWithFaces.src = `${BACKEND_URL}/static/${data.image_with_faces}`;
          }
        }
      }
    }
  });
  
  // This function will be injected into the page
  function injectImageSelector() {
    // Create indicator element to show the selection mode is active
    const indicator = document.createElement('div');
    indicator.id = 'twitter-image-selector-indicator';
    indicator.textContent = 'Click on any image to select it for deepfake detection';
    indicator.style.cssText = `
      position: fixed;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      background-color: rgba(29, 161, 242, 0.9);
      color: white;
      padding: 10px 20px;
      border-radius: 30px;
      z-index: 10000;
      font-family: 'Roboto', Arial, sans-serif;
      font-weight: 500;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      animation: slideIn 0.3s ease;
    `;
    document.body.appendChild(indicator);
    
    // Add animation styles
    const style = document.createElement('style');
    style.textContent = `
      @keyframes slideIn {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
      }
      @keyframes slideOut {
        from { transform: translateY(0); opacity: 1; }
        to { transform: translateY(-20px); opacity: 0; }
      }
      @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(29, 161, 242, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(29, 161, 242, 0); }
        100% { box-shadow: 0 0 0 0 rgba(29, 161, 242, 0); }
      }
    `;
    document.head.appendChild(style);
    
    // Start pulse animation after a delay
    setTimeout(() => {
      indicator.style.animation = 'pulse 2s infinite';
    }, 1000);
    
    // Add hover effect to images
    const imgHoverStyle = document.createElement('style');
    imgHoverStyle.textContent = `
      img:hover {
        outline: 3px solid rgba(29, 161, 242, 0.8) !important;
        cursor: pointer !important;
      }
    `;
    document.head.appendChild(imgHoverStyle);
    
    // Add temporary event listener
    document.addEventListener('click', imageClickHandler);
    
    // Escape key to cancel
    document.addEventListener('keydown', function escHandler(e) {
      if (e.key === 'Escape') {
        cleanup();
        document.removeEventListener('keydown', escHandler);
      }
    });
    
    // Handle image clicks
    function imageClickHandler(e) {
      // Check if the clicked element is an image
      if (e.target.tagName === 'IMG') {
        // Don't process tiny images (likely icons)
        if (e.target.width < 50 || e.target.height < 50) return;
        
        e.preventDefault();
        e.stopPropagation();
        
        // Get the best quality image URL
        let imageUrl = e.target.src;
        
        // Store in Chrome storage
        chrome.storage.local.set({
          'selectedImage': {
            url: imageUrl,
            timestamp: new Date().getTime()
          }
        });
        
        // Clean up
        cleanup();
        
        // Notify the user
        const notification = document.createElement('div');
        notification.style.cssText = `
          position: fixed;
          top: 20px;
          right: 20px;
          background-color: rgba(70, 203, 94, 0.95);
          color: white;
          padding: 15px;
          border-radius: 8px;
          z-index: 10000;
          font-family: 'Roboto', Arial, sans-serif;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
          animation: slideIn 0.3s ease;
          display: flex;
          align-items: center;
          gap: 10px;
        `;
        notification.innerHTML = '<i class="fas fa-check-circle" style="font-size: 20px;"></i><div><strong>Image selected!</strong><br><span style="font-size: 13px;">Open the extension again to view it.</span></div>';
        document.body.appendChild(notification);
        
        // Remove notification after 3 seconds
        setTimeout(() => {
          notification.style.animation = 'slideOut 0.3s ease forwards';
          setTimeout(() => {
            notification.remove();
          }, 300);
        }, 3000);
      }
    }
    
    // Clean up function
    function cleanup() {
      document.removeEventListener('click', imageClickHandler);
      const indicator = document.getElementById('twitter-image-selector-indicator');
      document.head.removeChild(imgHoverStyle);
      if (indicator) {
        indicator.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => {
          indicator.remove();
        }, 300);
      }
    }
  };