import React, { useState, useEffect, useRef } from "react";
import { 
  Box, 
  Button, 
  Typography, 
  CircularProgress, 
  Alert, 
  AlertTitle, 
  Tabs, 
  Tab, 
  Paper,
  Grid,
  Divider,
  Fade,
  Card,
  CardContent,
  Snackbar,
  LinearProgress,
  useTheme,
  alpha,
  Avatar,
  Stack,
  IconButton,
  Chip
} from "@mui/material";
import { useDropzone } from "react-dropzone";
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ImageIcon from '@mui/icons-material/Image';
import AssessmentIcon from '@mui/icons-material/Assessment';
import HistoryIcon from '@mui/icons-material/History';
import AnalyticsDashboard from "./AnalyticsDashboard";
import ModelExplainabilityPanel from "./ModelExplainabilityPanel";
import { useModels } from "../contexts/ModelsContext";
import InsertPhotoIcon from '@mui/icons-material/InsertPhoto';
import PasteIcon from '@mui/icons-material/ContentPaste';
import ReplayIcon from '@mui/icons-material/Replay';
import TaskAltIcon from '@mui/icons-material/TaskAlt';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';

const ModelResultBox = ({ modelResult }) => {
  const theme = useTheme();
  
  if (!modelResult) return null;
  
  const { model, label, confidence, processing_time } = modelResult;
  const isReal = label === 'Real';
  
  return (
    <Box 
      sx={{ 
        position: 'relative',
        borderRadius: 3,
        overflow: 'hidden',
        mb: 2,
        background: isReal 
          ? `linear-gradient(135deg, ${alpha(theme.palette.success.dark, 0.8)}, ${alpha(theme.palette.success.main, 0.6)})` 
          : `linear-gradient(135deg, ${alpha(theme.palette.error.dark, 0.8)}, ${alpha(theme.palette.error.main, 0.6)})`,
        backdropFilter: 'blur(10px)',
        boxShadow: isReal
          ? `0 8px 32px ${alpha(theme.palette.success.main, 0.3)}`
          : `0 8px 32px ${alpha(theme.palette.error.main, 0.3)}`,
        border: `1px solid ${alpha(isReal ? theme.palette.success.light : theme.palette.error.light, 0.3)}`
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          opacity: 0.05,
          backgroundImage: isReal
            ? 'url("data:image/svg+xml,%3Csvg width=\'100\' height=\'100\' viewBox=\'0 0 100 100\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cpath d=\'M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z\' fill=\'%23ffffff\' fill-opacity=\'1\' fill-rule=\'evenodd\'/%3E%3C/svg%3E")'
            : 'url("data:image/svg+xml,%3Csvg width=\'100\' height=\'100\' viewBox=\'0 0 100 100\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cpath d=\'M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z\' fill=\'%23ffffff\' fill-opacity=\'1\' fill-rule=\'evenodd\'/%3E%3C/svg%3E")'
        }}
      />

      <Box p={3}>
        <Box display="flex" alignItems="center" mb={1.5}>
          <Avatar
            sx={{ 
              bgcolor: 'transparent',
              border: `2px solid ${isReal ? theme.palette.success.light : theme.palette.error.light}`,
              mr: 1.5,
              width: 36,
              height: 36
            }}
          >
            {isReal ? (
              <TaskAltIcon color="inherit" />
            ) : (
              <ErrorOutlineIcon color="inherit" />
            )}
          </Avatar>
        
          <Box flex={1}>
            <Typography variant="body2" color="rgba(255,255,255,0.7)">
              {model}
            </Typography>
            
            <Typography 
              variant="h5" 
              fontWeight="bold"
              color="white"
            >
              {label}
            </Typography>
          </Box>
        </Box>
        
        <Box mb={2}>
          <Box display="flex" justifyContent="space-between" mb={0.5}>
            <Typography variant="body2" color="rgba(255,255,255,0.9)">Confidence</Typography>
            <Typography variant="body2" fontWeight="bold" color="white">
              {(confidence * 100).toFixed(1)}%
            </Typography>
          </Box>
          
          <Box position="relative" height={6} bgcolor={alpha('#000', 0.2)} borderRadius={3} overflow="hidden">
            <Box 
              position="absolute"
              top={0}
              left={0}
              height="100%" 
              width={`${confidence * 100}%`}
              sx={{ 
                bgcolor: isReal ? 'success.light' : 'error.light',
                boxShadow: isReal 
                  ? `0 0 8px ${theme.palette.success.main}` 
                  : `0 0 8px ${theme.palette.error.main}`,
              }}
              borderRadius={3}
            />
          </Box>
        </Box>
        
        <Typography variant="caption" color="rgba(255,255,255,0.7)">
          Analysis completed in {(processing_time * 1000).toFixed(0)} ms
        </Typography>
      </Box>
    </Box>
  );
};

const ImageDrop = () => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [imagePath, setImagePath] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [error, setError] = useState(null);
  const [histogramPath, setHistogramPath] = useState(null);
  const [histogramStats, setHistogramStats] = useState({});
  const [processing, setProcessing] = useState(false);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [modelChecked, setModelChecked] = useState(false);
  const [modelAvailable, setModelAvailable] = useState(false);
  const [checkingModel, setCheckingModel] = useState(true);
  const [snackbarMessage, setSnackbarMessage] = useState("");
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const modelCheckAttempts = useRef(0);
  const maxModelCheckAttempts = 5;
  const theme = useTheme();

  const fileInputRef = useRef(null);
  const { selectedModels, availableModels } = useModels();
  const checkModelAvailability = () => {
    setCheckingModel(true);
    setModelAvailable(false);
    fetch('http://localhost:5000/api/model/status')
      .then(response => response.json())
      .then(data => {
        setModelChecked(true);
        setCheckingModel(false);
        
        if (data.status === 'ready') {
          setModelAvailable(true);
          console.log('Model is ready for use');
        } else {
          setModelAvailable(false);
          console.log('Model is not ready yet:', data.message || 'Unknown status');
          if (modelCheckAttempts.current < maxModelCheckAttempts) {
            modelCheckAttempts.current += 1;
            setTimeout(checkModelAvailability, 3000);
          }
        }
      })
      .catch(err => {
        console.error('Error checking model availability:', err);
        setModelChecked(true);
        setCheckingModel(false);
        setModelAvailable(false);
        if (modelCheckAttempts.current < maxModelCheckAttempts) {
          modelCheckAttempts.current += 1;
          setTimeout(checkModelAvailability, 3000);
        }
      });
  };
  const retryModelCheck = () => {
    modelCheckAttempts.current = 0;
    checkModelAvailability();
  };
  useEffect(() => {
    checkModelAvailability();
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    accept: { "image/*": [] },
    onDrop: (acceptedFiles) => {
      if (!modelAvailable) {
        setSnackbarMessage("Model is not ready yet. Please wait a moment.");
        setSnackbarOpen(true);
        return;
      }
      handleImageSelection(acceptedFiles[0]);
    },
    noClick: true,
    noKeyboard: true
  });

  const handleImageSelection = (file) => {
    if (!file) return;
    setError(null);
    setResults([]);
    setHistogramPath(null);
    setHistogramStats({});
    setImagePath(null);
    setAnalysisComplete(false);
    const reader = new FileReader();
    reader.onload = () => {
      setImagePreview(reader.result);
      setUploadedFile(file);
      setProcessing(true);
      predictImage(file);
    };
    reader.readAsDataURL(file);
  };

  const predictImage = (file) => {
    const formData = new FormData();
    formData.append('image', file);
    if (selectedModels && selectedModels.length > 0) {
      formData.append('models', selectedModels.join(','));
    }
    
    setProcessing(true);
    
    fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Error: ${response.status} ${response.statusText}`);
        }
        return response.json();
      })
      .then((data) => {
        console.log("Prediction result:", data);
        setImagePath(data.image);
        
        if (data.histogram_path) {
          setHistogramPath(data.histogram_path);
        }
        
        if (data.histogram_stats) {
          setHistogramStats(data.histogram_stats);
        }
        if (data.results && data.results.length > 0) {
          setResults(data.results);
        } else {
          setResults([data]);
        }
        const historyItem = {
          id: Date.now().toString(),
          image: data.image,
          prediction: data.label,
          timestamp: new Date().toLocaleString(),
        };
        
        setAnalysisHistory(prev => [historyItem, ...prev]);
        setAnalysisComplete(true);
      })
      .catch((err) => {
        console.error("Error during prediction:", err);
        setError(`Failed to process the image: ${err.message}`);
      })
      .finally(() => {
        setProcessing(false);
      });
  };

  const handlePaste = (event) => {
    if (!modelAvailable) {
      setSnackbarMessage("Model is not ready yet. Please wait a moment.");
      setSnackbarOpen(true);
      return;
    }

    if (event.clipboardData && event.clipboardData.items) {
      const items = event.clipboardData.items;
      
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          const blob = items[i].getAsFile();
          handleImageSelection(blob);
          break;
        }
      }
    }
  };

  const resetUpload = () => {
    setUploadedFile(null);
    setImagePreview(null);
    setResults([]);
    setImagePath(null);
    setTabValue(0);
    setError(null);
    setHistogramPath(null);
    setHistogramStats({});
    setProcessing(false);
    setAnalysisComplete(false);
  };

  const handleFileInputChange = (event) => {
    if (!modelAvailable) {
      setSnackbarMessage("Model is not ready yet. Please wait a moment.");
      setSnackbarOpen(true);
      return;
    }
    
    if (event.target.files && event.target.files[0]) {
      handleImageSelection(event.target.files[0]);
    }
  };

  useEffect(() => {
    window.addEventListener("paste", handlePaste);
    return () => {
      window.removeEventListener("paste", handlePaste);
    };
  }, [modelAvailable]); // Re-add event listener when model availability changes

  // Add useEffect to check for URL parameters
  useEffect(() => {
    // Check for URL parameters
    const queryParams = new URLSearchParams(window.location.search);
    const source = queryParams.get('source');
    const imageId = queryParams.get('imageId');
    
    // If coming from extension with an ID
    if (source === 'extension' && imageId) {
      console.log("Loading image from extension with ID:", imageId);
      loadExtensionImage(imageId);
    }
  }, []); // Empty dependency array means this runs once on component mount
  
  // Function to load an image from the extension
  const loadExtensionImage = (imageId) => {
    setProcessing(true);
    setError(null);
    
    // Fetch the image details from backend
    fetch(`http://localhost:5000/get_extension_image/${imageId}`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to get extension image: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        if (!data.success) {
          throw new Error(data.error || "Failed to load image");
        }
        
        // Image URL from backend
        const imageUrl = `http://localhost:5000${data.image_url}`;
        
        // Fetch the actual image as a blob
        return fetch(imageUrl)
          .then(response => {
            if (!response.ok) {
              throw new Error(`Failed to fetch image: ${response.status}`);
            }
            return response.blob();
          })
          .then(blob => {
            // Create a File object from the blob
            const file = new File([blob], data.filename || "extension-image.png", { 
              type: blob.type 
            });
            
            // Create a preview URL
            const imagePreviewUrl = URL.createObjectURL(blob);
            
            // Set the image preview and file
            setImagePreview(imagePreviewUrl);
            setUploadedFile(file);
            
            // If we already have analysis data, use it
            if (data.analysis) {
              console.log("Using analysis data from extension:", data.analysis);
              // Parse the analysis data into expected format
              setResults(Array.isArray(data.analysis) ? data.analysis : [data.analysis]);
              
              // Set other analysis data
              if (data.analysis.histogram_path) {
                setHistogramPath(data.analysis.histogram_path);
              }
              if (data.analysis.histogram_stats) {
                setHistogramStats(data.analysis.histogram_stats);
              }
              if (data.analysis.image) {
                setImagePath(data.analysis.image);
              }
              setAnalysisComplete(true);
              setProcessing(false);
            } else {
              // Otherwise, run prediction on the image
              console.log("Running prediction on extension image");
              predictImage(file);
            }
          });
      })
      .catch(error => {
        console.error("Error loading extension image:", error);
        setError(`Failed to load image from extension: ${error.message}`);
        setProcessing(false);
      });
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleCloseSnackbar = () => {
    setSnackbarOpen(false);
  };

  const reanalyzeFromHistory = async (historyItem) => {
    try {
      // Show loading
      setProcessing(true);
      
      // Fetch the image as a blob
      const imageUrl = `http://localhost:5000/static/${historyItem.image}`;
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error('Failed to fetch image from history');
      }
      
      const blob = await response.blob();
      const file = new File([blob], historyItem.image.split('/').pop() || 'history-image.png', { 
        type: blob.type || 'image/png'
      });
      
      // Create a preview
      const imagePreviewUrl = URL.createObjectURL(blob);
      
      // Set up for analysis
      setImagePreview(imagePreviewUrl);
      setUploadedFile(file);
      setTabValue(0);
      
      // Run prediction
      predictImage(file);
    } catch (error) {
      console.error('Error re-analyzing image from history:', error);
      setError(`Failed to re-analyze image: ${error.message}`);
      setProcessing(false);
    }
  };

  return (
    <div>
      {/* Main upload and display area */}
      <Card 
        elevation={0}
        sx={{ 
          mb: 4, 
          borderRadius: 3,
          backdropFilter: 'blur(10px)',
          backgroundColor: alpha(theme.palette.background.paper, 0.7)
        }}
      >
        <Box 
          sx={{
            borderTopLeftRadius: 12,
            borderTopRightRadius: 12,
            p: 3,
            position: 'relative',
            overflow: 'hidden',
            background: `linear-gradient(135deg, ${alpha(theme.palette.primary.dark, 0.8)} 0%, ${alpha(theme.palette.primary.main, 0.4)} 100%)`,
          }}
        >
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              opacity: 0.1,
              background: 'url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'100\' height=\'100\' viewBox=\'0 0 100 100\'%3E%3Cg fill-rule=\'evenodd\'%3E%3Cg fill=\'%23ffffff\' fill-opacity=\'0.12\'%3E%3Cpath opacity=\'.5\' d=\'M96 95h4v1h-4v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9zm-1 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm9-10v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm9-10v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9z\'/%3E%3Cpath d=\'M6 5V0H5v5H0v1h5v94h1V6h94V5H6z\'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")'
            }}
          />
          
          <Box position="relative">
            <Typography variant="h4" component="h1" className="font-bold mb-1" color="white" sx={{ textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}>
              Deepfake Detection Analysis
            </Typography>
            <Typography variant="body1" color="rgba(255,255,255,0.8)" sx={{ maxWidth: 600 }}>
              Upload or paste an image to analyze using AI. This system detects face manipulations using the EfficientNet architecture.
            </Typography>
            
            {checkingModel && (
              <Box sx={{ width: '100%', mt: 2 }}>
                <Typography variant="caption" color="rgba(255,255,255,0.9)">
                  Initializing AI model...
                </Typography>
                <LinearProgress 
                  color="secondary" 
                  sx={{ 
                    height: 4, 
                    borderRadius: 2,
                    bgcolor: alpha(theme.palette.common.white, 0.1),
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 2,
                      background: `linear-gradient(90deg, ${theme.palette.secondary.dark}, ${theme.palette.secondary.main})`,
                      boxShadow: `0 0 10px ${theme.palette.secondary.main}`
                    }
                  }} 
                />
              </Box>
            )}
            
            {modelChecked && !modelAvailable && !checkingModel && (
              <Alert 
                severity="warning" 
                sx={{ 
                  mt: 2,
                  backgroundColor: alpha(theme.palette.warning.dark, 0.9),
                  color: 'white',
                  '&:hover': { backgroundColor: alpha(theme.palette.warning.dark, 1) },
                  '& .MuiAlert-icon': { color: 'white' }
                }} 
                action={
                  <Button color="inherit" size="small" startIcon={<ReplayIcon />} onClick={retryModelCheck}>
                    Retry
                  </Button>
                }
              >
                EfficientNet model is not ready. Check if the backend server is running.
              </Alert>
            )}
          </Box>
        </Box>

        {!imagePreview ? (
          <Box
            {...getRootProps()}
            onClick={() => {
              if (modelAvailable) {
                fileInputRef.current.click();
              } else {
                setSnackbarMessage("Model is not ready yet. Please wait a moment.");
                setSnackbarOpen(true);
              }
            }}
            sx={{
              border: `2px dashed ${modelAvailable ? alpha(theme.palette.primary.main, 0.3) : alpha(theme.palette.text.disabled, 0.3)}`,
              borderRadius: 3,
              transition: 'all 0.3s ease',
              cursor: modelAvailable ? 'pointer' : 'not-allowed',
              '&:hover': {
                backgroundColor: modelAvailable ? alpha(theme.palette.primary.main, 0.05) : 'transparent',
                borderColor: modelAvailable ? theme.palette.primary.main : alpha(theme.palette.text.disabled, 0.3),
              },
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 6,
              margin: 3
            }}
          >
            <input {...getInputProps()} ref={fileInputRef} onChange={handleFileInputChange} className="hidden" disabled={!modelAvailable} />

            <Box
              sx={{
                width: 80,
                height: 80,
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mb: 3,
                background: modelAvailable 
                  ? `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)}, ${alpha(theme.palette.secondary.main, 0.1)})`
                  : alpha(theme.palette.text.disabled, 0.1),
                border: `1px solid ${modelAvailable ? alpha(theme.palette.primary.main, 0.3) : alpha(theme.palette.text.disabled, 0.3)}`,
                boxShadow: modelAvailable 
                  ? `0 4px 20px ${alpha(theme.palette.primary.main, 0.2)}`
                  : 'none'
              }}
            >
              <InsertPhotoIcon 
                sx={{ 
                  fontSize: 40,
                  color: modelAvailable ? theme.palette.primary.main : theme.palette.text.disabled,
                  filter: modelAvailable ? `drop-shadow(0 0 8px ${alpha(theme.palette.primary.main, 0.5)})` : 'none'
                }} 
              />
            </Box>

            <Typography 
              variant="h6" 
              textAlign="center" 
              mb={2}
              color={modelAvailable ? 'text.primary' : 'text.disabled'}
              fontWeight="medium"
            >
              {modelAvailable 
                ? "Drag & drop an image here or click to browse" 
                : "Waiting for AI model to be ready..."}
            </Typography>

            <Typography 
              variant="body2" 
              color={modelAvailable ? 'text.secondary' : 'text.disabled'} 
              textAlign="center" 
              mb={4}
              sx={{ maxWidth: 450 }}
            >
              {modelAvailable 
                ? "Upload your file or paste from clipboard (Ctrl+V) to begin analysis. We'll scan the image using advanced neural networks for deepfake detection."
                : "EfficientNet model is loading. Please wait a moment before analyzing images."}
            </Typography>

            <Stack direction="row" spacing={2}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<ImageIcon />}
                disabled={!modelAvailable}
                sx={{ 
                  borderRadius: 6,
                  px: 3,
                  py: 1
                }}
              >
                Select Image
              </Button>
              
              <Button
                variant="outlined"
                color="secondary"
                startIcon={<PasteIcon />}
                disabled={!modelAvailable}
                sx={{ 
                  borderRadius: 6,
                  px: 3,
                  py: 1
                }}
              >
                Paste from Clipboard
              </Button>
            </Stack>

            <Typography variant="caption" color="text.secondary" mt={3}>
              Supported formats: JPG, PNG, JPEG, BMP
            </Typography>
            
            {!modelAvailable && !checkingModel && (
              <Button 
                variant="outlined" 
                color="primary" 
                onClick={(e) => {
                  e.stopPropagation();
                  retryModelCheck();
                }}
                startIcon={<ReplayIcon />}
                sx={{ mt: 3, borderRadius: 6 }}
              >
                Recheck Model Availability
              </Button>
            )}
          </Box>
        ) : (
          <Box>
            <Box 
              sx={{ 
                borderBottom: `1px solid ${alpha(theme.palette.divider, 0.3)}`,
                backgroundColor: alpha(theme.palette.background.paper, 0.4)
              }}
            >
              <Tabs
                value={tabValue}
                onChange={handleTabChange}
                variant="scrollable"
                scrollButtons="auto"
                aria-label="analysis tabs"
                sx={{
                  '& .MuiTabs-indicator': {
                    backgroundColor: theme.palette.primary.main,
                    height: 3,
                    borderTopLeftRadius: 3,
                    borderTopRightRadius: 3,
                    boxShadow: `0 0 8px ${theme.palette.primary.main}`
                  },
                  '& .MuiTab-root': {
                    minHeight: 56,
                    fontSize: '0.95rem',
                    fontWeight: 600
                  },
                  '& .Mui-selected': {
                    color: `${theme.palette.primary.main} !important`,
                  }
                }}
              >
                <Tab 
                  icon={<ImageIcon />} 
                  label="Results" 
                  iconPosition="start" 
                  sx={{ px: 3 }}
                />
                <Tab 
                  icon={<AssessmentIcon />} 
                  label="Detailed Analysis" 
                  iconPosition="start"
                  sx={{ px: 3 }}
                />
                <Tab 
                  icon={<HistoryIcon />} 
                  label="History" 
                  iconPosition="start"
                  sx={{ px: 3 }}
                />
              </Tabs>
            </Box>

            {processing ? (
              <Box className="flex flex-col items-center justify-center p-20">
                <Box
                  sx={{
                    position: 'relative',
                    width: 100,
                    height: 100,
                    mb: 3
                  }}
                >
                  <CircularProgress 
                    size={100}
                    thickness={2}
                    sx={{
                      color: theme.palette.primary.main,
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      boxShadow: `0 0 20px ${alpha(theme.palette.primary.main, 0.5)}`
                    }}
                  />
                  <CircularProgress 
                    size={80}
                    thickness={4}
                    sx={{
                      color: theme.palette.secondary.main,
                      position: 'absolute',
                      top: 10,
                      left: 10,
                    }}
                  />
                </Box>
                <Typography variant="h6" fontWeight="medium" mb={1}>Processing Image</Typography>
                <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ maxWidth: 400 }}>
                  Analyzing with EfficientNet neural network. The AI is currently examining facial features, 
                  texture patterns, and noise inconsistencies to determine authenticity...
                </Typography>
              </Box>
            ) : error ? (
              <Box sx={{ p: 4 }}>
                <Alert 
                  severity="error" 
                  sx={{
                    backgroundColor: alpha(theme.palette.error.dark, 0.1),
                    border: `1px solid ${alpha(theme.palette.error.main, 0.2)}`,
                    borderRadius: 2,
                    '& .MuiAlert-icon': {
                      color: theme.palette.error.main
                    }
                  }}
                  action={
                    <Button 
                      color="error" 
                      variant="outlined" 
                      size="small" 
                      onClick={resetUpload}
                      sx={{ borderRadius: 6 }}
                    >
                      Try Again
                    </Button>
                  }
                >
                  <AlertTitle sx={{ fontWeight: 600 }}>Error</AlertTitle>
                  {error}
                </Alert>
              </Box>
            ) : (
              <Box className="p-4">
                {/* Tab 1: Results Overview */}
                <TabPanel value={tabValue} index={0}>
                  <Fade in={tabValue === 0} timeout={800}>
                    <div>
                      <Grid container spacing={4}>
                        <Grid item xs={12} md={5}>
                          <Card 
                            elevation={0} 
                            sx={{
                              mb: 4, 
                              overflow: 'hidden',
                              border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                              backdropFilter: 'blur(10px)',
                              backgroundColor: alpha(theme.palette.background.paper, 0.4)
                            }}
                          >
                            <CardContent sx={{ p: 0 }}>
                              <Box sx={{ p: 2, pb: 1 }}>
                                <Typography variant="subtitle1" fontWeight="600" sx={{ mb: 1 }}>
                                  Analyzed Image
                                </Typography>
                              </Box>
                              
                              <Box 
                                sx={{
                                  position: 'relative',
                                  overflow: 'hidden'
                                }}
                              >
                                <Box 
                                  component="img"
                                  src={imagePath ? `http://localhost:5000/static/${imagePath}` : imagePreview}
                                  alt="Analyzed"
                                  sx={{
                                    width: '100%',
                                    display: 'block'
                                  }}
                                />
                                
                                <Box
                                  sx={{
                                    position: 'absolute',
                                    bottom: 0,
                                    left: 0,
                                    right: 0,
                                    p: 1.5,
                                    backdropFilter: 'blur(10px)',
                                    backgroundColor: alpha(theme.palette.background.paper, 0.7),
                                    borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`
                                  }}
                                >
                                  <Typography variant="body2" color="text.secondary">
                                    Analysis Timestamp: {new Date().toLocaleString()}
                                  </Typography>
                                </Box>
                              </Box>
                              
                              {histogramPath && (
                                <Box sx={{ p: 2, pt: 2 }}>
                                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                    Color Distribution Analysis
                                  </Typography>
                                  <Box 
                                    component="img"
                                    src={`http://localhost:5000/static/${histogramPath}`}
                                    alt="Histogram"
                                    sx={{
                                      width: '100%',
                                      borderRadius: 2,
                                      border: `1px solid ${alpha(theme.palette.divider, 0.1)}`
                                    }}
                                  />
                                </Box>
                              )}
                            </CardContent>
                          </Card>
                        </Grid>
                        
                        <Grid item xs={12} md={7}>
                          <Card 
                            elevation={0}
                            sx={{ 
                              p: 3,
                              border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                              backdropFilter: 'blur(10px)',
                              backgroundColor: alpha(theme.palette.background.paper, 0.4)
                            }}
                          >
                            <Box sx={{ 
                              display: 'flex', 
                              alignItems: 'center', 
                              justifyContent: 'space-between',
                              mb: 3, 
                              pb: 1, 
                              borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}` 
                            }}>
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <Box
                                  sx={{
                                    width: 30,
                                    height: 30,
                                    borderRadius: '50%',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    mr: 1.5,
                                    background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)}, ${alpha(theme.palette.secondary.main, 0.1)})`,
                                    border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
                                  }}
                                >
                                  <AssessmentIcon 
                                    sx={{ 
                                      fontSize: 18,
                                      color: theme.palette.primary.main
                                    }}
                                  />
                                </Box>
                                <Typography variant="h6" fontWeight="600">
                                  Detection Results
                                </Typography>
                              </Box>
                              
                              {/* Add Re-run Analysis Button */}
                              {!processing && analysisComplete && (
                                <Button
                                  variant="outlined"
                                  color="secondary"
                                  size="small"
                                  startIcon={<ReplayIcon />}
                                  onClick={resetUpload}
                                  sx={{ borderRadius: 8 }}
                                >
                                  Re-run Analysis
                                </Button>
                              )}
                            </Box>
                            
                            {results.length > 0 ? (
                              <div>
                                {results.map((result, index) => (
                                  <div key={index}>
                                    {/* Check if we have face data */}
                                    {result.face_path && (
                                      <Box
                                        sx={{ 
                                          display: 'flex', 
                                          alignItems: 'center', 
                                          mb: 2,
                                          p: 2,
                                          borderRadius: 3,
                                          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                                          backdropFilter: 'blur(5px)',
                                          backgroundColor: alpha(theme.palette.background.default, 0.4)
                                        }}
                                      >
                                        <Box 
                                          component="img"
                                          src={`http://localhost:5000/static/${result.face_path}`}
                                          alt={`Face ${index + 1}`}
                                          sx={{
                                            width: 90,
                                            height: 90,
                                            objectFit: 'cover',
                                            borderRadius: 2,
                                            marginRight: 2,
                                            border: `2px solid ${theme.palette.primary.main}`,
                                            boxShadow: `0 4px 12px ${alpha(theme.palette.primary.main, 0.3)}`
                                          }}
                                        />
                                        
                                        <Box>
                                          <Typography variant="subtitle1" fontWeight="600">
                                            Face #{index + 1}
                                          </Typography>
                                          {result.position && (
                                            <Box sx={{ mt: 0.5 }}>
                                              <Typography variant="body2" color="text.secondary">
                                                Position: ({result.position[0]}, {result.position[1]})
                                              </Typography>
                                              <Typography variant="body2" color="text.secondary">
                                                Size: {result.position[2]}×{result.position[3]}px
                                              </Typography>
                                            </Box>
                                          )}
                                        </Box>
                                      </Box>
                                    )}
                                    
                                    {/* Adapt to handle both old and new API response formats */}
                                    {result.models ? (
                                      result.models.map((modelResult, modelIndex) => (
                                        <ModelResultBox 
                                          key={modelIndex} 
                                          modelResult={modelResult} 
                                        />
                                      ))
                                    ) : (
                                      <ModelResultBox 
                                        modelResult={{
                                          model: result.model_name || result.model || "EfficientNet",
                                          label: result.label || "Unknown",
                                          confidence: result.confidence || 0,
                                          processing_time: result.prediction_time || result.processing_time || 0
                                        }} 
                                      />
                                    )}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <Box 
                                sx={{ 
                                  textAlign: 'center', 
                                  py: 6,
                                  borderRadius: 2,
                                  backgroundColor: alpha(theme.palette.background.default, 0.3)
                                }}
                              >
                                <Typography variant="body1" color="text.secondary">
                                  No results available yet. Please wait for analysis to complete.
                                </Typography>
                              </Box>
                            )}
                          </Card>
                        </Grid>
                      </Grid>
                    </div>
                  </Fade>
                </TabPanel>
                
                {/* Tab 2: Detailed Analysis */}
                <TabPanel value={tabValue} index={1}>
                  <Fade in={tabValue === 1}>
                    <div>
                      <ModelExplainabilityPanel 
                        results={results} 
                        histogramStats={histogramStats} 
                      />
                    </div>
                  </Fade>
                </TabPanel>
                
                {/* Tab 3: History */}
                <TabPanel value={tabValue} index={2}>
                  <Fade in={tabValue === 2}>
                    <div>
                      <Card
                        elevation={0}
                        sx={{ 
                          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                          backdropFilter: 'blur(10px)',
                          backgroundColor: alpha(theme.palette.background.paper, 0.4)
                        }}
                      >
                        <CardContent>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, pb: 1, borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}` }}>
                            <Box
                              sx={{
                                width: 30,
                                height: 30,
                                borderRadius: '50%',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                mr: 1.5,
                                background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)}, ${alpha(theme.palette.secondary.main, 0.1)})`,
                                border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
                              }}
                            >
                              <HistoryIcon 
                                sx={{ 
                                  fontSize: 18,
                                  color: theme.palette.primary.main
                                }}
                              />
                            </Box>
                            <Typography variant="h6" fontWeight="600">
                              Analysis History
                            </Typography>
                          </Box>
                          
                          {analysisHistory.length > 0 ? (
                            <Grid container spacing={2}>
                              {analysisHistory.map((item) => (
                                <Grid item xs={12} sm={6} md={4} key={item.id}>
                                  <Paper 
                                    elevation={0}
                                    sx={{ 
                                      p: 0, 
                                      overflow: 'hidden',
                                      cursor: 'pointer',
                                      borderRadius: 3,
                                      border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                                      backdropFilter: 'blur(5px)',
                                      backgroundColor: alpha(theme.palette.background.default, 0.5),
                                      transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                                      '&:hover': {
                                        transform: 'translateY(-4px)',
                                        boxShadow: `0 6px 20px ${alpha(theme.palette.common.black, 0.2)}`
                                      }
                                    }}
                                    onClick={() => reanalyzeFromHistory(item)}
                                  >
                                    <Box 
                                      sx={{ 
                                        position: 'relative',
                                        height: 140,
                                        overflow: 'hidden'
                                      }}
                                    >
                                      <Box 
                                        component="img"
                                        src={`http://localhost:5000/static/${item.image}`}
                                        alt="History item"
                                        sx={{
                                          width: '100%',
                                          height: '100%',
                                          objectFit: 'cover',
                                        }}
                                      />
                                      <Box
                                        sx={{
                                          position: 'absolute',
                                          top: 10,
                                          right: 10,
                                          borderRadius: 6,
                                          px: 1.5,
                                          py: 0.5,
                                          backgroundColor: item.prediction === 'Real' 
                                            ? alpha(theme.palette.success.main, 0.9)
                                            : alpha(theme.palette.error.main, 0.9),
                                          backdropFilter: 'blur(4px)'
                                        }}
                                      >
                                        <Typography 
                                          variant="caption" 
                                          fontWeight="bold"
                                          color="white"
                                        >
                                          {item.prediction}
                                        </Typography>
                                      </Box>
                                    </Box>
                                    
                                    <Box sx={{ p: 2 }}>
                                      <Typography variant="caption" display="block" color="text.secondary">
                                        {item.timestamp}
                                      </Typography>
                                      
                                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
                                        <Typography variant="body2" sx={{ color: theme.palette.text.secondary }}>
                                          EfficientNet
                                        </Typography>
                                        <Button
                                          size="small"
                                          startIcon={<ReplayIcon fontSize="small" />}
                                          variant="outlined"
                                          color="secondary"
                                          sx={{ borderRadius: 8, py: 0.5 }}
                                        >
                                          Re-analyze
                                        </Button>
                                      </Box>
                                    </Box>
                                  </Paper>
                                </Grid>
                              ))}
                            </Grid>
                          ) : (
                            <Box 
                              sx={{ 
                                textAlign: 'center', 
                                py: 6,
                                borderRadius: 2,
                                backgroundColor: alpha(theme.palette.background.paper, 0.2)
                              }}
                            >
                              <HistoryIcon sx={{ fontSize: 40, color: alpha(theme.palette.text.secondary, 0.5), mb: 1 }} />
                              <Typography variant="body1" color="text.secondary">
                                No analysis history yet. Images you analyze will appear here.
                              </Typography>
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </div>
                  </Fade>
                </TabPanel>
              </Box>
            )}

            {analysisComplete && (
              <Box 
                sx={{
                  p: 3, 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center', 
                  borderTop: `1px solid ${alpha(theme.palette.divider, 0.1)}`
                }}
              >
                <Typography variant="body2" color="text.secondary">
                  Analysis completed using EfficientNet neural network
                </Typography>
                
                <Button
                  variant="contained"
                  color="primary"
                  onClick={resetUpload}
                  startIcon={<ImageIcon />}
                  sx={{ borderRadius: 8 }}
                >
                  Analyze New Image
                </Button>
              </Box>
            )}
          </Box>
        )}
      </Card>
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        message={snackbarMessage}
        action={
          <Button color="secondary" size="small" onClick={handleCloseSnackbar}>
            Close
          </Button>
        }
        sx={{
          '& .MuiSnackbarContent-root': {
            backgroundColor: alpha(theme.palette.background.paper, 0.9),
            backdropFilter: 'blur(10px)',
            border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
            color: theme.palette.text.primary
          }
        }}
      />
    </div>
  );
};
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );
}

export default ImageDrop;
