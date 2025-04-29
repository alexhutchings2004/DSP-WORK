import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Card, 
  CardContent, 
  Typography, 
  Paper,
  Tooltip,
  IconButton,
  Chip,
  Collapse,
  Button,
  Divider,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import MemoryIcon from '@mui/icons-material/Memory';
import SpeedIcon from '@mui/icons-material/Speed';
import BuildIcon from '@mui/icons-material/Build';
import SchemaIcon from '@mui/icons-material/Schema';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { useModels } from '../contexts/ModelsContext';

export default function ModelSelector() {
  const { availableModels, selectedModels } = useModels();
  const [modelDetails, setModelDetails] = useState({});
  const [showDetails, setShowDetails] = useState(false);
  const fetchModelInfo = async (modelId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/model/${modelId}/info`);
      if (response.ok) {
        const data = await response.json();
        setModelDetails(prev => ({
          ...prev,
          [modelId]: data
        }));
      }
    } catch (error) {
      console.error(`Error fetching model ${modelId} info:`, error);
    }
  };
  useEffect(() => {
    if (availableModels.length > 0) {
      fetchModelInfo(availableModels[0].id);
    }
  }, [availableModels]);
  if (availableModels.length === 0) {
    return (
      <Card className="mb-8">
        <CardContent>
          <Typography variant="body2" color="textSecondary" className="text-center">
            Loading EfficientNet model...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const model = availableModels[0]; // We only have one model
  const details = modelDetails[model.id] || {};

  return (
    <Card className="mb-8">
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" component="h2">
            Deepfake Detection Model
          </Typography>
          
          <Chip
            label="Active Model"
            color="primary"
            icon={<CheckCircleIcon />}
          />
        </Box>
        
        <Paper variant="outlined" className="p-4 mt-2">
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box>
              <Typography variant="h6" className="font-semibold">
                {model.name}
              </Typography>
              
              <Typography variant="body2" color="textSecondary" className="mt-1">
                {model.description || "Advanced EfficientNet model for deepfake detection"}
              </Typography>
              
              {details.accuracy && (
                <Typography variant="body2" className="mt-2 flex items-center">
                  <span className="font-semibold mr-1">Accuracy:</span> 
                  {(details.accuracy * 100).toFixed(1)}%
                </Typography>
              )}
              
              <Typography variant="body2" className="mt-1" color="primary">
                Framework: {details?.framework || "PyTorch"}
              </Typography>
            </Box>
            
            <Button 
              variant="outlined" 
              size="small" 
              endIcon={showDetails ? undefined : <ExpandMoreIcon />}
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? 'Hide Details' : 'View Details'}
            </Button>
          </Box>
          
          <Collapse in={showDetails} timeout="auto" unmountOnExit>
            <Box mt={3}>
              <Divider />
              
              <Box mt={2} p={2} bgcolor="background.paper" borderRadius={1}>
                <Typography variant="subtitle1" gutterBottom>
                  <AutoAwesomeIcon sx={{ mr: 1, verticalAlign: 'middle', fontSize: '1.2rem' }} />
                  Model Architecture
                </Typography>
                <Typography variant="body2" paragraph>
                  This model is based on EfficientNet-B0 architecture with specialized layers for deepfake detection.
                  EfficientNet uses a compound scaling method that uniformly scales network depth, width, and resolution
                  for optimal performance and efficiency.
                </Typography>
                
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={12} sm={6}>
                    <List dense disablePadding>
                      <ListItem>
                        <ListItemIcon sx={{ minWidth: 36 }}>
                          <MemoryIcon fontSize="small" />
                        </ListItemIcon>
                        <ListItemText 
                          primary={`Parameters: ${details.parameters_count || '5.3M'}`}
                          secondary="Efficient parameter utilization"
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon sx={{ minWidth: 36 }}>
                          <SchemaIcon fontSize="small" />
                        </ListItemIcon>
                        <ListItemText 
                          primary={`Input Size: ${details.input_size || '224 x 224 x 3'}`}
                          secondary="RGB image input"
                        />
                      </ListItem>
                    </List>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <List dense disablePadding>
                      <ListItem>
                        <ListItemIcon sx={{ minWidth: 36 }}>
                          <SpeedIcon fontSize="small" />
                        </ListItemIcon>
                        <ListItemText 
                          primary={`Inference Speed: ${details.inference_speed || '~150ms'}/image`}
                          secondary="On server-grade GPU"
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemIcon sx={{ minWidth: 36 }}>
                          <BuildIcon fontSize="small" />
                        </ListItemIcon>
                        <ListItemText 
                          primary={`Fine-tuned on: ${details.training_dataset || 'Custom deepfake dataset'}`}
                          secondary="With data augmentation"
                        />
                      </ListItem>
                    </List>
                  </Grid>
                </Grid>
                
                <Typography variant="subtitle1" gutterBottom>
                  <AutoAwesomeIcon sx={{ mr: 1, verticalAlign: 'middle', fontSize: '1.2rem' }} />
                  Detection Capabilities
                </Typography>
                
                <Typography variant="body2" paragraph>
                  The EfficientNet model has been enhanced with specialized layers for deepfake detection:
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Paper variant="outlined" sx={{ p: 1, mb: 1 }}>
                      <Typography variant="subtitle2">Texture Analysis</Typography>
                      <Typography variant="body2" color="textSecondary">
                        Detects inconsistencies in skin texture and unnatural smoothness
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Paper variant="outlined" sx={{ p: 1, mb: 1 }}>
                      <Typography variant="subtitle2">Frequency Domain Analysis</Typography>
                      <Typography variant="body2" color="textSecondary">
                        Identifies artifacts in frequency spectra common in generated images
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Paper variant="outlined" sx={{ p: 1, mb: 1 }}>
                      <Typography variant="subtitle2">Face Region Analysis</Typography>
                      <Typography variant="body2" color="textSecondary">
                        Evaluates individual facial regions for manipulation evidence
                      </Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Paper variant="outlined" sx={{ p: 1, mb: 1 }}>
                      <Typography variant="subtitle2">Attention Mechanism</Typography>
                      <Typography variant="body2" color="textSecondary">
                        Focuses on the most discriminative regions for making decisions
                      </Typography>
                    </Paper>
                  </Grid>
                </Grid>
                
                <Box mt={2}>
                  <Typography variant="caption" color="textSecondary">
                    Model last updated: {details.last_updated || "April 2025"}
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Collapse>
        </Paper>
      </CardContent>
    </Card>
  );
}