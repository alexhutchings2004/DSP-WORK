import React from 'react';
import { 
  Box, 
  Card, 
  CardContent, 
  Typography, 
  Grid, 
  Divider,
  LinearProgress,
  Chip,
  Paper
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';

export default function ComparisonView({ results, image }) {
  if (!results || results.length === 0 || !results[0].models || results[0].models.length <= 1) {
    return null;
  }
  const faceResults = results[0];
  const modelResults = faceResults.models;
  const consistentResult = modelResults.every(
    (result, i, arr) => i === 0 || result.label === arr[0].label
  );
  const highestConfidence = modelResults.reduce(
    (max, result) => Math.max(max, result.confidence), 
    0
  );
  
  const highestConfidenceModel = modelResults.find(
    result => result.confidence === highestConfidence
  );
  
  return (
    <Card className="mb-8">
      <CardContent>
        <Typography variant="h6" component="h2" className="mb-4">
          Model Comparison Analysis
        </Typography>
        
        <Box className="bg-gray-800 bg-opacity-30 p-4 rounded-lg mb-6">
          <Box display="flex" alignItems="center" mb={2}>
            <Box 
              component="img" 
              src={`http://localhost:5000/${faceResults.face_path}`}
              alt="Analyzed Face" 
              sx={{ 
                width: 80, 
                height: 80, 
                objectFit: 'cover',
                borderRadius: 2,
                marginRight: 2
              }}
            />
            
            <Box>
              <Typography variant="subtitle1">
                Model Consensus
                {consistentResult ? (
                  <Chip 
                    icon={<CheckCircleIcon />} 
                    label="Consistent" 
                    color="success" 
                    size="small" 
                    className="ml-2"
                  />
                ) : (
                  <Chip 
                    icon={<CancelIcon />} 
                    label="Inconsistent" 
                    color="warning" 
                    size="small" 
                    className="ml-2"
                  />
                )}
              </Typography>
              
              <Typography variant="body2" color="textSecondary">
                {consistentResult 
                  ? `All models agree: ${modelResults[0].label}`
                  : `Models have different opinions on this image`}
              </Typography>
              
              <Typography variant="caption" color="textSecondary" className="mt-1 block">
                Highest confidence: {(highestConfidence * 100).toFixed(1)}% 
                ({highestConfidenceModel?.model || 'Unknown'})
              </Typography>
            </Box>
          </Box>
        </Box>

        <Grid container spacing={3}>
          {modelResults.map((result, index) => (
            <Grid item xs={12} md={6} lg={4} key={result.model_id || index}>
              <Paper 
                elevation={3} 
                className="h-full flex flex-col"
                sx={{ 
                  bgcolor: result.label === 'Real' ? 'success.dark' : 'error.dark',
                  opacity: 0.9,
                  transition: 'transform 0.2s',
                  '&:hover': {
                    transform: 'scale(1.02)',
                    opacity: 1
                  }
                }}
              >
                <Box className="p-3 flex justify-between items-center">
                  <Typography variant="subtitle1" className="font-bold">
                    {result.model}
                  </Typography>
                  <Chip 
                    label={result.label} 
                    color={result.label === 'Real' ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
                
                <Divider />
                
                <Box className="p-4 flex-grow">
                  <Box className="mb-3">
                    <Box display="flex" justifyContent="space-between" mb={0.5}>
                      <Typography variant="body2">Confidence</Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {(result.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={result.confidence * 100} 
                      color={result.label === 'Real' ? 'success' : 'error'}
                      sx={{ height: 8, borderRadius: 1 }}
                    />
                  </Box>
                  
                  <Box className="mb-3">
                    <Typography variant="body2" className="mb-1">
                      Processing Time: {(result.processing_time * 1000).toFixed(0)} ms
                    </Typography>
                  </Box>
                  
                  {result.feature_importance && (
                    <Box>
                      <Typography variant="body2" className="mb-2">
                        Key Features
                      </Typography>
                      
                      {Object.entries(result.feature_importance)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 3)
                        .map(([feature, value]) => (
                          <Box key={feature} className="mb-1">
                            <Box display="flex" justifyContent="space-between" alignItems="center">
                              <Typography variant="caption">
                                {feature.replace(/_/g, ' ')}
                              </Typography>
                              <Typography variant="caption">
                                {(value * 100).toFixed(0)}%
                              </Typography>
                            </Box>
                            <LinearProgress 
                              variant="determinate" 
                              value={value * 100} 
                              sx={{ 
                                height: 4, 
                                borderRadius: 1,
                                bgcolor: 'rgba(255,255,255,0.1)'
                              }}
                            />
                          </Box>
                        ))
                      }
                    </Box>
                  )}
                </Box>
                
                <Box className="p-3 bg-black bg-opacity-20">
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Box 
                        component="img" 
                        src={`http://localhost:5000/${result.gradcam_path}`} 
                        alt="GradCAM" 
                        sx={{ 
                          width: '100%', 
                          height: 100, 
                          objectFit: 'cover',
                          borderRadius: 1
                        }}
                      />
                      <Typography 
                        variant="caption" 
                        align="center" 
                        display="block" 
                        mt={0.5}
                      >
                        Attention Map
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Box 
                        component="img" 
                        src={`http://localhost:5000/${result.noise_path}`} 
                        alt="Noise Pattern" 
                        sx={{ 
                          width: '100%', 
                          height: 100, 
                          objectFit: 'cover',
                          borderRadius: 1
                        }}
                      />
                      <Typography 
                        variant="caption" 
                        align="center" 
                        display="block" 
                        mt={0.5}
                      >
                        Noise Analysis
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
}