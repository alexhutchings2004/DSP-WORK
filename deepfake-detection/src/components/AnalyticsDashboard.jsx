import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Divider,
  Paper,
  Tooltip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  useTheme,
  alpha,
  Avatar
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import AssessmentIcon from '@mui/icons-material/Assessment';
import SpeedIcon from '@mui/icons-material/Speed';
import FingerprintIcon from '@mui/icons-material/Fingerprint';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import PieChartIcon from '@mui/icons-material/PieChart';

export default function AnalyticsDashboard({ results, histogramStats }) {
  const theme = useTheme();
  if (!results || results.length === 0) {
    return (
      <Card sx={{ mb: 4, borderRadius: 3, overflow: 'hidden' }}>
        <CardContent sx={{ p: 4, textAlign: 'center' }}>
          <AnalyticsIcon sx={{ fontSize: 60, color: alpha(theme.palette.primary.main, 0.2), mb: 2 }} />
          <Typography variant="h6" color="textSecondary">
            No analysis data available
          </Typography>
          <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
            Please process an image to view detailed analytics
          </Typography>
        </CardContent>
      </Card>
    );
  }
  const result = results[0];
  const isOldFormat = result.models && Array.isArray(result.models);
  let modelResult;
  let faceResult;
  
  if (isOldFormat) {
    faceResult = result;
    modelResult = result.models[0]; // Use the first model's result
  } else {
    faceResult = result;
    modelResult = {
      label: result.label || "Unknown",
      confidence: result.confidence || 0,
      raw_confidence: result.raw_confidence || 0,
      processing_time: result.prediction_time || result.processing_time || 0,
      model: result.model_name || result.model || "Unknown",
      gradcam_path: result.gradcam || ""
    };
  }
  const noiseMetrics = faceResult.noise_metrics || {};
  const featureImportance = isOldFormat 
    ? (modelResult.feature_importance || {}) 
    : (result.feature_importance || {});
  const sortedFeatures = Object.entries(featureImportance)
    .sort((a, b) => b[1] - a[1]);
  const noiseByRegion = noiseMetrics.regions || {};
  const histogramData = histogramStats || {};
  const redMean = histogramData?.red?.mean || 0;
  const greenMean = histogramData?.green?.mean || 0;
  const blueMean = histogramData?.blue?.mean || 0;

  const dominantChannel = [
    { name: 'Red', value: redMean },
    { name: 'Green', value: greenMean },
    { name: 'Blue', value: blueMean }
  ].sort((a, b) => b.value - a.value)[0];
  const grayscaleMean = histogramData?.overall?.brightness || 0;
  const grayscaleStd = histogramData?.overall?.contrast || 0;
  const statisticalConsistency = grayscaleStd ? grayscaleMean / grayscaleStd : 0;
  const overallNoiseEnergy = noiseMetrics.overall?.energy || 0.2; // Default for visualization
  const getConfidenceExplanation = (confidence) => {
    if (confidence > 0.85) {
      return "Very high confidence. The model has detected strong indicators supporting this classification.";
    } else if (confidence > 0.7) {
      return "High confidence. Several key features strongly align with this classification.";
    } else if (confidence > 0.55) {
      return "Moderate confidence. Some indicators present but not all features align strongly.";
    } else {
      return "Lower confidence. The prediction should be considered with caution.";
    }
  };
  const gradcamPath = modelResult.gradcam_path || result.gradcam || "";
  const noisePath = faceResult.noise_path || result.noise_path || "";

  return (
    <Card sx={{ 
      mb: 4, 
      borderRadius: 3, 
      overflow: 'hidden', 
      boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.1)}`,
      border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
    }}>
      <Box
        sx={{
          p: 3,
          background: `linear-gradient(120deg, ${alpha(theme.palette.background.paper, 0.9)}, ${alpha(theme.palette.background.paper, 0.7)})`,
          backdropFilter: 'blur(10px)',
          borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`
        }}
      >
        <Box display="flex" alignItems="center">
          <Avatar
            sx={{
              bgcolor: alpha(theme.palette.primary.main, 0.1),
              color: theme.palette.primary.main,
              mr: 2,
              width: 48,
              height: 48
            }}
          >
            <AssessmentIcon />
          </Avatar>
          <Box>
            <Typography variant="h5" fontWeight="bold" component="h2">
              Detailed Analytics
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Advanced metrics and visualizations for deepfake detection analysis
            </Typography>
          </Box>
        </Box>
      </Box>

      <CardContent sx={{ p: 3, backgroundColor: alpha(theme.palette.background.default, 0.5) }}>
        <Grid container spacing={3}>
          {/* Prediction Summary Card */}
          <Grid item xs={12}>
            <Paper 
              elevation={0} 
              sx={{
                p: 0,
                borderRadius: 3,
                overflow: 'hidden',
                border: `1px solid ${alpha(modelResult.label === 'Real' ? theme.palette.success.main : theme.palette.error.main, 0.2)}`,
                boxShadow: `0 4px 20px ${alpha(modelResult.label === 'Real' ? theme.palette.success.main : theme.palette.error.main, 0.1)}`
              }}
            >
              <Box 
                sx={{
                  p: 3,
                  background: modelResult.label === 'Real' 
                    ? `linear-gradient(135deg, ${alpha(theme.palette.success.dark, 0.9)}, ${alpha(theme.palette.success.main, 0.7)})`
                    : `linear-gradient(135deg, ${alpha(theme.palette.error.dark, 0.9)}, ${alpha(theme.palette.error.main, 0.7)})`,
                  color: 'white'
                }}
              >
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} md={7}>
                    <Box display="flex" alignItems="center">
                      <Avatar
                        sx={{
                          bgcolor: 'white',
                          color: modelResult.label === 'Real' ? theme.palette.success.main : theme.palette.error.main,
                          mr: 2
                        }}
                      >
                        {modelResult.label === 'Real' ? <CheckCircleIcon /> : <CancelIcon />}
                      </Avatar>
                      <Box>
                        <Typography variant="h5" fontWeight="bold" sx={{ textShadow: '0 2px 4px rgba(0,0,0,0.2)' }}>
                          {modelResult.label === 'Real' ? 'Authentic Image' : 'Deepfake Detected'}
                        </Typography>
                        <Typography variant="body2" sx={{ opacity: 0.9, mt: 0.5 }}>
                          {modelResult.label === 'Real' 
                            ? 'Analysis indicates this is likely an authentic, unaltered image' 
                            : 'Analysis suggests this image has been manipulated or AI-generated'}
                        </Typography>
                      </Box>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={5}>
                    <Box>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                        <Typography variant="body2" fontWeight="bold">Confidence</Typography>
                        <Typography variant="body1" fontWeight="bold">
                          {(modelResult.confidence * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={modelResult.confidence * 100} 
                        sx={{
                          height: 12,
                          borderRadius: 6,
                          bgcolor: alpha('white', 0.2),
                          '& .MuiLinearProgress-bar': {
                            bgcolor: 'white',
                            boxShadow: '0 0 10px rgba(255,255,255,0.5)'
                          }
                        }}
                      />
                      <Typography variant="caption" sx={{ display: 'block', mt: 0.5, fontStyle: 'italic' }}>
                        {getConfidenceExplanation(modelResult.confidence)}
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>
              
              {/* Key metrics row */}
              <Box px={3} py={2} sx={{ bgcolor: alpha(theme.palette.background.paper, 0.5) }}>
                <Grid container spacing={2}>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="caption" color="textSecondary">
                      Processing Time
                    </Typography>
                    <Typography variant="body1" fontWeight="medium">
                      {(modelResult.processing_time * 1000).toFixed(0)} ms
                    </Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="caption" color="textSecondary">
                      Model Type
                    </Typography>
                    <Typography variant="body1" fontWeight="medium">
                      {modelResult.model || 'EfficientNet'}
                    </Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="caption" color="textSecondary">
                      Noise Energy
                    </Typography>
                    <Typography variant="body1" fontWeight="medium">
                      {(overallNoiseEnergy * 100).toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="caption" color="textSecondary">
                      Raw Prediction
                    </Typography>
                    <Typography variant="body1" fontWeight="medium">
                      {modelResult.raw_confidence?.toFixed(4) || '0.5000'}
                    </Typography>
                  </Grid>
                </Grid>
              </Box>
            </Paper>
          </Grid>
          
          {/* Feature Importance Analysis */}
          <Grid item xs={12} md={6}>
            <Paper 
              elevation={0} 
              sx={{ 
                p: 3, 
                height: '100%',
                borderRadius: 3,
                overflow: 'hidden',
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                backdropFilter: 'blur(5px)',
                background: alpha(theme.palette.background.paper, 0.7)
              }}
            >
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Box display="flex" alignItems="center">
                  <Avatar
                    sx={{
                      bgcolor: alpha(theme.palette.primary.main, 0.1),
                      color: theme.palette.primary.main,
                      width: 36,
                      height: 36,
                      mr: 1.5
                    }}
                  >
                    <FingerprintIcon fontSize="small" />
                  </Avatar>
                  <Typography variant="h6" fontWeight="bold">
                    Feature Analysis
                  </Typography>
                </Box>
                <Tooltip title="Features that most influenced the model's decision">
                  <InfoIcon fontSize="small" color="action" />
                </Tooltip>
              </Box>
              
              <Typography variant="body2" color="textSecondary" paragraph>
                Regions that influenced the model's decision, ranked by importance
              </Typography>
              
              {sortedFeatures.length > 0 ? (
                <Box sx={{ mt: 2, mb: 2 }}>
                  {sortedFeatures.slice(0, 5).map(([feature, value], index) => (
                    <Box key={feature} sx={{ mb: 2 }}>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                        <Box display="flex" alignItems="center">
                          <Box 
                            sx={{ 
                              width: 24, 
                              height: 24, 
                              borderRadius: '50%', 
                              bgcolor: alpha(theme.palette.primary.main, 0.1),
                              color: theme.palette.primary.main,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              mr: 1.5,
                              fontSize: '0.75rem',
                              fontWeight: 'bold'
                            }}
                          >
                            {index + 1}
                          </Box>
                          <Typography variant="body2" fontWeight="medium">
                            {feature.replace(/_/g, ' ')}
                          </Typography>
                        </Box>
                        <Chip
                          label={`${(value * 100).toFixed(0)}%`}
                          size="small"
                          sx={{
                            fontWeight: 'bold',
                            bgcolor: index === 0 
                              ? alpha(theme.palette.primary.main, 0.1)
                              : alpha(theme.palette.grey[500], 0.1),
                            color: index === 0 
                              ? theme.palette.primary.main
                              : theme.palette.grey[700]
                          }}
                        />
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={value * 100} 
                        color={index === 0 ? "primary" : "secondary"}
                        sx={{ 
                          height: 8, 
                          borderRadius: 4,
                          bgcolor: alpha(theme.palette.grey[300], 0.4),
                          '& .MuiLinearProgress-bar': {
                            borderRadius: 4,
                            background: index === 0
                              ? `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.primary.light})`
                              : undefined,
                            boxShadow: index === 0
                              ? `0 0 10px ${alpha(theme.palette.primary.main, 0.4)}`
                              : undefined
                          }
                        }}
                      />
                    </Box>
                  ))}

                  {sortedFeatures.length > 5 && (
                    <Typography variant="caption" color="textSecondary" sx={{ display: 'block', textAlign: 'center', mt: 2 }}>
                      + {sortedFeatures.length - 5} more features with lower importance
                    </Typography>
                  )}
                </Box>
              ) : (
                <Box 
                  sx={{ 
                    p: 3, 
                    textAlign: 'center',
                    bgcolor: alpha(theme.palette.background.paper, 0.4),
                    borderRadius: 2
                  }}
                >
                  <Typography variant="body2" color="textSecondary">
                    No feature importance data available
                  </Typography>
                </Box>
              )}
              
              {Object.keys(noiseByRegion).length > 0 && (
                <>
                  <Divider sx={{ my: 3 }} />
                  
                  <Typography variant="subtitle1" fontWeight="medium" sx={{ mb: 1.5 }}>
                    Noise Analysis by Region
                  </Typography>
                  
                  <TableContainer sx={{ 
                    borderRadius: 2, 
                    overflow: 'hidden',
                    border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                  }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow sx={{ 
                          bgcolor: alpha(theme.palette.background.default, 0.5),
                        }}>
                          <TableCell sx={{ fontWeight: 'bold' }}>Region</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 'bold' }}>Energy</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 'bold' }}>Variance</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(noiseByRegion)
                          .sort(([, a], [, b]) => b.energy - a.energy)
                          .slice(0, 4)
                          .map(([region, metrics]) => (
                          <TableRow key={region} sx={{
                            '&:last-child td, &:last-child th': { border: 0 },
                            '&:nth-of-type(odd)': {
                              bgcolor: alpha(theme.palette.background.default, 0.3)
                            }
                          }}>
                            <TableCell component="th" scope="row">
                              {region.replace(/_/g, ' ')}
                            </TableCell>
                            <TableCell align="right">
                              <Box display="flex" alignItems="center" justifyContent="flex-end">
                                <Box 
                                  sx={{ 
                                    width: 50, 
                                    mr: 1,
                                    '& .MuiLinearProgress-root': {
                                      height: 6,
                                      borderRadius: 3,
                                      bgcolor: alpha(theme.palette.grey[300], 0.4)
                                    }
                                  }}
                                >
                                  <LinearProgress 
                                    variant="determinate" 
                                    value={metrics.energy * 100}
                                    color={metrics.energy > 0.5 ? "error" : "success"}
                                  />
                                </Box>
                                {(metrics.energy * 100).toFixed(1)}%
                              </Box>
                            </TableCell>
                            <TableCell align="right">
                              {metrics.std.toFixed(2)}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </>
              )}
            </Paper>
          </Grid>
          
          {/* Color Analysis & Visualizations */}
          <Grid item xs={12} md={6}>
            <Paper 
              elevation={0} 
              sx={{ 
                p: 3, 
                height: '100%',
                borderRadius: 3,
                overflow: 'hidden',
                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                backdropFilter: 'blur(5px)',
                background: alpha(theme.palette.background.paper, 0.7)
              }}
            >
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Box display="flex" alignItems="center">
                  <Avatar
                    sx={{
                      bgcolor: alpha(theme.palette.secondary.main, 0.1),
                      color: theme.palette.secondary.main,
                      width: 36,
                      height: 36,
                      mr: 1.5
                    }}
                  >
                    <PieChartIcon fontSize="small" />
                  </Avatar>
                  <Typography variant="h6" fontWeight="bold">
                    Visual Analysis
                  </Typography>
                </Box>
              </Box>
              
              <Grid container spacing={3}>
                {/* GradCAM visualization */}
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" gutterBottom>Attention Heatmap</Typography>
                  <Box 
                    sx={{ 
                      position: 'relative',
                      borderRadius: 2,
                      overflow: 'hidden',
                      boxShadow: `0 2px 8px ${alpha(theme.palette.common.black, 0.15)}`
                    }}
                  >
                    {gradcamPath ? (
                      <Box 
                        component="img" 
                        src={`http://localhost:5000/static/${gradcamPath}`}
                        alt="GradCAM" 
                        sx={{ 
                          width: '100%', 
                          height: 180, 
                          objectFit: 'cover',
                          display: 'block'
                        }}
                      />
                    ) : (
                      <Box 
                        sx={{ 
                          width: '100%', 
                          height: 180, 
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          bgcolor: alpha(theme.palette.background.default, 0.5)
                        }}
                      >
                        <Typography variant="body2" color="textSecondary">
                          No heatmap available
                        </Typography>
                      </Box>
                    )}
                  </Box>
                  <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 1 }}>
                    Highlights regions that influenced the model's decision
                  </Typography>
                </Grid>
                
                {/* Noise pattern visualization */}
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" gutterBottom>Noise Pattern</Typography>
                  <Box 
                    sx={{ 
                      position: 'relative',
                      borderRadius: 2,
                      overflow: 'hidden',
                      boxShadow: `0 2px 8px ${alpha(theme.palette.common.black, 0.15)}`
                    }}
                  >
                    {noisePath ? (
                      <Box 
                        component="img" 
                        src={`http://localhost:5000/static/${noisePath}`}
                        alt="Noise Pattern" 
                        sx={{ 
                          width: '100%', 
                          height: 180, 
                          objectFit: 'cover',
                          display: 'block'
                        }}
                      />
                    ) : (
                      <Box 
                        sx={{ 
                          width: '100%', 
                          height: 180, 
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          bgcolor: alpha(theme.palette.background.default, 0.5)
                        }}
                      >
                        <Typography variant="body2" color="textSecondary">
                          No noise analysis available
                        </Typography>
                      </Box>
                    )}
                  </Box>
                  <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 1 }}>
                    Revealing noise patterns that may indicate manipulation
                  </Typography>
                </Grid>
              </Grid>
              
              <Divider sx={{ my: 3 }} />
              
              {/* Color analysis */}
              <Typography variant="subtitle1" fontWeight="medium" sx={{ mb: 2 }}>
                Color Channel Analysis
              </Typography>
              
              <Grid container spacing={1} sx={{ mb: 2 }}>
                <Grid item xs={4}>
                  <Box 
                    sx={{ 
                      p: 1.5, 
                      borderRadius: 2, 
                      bgcolor: alpha(theme.palette.error.main, 0.1),
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                  >
                    <Box 
                      sx={{
                        width: 30,
                        height: 30,
                        bgcolor: theme.palette.error.main,
                        borderRadius: '50%',
                        mb: 1,
                        boxShadow: `0 0 10px ${alpha(theme.palette.error.main, 0.5)}`
                      }}
                    />
                    <Typography variant="subtitle2" align="center" color={theme.palette.error.dark}>Red</Typography>
                    <Typography variant="h6" align="center" fontWeight="bold">{redMean.toFixed(1)}</Typography>
                  </Box>
                </Grid>
                <Grid item xs={4}>
                  <Box 
                    sx={{ 
                      p: 1.5, 
                      borderRadius: 2, 
                      bgcolor: alpha(theme.palette.success.main, 0.1),
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                  >
                    <Box 
                      sx={{
                        width: 30,
                        height: 30,
                        bgcolor: theme.palette.success.main,
                        borderRadius: '50%',
                        mb: 1,
                        boxShadow: `0 0 10px ${alpha(theme.palette.success.main, 0.5)}`
                      }}
                    />
                    <Typography variant="subtitle2" align="center" color={theme.palette.success.dark}>Green</Typography>
                    <Typography variant="h6" align="center" fontWeight="bold">{greenMean.toFixed(1)}</Typography>
                  </Box>
                </Grid>
                <Grid item xs={4}>
                  <Box 
                    sx={{ 
                      p: 1.5, 
                      borderRadius: 2, 
                      bgcolor: alpha(theme.palette.info.main, 0.1),
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                  >
                    <Box 
                      sx={{
                        width: 30,
                        height: 30,
                        bgcolor: theme.palette.info.main,
                        borderRadius: '50%',
                        mb: 1,
                        boxShadow: `0 0 10px ${alpha(theme.palette.info.main, 0.5)}`
                      }}
                    />
                    <Typography variant="subtitle2" align="center" color={theme.palette.info.dark}>Blue</Typography>
                    <Typography variant="h6" align="center" fontWeight="bold">{blueMean.toFixed(1)}</Typography>
                  </Box>
                </Grid>
              </Grid>
              
              <Box 
                sx={{ 
                  p: 2, 
                  mt: 1,
                  borderRadius: 2,
                  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                  bgcolor: alpha(theme.palette.background.default, 0.3)
                }}
              >
                <Typography variant="subtitle2" gutterBottom>
                  Analysis Insights
                </Typography>

                <Box sx={{ mb: 1 }}>
                  <Typography variant="body2" display="flex" alignItems="center" sx={{ mb: 0.5 }}>
                    <TrendingUpIcon fontSize="small" sx={{ mr: 1, color: theme.palette.primary.main }} />
                    <strong>Dominant Channel:</strong> {dominantChannel.name} ({dominantChannel.value.toFixed(1)})
                  </Typography>
                </Box>
                
                <Box sx={{ mb: 1 }}>
                  <Typography variant="body2" sx={{ mb: 0.5 }}>
                    <strong>Statistical Consistency:</strong> {statisticalConsistency.toFixed(2)}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    {statisticalConsistency > 2 
                      ? "High consistency - typical of natural images" 
                      : "Low consistency - may indicate synthetic generation"}
                  </Typography>
                </Box>
                
                <Box>
                  <Typography variant="body2" sx={{ mb: 0.5 }}>
                    <strong>Key Finding:</strong> {modelResult.label === 'Real' 
                      ? "Natural texture patterns consistent with authentic imagery" 
                      : "Inconsistent noise patterns detected in key facial regions"}
                  </Typography>
                </Box>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}