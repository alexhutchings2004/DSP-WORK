import React from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Grid, 
  Divider, 
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Card,
  CardContent,
  useTheme,
  alpha,
  Avatar,
  Tooltip,
  Chip
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InfoIcon from '@mui/icons-material/Info';
import VisibilityIcon from '@mui/icons-material/Visibility';
import TuneIcon from '@mui/icons-material/Tune';
import FaceIcon from '@mui/icons-material/Face';
import WavesIcon from '@mui/icons-material/Waves';
import ColorLensIcon from '@mui/icons-material/ColorLens';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import { styled } from '@mui/material/styles';
const AnalysisCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  borderRadius: theme.shape.borderRadius * 2,
  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
  background: alpha(theme.palette.background.paper, 0.7),
  backdropFilter: 'blur(10px)',
  boxShadow: `0 4px 20px ${alpha(theme.palette.common.black, 0.05)}`,
  transition: 'transform 0.2s ease, box-shadow 0.2s ease',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: `0 8px 30px ${alpha(theme.palette.common.black, 0.1)}`
  }
}));
const FrequencyAnalysis = ({ frequencyData }) => {
  const theme = useTheme();
  
  if (!frequencyData) return null;
  
  const { low_freq_power, mid_freq_power, high_freq_power, num_spectral_peaks, freq_analysis_path } = frequencyData;
  
  const frequencies = [
    { name: 'Low Freq', value: low_freq_power },
    { name: 'Mid Freq', value: mid_freq_power },
    { name: 'High Freq', value: high_freq_power },
  ];
  const maxValue = Math.max(...frequencies.map(f => f.value));
  const normalizedFrequencies = frequencies.map(f => ({
    ...f,
    normalizedValue: f.value / (maxValue || 1)
  }));
  
  return (
    <Box>
      <Box display="flex" alignItems="center" mb={2}>
        <Avatar sx={{ bgcolor: alpha(theme.palette.secondary.main, 0.1), color: theme.palette.secondary.main, mr: 1.5 }}>
          <WavesIcon />
        </Avatar>
        <Typography variant="h6" fontWeight="bold">
          Frequency Analysis
        </Typography>
        <Tooltip title="Frequency analysis helps detect manipulation artifacts not visible to the human eye">
          <InfoIcon fontSize="small" sx={{ ml: 1, color: alpha(theme.palette.text.secondary, 0.7) }} />
        </Tooltip>
      </Box>
      
      <Typography variant="body2" color="textSecondary" paragraph>
        AI-generated images often show unusual patterns in the frequency domain that differ from real photographs.
      </Typography>
      
      {/* Frequency spectrum visualization */}
      {freq_analysis_path && (
        <Box 
          sx={{
            position: 'relative',
            mb: 3,
            borderRadius: 2,
            overflow: 'hidden',
            boxShadow: `0 4px 20px ${alpha(theme.palette.common.black, 0.15)}`
          }}
        >
          <Box 
            component="img"
            src={`http://localhost:5000/static/${freq_analysis_path}`}
            alt="Frequency Analysis"
            sx={{
              width: '100%',
              display: 'block'
            }}
          />
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              backgroundColor: alpha(theme.palette.background.paper, 0.6),
              backdropFilter: 'blur(4px)',
              p: 1
            }}
          >
            <Typography variant="caption" fontWeight="medium">
              Frequency Spectrum
            </Typography>
          </Box>
        </Box>
      )}
      
      {/* Frequency bands analysis */}
      <Box 
        sx={{ 
          p: 2, 
          borderRadius: 2, 
          bgcolor: alpha(theme.palette.background.default, 0.4),
          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`
        }}
      >
        <Typography variant="subtitle2" gutterBottom fontWeight="medium">
          Frequency Band Distribution
        </Typography>
        
        {normalizedFrequencies.map((freq, index) => (
          <Box key={freq.name} sx={{ mb: index !== normalizedFrequencies.length - 1 ? 2 : 0 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
              <Typography variant="body2" fontWeight="medium" sx={{ display: 'flex', alignItems: 'center' }}>
                <Box 
                  sx={{ 
                    width: 8, 
                    height: 8, 
                    borderRadius: '50%', 
                    mr: 1,
                    bgcolor: index === 0 
                      ? theme.palette.info.main 
                      : index === 1 
                        ? theme.palette.success.main 
                        : theme.palette.warning.main
                  }} 
                />
                {freq.name}
              </Typography>
              <Typography variant="body2" fontWeight="bold">
                {freq.value.toFixed(2)}
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={freq.normalizedValue * 100}
              sx={{
                height: 8,
                borderRadius: 4,
                bgcolor: alpha(theme.palette.grey[300], 0.4),
                '& .MuiLinearProgress-bar': {
                  borderRadius: 4,
                  background: index === 0 
                    ? `linear-gradient(90deg, ${theme.palette.info.dark}, ${theme.palette.info.main})`
                    : index === 1 
                      ? `linear-gradient(90deg, ${theme.palette.success.dark}, ${theme.palette.success.main})`
                      : `linear-gradient(90deg, ${theme.palette.warning.dark}, ${theme.palette.warning.main})`,
                  boxShadow: `0 0 10px ${alpha(
                    index === 0 
                      ? theme.palette.info.main 
                      : index === 1 
                        ? theme.palette.success.main 
                        : theme.palette.warning.main, 
                    0.3)}`
                }
              }}
            />
          </Box>
        ))}
      </Box>
      
      <Box 
        sx={{ 
          mt: 2, 
          p: 2, 
          borderRadius: 2,
          bgcolor: num_spectral_peaks > 5 
            ? alpha(theme.palette.warning.main, 0.1) 
            : alpha(theme.palette.success.main, 0.1),
          borderLeft: `4px solid ${num_spectral_peaks > 5 ? theme.palette.warning.main : theme.palette.success.main}`
        }}
      >
        <Typography variant="body2" display="flex" alignItems="center">
          <span style={{ fontWeight: 'bold', marginRight: '4px' }}>Spectral Peak Count:</span> 
          {num_spectral_peaks} 
          <Chip 
            size="small" 
            label={num_spectral_peaks > 5 ? 'High' : 'Normal'} 
            color={num_spectral_peaks > 5 ? 'warning' : 'success'} 
            sx={{ ml: 1, height: 20, '& .MuiChip-label': { px: 1, py: 0 } }}
          />
        </Typography>
        <Typography variant="caption" color="textSecondary">
          {num_spectral_peaks > 5 
            ? 'High spectral peak count may indicate manipulation or AI generation' 
            : 'Normal spectral peak count is consistent with natural images'}
        </Typography>
      </Box>
    </Box>
  );
};
const FaceRegionAnalysis = ({ faceData }) => {
  const theme = useTheme();
  
  if (!faceData || !faceData.feature_analysis || !faceData.feature_analysis.region_scores) return null;
  
  const { region_scores, overall_score } = faceData.feature_analysis;
  const sortedRegions = Object.entries(region_scores)
    .sort(([, a], [, b]) => b.manipulation_score - a.manipulation_score);
  
  return (
    <Box>
      <Box display="flex" alignItems="center" mb={2}>
        <Avatar sx={{ bgcolor: alpha(theme.palette.primary.main, 0.1), color: theme.palette.primary.main, mr: 1.5 }}>
          <FaceIcon />
        </Avatar>
        <Typography variant="h6" fontWeight="bold">
          Facial Region Analysis
        </Typography>
        <Tooltip title="Analysis of specific facial regions to detect localized manipulation">
          <InfoIcon fontSize="small" sx={{ ml: 1, color: alpha(theme.palette.text.secondary, 0.7) }} />
        </Tooltip>
      </Box>
      
      <Typography variant="body2" color="textSecondary" paragraph>
        Deepfake detection is performed across different facial regions. Higher scores indicate higher likelihood of manipulation.
      </Typography>
      
      <Box 
        sx={{ 
          mb: 4, 
          p: 2, 
          borderRadius: 2, 
          bgcolor: alpha(
            overall_score > 0.6 
              ? theme.palette.error.main 
              : overall_score > 0.4 
                ? theme.palette.warning.main 
                : theme.palette.success.main, 
            0.1
          ),
          border: `1px solid ${alpha(
            overall_score > 0.6 
              ? theme.palette.error.main 
              : overall_score > 0.4 
                ? theme.palette.warning.main 
                : theme.palette.success.main, 
            0.2
          )}`
        }}
      >
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="subtitle2">Overall Manipulation Score</Typography>
          <Typography variant="subtitle1" fontWeight="bold">
            {(overall_score * 100).toFixed(1)}%
          </Typography>
        </Box>
        
        <LinearProgress
          variant="determinate"
          value={overall_score * 100}
          sx={{
            height: 12,
            borderRadius: 6,
            bgcolor: alpha(theme.palette.grey[300], 0.4),
            '& .MuiLinearProgress-bar': {
              borderRadius: 6,
              background: overall_score > 0.6 
                ? `linear-gradient(90deg, ${theme.palette.error.dark}, ${theme.palette.error.main})` 
                : overall_score > 0.4 
                  ? `linear-gradient(90deg, ${theme.palette.warning.dark}, ${theme.palette.warning.main})` 
                  : `linear-gradient(90deg, ${theme.palette.success.dark}, ${theme.palette.success.main})`,
              boxShadow: `0 0 10px ${alpha(
                overall_score > 0.6 
                  ? theme.palette.error.main 
                  : overall_score > 0.4 
                    ? theme.palette.warning.main 
                    : theme.palette.success.main, 
                0.3)}`
            }
          }}
        />
        
        <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
          {overall_score > 0.6 
            ? 'High manipulation score indicates likely artificial generation or tampering' 
            : overall_score > 0.4 
              ? 'Medium score suggests possible manipulation, but not conclusive' 
              : 'Low score is consistent with authentic, unmodified facial features'}
        </Typography>
      </Box>
      
      <Grid container spacing={2}>
        {sortedRegions.map(([regionName, data], index) => (
          <Grid item xs={12} sm={6} key={regionName}>
            <Box sx={{ 
              p: 2, 
              borderRadius: 2,
              bgcolor: alpha(theme.palette.background.default, 0.4),
              border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
              '&:hover': {
                bgcolor: alpha(theme.palette.background.paper, 0.3)
              }
            }}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="body2" fontWeight="medium" sx={{ textTransform: 'capitalize' }}>
                  {regionName.replace(/_/g, ' ')}
                </Typography>
                <Chip 
                  size="small" 
                  label={`${(data.manipulation_score * 100).toFixed(0)}%`}
                  color={data.manipulation_score > 0.6 ? 'error' : data.manipulation_score > 0.4 ? 'warning' : 'success'}
                  variant="outlined"
                />
              </Box>
              
              <LinearProgress
                variant="determinate"
                value={data.manipulation_score * 100}
                sx={{
                  height: 6,
                  borderRadius: 3,
                  bgcolor: alpha(theme.palette.grey[300], 0.4),
                  '& .MuiLinearProgress-bar': {
                    bgcolor: data.manipulation_score > 0.6 
                      ? theme.palette.error.main 
                      : data.manipulation_score > 0.4 
                        ? theme.palette.warning.main 
                        : theme.palette.success.main
                  }
                }}
              />
              
              {/* Feature details in smaller text */}
              {data.features && (
                <Box 
                  sx={{ 
                    mt: 1.5, 
                    p: 1, 
                    borderRadius: 1, 
                    bgcolor: alpha(theme.palette.background.default, 0.5),
                    display: 'flex',
                    justifyContent: 'space-between'
                  }}
                >
                  <Tooltip title="Entropy measures randomness of pixel values - synthetic images often have lower entropy">
                    <Typography variant="caption">
                      <span style={{ fontWeight: 'bold' }}>Entropy:</span> {data.features.entropy.toFixed(2)}
                    </Typography>
                  </Tooltip>
                  <Tooltip title="Edge strength measures sharpness of facial features - manipulated images may have unusual edge patterns">
                    <Typography variant="caption">
                      <span style={{ fontWeight: 'bold' }}>Edge:</span> {data.features.edge_strength.toFixed(2)}
                    </Typography>
                  </Tooltip>
                </Box>
              )}
            </Box>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};
const ColorAnalysis = ({ histogramStats }) => {
  const theme = useTheme();
  
  if (!histogramStats || (!histogramStats.red && !histogramStats.luminance)) return null;
  const colorValues = {
    red: histogramStats.red?.mean || 0,
    green: histogramStats.green?.mean || 0,
    blue: histogramStats.blue?.mean || 0
  };
  
  const dominantChannel = Object.entries(colorValues)
    .sort((a, b) => b[1] - a[1])[0][0];
  const maxValue = Math.max(...Object.values(colorValues));
  const minValue = Math.min(...Object.values(colorValues));
  const colorBalance = minValue > 0 ? minValue / maxValue : 0;
  
  return (
    <Box>
      <Box display="flex" alignItems="center" mb={2}>
        <Avatar sx={{ bgcolor: alpha(theme.palette.info.main, 0.1), color: theme.palette.info.main, mr: 1.5 }}>
          <ColorLensIcon />
        </Avatar>
        <Typography variant="h6" fontWeight="bold">
          Color Analysis
        </Typography>
        <Tooltip title="Color distribution analysis can reveal inconsistencies in AI-generated images">
          <InfoIcon fontSize="small" sx={{ ml: 1, color: alpha(theme.palette.text.secondary, 0.7) }} />
        </Tooltip>
      </Box>
      
      <Typography variant="body2" color="textSecondary" paragraph>
        Statistical analysis of color channels helps identify inconsistencies typical of synthetic images.
      </Typography>
      
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Box sx={{ 
            p: 2.5, 
            borderRadius: 2,
            height: '100%',
            bgcolor: alpha(theme.palette.background.default, 0.4),
            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`
          }}>
            <Typography variant="subtitle2" gutterBottom>
              Brightness & Contrast
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Box sx={{ textAlign: 'center', p: 1, borderRadius: 2, bgcolor: alpha(theme.palette.common.white, 0.1), mb: 1 }}>
                  <Typography variant="h5" fontWeight="bold" gutterBottom>
                    {histogramStats.luminance?.mean.toFixed(0)}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Mean Brightness
                  </Typography>
                </Box>
                <Typography variant="caption" color="textSecondary" align="center" display="block">
                  {histogramStats.luminance?.mean > 200 ? 'Very bright' : 
                   histogramStats.luminance?.mean > 150 ? 'Bright' :
                   histogramStats.luminance?.mean > 100 ? 'Medium' :
                   histogramStats.luminance?.mean > 50 ? 'Dark' : 'Very dark'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Box sx={{ textAlign: 'center', p: 1, borderRadius: 2, bgcolor: alpha(theme.palette.common.white, 0.1), mb: 1 }}>
                  <Typography variant="h5" fontWeight="bold" gutterBottom>
                    {histogramStats.luminance?.std_dev.toFixed(0)}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Contrast (StdDev)
                  </Typography>
                </Box>
                <Typography variant="caption" color="textSecondary" align="center" display="block">
                  {histogramStats.luminance?.std_dev > 70 ? 'High contrast' : 
                   histogramStats.luminance?.std_dev > 40 ? 'Normal contrast' : 'Low contrast'}
                </Typography>
              </Grid>
            </Grid>
            
            {histogramStats.luminance?.std_dev < 35 && (
              <Box 
                sx={{ 
                  mt: 2, 
                  p: 1.5, 
                  borderRadius: 1, 
                  bgcolor: alpha(theme.palette.warning.main, 0.1),
                  borderLeft: `4px solid ${theme.palette.warning.main}`
                }}
              >
                <Typography variant="caption" color="textSecondary">
                  Low contrast may indicate image manipulation or AI generation
                </Typography>
              </Box>
            )}
          </Box>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Box sx={{ 
            p: 2.5, 
            borderRadius: 2,
            height: '100%',
            bgcolor: alpha(theme.palette.background.default, 0.4),
            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`
          }}>
            <Typography variant="subtitle2" gutterBottom>
              Color Channel Balance
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={4}>
                <Box 
                  sx={{ 
                    p: 1, 
                    textAlign: 'center', 
                    bgcolor: alpha(theme.palette.error.main, 0.1),
                    borderRadius: 2,
                    border: dominantChannel === 'red' ? `2px solid ${theme.palette.error.main}` : 'none'
                  }}
                >
                  <Typography variant="caption" color="error" fontWeight="bold" sx={{ textTransform: 'uppercase' }}>
                    Red
                  </Typography>
                  <Typography variant="h6" fontWeight="medium" color="error.dark">
                    {colorValues.red.toFixed(0)}
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box 
                  sx={{ 
                    p: 1, 
                    textAlign: 'center', 
                    bgcolor: alpha(theme.palette.success.main, 0.1),
                    borderRadius: 2,
                    border: dominantChannel === 'green' ? `2px solid ${theme.palette.success.main}` : 'none'
                  }}
                >
                  <Typography variant="caption" color="success" fontWeight="bold" sx={{ textTransform: 'uppercase' }}>
                    Green
                  </Typography>
                  <Typography variant="h6" fontWeight="medium" color="success.dark">
                    {colorValues.green.toFixed(0)}
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={4}>
                <Box 
                  sx={{ 
                    p: 1, 
                    textAlign: 'center', 
                    bgcolor: alpha(theme.palette.info.main, 0.1),
                    borderRadius: 2,
                    border: dominantChannel === 'blue' ? `2px solid ${theme.palette.info.main}` : 'none'
                  }}
                >
                  <Typography variant="caption" color="info" fontWeight="bold" sx={{ textTransform: 'uppercase' }}>
                    Blue
                  </Typography>
                  <Typography variant="h6" fontWeight="medium" color="info.dark">
                    {colorValues.blue.toFixed(0)}
                  </Typography>
                </Box>
              </Grid>
            </Grid>
            
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <span>Color Balance:</span>
                <span>{(colorBalance * 100).toFixed(0)}%</span>
              </Typography>
              <LinearProgress
                variant="determinate"
                value={colorBalance * 100}
                sx={{
                  height: 6,
                  borderRadius: 3,
                  bgcolor: alpha(theme.palette.grey[300], 0.4),
                  '& .MuiLinearProgress-bar': {
                    background: `linear-gradient(90deg, 
                      ${theme.palette.error.main}, 
                      ${theme.palette.success.main}, 
                      ${theme.palette.info.main})`,
                  }
                }}
              />
              <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 0.5 }}>
                {colorBalance > 0.8 ? 'Well-balanced colors typical of natural images' : 
                 colorBalance > 0.5 ? 'Moderate color balance' : 'Unbalanced colors may indicate manipulation'}
              </Typography>
            </Box>
          </Box>
        </Grid>
      </Grid>
      
      <Accordion 
        elevation={0}
        sx={{ 
          '&.MuiPaper-root': { 
            bgcolor: alpha(theme.palette.background.default, 0.4),
            border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
            borderRadius: '8px !important',
            overflow: 'hidden',
          },
          '&::before': {
            display: 'none',
          }
        }}
      >
        <AccordionSummary 
          expandIcon={<ExpandMoreIcon />}
          sx={{ 
            borderRadius: '8px !important',
            '&.Mui-expanded': {
              borderBottom: `1px solid ${alpha(theme.palette.divider, 0.1)}`
            }
          }}
        >
          <Typography fontWeight="medium">Detailed RGB Channel Statistics</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            {['red', 'green', 'blue'].map(channel => histogramStats[channel] && (
              <Grid item xs={12} sm={4} key={channel}>
                <Box 
                  sx={{ 
                    p: 1.5, 
                    borderRadius: 2,
                    bgcolor: alpha(
                      channel === 'red' ? theme.palette.error.main : 
                      channel === 'green' ? theme.palette.success.main : 
                      theme.palette.info.main, 0.05
                    ),
                    border: `1px solid ${alpha(
                      channel === 'red' ? theme.palette.error.main : 
                      channel === 'green' ? theme.palette.success.main : 
                      theme.palette.info.main, 0.2
                    )}`,
                    height: '100%'
                  }}
                >
                  <Typography 
                    variant="subtitle2" 
                    sx={{ 
                      textTransform: 'uppercase',
                      color: channel === 'red' ? theme.palette.error.main : 
                             channel === 'green' ? theme.palette.success.main : 
                             theme.palette.info.main
                    }}
                  >
                    {channel}
                  </Typography>
                  
                  <Grid container spacing={2} sx={{ mt: 0.5 }}>
                    <Grid item xs={6}>
                      <Typography variant="caption" display="block" color="textSecondary">
                        Mean
                      </Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {histogramStats[channel].mean.toFixed(1)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" display="block" color="textSecondary">
                        StdDev
                      </Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {histogramStats[channel].std_dev.toFixed(1)}
                      </Typography>
                    </Grid>
                  </Grid>
                  
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="caption" display="block" color="textSecondary">
                      Range
                    </Typography>
                    <Typography variant="body2">
                      {histogramStats[channel].min.toFixed(0)} - {histogramStats[channel].max.toFixed(0)}
                    </Typography>
                  </Box>
                </Box>
              </Grid>
            ))}
          </Grid>
          
          {(histogramStats.red?.std_dev < 35 || histogramStats.green?.std_dev < 35 || histogramStats.blue?.std_dev < 35) && (
            <Box 
              sx={{ 
                mt: 2, 
                p: 1.5,
                borderRadius: 1, 
                bgcolor: alpha(theme.palette.warning.main, 0.1),
                border: `1px solid ${alpha(theme.palette.warning.main, 0.2)}`,
              }}
            >
              <Typography variant="body2" color="warning.dark" fontWeight="medium">
                Low variation detected in {[
                  histogramStats.red?.std_dev < 35 ? 'Red' : null,
                  histogramStats.green?.std_dev < 35 ? 'Green' : null,
                  histogramStats.blue?.std_dev < 35 ? 'Blue' : null
                ].filter(Boolean).join(', ')} channel{histogramStats.red?.std_dev < 35 && histogramStats.green?.std_dev < 35 && histogramStats.blue?.std_dev < 35 ? 's' : ''}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Low variation in color channels may indicate synthetic generation or image manipulation
              </Typography>
            </Box>
          )}
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};
const NoisePatternAnalysis = ({ noisePath }) => {
  const theme = useTheme();
  
  if (!noisePath) return null;
  
  return (
    <Box>
      <Box display="flex" alignItems="center" mb={2}>
        <Avatar sx={{ bgcolor: alpha(theme.palette.error.main, 0.1), color: theme.palette.error.main, mr: 1.5 }}>
          <TuneIcon />
        </Avatar>
        <Typography variant="h6" fontWeight="bold">
          Noise Pattern Analysis
        </Typography>
        <Tooltip title="Analyzing noise patterns helps detect traces of AI manipulation">
          <InfoIcon fontSize="small" sx={{ ml: 1, color: alpha(theme.palette.text.secondary, 0.7) }} />
        </Tooltip>
      </Box>
      
      <Typography variant="body2" color="textSecondary" paragraph>
        Natural images have distinct noise patterns. AI-generated or manipulated images often show unnatural noise signatures.
      </Typography>
      
      <Box 
        sx={{
          position: 'relative',
          borderRadius: 2,
          overflow: 'hidden',
          boxShadow: `0 4px 20px ${alpha(theme.palette.common.black, 0.15)}`,
          mb: 2
        }}
      >
        <Box 
          component="img"
          src={`http://localhost:5000/static/${noisePath}`}
          alt="Noise Analysis"
          sx={{
            width: '100%',
            display: 'block'
          }}
        />
        
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            backgroundColor: alpha(theme.palette.background.paper, 0.6),
            backdropFilter: 'blur(4px)',
            p: 1,
            display: 'flex',
            alignItems: 'center'
          }}
        >
          <VisibilityIcon fontSize="small" sx={{ mr: 0.5 }} />
          <Typography variant="caption" fontWeight="medium">
            Noise Pattern Visualization
          </Typography>
        </Box>
      </Box>
      
      <Box 
        sx={{ 
          p: 2, 
          borderRadius: 2, 
          bgcolor: alpha(theme.palette.background.default, 0.4),
          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`
        }}
      >
        <Typography variant="subtitle2" gutterBottom fontWeight="medium">
          Key Indicators
        </Typography>
        
        <Grid container spacing={1.5}>
          <Grid item xs={12}>
            <Box 
              sx={{ 
                p: 1.5, 
                borderRadius: 1, 
                bgcolor: alpha(theme.palette.info.main, 0.05),
                borderLeft: `4px solid ${theme.palette.info.main}`
              }}
            >
              <Typography variant="body2">
                <span style={{ fontWeight: 'bold' }}>Pattern Regularity:</span> Uniform grid-like patterns often indicate AI-generated images, while natural images typically have random, grain-like noise patterns.
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12}>
            <Box 
              sx={{ 
                p: 1.5, 
                borderRadius: 1, 
                bgcolor: alpha(theme.palette.info.main, 0.05),
                borderLeft: `4px solid ${theme.palette.info.main}`
              }}
            >
              <Typography variant="body2">
                <span style={{ fontWeight: 'bold' }}>Local Consistency:</span> Look for abnormal noise levels in specific regions that differ from the rest of the image, which may indicate localized editing.
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
};
const ModelExplainabilityPanel = ({ results, histogramStats }) => {
  const theme = useTheme();
  
  if (!results || results.length === 0) {
    return (
      <AnalysisCard sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" color="textSecondary">
          No analysis results available
        </Typography>
        <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
          The analysis didn't return valid results. Please try again or check if the model is working properly.
        </Typography>
      </AnalysisCard>
    );
  }
  const mainResult = results[0];
  const frequencyData = mainResult.frequency_analysis;
  const gradcamPath = mainResult.gradcam || mainResult.gradcam_path;
  const noisePath = mainResult.noise_path;
  const faceDetected = mainResult.face_path || (mainResult.face_regions && mainResult.face_regions.length > 0);
  
  const predictionLabel = mainResult.label || "Unknown";
  const predictionConfidence = mainResult.confidence;
  if (predictionConfidence === undefined || predictionConfidence === null || isNaN(predictionConfidence)) {
    return (
      <AnalysisCard sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" color="error">
          Error in Analysis Data
        </Typography>
        <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
          The analysis results contain invalid confidence values. Please try again with a different image.
        </Typography>
      </AnalysisCard>
    );
  }
  
  const isReal = predictionLabel === 'Real';
  
  return (
    <Box>
      {/* Prediction Summary Card */}
      <Card 
        elevation={0}
        sx={{ 
          mb: 4,
          borderRadius: 3,
          overflow: 'hidden',
          border: `1px solid ${alpha(isReal ? theme.palette.success.main : theme.palette.error.main, 0.3)}`,
          boxShadow: `0 8px 32px ${alpha(theme.palette.common.black, 0.1)}`
        }}
      >
        <Box 
          sx={{
            p: 3,
            background: isReal
              ? `linear-gradient(135deg, ${alpha(theme.palette.success.dark, 0.95)}, ${alpha(theme.palette.success.main, 0.85)})`
              : `linear-gradient(135deg, ${alpha(theme.palette.error.dark, 0.95)}, ${alpha(theme.palette.error.main, 0.85)})`,
            color: theme.palette.common.white,
            position: 'relative',
            overflow: 'hidden'
          }}
        >
          {/* Background pattern */}
          <Box 
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              opacity: 0.05,
              backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'%3E%3Cg fill='%23ffffff' fill-opacity='1'%3E%3Cpath fill-rule='evenodd' d='M0 0h40v40H0V0zm40 40h40v40H40V40zm0-40h2l-2 2V0zm0 4l4-4h2l-6 6V4zm0 4l8-8h2L40 10V8zm0 4L52 0h2L40 14v-2zm0 4L56 0h2L40 18v-2zm0 4L60 0h2L40 22v-2zm0 4L64 0h2L40 26v-2zm0 4L68 0h2L40 30v-2zm0 4L72 0h2L40 34v-2zm0 4L76 0h2L40 38v-2zm0 4L80 0v2L42 40h-2zm4 0L80 4v2L46 40h-2zm4 0L80 8v2L50 40h-2zm4 0l28-28v2L54 40h-2zm4 0l24-24v2L58 40h-2zm4 0l20-20v2L62 40h-2zm4 0l16-16v2L66 40h-2zm4 0l12-12v2L70 40h-2zm4 0l8-8v2l-6 6h-2zm4 0l4-4v2l-2 2h-2z'/%3E%3C/g%3E%3C/svg%3E\")"
            }}
          />
          
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={7}>
              <Box display="flex" alignItems="center">
                <Avatar
                  sx={{
                    bgcolor: theme.palette.common.white,
                    color: isReal ? theme.palette.success.main : theme.palette.error.main,
                    mr: 2,
                    width: 56,
                    height: 56,
                    boxShadow: `0 4px 10px ${alpha('#000000', 0.2)}`
                  }}
                >
                  {isReal ? <CheckCircleOutlineIcon fontSize="large" /> : <ErrorOutlineIcon fontSize="large" />}
                </Avatar>
                <Box>
                  <Typography variant="overline" sx={{ opacity: 0.9, letterSpacing: 1 }}>
                    Analysis Result
                  </Typography>
                  <Typography variant="h4" fontWeight="bold" sx={{ textShadow: '0 2px 4px rgba(0,0,0,0.2)' }}>
                    {isReal ? 'Authentic Image' : 'Deepfake Detected'}
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 0.5, opacity: 0.9 }}>
                    {isReal 
                      ? 'Analysis indicates this is likely an authentic, unaltered image' 
                      : 'Analysis suggests this image has been manipulated or AI-generated'}
                  </Typography>
                </Box>
              </Box>
            </Grid>
            
            <Grid item xs={12} md={5}>
              <Box>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2" fontWeight="medium">
                    Model Confidence
                  </Typography>
                  <Typography variant="h6" fontWeight="bold">
                    {(predictionConfidence * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={predictionConfidence * 100} 
                  sx={{
                    height: 10,
                    borderRadius: 5,
                    bgcolor: alpha(theme.palette.common.white, 0.3),
                    '& .MuiLinearProgress-bar': {
                      bgcolor: theme.palette.common.white,
                      boxShadow: `0 0 10px ${alpha(theme.palette.common.white, 0.5)}`
                    }
                  }}
                />
                <Typography variant="caption" sx={{ display: 'block', mt: 1, fontStyle: 'italic' }}>
                  {predictionConfidence > 0.85 
                    ? "Very high confidence level - strong indicators supporting this classification"
                    : predictionConfidence > 0.7 
                      ? "High confidence level - multiple indicators align with this classification"
                      : predictionConfidence > 0.55
                        ? "Moderate confidence level - some indicators present but not definitive"
                        : "Lower confidence level - prediction should be treated with caution"
                  }
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Box>
        
        <Box p={3} sx={{ bgcolor: alpha(theme.palette.background.paper, 0.7), backdropFilter: 'blur(10px)' }}>
          <Typography variant="h6" gutterBottom fontWeight="bold">
            Model Interpretation
          </Typography>
          
          <Typography variant="body2" paragraph>
            {isReal ? (
              <>
                The EfficientNet model has classified this image as <strong>authentic</strong> with {(predictionConfidence * 100).toFixed(1)}% confidence. 
                The analysis shows natural facial feature consistency, appropriate noise distribution,
                and frequency characteristics that align with authentic photographs rather than 
                AI-generated or manipulated content.
              </>
            ) : (
              <>
                The EfficientNet model has classified this image as <strong>manipulated</strong> with {(predictionConfidence * 100).toFixed(1)}% confidence. 
                This assessment is based on multiple factors including inconsistencies in facial features, 
                unnatural noise patterns, and unusual frequency distribution in the image that are 
                characteristic of AI-generated or manipulated content.
              </>
            )}
          </Typography>
          
          <Box 
            sx={{ 
              p: 2, 
              mt: 1, 
              borderRadius: 2,
              bgcolor: alpha(theme.palette.info.main, 0.1),
              border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`
            }}
          >
            <Typography variant="body2" display="flex" alignItems="center" color="info.dark">
              <InfoIcon fontSize="small" sx={{ mr: 1 }} />
              <strong>Note:</strong> No AI detection system is 100% accurate. Multiple verification methods should be used for critical assessments.
            </Typography>
          </Box>
        </Box>
      </Card>

      {/* Analysis Grid */}
      <Typography variant="h5" fontWeight="bold" gutterBottom sx={{ mb: 3, pl: 1 }}>
        Detailed Analysis Components
      </Typography>
      
      <Grid container spacing={3}>
        {/* GradCAM visualization */}
        {gradcamPath && (
          <Grid item xs={12} md={6}>
            <AnalysisCard>
              <Box display="flex" alignItems="center" mb={2}>
                <Avatar sx={{ bgcolor: alpha(theme.palette.warning.main, 0.1), color: theme.palette.warning.main, mr: 1.5 }}>
                  <VisibilityIcon />
                </Avatar>
                <Typography variant="h6" fontWeight="bold">
                  Attention Heatmap
                </Typography>
                <Tooltip title="Shows regions the model focused on when making its decision">
                  <InfoIcon fontSize="small" sx={{ ml: 1, color: alpha(theme.palette.text.secondary, 0.7) }} />
                </Tooltip>
              </Box>
              
              <Typography variant="body2" color="textSecondary" paragraph>
                The heatmap reveals areas that influenced the model's decision most strongly. Red areas had the greatest impact.
              </Typography>
              
              <Box 
                sx={{
                  position: 'relative',
                  borderRadius: 2,
                  overflow: 'hidden',
                  boxShadow: `0 4px 20px ${alpha(theme.palette.common.black, 0.15)}`,
                  mb: 2
                }}
              >
                <Box 
                  component="img"
                  src={`http://localhost:5000/static/${gradcamPath}`}
                  alt="Model Attention Map"
                  sx={{
                    width: '100%',
                    display: 'block'
                  }}
                />
              </Box>
              
              <Box 
                sx={{ 
                  p: 2, 
                  borderRadius: 2, 
                  bgcolor: alpha(theme.palette.background.default, 0.4),
                  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`
                }}
              >
                <Typography variant="body2" fontWeight="medium">
                  Interpretation Guide
                </Typography>
                <Grid container spacing={1} sx={{ mt: 1 }}>
                  <Grid item xs={6}>
                    <Box display="flex" alignItems="center">
                      <Box 
                        sx={{ 
                          width: 12, 
                          height: 12, 
                          borderRadius: '50%', 
                          mr: 1, 
                          bgcolor: theme.palette.error.main 
                        }} 
                      />
                      <Typography variant="caption">High attention</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6}>
                    <Box display="flex" alignItems="center">
                      <Box 
                        sx={{ 
                          width: 12, 
                          height: 12, 
                          borderRadius: '50%', 
                          mr: 1, 
                          bgcolor: theme.palette.warning.light 
                        }} 
                      />
                      <Typography variant="caption">Medium attention</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6}>
                    <Box display="flex" alignItems="center">
                      <Box 
                        sx={{ 
                          width: 12, 
                          height: 12, 
                          borderRadius: '50%', 
                          mr: 1, 
                          bgcolor: theme.palette.info.light 
                        }} 
                      />
                      <Typography variant="caption">Lower attention</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6}>
                    <Box display="flex" alignItems="center">
                      <Box 
                        sx={{ 
                          width: 12, 
                          height: 12, 
                          borderRadius: '50%', 
                          mr: 1, 
                          bgcolor: theme.palette.success.light 
                        }} 
                      />
                      <Typography variant="caption">Minimal attention</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>
            </AnalysisCard>
          </Grid>
        )}
        
        {/* Noise pattern analysis */}
        {noisePath && (
          <Grid item xs={12} md={6}>
            <AnalysisCard>
              <NoisePatternAnalysis noisePath={noisePath} />
            </AnalysisCard>
          </Grid>
        )}
        
        {/* Face region analysis */}
        {faceDetected && (
          <Grid item xs={12} md={6}>
            <AnalysisCard>
              <FaceRegionAnalysis faceData={mainResult} />
            </AnalysisCard>
          </Grid>
        )}
        
        {/* Frequency domain analysis */}
        {frequencyData && !frequencyData.error && (
          <Grid item xs={12} md={6}>
            <AnalysisCard>
              <FrequencyAnalysis frequencyData={frequencyData} />
            </AnalysisCard>
          </Grid>
        )}
        
        {/* Color statistics analysis */}
        {histogramStats && Object.keys(histogramStats).length > 0 && (
          <Grid item xs={12}>
            <AnalysisCard>
              <ColorAnalysis histogramStats={histogramStats} />
            </AnalysisCard>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default ModelExplainabilityPanel;