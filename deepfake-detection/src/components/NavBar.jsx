import React from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Link, 
  Box, 
  Button, 
  Chip,
  IconButton, 
  Tooltip,
  useTheme,
  alpha,
  Container,
  Avatar
} from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import InfoIcon from '@mui/icons-material/Info';
import ShieldIcon from '@mui/icons-material/Shield';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import BiotechIcon from '@mui/icons-material/Biotech';
import { useModels } from '../contexts/ModelsContext';

const Navbar = () => {
  const { 
    availableModels, 
    selectedModels, 
    comparisonMode,
    toggleComparisonMode
  } = useModels();
  
  const theme = useTheme();

  return (
    <AppBar 
      position="sticky" 
      elevation={0}
      sx={{ 
        backgroundColor: 'transparent',
        backdropFilter: 'blur(20px)',
        borderBottom: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
        boxShadow: `0 4px 30px ${alpha(theme.palette.common.black, 0.1)}`,
        background: `linear-gradient(to bottom, ${alpha(theme.palette.background.paper, 0.8)}, ${alpha(theme.palette.background.paper, 0.6)})`
      }}
    >
      <Container maxWidth="xl">
        <Toolbar sx={{ py: 1 }}>
          {/* Logo & App Title */}
          <Box display="flex" alignItems="center">
            <Box
              sx={{
                width: 44,
                height: 44,
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.8)}, ${alpha(theme.palette.primary.dark, 0.9)})`,
                boxShadow: `0 4px 12px ${alpha(theme.palette.primary.main, 0.4)}`,
                mr: 1.5
              }}
            >
              <ShieldIcon 
                sx={{ 
                  fontSize: 28, 
                  color: 'white',
                  filter: 'drop-shadow(0 0 8px rgba(124, 58, 237, 0.5))'
                }} 
              />
            </Box>
            
            <Box>
              <Typography 
                variant="h5" 
                fontWeight="bold"
                sx={{
                  background: 'linear-gradient(90deg, #7c3aed, #06b6d4)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  textShadow: '0px 2px 4px rgba(0,0,0,0.2)',
                  lineHeight: 1.2
                }}
              >
                DeepFake Detector
              </Typography>
              <Chip 
                label="AI-Powered" 
                size="small" 
                color="secondary" 
                sx={{ 
                  fontWeight: 'bold',
                  background: 'linear-gradient(45deg, #06b6d4, #0891b2)',
                  borderColor: 'transparent',
                  height: 20,
                  '& .MuiChip-label': {
                    px: 1,
                    py: 0
                  }
                }} 
              />
            </Box>
          </Box>
          
          {/* Center section - model badges */}
          <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center' }}>
            {availableModels.length > 0 && (
              <Box 
                sx={{ 
                  display: { xs: 'none', md: 'flex' }, 
                  gap: 2, 
                  alignItems: 'center', 
                  backgroundColor: alpha(theme.palette.background.paper, 0.4),
                  backdropFilter: 'blur(8px)',
                  border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                  borderRadius: 8,
                  py: 0.5,
                  px: 2,
                }}
              >
                <Box sx={{ display: "flex", alignItems: "center" }}>
                  <BiotechIcon fontSize="small" sx={{ mr: 1, color: alpha(theme.palette.text.secondary, 0.8) }} />
                  <Typography variant="body2" color="text.secondary" fontWeight="medium">
                    Models:
                  </Typography>
                </Box>
                
                {availableModels.map(model => (
                  <Button
                    key={model.id}
                    variant={selectedModels.includes(model.id) ? "contained" : "outlined"}
                    size="small"
                    color={model.id === "efficientnet" ? "primary" : "secondary"}
                    sx={{ 
                      borderRadius: 6,
                      fontSize: '0.75rem',
                      py: 0.5,
                      px: 1.5,
                      minWidth: 0,
                      textTransform: 'none',
                      fontWeight: 600,
                      boxShadow: selectedModels.includes(model.id) ? 
                        `0 2px 8px ${alpha(model.id === "efficientnet" ? theme.palette.primary.main : theme.palette.secondary.main, 0.4)}` : 'none'
                    }}
                  >
                    {model.name || model.id}
                  </Button>
                ))}
              </Box>
            )}
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {availableModels.length > 0 && (
              <Tooltip title={comparisonMode ? "Disable model comparison" : "Enable model comparison"}>
                <Button
                  color={comparisonMode ? "primary" : "secondary"}
                  variant={comparisonMode ? "contained" : "outlined"}
                  startIcon={<CompareArrowsIcon />}
                  onClick={toggleComparisonMode}
                  size="small"
                  sx={{ 
                    borderRadius: '20px',
                    px: 2,
                    py: 0.5
                  }}
                >
                  {comparisonMode ? "Comparison Active" : "Compare Models"}
                </Button>
              </Tooltip>
            )}

            <Tooltip title="About this tool">
              <IconButton 
                sx={{ 
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  '&:hover': {
                    bgcolor: alpha(theme.palette.primary.main, 0.2),
                  },
                  transition: 'all 0.2s ease'
                }}
              >
                <InfoIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="GitHub Repository">
              <IconButton 
                component={Link}
                href="https://github.com/yourusername/deepfake-detection"
                target="_blank"
                rel="noopener noreferrer"
                sx={{ 
                  bgcolor: alpha(theme.palette.secondary.main, 0.1),
                  '&:hover': {
                    bgcolor: alpha(theme.palette.secondary.main, 0.2),
                  },
                  transition: 'all 0.2s ease'
                }}
              >
                <GitHubIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            
            {/* User Avatar - Optional */}
            <Avatar
              alt="User"
              sx={{ 
                width: 36, 
                height: 36,
                display: { xs: 'none', sm: 'flex' },
                border: `2px solid ${alpha(theme.palette.primary.main, 0.6)}`,
                boxShadow: `0 0 0 2px ${theme.palette.background.paper}`,
                background: `linear-gradient(135deg, ${alpha(theme.palette.primary.light, 0.7)}, ${alpha(theme.palette.secondary.light, 0.7)})`,
                cursor: 'pointer'
              }}
            >
              A
            </Avatar>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Navbar;
