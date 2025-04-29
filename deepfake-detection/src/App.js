import React from 'react';
import { CssBaseline, Box, ThemeProvider, alpha } from '@mui/material';
import Navbar from './components/NavBar'; // Fixed casing to match actual file name
import ImageDrop from './components/ImageDrop';
import ModelsProvider, { useModels } from './contexts/ModelsContext.jsx'; // Fixed file extension and import structure
import theme from './theme/theme';
import './App.css'; // Added back now that the file exists

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ModelsProvider>
        <Box
          sx={{
            minHeight: '100vh',
            position: 'relative',
            overflow: 'hidden',
            backgroundColor: theme.palette.background.default,
            background: `radial-gradient(circle at 10% 20%, ${alpha('#3b82f6', 0.03)} 0%, ${alpha('#7c3aed', 0.01)} 90.2%)`,
            "&::before": {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 0,
              opacity: 0.4,
              backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100'%3E%3Cg fill-rule='evenodd'%3E%3Cg fill='%237c3aed' fill-opacity='0.03'%3E%3Cpath opacity='.5' d='M96 95h4v1h-4v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4h-9v4h-1v-4H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15v-9H0v-1h15V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h9V0h1v15h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9h4v1h-4v9zm-1 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm9-10v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm9-10v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm9-10v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-10 0v-9h-9v9h9zm-9-10h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9zm10 0h9v-9h-9v9z'/%3E%3Cpath d='M6 5V0H5v5H0v1h5v94h1V6h94V5H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E\")",
            },
            "&::after": {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 0,
              background: `
                radial-gradient(circle at 50% 0%, ${alpha('#7c3aed', 0.15)}, transparent 25%),
                radial-gradient(circle at 15% 50%, ${alpha('#06b6d4', 0.1)}, transparent 35%)
              `
            }
          }}
        >
          <Navbar />
          <Box
            sx={{
              paddingTop: 3,
              paddingX: { xs: 2, sm: 4, md: 6, lg: 8 },
              position: 'relative',
              zIndex: 1,
              maxWidth: '1600px',
              margin: '0 auto',
              pb: 8
            }}
          >
            <ImageDrop />
          </Box>
          
          {/* Decorative Elements */}
          <Box
            sx={{
              position: 'absolute',
              width: '300px',
              height: '300px',
              borderRadius: '50%',
              background: `radial-gradient(circle, ${alpha('#7c3aed', 0.1)} 0%, ${alpha('#7c3aed', 0)} 70%)`,
              filter: 'blur(60px)',
              top: '-100px',
              right: '-100px',
              zIndex: 0
            }}
          />
          
          <Box
            sx={{
              position: 'absolute',
              width: '400px',
              height: '400px',
              borderRadius: '50%',
              background: `radial-gradient(circle, ${alpha('#06b6d4', 0.08)} 0%, ${alpha('#06b6d4', 0)} 70%)`,
              filter: 'blur(80px)',
              bottom: '-150px',
              left: '-150px',
              zIndex: 0
            }}
          />
        </Box>
      </ModelsProvider>
    </ThemeProvider>
  );
}

export default App;
